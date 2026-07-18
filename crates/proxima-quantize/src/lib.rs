//! Vector compression for Proxima: scalar and product quantization.
//!
//! Both quantizers answer the same question — "how do I keep ten times more
//! vectors in RAM" — with different trade-offs, and both are *lossy*, which
//! means the only honest way to present them is with a measured recall figure.
//! `tests/recall.rs` produces one, and the summary is:
//!
//! | quantizer | code size (dim 128) | compression | recall@10 vs exact |
//! |---|--:|--:|--:|
//! | [`ScalarQuantizer`] (u8, per-dimension) | 128 B | 4x | 0.972 |
//! | [`ScalarQuantizer`] (u8, global range) | 128 B | 4x | 0.960 |
//! | [`ProductQuantizer`] `m = 32` | 32 B | 16x | 0.473 |
//! | [`ProductQuantizer`] `m = 16` | 16 B | 32x | 0.308 |
//! | [`ProductQuantizer`] `m = 8` | 8 B | 64x | 0.219 |
//! | [`ProductQuantizer`] `m = 8`, top-100 reranked exactly | 8 B | 64x | 0.833 |
//!
//! Measured on 5 000 clustered 128-dimensional vectors, 200 queries, seeded
//! generator, `cargo test --release -p proxima-quantize`. Those numbers are
//! asserted by the test suite with a small margin, so they cannot silently rot.
//!
//! The absolute PQ figures are low because the fixture is deliberately hostile
//! to it — isotropic Gaussian clusters in 128 dimensions have no
//! low-dimensional structure for a subspace codebook to exploit, and `tests/
//! recall.rs` explains why at length. The *shape* is what transfers: accuracy
//! falls steeply as `m` shrinks, and reranking the PQ shortlist with exact
//! distances recovers most of the loss for a hundred extra distance
//! evaluations per query. That combination — aggressive PQ to shortlist, exact
//! distances to rank — is how 64x compression is actually deployed, and it is
//! why the bare 0.219 is not the number to design against.
//!
//! # Choosing between them
//!
//! Use [`ScalarQuantizer`] when 4x is enough: it needs no meaningful training,
//! its error is uniform and bounded by half a lattice step, and its decode is
//! one multiply-add. Use [`ProductQuantizer`] when you need 16x or more and can
//! afford a training pass and a codebook. There is no configuration under which
//! SQ beats PQ on compression or PQ beats SQ on accuracy.
//!
//! ```
//! use proxima_core::{Metric, VectorSpace};
//! use proxima_quantize::{PqConfig, ProductQuantizer};
//!
//! let space = VectorSpace::new(16, Metric::Euclidean)?;
//! let training: Vec<f32> = (0..2048)
//!     .map(|i| ((i % 16) as f32 * 0.7 + (i / 16) as f32 * 0.01).sin())
//!     .collect();
//!
//! let pq = ProductQuantizer::train(&space, &training, PqConfig::new(4).with_seed(7))?;
//! let query = &training[..16];
//!
//! // One table per query; then every candidate costs four table lookups.
//! let table = pq.distance_table(query)?;
//! let mut best = f32::INFINITY;
//! for vector in training.chunks_exact(16) {
//!     best = best.min(table.squared_l2(&pq.encode(vector)?));
//! }
//! assert!(best < 1e-2, "the query is in the corpus, so it must match itself");
//! # Ok::<(), proxima_core::ProximaError>(())
//! ```
#![forbid(unsafe_code)]

pub mod kmeans;
pub mod product;
pub mod scalar;

pub use kmeans::{kmeans, Clustering};
pub use product::{DistanceTable, PqConfig, ProductQuantizer, CENTROIDS_PER_SUBSPACE};
pub use scalar::{CodeType, RangeMode, ScalarQuantizer};
