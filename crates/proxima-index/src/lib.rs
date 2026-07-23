//! Vector indexes: exhaustive, approximate, and the harness that tells them
//! apart.
//!
//! Two implementations of one trait:
//!
//! * [`FlatIndex`] scans everything. Exact, trivially correct, and genuinely
//!   the right choice below a few tens of thousands of vectors. It is also the
//!   ground truth that every recall number in this repository is computed
//!   against.
//! * [`HnswIndex`] is a hierarchical navigable small world graph. Sub-linear,
//!   approximate, and tunable along an explicit recall/latency curve.
//!
//! # The claim this crate makes, and how to check it
//!
//! Semantic search infrastructure is usually a black box: you get results, you
//! have no idea what you did not get. [`recall_at_k`] is the answer, and it is
//! a function you can call on your own corpus in three lines rather than a
//! benchmark someone else ran on theirs:
//!
//! ```
//! use proxima_core::{Metadata, Metric, VectorId, VectorSpace};
//! use proxima_index::{recall_at_k, FlatIndex, HnswConfig, HnswIndex, VectorIndex};
//!
//! let space = VectorSpace::new(16, Metric::Cosine)?;
//! let mut flat = FlatIndex::new(space.clone());
//! let mut hnsw = HnswIndex::new(space, HnswConfig::new().with_seed(11))?;
//!
//! // A corpus with cluster structure, and every twentieth vector as a query.
//! let mut queries = Vec::new();
//! for i in 0..3000u64 {
//!     let cluster = (i % 30) as f32;
//!     let vector: Vec<f32> = (0..16)
//!         .map(|d| (cluster * 0.7 + d as f32 * 0.3).sin() + (i as f32 * 0.0007) * d as f32)
//!         .collect();
//!     flat.insert(VectorId::new(i), &vector, Metadata::new())?;
//!     hnsw.insert(VectorId::new(i), &vector, Metadata::new())?;
//!     if i % 20 == 0 {
//!         queries.extend_from_slice(&vector);
//!     }
//! }
//!
//! let report = recall_at_k(&hnsw, &flat, &queries, 10)?;
//! assert!(report.mean >= 0.95, "{report}");
//! assert!(hnsw.stats().mean_degree() > 8.0);
//! # Ok::<(), proxima_core::ProximaError>(())
//! ```
//!
//! # Choosing between them
//!
//! | | [`FlatIndex`] | [`HnswIndex`] |
//! |---|---|---|
//! | query | `O(n * dim)` | `O(ef * log n * dim)` |
//! | insert | `O(dim)` | `O(ef_construction * log n * dim)` |
//! | memory beyond the vectors | none | `~8 * M` bytes per vector |
//! | recall | 1.0, by construction | measured; see `BENCHMARKS.md` |
//! | filter cost | none | selective filters cost recall or latency |
//!
//! Measured on the benchmark machine at dimension 128: a flat scan of 50 000
//! vectors takes 1 049 µs and the graph takes 54.8 µs at `ef_search = 64` for
//! recall@10 of 0.988, having cost 6.1 s to build. Query latency alone crosses
//! over at about 1 000 vectors; total work crosses over once the graph has
//! served enough queries to repay its build, which is roughly 58 000 queries at
//! 1 000 vectors and 6 100 at 50 000. `BENCHMARKS.md` has the full tables and
//! the method.
#![forbid(unsafe_code)]

pub mod flat;
pub mod hnsw;
pub mod index;
pub mod ivfpq;
pub mod recall;

pub use flat::FlatIndex;
pub use hnsw::{HnswConfig, HnswGraph, HnswGraphParts, HnswIndex};
pub use index::{AcceptAll, Filter, IndexStats, VectorIndex};
pub use ivfpq::{IvfPqConfig, IvfPqIndex, IvfPqParts, IvfPqPartsOwned, TrainingReport};
pub use recall::{recall_at_k, recall_at_k_filtered, RecallReport};
