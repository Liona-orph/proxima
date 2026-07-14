//! The vocabulary of the Proxima vector search engine.
//!
//! This crate holds everything the index implementations and the quantizers
//! agree on, and nothing that searches: identifiers, the flat vector store,
//! the distance kernels, and the bounded structure that ranked results come
//! out of. It has one dependency (`thiserror`) and no opinion about how a
//! search is performed.
//!
//! # The four decisions worth knowing about
//!
//! **Vectors live in one flat `Vec<f32>`, addressed by slot.** A
//! `Vec<Vec<f32>>` would put a pointer chase in front of every distance
//! evaluation, and a graph search is nothing but unpredictable distance
//! evaluations. [`VectorStore`] has the full argument.
//!
//! **Slots are recycled.** [`SlotAllocator`] hands deleted slots back out, so
//! an index whose live set is constant has a constant footprint no matter how
//! many times it has been written to.
//!
//! **Distances are ranking scores, not distances.** [`Metric::Euclidean`]
//! returns the squared value and [`Metric::InnerProduct`] returns the negated
//! one, because both are monotone in the real thing and the search never looks
//! at the magnitude. [`Metric::to_true_distance`] converts for display.
//!
//! **Cosine is enforced, not conventional.** [`VectorSpace::prepare`]
//! normalises when the metric asks it to, so that [`VectorSpace::distance`] can
//! run the two-norms-free kernel. Getting this wrong produces plausible wrong
//! answers rather than errors, which is why it is the API's job and not the
//! caller's.
//!
//! ```
//! use proxima_core::{Metadata, Metric, TopK, VectorId, VectorSpace, VectorStore};
//!
//! let space = VectorSpace::new(3, Metric::Euclidean)?;
//! let mut store = VectorStore::with_capacity(space.clone(), 3);
//!
//! for (id, v) in [(1u64, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0]), (3, [0.9, 0.1, 0.0])] {
//!     let prepared = space.prepare(&v)?;
//!     store.insert(VectorId::new(id), &prepared, Metadata::new())?;
//! }
//!
//! // An exhaustive scan, written the way an index writes it: score every
//! // candidate, keep only the best two.
//! let query = space.prepare(&[1.0, 0.0, 0.0])?;
//! let mut top = TopK::new(2);
//! for (id, slot) in store.iter() {
//!     top.push(id, space.distance(&query, store.vector(slot)));
//! }
//!
//! let hits = top.into_sorted_vec();
//! assert_eq!(hits[0].id, VectorId::new(1));
//! assert_eq!(hits[1].id, VectorId::new(3));
//! // Euclidean scores are squared; ask for the presentable number.
//! assert!((Metric::Euclidean.to_true_distance(hits[1].distance) - 0.1414).abs() < 1e-3);
//! # Ok::<(), proxima_core::ProximaError>(())
//! ```
#![forbid(unsafe_code)]

pub mod distance;
pub mod error;
pub mod id;
pub mod metadata;
pub mod space;
pub mod storage;
pub mod topk;

pub use distance::{Metric, LANES};
pub use error::{ProximaError, Result};
pub use id::{Slot, SlotAllocator, VectorId, MAX_SLOTS};
pub use metadata::{Metadata, MetadataValue};
pub use space::VectorSpace;
pub use storage::{UnmappedSlots, VectorStore};
pub use topk::{SearchResult, TopK};
