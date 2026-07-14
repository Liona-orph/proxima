//! The single error type the whole engine returns.

use crate::VectorId;

/// Result alias used throughout Proxima.
pub type Result<T> = std::result::Result<T, ProximaError>;

/// Everything that can go wrong inside Proxima.
///
/// One enum for the whole workspace rather than one per crate: a caller that
/// wraps an index behind a service wants a single `match`, and layered error
/// types would mean a `From` impl per boundary that erases exactly the fields
/// worth logging.
///
/// Every variant carries enough context to diagnose the fault **from a single
/// log line**. That is the design rule for this type: no variant is allowed to
/// say only "dimension mismatch", it has to say which dimensions and where. If
/// you find yourself adding a `String` message, add fields instead.
///
/// ```
/// use proxima_core::{ProximaError, VectorId};
///
/// let err = ProximaError::DimensionMismatch { expected: 768, actual: 512 };
/// assert_eq!(
///     err.to_string(),
///     "dimension mismatch: index holds 768-dimensional vectors, got 512"
/// );
///
/// let err = ProximaError::UnknownId { id: VectorId::new(42) };
/// assert_eq!(err.to_string(), "unknown vector id 42");
/// ```
// No `Eq`: `NonFinite` carries an `f32`, and the whole point of that variant is
// that the value may be NaN, which is not equal to itself. `PartialEq` is kept
// because tests compare errors and the NaN case is matched structurally.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ProximaError {
    /// A vector was presented whose length is not the index's dimension.
    ///
    /// By far the most common integration mistake — usually an embedding model
    /// swapped for one with a different output width — so it names both sides.
    #[error("dimension mismatch: index holds {expected}-dimensional vectors, got {actual}")]
    DimensionMismatch {
        /// Dimension the index was configured with.
        expected: usize,
        /// Dimension of the vector that was supplied.
        actual: usize,
    },

    /// A lookup, update or delete named an id the index does not hold.
    ///
    /// Also returned for an id that was inserted and then removed: a tombstoned
    /// vector is indistinguishable from one that never existed, by design.
    #[error("unknown vector id {id}")]
    UnknownId {
        /// The id that was not found.
        id: VectorId,
    },

    /// An id that is already present was inserted again.
    ///
    /// Insert is not an upsert. Silently replacing would hide a duplicated
    /// ingest pipeline, which is the far more likely cause; callers who really
    /// do want replace semantics have [`VectorIndex::upsert`] (in
    /// `proxima-index`) or can call `remove` first.
    ///
    /// [`VectorIndex::upsert`]: https://docs.rs/proxima-index
    #[error("duplicate vector id {id}")]
    DuplicateId {
        /// The id that was already present.
        id: VectorId,
    },

    /// A structural limit was reached.
    ///
    /// Slots are addressed by `u32`, so an index tops out at `u32::MAX - 1`
    /// live vectors; the variant names the limit rather than the count so the
    /// log line tells you which limit without a lookup.
    #[error("capacity exceeded: {what} limit is {limit}")]
    Capacity {
        /// Which structure ran out of room, e.g. `"slot allocator"`.
        what: &'static str,
        /// The limit that was hit.
        limit: usize,
    },

    /// A vector contains a value that is not finite.
    ///
    /// Rejected at the boundary rather than tolerated. A single NaN in stored
    /// data makes every distance involving that vector NaN, which sorts to the
    /// bottom of the result heap and so silently removes the vector from every
    /// search — the worst kind of bug, because recall degrades without an
    /// error. See the module docs of [`crate::distance`] for the full policy.
    #[error("vector component {index} is not finite ({value})")]
    NonFinite {
        /// Position of the offending component within the vector.
        index: usize,
        /// The value found there, formatted for the log.
        value: f32,
    },

    /// Persisted or in-memory state failed an internal consistency check.
    ///
    /// Names the structure and the offending value, because the useful question
    /// is always "which invariant, and how far off".
    #[error("corrupt {structure}: {detail}")]
    Corruption {
        /// The structure that failed its check, e.g. `"hnsw layer 0"`.
        structure: &'static str,
        /// What specifically was wrong.
        detail: String,
    },

    /// A configuration value is outside its legal range, or a combination of
    /// values is inconsistent.
    ///
    /// Always reported at construction time, never at query time: a
    /// misconfigured index that builds and then quietly returns bad results is
    /// the failure this variant exists to make impossible.
    #[error("invalid configuration: {parameter} = {value} ({reason})")]
    Configuration {
        /// Name of the parameter, matching the field name in the config struct.
        parameter: &'static str,
        /// The offending value, formatted.
        value: String,
        /// Why it is not acceptable, phrased as the constraint that was broken.
        reason: &'static str,
    },

    /// An invariant that this crate is responsible for was violated.
    ///
    /// Reaching this is a bug in Proxima, not in the caller. It exists so that
    /// internal invariants can be *checked* on paths where panicking would take
    /// down a caller's server.
    #[error("internal invariant violated in {location}: {detail}")]
    Internal {
        /// Module or function where the invariant is maintained.
        location: &'static str,
        /// The invariant, stated as what was expected.
        detail: String,
    },

    /// A filesystem operation failed.
    ///
    /// Flattened from [`std::io::Error`] rather than wrapping it, because this
    /// enum is `Clone + PartialEq` and `io::Error` is neither. The cost is that
    /// the `ErrorKind` is only available as text; the gain is that a caller can
    /// still compare, clone and log errors uniformly, which every layer above
    /// `proxima-store` relies on. The `path` and `operation` are what actually
    /// make an I/O failure diagnosable, and both survive.
    #[error("{operation} failed on {path}: {detail}")]
    Io {
        /// What was being attempted, e.g. `"append to wal"`.
        operation: &'static str,
        /// The path involved, or `"<unnamed>"` for an in-memory target.
        path: String,
        /// The underlying `io::Error`, rendered.
        detail: String,
    },
}

impl ProximaError {
    /// Build a [`ProximaError::Configuration`] without formatting at the call site.
    ///
    /// Config validation is a long list of one-line checks; this keeps each one
    /// to a single readable line.
    pub fn config<V: std::fmt::Display>(
        parameter: &'static str,
        value: V,
        reason: &'static str,
    ) -> Self {
        ProximaError::Configuration {
            parameter,
            value: value.to_string(),
            reason,
        }
    }

    /// Build a [`ProximaError::Io`] from an `io::Error` and its context.
    ///
    /// Every I/O call site in the workspace goes through this, so that the
    /// operation and the path are never lost — an `io::Error` on its own says
    /// "No such file or directory" and nothing about which file.
    pub fn io(operation: &'static str, path: impl std::fmt::Display, err: &std::io::Error) -> Self {
        ProximaError::Io {
            operation,
            path: path.to_string(),
            detail: err.to_string(),
        }
    }

    /// Build a [`ProximaError::Corruption`] without formatting at the call site.
    pub fn corrupt(structure: &'static str, detail: impl std::fmt::Display) -> Self {
        ProximaError::Corruption {
            structure,
            detail: detail.to_string(),
        }
    }

    /// `true` if the error is the caller's fault rather than Proxima's.
    ///
    /// A service layer maps this straight onto 4xx versus 5xx, which is the
    /// only reason the distinction is exposed.
    pub fn is_caller_error(&self) -> bool {
        matches!(
            self,
            ProximaError::DimensionMismatch { .. }
                | ProximaError::UnknownId { .. }
                | ProximaError::DuplicateId { .. }
                | ProximaError::NonFinite { .. }
                | ProximaError::Configuration { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    // Exact float comparison is deliberate here: every constant below is
    // exactly representable and the point is that the kernel returns it
    // bit-for-bit.
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn every_variant_names_its_operands() {
        // The rule for this type is that the Display output alone is enough to
        // diagnose the fault. These assertions are the executable form of it.
        let cases: Vec<(ProximaError, &[&str])> = vec![
            (
                ProximaError::DimensionMismatch {
                    expected: 768,
                    actual: 512,
                },
                &["768", "512"],
            ),
            (
                ProximaError::UnknownId {
                    id: VectorId::new(7),
                },
                &["7"],
            ),
            (
                ProximaError::DuplicateId {
                    id: VectorId::new(9),
                },
                &["9"],
            ),
            (
                ProximaError::Capacity {
                    what: "slot allocator",
                    limit: 4_294_967_294,
                },
                &["slot allocator", "4294967294"],
            ),
            (
                ProximaError::NonFinite {
                    index: 3,
                    value: f32::NAN,
                },
                &["3", "NaN"],
            ),
            (
                ProximaError::Corruption {
                    structure: "hnsw layer 0",
                    detail: "neighbour 12 out of range".into(),
                },
                &["hnsw layer 0", "neighbour 12"],
            ),
            (
                ProximaError::config("ef_search", 0, "must be at least k"),
                &["ef_search", "0", "at least k"],
            ),
            (
                ProximaError::Internal {
                    location: "hnsw::connect",
                    detail: "level 4 above max 3".into(),
                },
                &["hnsw::connect", "level 4"],
            ),
        ];
        for (err, fragments) in cases {
            let rendered = err.to_string();
            for fragment in fragments {
                assert!(
                    rendered.contains(fragment),
                    "{rendered:?} does not mention {fragment:?}"
                );
            }
        }
    }

    #[test]
    fn caller_errors_are_separated_from_bugs() {
        assert!(ProximaError::UnknownId {
            id: VectorId::new(1)
        }
        .is_caller_error());
        assert!(!ProximaError::Internal {
            location: "x",
            detail: String::new()
        }
        .is_caller_error());
    }
}
