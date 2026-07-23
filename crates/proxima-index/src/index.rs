//! The [`VectorIndex`] trait, the [`Filter`] predicate and [`IndexStats`].

use proxima_core::{Metadata, Metric, ProximaError, Result, SearchResult, VectorId};

/// A predicate evaluated against a candidate's metadata during search.
///
/// # Where this is applied, and why it matters
///
/// A filter is applied **during** traversal, not to the result list afterwards.
/// The difference is not an optimisation, it is a correctness property, and it
/// is spelled out in the [`HnswIndex`](crate::HnswIndex) docs. In short:
/// post-filtering a `k`-element result list returns fewer than `k` results
/// whenever the filter rejects anything, and there is no `k` you can search for
/// that fixes it in general.
///
/// Implemented for any `Fn(VectorId, &Metadata) -> bool`, so a closure is a
/// filter:
///
/// ```
/// use proxima_core::{Metadata, VectorId};
/// use proxima_index::Filter;
///
/// let recent = |_id: VectorId, meta: &Metadata| meta.int("year").is_some_and(|y| y >= 2024);
/// assert!(recent.accepts(VectorId::new(1), &Metadata::new().with("year", 2025_i64)));
/// assert!(!recent.accepts(VectorId::new(1), &Metadata::new()));
/// ```
pub trait Filter {
    /// `true` if this candidate may appear in the results.
    ///
    /// Called once per candidate examined — tens of thousands of times per
    /// query on a large graph — so it must be cheap. Anything that needs a
    /// network round trip or a lock belongs outside the search, applied to the
    /// returned ids.
    fn accepts(&self, id: VectorId, metadata: &Metadata) -> bool;
}

/// The filter that accepts everything.
///
/// A unit struct rather than `Option<&dyn Filter>` throughout: the branch is
/// predictable and the call devirtualises for the concrete-typed helpers, and
/// it keeps one code path in the search loops instead of two.
#[derive(Debug, Clone, Copy, Default)]
pub struct AcceptAll;

impl Filter for AcceptAll {
    #[inline]
    fn accepts(&self, _id: VectorId, _metadata: &Metadata) -> bool {
        true
    }
}

impl<F> Filter for F
where
    F: Fn(VectorId, &Metadata) -> bool,
{
    #[inline]
    fn accepts(&self, id: VectorId, metadata: &Metadata) -> bool {
        self(id, metadata)
    }
}

/// A snapshot of an index's size and shape.
///
/// Everything here is O(1) or O(slots) to compute and none of it is cached, so
/// calling [`VectorIndex::stats`] in a hot loop is a mistake; calling it once
/// per minute from a metrics exporter is exactly what it is for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStats {
    /// Vectors that are live and searchable.
    pub len: usize,
    /// Tombstoned vectors: removed, still occupying a slot and still traversed
    /// through. Always zero for [`FlatIndex`](crate::FlatIndex).
    ///
    /// The ratio of this to `len` is the number to alert on: past roughly 20 %
    /// the graph is spending a fifth of its distance evaluations on vectors
    /// that can never be returned, and it is time to
    /// [`compact`](crate::HnswIndex::compact).
    pub deleted: usize,
    /// Dimension of the space.
    pub dim: usize,
    /// Metric of the space.
    pub metric: Metric,
    /// Number of graph layers, or 1 for a flat index.
    pub layers: usize,
    /// Nodes present on each layer, index 0 being the base layer.
    ///
    /// Should decay by roughly a factor of `M` per level. A layer histogram
    /// that decays much faster or slower than that means the level generator
    /// and `M` disagree, which is a configuration bug that costs recall.
    pub nodes_per_layer: Vec<usize>,
    /// Directed edges on the base layer.
    pub edges: usize,
    /// Bytes held by the vector array.
    pub vector_bytes: usize,
    /// Bytes held by the graph, if any.
    pub graph_bytes: usize,
}

impl IndexStats {
    /// Mean out-degree on the base layer.
    ///
    /// The single most diagnostic number about an HNSW graph: it should settle
    /// near `M` for a well-built index. Well below means the pruning heuristic
    /// is discarding too much; at exactly `2 * M` for most nodes, the graph is
    /// saturated and `M` is probably too small for the corpus.
    pub fn mean_degree(&self) -> f32 {
        if self.len + self.deleted == 0 {
            return 0.0;
        }
        self.edges as f32 / (self.len + self.deleted) as f32
    }

    /// Total heap bytes: vectors plus graph.
    pub fn total_bytes(&self) -> usize {
        self.vector_bytes + self.graph_bytes
    }

    /// Bytes per live vector, graph included.
    ///
    /// The number to compare against `dim * 4`. An HNSW index at `M = 16` costs
    /// roughly 128 bytes of graph per vector, which is 4 % overhead at
    /// dimension 768 and 100 % overhead at dimension 32 — the second case is
    /// where [`FlatIndex`](crate::FlatIndex) starts winning outright.
    pub fn bytes_per_vector(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        self.total_bytes() as f32 / self.len as f32
    }
}

/// The operations every Proxima index supports.
///
/// # What an implementation must guarantee
///
/// * **`search` returns at most `k` results, ordered nearest first**, and every
///   returned id is live and accepted by the filter. It may return *fewer* than
///   `k` only when fewer than `k` live vectors pass the filter.
/// * **`search` is deterministic.** The same index, query, `k` and filter give
///   the same results in the same order, in every process and on every run.
///   Ties in distance are broken by id.
/// * **Scores are the space's ranking scores**, so Euclidean distances are
///   squared and inner products negated. See
///   [`Metric::to_true_distance`](proxima_core::Metric::to_true_distance).
///
/// What an implementation does *not* have to guarantee is that the results are
/// the true nearest neighbours. [`FlatIndex`](crate::FlatIndex) is exact;
/// [`HnswIndex`](crate::HnswIndex) is approximate, and
/// [`recall_at_k`](crate::recall_at_k) is how you find out by how much.
///
/// The trait is object safe, so `Box<dyn VectorIndex>` works and a caller can
/// swap a flat index for a graph one behind the same handle — which is the
/// intended migration path as a corpus grows.
pub trait VectorIndex {
    /// Add a vector.
    ///
    /// The vector is validated and, if the metric requires it, normalised;
    /// callers pass raw vectors and do not need to know which metrics
    /// normalise.
    ///
    /// Returns [`ProximaError::DuplicateId`] if `id` is already present. This
    /// is not an upsert; see [`VectorIndex::upsert`].
    fn insert(&mut self, id: VectorId, vector: &[f32], metadata: Metadata) -> Result<()>;

    /// Remove a vector.
    ///
    /// Returns [`ProximaError::UnknownId`] if it is not present.
    fn remove(&mut self, id: VectorId) -> Result<()>;

    /// The `k` nearest live vectors accepted by `filter`, nearest first.
    fn search(&self, query: &[f32], k: usize, filter: &dyn Filter) -> Result<Vec<SearchResult>>;

    /// `true` if `id` is live in this index.
    fn contains(&self, id: VectorId) -> bool;

    /// Number of live vectors.
    fn len(&self) -> usize;

    /// `true` if there are no live vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A snapshot of the index's size and shape.
    fn stats(&self) -> IndexStats;

    /// Insert, replacing any existing vector with the same id.
    ///
    /// Implemented as remove-then-insert. On a graph index that means the
    /// vector's old node is tombstoned and a new one is linked in, so an index
    /// that is upserted heavily accumulates tombstones and needs periodic
    /// compaction — which is the honest cost of upsert on a graph, and the
    /// reason it is a separate method rather than `insert`'s behaviour.
    fn upsert(&mut self, id: VectorId, vector: &[f32], metadata: Metadata) -> Result<()> {
        if self.contains(id) {
            self.remove(id)?;
        }
        self.insert(id, vector, metadata)
    }

    /// [`VectorIndex::search`] with no filter.
    fn search_all(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search(query, k, &AcceptAll)
    }
}

/// Shared validation of `k`, so both implementations reject the same inputs.
///
/// `k == 0` is legal and returns an empty result: it is a meaningful degenerate
/// query and every caller that has to special-case it is a caller that will
/// eventually forget to.
pub(crate) fn check_k(k: usize) -> Result<()> {
    // `k` above this cannot be satisfied and almost certainly means a `usize`
    // underflow at the call site, which is worth catching loudly.
    const MAX_K: usize = 1 << 24;
    if k > MAX_K {
        return Err(ProximaError::config(
            "k",
            k,
            "exceeds the 16777216 result ceiling",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    // Exact float comparison is deliberate here: the constants are exactly
    // representable and the point is that the accessor returns them
    // bit-for-bit.
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn closures_are_filters() {
        let filter = |id: VectorId, _meta: &Metadata| id.get() % 2 == 0;
        assert!(filter.accepts(VectorId::new(2), &Metadata::new()));
        assert!(!filter.accepts(VectorId::new(3), &Metadata::new()));
    }

    #[test]
    fn accept_all_accepts_all() {
        assert!(AcceptAll.accepts(VectorId::new(u64::MAX), &Metadata::new()));
    }

    #[test]
    fn stats_arithmetic_survives_an_empty_index() {
        let stats = IndexStats {
            len: 0,
            deleted: 0,
            dim: 4,
            metric: Metric::Euclidean,
            layers: 1,
            nodes_per_layer: vec![0],
            edges: 0,
            vector_bytes: 0,
            graph_bytes: 0,
        };
        assert_eq!(stats.mean_degree(), 0.0);
        assert_eq!(stats.bytes_per_vector(), 0.0);
    }

    #[test]
    fn absurd_k_is_rejected() {
        assert!(check_k(0).is_ok());
        assert!(check_k(1 << 24).is_ok());
        assert!(matches!(
            check_k(usize::MAX),
            Err(ProximaError::Configuration { parameter: "k", .. })
        ));
    }
}
