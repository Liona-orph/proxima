//! [`FlatIndex`]: exhaustive search, and the reference every other index is
//! measured against.

use proxima_core::{
    Metadata, ProximaError, Result, SearchResult, TopK, VectorId, VectorSpace, VectorStore,
};

use crate::index::{check_k, Filter, IndexStats, VectorIndex};

/// Exhaustive nearest-neighbour search over a flat vector array.
///
/// # This is not a placeholder
///
/// It is easy to read "brute force" as "the version we ship before the real
/// one". It is neither of the two things it looks like:
///
/// **It is the ground truth.** Recall is defined as agreement with the exact
/// answer, so measuring an approximate index requires an exact one to measure
/// against. [`recall_at_k`](crate::recall_at_k) takes a `FlatIndex` for exactly
/// this reason, and without it every accuracy claim in this repository would be
/// unfalsifiable.
///
/// **It is often the right choice below a few tens of thousands of vectors.**
/// An exhaustive scan is a linear read of one contiguous array at close to
/// memory bandwidth (23.3 GiB/s measured), with no pointer chasing, no graph to
/// build and no parameters to get wrong.
///
/// The *query latency* crossover on the benchmark machine is around 1 000
/// vectors of dimension 128 — but that is the wrong number to decide on,
/// because HNSW's build cost has to be amortised. At 1 000 vectors the graph
/// costs 75 ms to build and saves 1.3 µs per query, so it needs about 58 000
/// queries before it is ahead; at 50 000 vectors it needs about 6 100. If your
/// corpus is a product catalogue that is rebuilt nightly and queried a few
/// thousand times a day, `FlatIndex` is not a compromise. See `BENCHMARKS.md`.
///
/// # Guarantees
///
/// Exact, deterministic, and unaffected by filter selectivity: a filter that
/// rejects 99.9 % of the corpus costs exactly the same as one that rejects
/// nothing, because both scan everything. That property is worth remembering —
/// it is the one thing flat search does better than a graph.
///
/// Complexity: `O(n * dim)` per query, `O(1)` amortised per insert and delete.
///
/// ```
/// use proxima_core::{Metadata, Metric, VectorId, VectorSpace};
/// use proxima_index::{FlatIndex, VectorIndex};
///
/// let space = VectorSpace::new(2, Metric::Euclidean)?;
/// let mut index = FlatIndex::new(space);
///
/// index.insert(VectorId::new(1), &[0.0, 0.0], Metadata::new())?;
/// index.insert(VectorId::new(2), &[1.0, 1.0], Metadata::new().with("tag", "b"))?;
/// index.insert(VectorId::new(3), &[5.0, 5.0], Metadata::new().with("tag", "b"))?;
///
/// let hits = index.search_all(&[0.9, 0.9], 2)?;
/// assert_eq!(hits[0].id, VectorId::new(2));
/// assert_eq!(hits[1].id, VectorId::new(1));
///
/// // Filters run during the scan, so `k` results are still `k` results.
/// let tagged = index.search(&[0.9, 0.9], 2, &|_id, m: &Metadata| m.text("tag") == Some("b"))?;
/// assert_eq!(tagged.len(), 2);
/// assert_eq!(tagged[0].id, VectorId::new(2));
/// assert_eq!(tagged[1].id, VectorId::new(3));
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
#[derive(Debug, Clone)]
pub struct FlatIndex {
    store: VectorStore,
}

impl FlatIndex {
    /// An empty index over `space`.
    pub fn new(space: VectorSpace) -> Self {
        FlatIndex {
            store: VectorStore::new(space),
        }
    }

    /// An empty index with room for `capacity` vectors reserved.
    pub fn with_capacity(space: VectorSpace, capacity: usize) -> Self {
        FlatIndex {
            store: VectorStore::with_capacity(space, capacity),
        }
    }

    /// The space this index is over.
    pub fn space(&self) -> &VectorSpace {
        self.store.space()
    }

    /// The stored (prepared) vector for `id`.
    ///
    /// Under [`Metric::Cosine`](proxima_core::Metric::Cosine) this is the
    /// *normalised* vector, not the one that was passed to
    /// [`VectorIndex::insert`]. Proxima does not keep the original: it would
    /// double the memory for a copy nothing reads.
    pub fn vector(&self, id: VectorId) -> Option<&[f32]> {
        self.store.get(id)
    }

    /// Metadata for `id`.
    pub fn metadata(&self, id: VectorId) -> Option<&Metadata> {
        self.store.metadata_of(id)
    }

    /// Replace the metadata for `id` without touching its vector.
    pub fn set_metadata(&mut self, id: VectorId, metadata: Metadata) -> Result<()> {
        self.store.set_metadata(id, metadata)
    }

    /// Live ids, in insertion (slot) order.
    pub fn ids(&self) -> impl Iterator<Item = VectorId> + '_ {
        self.store.iter().map(|(id, _)| id)
    }

    /// Drop every vector, keeping the space.
    pub fn clear(&mut self) {
        self.store.clear();
    }

    /// The underlying store, for callers building their own scan.
    pub fn store(&self) -> &VectorStore {
        &self.store
    }

    /// Wrap an existing store.
    ///
    /// The whole of a flat index's state *is* its store, so this is also its
    /// deserialisation constructor: `proxima-store` restores the store and
    /// hands it over. There is nothing else to check, which is the reason the
    /// snapshot format has no flat-specific section.
    pub fn from_store(store: VectorStore) -> Self {
        FlatIndex { store }
    }
}

impl VectorIndex for FlatIndex {
    fn insert(&mut self, id: VectorId, vector: &[f32], metadata: Metadata) -> Result<()> {
        let prepared = self.store.space().prepare(vector)?;
        self.store.insert(id, &prepared, metadata)?;
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<()> {
        // Flat search has no edges pointing at the slot, so it can be recycled
        // immediately. This is the whole reason `VectorStore` splits `remove`
        // from `detach`: a graph index cannot do this.
        self.store.remove(id)?;
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize, filter: &dyn Filter) -> Result<Vec<SearchResult>> {
        check_k(k)?;
        let space = self.store.space();
        let prepared = space.prepare(query)?;

        // Bounded heap, not sort-then-truncate: see `proxima_core::topk`. On a
        // million-vector scan this is the difference between one 16-element
        // heap and a million-element vector.
        let mut top = TopK::new(k);
        for (id, slot) in self.store.iter() {
            if !filter.accepts(id, self.store.metadata(slot)) {
                continue;
            }
            top.push(id, space.distance(&prepared, self.store.vector(slot)));
        }
        Ok(top.into_sorted_vec())
    }

    fn contains(&self, id: VectorId) -> bool {
        self.store.contains(id)
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn stats(&self) -> IndexStats {
        IndexStats {
            len: self.store.len(),
            // A flat index frees slots on delete, so it never has tombstones.
            deleted: 0,
            dim: self.store.dim(),
            metric: self.store.space().metric(),
            layers: 1,
            nodes_per_layer: vec![self.store.len()],
            edges: 0,
            vector_bytes: self.store.vector_bytes(),
            graph_bytes: 0,
        }
    }
}

impl FlatIndex {
    /// Exact nearest neighbours as raw ids, for the recall harness.
    ///
    /// Skips the [`SearchResult`] wrapper because the harness only ever
    /// compares id sets, and building `Vec<SearchResult>` for a hundred queries
    /// against ten thousand vectors is measurable in the harness's own runtime.
    pub(crate) fn exact_ids(
        &self,
        query: &[f32],
        k: usize,
        filter: &dyn Filter,
    ) -> Result<Vec<VectorId>> {
        Ok(self
            .search(query, k, filter)?
            .into_iter()
            .map(|hit| hit.id)
            .collect())
    }
}

/// Convenience constructor used throughout the tests and benchmarks.
impl TryFrom<(&VectorSpace, &[f32])> for FlatIndex {
    type Error = ProximaError;

    /// Build a flat index from a flat `n * dim` array, assigning ids `0..n`.
    fn try_from((space, data): (&VectorSpace, &[f32])) -> Result<Self> {
        let dim = space.dim();
        if dim == 0 || data.len() % dim != 0 {
            return Err(ProximaError::config(
                "data",
                data.len(),
                "must be a multiple of the space dimension",
            ));
        }
        let count = data.len() / dim;
        let mut index = FlatIndex::with_capacity(space.clone(), count);
        for (i, vector) in data.chunks_exact(dim).enumerate() {
            index.insert(VectorId::new(i as u64), vector, Metadata::new())?;
        }
        Ok(index)
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
    use proxima_core::Metric;

    fn index(dim: usize, metric: Metric) -> FlatIndex {
        FlatIndex::new(VectorSpace::new(dim, metric).unwrap())
    }

    #[test]
    fn round_trip() {
        let mut index = index(2, Metric::Euclidean);
        index
            .insert(VectorId::new(1), &[3.0, 4.0], Metadata::new())
            .unwrap();
        assert!(index.contains(VectorId::new(1)));
        assert_eq!(index.vector(VectorId::new(1)), Some(&[3.0, 4.0][..]));
        assert_eq!(index.search_all(&[3.0, 4.0], 1).unwrap()[0].distance, 0.0);
        index.remove(VectorId::new(1)).unwrap();
        assert!(!index.contains(VectorId::new(1)));
        assert!(index.is_empty());
    }

    #[test]
    fn empty_index_returns_nothing_rather_than_an_error() {
        let index = index(3, Metric::Euclidean);
        assert!(index.search_all(&[1.0, 2.0, 3.0], 10).unwrap().is_empty());
    }

    #[test]
    fn k_larger_than_the_index_returns_everything() {
        let mut index = index(1, Metric::Euclidean);
        for i in 0..3u64 {
            index
                .insert(VectorId::new(i), &[i as f32], Metadata::new())
                .unwrap();
        }
        assert_eq!(index.search_all(&[0.0], 100).unwrap().len(), 3);
    }

    #[test]
    fn k_zero_returns_nothing() {
        let mut index = index(1, Metric::Euclidean);
        index
            .insert(VectorId::new(1), &[0.0], Metadata::new())
            .unwrap();
        assert!(index.search_all(&[0.0], 0).unwrap().is_empty());
    }

    #[test]
    fn all_identical_vectors_tie_break_on_id() {
        let mut index = index(2, Metric::Euclidean);
        for i in [9u64, 4, 7, 1] {
            index
                .insert(VectorId::new(i), &[1.0, 1.0], Metadata::new())
                .unwrap();
        }
        let ids: Vec<u64> = index
            .search_all(&[1.0, 1.0], 3)
            .unwrap()
            .into_iter()
            .map(|h| h.id.get())
            .collect();
        assert_eq!(ids, vec![1, 4, 7]);
    }

    #[test]
    fn duplicate_insert_is_an_error() {
        let mut index = index(1, Metric::Euclidean);
        index
            .insert(VectorId::new(1), &[0.0], Metadata::new())
            .unwrap();
        assert!(matches!(
            index.insert(VectorId::new(1), &[1.0], Metadata::new()),
            Err(ProximaError::DuplicateId { .. })
        ));
        // ...but upsert replaces.
        index
            .upsert(VectorId::new(1), &[1.0], Metadata::new())
            .unwrap();
        assert_eq!(index.vector(VectorId::new(1)), Some(&[1.0][..]));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn cosine_vectors_are_normalised_on_the_way_in() {
        let mut index = index(2, Metric::Cosine);
        index
            .insert(VectorId::new(1), &[3.0, 4.0], Metadata::new())
            .unwrap();
        let stored = index.vector(VectorId::new(1)).unwrap();
        assert!((stored[0] - 0.6).abs() < 1e-6 && (stored[1] - 0.8).abs() < 1e-6);
        // ...and so is the query, so magnitude cannot affect the ranking.
        let a = index.search_all(&[30.0, 40.0], 1).unwrap()[0].distance;
        let b = index.search_all(&[0.3, 0.4], 1).unwrap()[0].distance;
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn filter_runs_during_the_scan_so_k_results_are_k_results() {
        let mut index = index(1, Metric::Euclidean);
        for i in 0..100u64 {
            let meta = Metadata::new().with("keep", i % 10 == 0);
            index.insert(VectorId::new(i), &[i as f32], meta).unwrap();
        }
        let hits = index
            .search(&[0.0], 5, &|_id, m: &Metadata| m.bool("keep") == Some(true))
            .unwrap();
        assert_eq!(hits.len(), 5);
        let ids: Vec<u64> = hits.into_iter().map(|h| h.id.get()).collect();
        assert_eq!(ids, vec![0, 10, 20, 30, 40]);
    }

    #[test]
    fn dimension_mismatch_is_caught_on_both_sides() {
        let mut index = index(3, Metric::Euclidean);
        assert!(matches!(
            index.insert(VectorId::new(1), &[1.0], Metadata::new()),
            Err(ProximaError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            index.search_all(&[1.0, 2.0], 1),
            Err(ProximaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn stats_report_no_tombstones_and_no_graph() {
        let mut index = index(4, Metric::Euclidean);
        for i in 0..8u64 {
            index
                .insert(VectorId::new(i), &[i as f32; 4], Metadata::new())
                .unwrap();
        }
        index.remove(VectorId::new(3)).unwrap();
        let stats = index.stats();
        assert_eq!(stats.len, 7);
        assert_eq!(stats.deleted, 0);
        assert_eq!(stats.graph_bytes, 0);
        assert_eq!(stats.layers, 1);
        assert!(stats.vector_bytes >= 8 * 4 * 4);
    }

    #[test]
    fn removal_recycles_slots() {
        let mut index = index(2, Metric::Euclidean);
        for i in 0..4u64 {
            index
                .insert(VectorId::new(i), &[i as f32; 2], Metadata::new())
                .unwrap();
        }
        let bytes = index.stats().vector_bytes;
        for round in 0..200u64 {
            index.remove(VectorId::new(round % 4)).unwrap();
            index
                .insert(
                    VectorId::new(round % 4),
                    &[round as f32; 2],
                    Metadata::new(),
                )
                .unwrap();
        }
        assert_eq!(index.stats().vector_bytes, bytes);
    }

    #[test]
    fn try_from_builds_from_a_flat_array() {
        let space = VectorSpace::new(2, Metric::Euclidean).unwrap();
        let index = FlatIndex::try_from((&space, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0][..])).unwrap();
        assert_eq!(index.len(), 3);
        assert_eq!(
            index.search_all(&[2.0, 2.0], 1).unwrap()[0].id,
            VectorId::new(2)
        );
        assert!(FlatIndex::try_from((&space, &[0.0, 0.0, 1.0][..])).is_err());
    }
}
