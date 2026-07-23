//! [`HnswIndex`]: a hierarchical navigable small world graph.
//!
//! The approximate index. Everything in this module exists to make greedy
//! search over a graph find the true nearest neighbours nearly all of the time,
//! and "nearly" is measured by [`recall_at_k`](crate::recall_at_k) rather than
//! asserted.

use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use proxima_core::{
    Metadata, ProximaError, Result, SearchResult, Slot, TopK, VectorId, VectorSpace, VectorStore,
};

use crate::index::{check_k, Filter, IndexStats, VectorIndex};

/// Marks an unused neighbour position.
///
/// `u32::MAX` rather than `Option<u32>`: the base layer's adjacency array is
/// `slot_count * 2M` entries, so eight bytes per slot instead of four would
/// double the graph's footprint to save one comparison. `Slot` is documented as
/// topping out one below this value for exactly this reason.
const SENTINEL: u32 = u32::MAX;

/// Build and search parameters.
///
/// # What each one costs
///
/// | parameter | raises | costs |
/// |---|---|---|
/// | `m` | recall, robustness of the graph | memory (`~8M` bytes/vector), build time |
/// | `ef_construction` | graph quality, so recall at every `ef_search` | build time only |
/// | `ef_search` | recall | query latency, linearly |
///
/// `ef_search` is the runtime dial and the other two are baked in at build
/// time. That asymmetry is the practical reason to build with a generous
/// `ef_construction`: it is paid once and it raises the whole recall/latency
/// curve, whereas `ef_search` only moves you along it.
///
/// ```
/// use proxima_index::HnswConfig;
///
/// let config = HnswConfig::new()
///     .with_m(32)
///     .with_ef_construction(400)
///     .with_ef_search(128)
///     .with_seed(42);
/// assert!(config.validate().is_ok());
///
/// // ef_search below k would silently cap recall, so it is rejected outright.
/// assert!(HnswConfig::new().with_ef_search(0).validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswConfig {
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    max_level: usize,
    seed: u64,
    filter_ef_limit: usize,
    exhaustive_filter_fallback: bool,
    keep_pruned_connections: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        HnswConfig {
            // 16 is the value the HNSW paper recommends for most datasets and
            // the one every published benchmark uses as its baseline; departing
            // from it should be a measured decision, so it is the default.
            m: 16,
            // 200 is enough that the graph quality curve has flattened on every
            // corpus in the benchmark suite. Raising it to 400 changed measured
            // recall@10 by under half a point and doubled build time.
            ef_construction: 200,
            ef_search: 64,
            // 16 levels covers 16^16 vectors at M = 16 -- far beyond the u32
            // slot ceiling. The cap exists only to bound the level array's type
            // and to stop a pathological draw from allocating 200 empty layers.
            max_level: 16,
            seed: 0x50B0_1A15,
            // Four doublings from ef_search = 64 reaches 1024, which is where
            // the escalation stops paying for itself and the exhaustive
            // fallback is cheaper. See `search` for the whole story.
            filter_ef_limit: 1024,
            exhaustive_filter_fallback: true,
            keep_pruned_connections: true,
        }
    }
}

impl HnswConfig {
    /// The defaults, which are the HNSW paper's recommended values.
    pub fn new() -> Self {
        HnswConfig::default()
    }

    /// Neighbours kept per node above the base layer.
    ///
    /// The base layer keeps `2 * m`, following the paper: the base layer holds
    /// every vector and carries the whole burden of the final refinement, and
    /// its higher degree is what stops greedy descent from terminating in a
    /// local minimum.
    pub const fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Size of the dynamic candidate list during construction.
    pub const fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Default size of the dynamic candidate list during search.
    pub const fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Seed for the level generator.
    ///
    /// Two indexes built from the same seed with the same insertion order are
    /// byte-identical graphs. This is a guarantee, not an observation: the
    /// generator is ChaCha8, whose output stream is specified, and nothing else
    /// in construction consults a source of randomness or iterates a hash map.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Cap on how far a filtered search escalates `ef` before falling back.
    pub const fn with_filter_ef_limit(mut self, limit: usize) -> Self {
        self.filter_ef_limit = limit;
        self
    }

    /// Whether a filtered search that still cannot fill `k` falls back to an
    /// exhaustive scan.
    ///
    /// Turning this off converts a worst-case `O(n)` query into a result set
    /// that is quietly short of `k`. That is occasionally the right trade for a
    /// latency-critical service, and it should be a decision, so it is a flag
    /// rather than a hard-coded behaviour.
    pub const fn with_exhaustive_filter_fallback(mut self, enabled: bool) -> Self {
        self.exhaustive_filter_fallback = enabled;
        self
    }

    /// Whether neighbour selection tops up with pruned candidates.
    ///
    /// With this off, the diversity heuristic can leave a node with far fewer
    /// than `m` neighbours, which saves a little memory and costs recall on
    /// clustered data. On by default; exposed so the effect can be measured.
    pub const fn with_keep_pruned_connections(mut self, enabled: bool) -> Self {
        self.keep_pruned_connections = enabled;
        self
    }

    /// Neighbours per node above the base layer.
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Neighbours per node on the base layer, `2 * m`.
    pub const fn m0(&self) -> usize {
        self.m * 2
    }

    /// Dynamic candidate list size during construction.
    pub const fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// Default dynamic candidate list size during search.
    pub const fn ef_search(&self) -> usize {
        self.ef_search
    }

    /// Level generator seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Maximum layer index.
    pub const fn max_level(&self) -> usize {
        self.max_level
    }

    /// Cap on `ef` escalation under a filter.
    pub const fn filter_ef_limit(&self) -> usize {
        self.filter_ef_limit
    }

    /// Whether a filtered search falls back to an exhaustive scan.
    pub const fn exhaustive_filter_fallback(&self) -> bool {
        self.exhaustive_filter_fallback
    }

    /// Whether neighbour selection tops up with pruned candidates.
    pub const fn keep_pruned_connections(&self) -> bool {
        self.keep_pruned_connections
    }

    /// Check every parameter, so that a misconfigured index fails at
    /// construction rather than by returning bad results.
    pub fn validate(&self) -> Result<()> {
        if self.m < 2 {
            return Err(ProximaError::config(
                "m",
                self.m,
                "must be at least 2; a graph with degree 1 is a set of paths",
            ));
        }
        if self.m > 512 {
            return Err(ProximaError::config(
                "m",
                self.m,
                "above 512 the neighbour heuristic's O(m^2) inner loop dominates build time",
            ));
        }
        if self.ef_construction < self.m {
            return Err(ProximaError::config(
                "ef_construction",
                self.ef_construction,
                "must be at least m, or there are not enough candidates to select m neighbours from",
            ));
        }
        if self.ef_search == 0 {
            return Err(ProximaError::config(
                "ef_search",
                self.ef_search,
                "must be at least 1",
            ));
        }
        if self.max_level == 0 || self.max_level > 30 {
            return Err(ProximaError::config(
                "max_level",
                self.max_level,
                "must be between 1 and 30",
            ));
        }
        if self.filter_ef_limit < self.ef_search {
            return Err(ProximaError::config(
                "filter_ef_limit",
                self.filter_ef_limit,
                "must be at least ef_search, or escalation could not happen",
            ));
        }
        Ok(())
    }

    /// The level-generator scale, `1 / ln(m)`.
    ///
    /// This is the value that makes the layer sizes decay by a factor of `m`,
    /// which is what makes the descent `O(log n)`. It is derived from `m`
    /// rather than exposed, because the two are only correct together and an
    /// index whose `m_L` and `m` disagree loses recall in a way that is very
    /// hard to attribute.
    fn level_scale(&self) -> f64 {
        1.0 / (self.m as f64).ln()
    }
}

/// A `(distance, slot)` pair ordered nearest-first-when-reversed.
///
/// Its `Ord` is the reason search is reproducible: distances are compared with
/// `total_cmp` (a genuine total order, NaN-safe) and ties broken by slot, so no
/// heap operation can depend on insertion order.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    distance: f32,
    slot: Slot,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.slot.cmp(&other.slot))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A fast hasher for `u32` slot numbers.
///
/// `DefaultHasher` is SipHash-1-3, which is a good choice when the keys are
/// attacker-controlled and a bad one when they are dense integers generated by
/// this crate: it costs more than the hash table probe it protects. This is a
/// single multiply by an odd constant plus a xorshift, which spreads sequential
/// slot numbers across the table well enough that the probe sequences stay
/// short.
#[derive(Default)]
struct SlotHasher(u64);

impl Hasher for SlotHasher {
    fn write(&mut self, bytes: &[u8]) {
        // Only `write_u32` is ever used; this exists to satisfy the trait and
        // defers to the standard hasher so it cannot be silently wrong.
        let mut fallback = DefaultHasher::new();
        fallback.write(bytes);
        self.0 = fallback.finish();
    }

    fn write_u32(&mut self, value: u32) {
        self.0 = u64::from(value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 29)
    }
}

/// Nodes already examined by one search.
///
/// A `HashSet` rather than a bitmap indexed by slot: a bitmap is `O(n)` to
/// allocate and zero, so on a million-vector index it would cost about a
/// millisecond per query to prepare a structure that a typical search touches a
/// few thousand entries of. Membership testing is the only operation, so the
/// set's iteration order — the usual source of non-determinism — is never
/// observed.
type VisitedSet = HashSet<u32, BuildHasherDefault<SlotHasher>>;

/// A hierarchical navigable small world index.
///
/// # How it works, in the order it matters
///
/// **Layers.** Each vector is assigned a level from an exponential
/// distribution, so layer `l` holds roughly `n / m^l` vectors and the top layer
/// holds one. A search starts at the top, greedily walks to the local minimum
/// on each layer, and drops down — a coarse-to-fine descent that reaches the
/// query's neighbourhood in `O(log n)` hops instead of scanning.
///
/// **The neighbour selection heuristic.** When a new node picks its `m`
/// neighbours from the `ef_construction` candidates it found, it does *not*
/// simply keep the `m` closest. A candidate `c` is discarded if some
/// already-selected neighbour `r` is closer to `c` than the new node is: the
/// path `new -> r -> c` already reaches `c` cheaply, so a direct edge is
/// redundant. What that buys is **diversity**. Keeping the `m` nearest in a
/// clustered corpus points every edge into the same cluster; the graph becomes
/// a set of well-connected islands with almost no edges between them, and
/// greedy search — which only ever moves to a *closer* node — gets trapped in
/// the wrong island and returns confidently wrong answers. The heuristic costs
/// `O(m^2)` distance evaluations per node per layer and is the single largest
/// contributor to recall in this implementation; `tests/hnsw.rs` measures the
/// difference by turning it off.
///
/// **Bidirectional links.** Every edge is added in both directions, because a
/// search that arrives from the other side needs to be able to leave. When that
/// pushes a neighbour past its degree cap, its list is re-selected with the
/// same heuristic rather than truncated to the nearest — truncation would undo
/// the diversity the heuristic just established.
///
/// # Deletion and compaction
///
/// [`VectorIndex::remove`] is a **tombstone**: the id is unmapped and the node
/// stops being returned, but its slot, its vector and all of its edges stay.
/// They have to. The node is very likely on the shortest path between two live
/// nodes, and cutting it out would disconnect the graph — repairing the graph
/// properly means re-running neighbour selection for every node that pointed at
/// it, which is most of the cost of a rebuild for one deletion.
///
/// The cost is that a tombstoned node is still visited, still costs a distance
/// evaluation, and still occupies `dim * 4` bytes. [`HnswIndex::compact`]
/// reclaims it all by rebuilding from the live vectors. Watch
/// `stats().deleted / stats().len` and compact past roughly 20 %.
///
/// Compaction is **not** a no-op on the graph: it reinserts in slot order,
/// which is generally not the original insertion order, so the rebuilt graph
/// differs from the original. Results stay deterministic — the same sequence of
/// operations always produces the same graph — but "compact" does not mean
/// "unchanged".
///
/// # Filtering
///
/// See [`HnswIndex::search`], which documents the recall trap and this
/// implementation's answer to it.
///
/// ```
/// use proxima_core::{Metadata, Metric, VectorId, VectorSpace};
/// use proxima_index::{HnswConfig, HnswIndex, VectorIndex};
///
/// let space = VectorSpace::new(4, Metric::Euclidean)?;
/// let mut index = HnswIndex::new(space, HnswConfig::new().with_seed(7))?;
///
/// for i in 0..1000u64 {
///     let t = i as f32 * 0.01;
///     index.insert(
///         VectorId::new(i),
///         &[t, t.sin(), t.cos(), -t],
///         Metadata::new().with("even", i % 2 == 0),
///     )?;
/// }
///
/// let query = [5.0, 5.0f32.sin(), 5.0f32.cos(), -5.0];
/// let hits = index.search_all(&query, 5)?;
/// assert_eq!(hits.len(), 5);
/// assert_eq!(hits[0].id, VectorId::new(500));
///
/// // The filter runs during traversal, so five results means five results.
/// let even = index.search(&query, 5, &|_id, m: &Metadata| m.bool("even") == Some(true))?;
/// assert_eq!(even.len(), 5);
/// assert!(even.iter().all(|hit| hit.id.get() % 2 == 0));
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
#[derive(Debug, Clone)]
pub struct HnswIndex {
    store: VectorStore,
    config: HnswConfig,
    /// Level of each slot.
    levels: Vec<u8>,
    /// Base layer adjacency, `slot_count * m0` entries padded with [`SENTINEL`].
    ///
    /// One flat array rather than a `Vec<Vec<u32>>` for the same reason the
    /// vectors are flat: this is read once per visited node, and a pointer
    /// chase in front of it would be a second cache miss on the critical path.
    level0: Vec<u32>,
    /// Upper layer adjacency, one entry per slot holding `level * m` links.
    ///
    /// Here the `Vec<Vec<u32>>` *is* right: only about `1/m` of nodes have a
    /// level above zero, so the outer array is mostly empty `Vec`s costing no
    /// allocation, and a flat arena would either waste `max_level * m` entries
    /// per node or need an offset table and a compaction story of its own.
    upper: Vec<Vec<u32>>,
    entry: Option<Slot>,
    entry_level: usize,
    /// Level generator. Advanced exactly once per insert, which is what makes
    /// "same seed, same insertion order, same graph" true.
    rng: ChaCha8Rng,
    /// How many times `rng` has been drawn from.
    ///
    /// Carried purely so that a restored index can fast-forward the generator
    /// to where it was. Without it a reloaded index would draw the *first*
    /// level again for its next insert, and "same seed, same order, same graph"
    /// would hold for a fresh build and quietly stop holding across a restart —
    /// the worst kind of guarantee.
    draws: u64,
}

/// A borrowed view of an [`HnswIndex`]'s graph, for serialisation.
///
/// The graph is the part of an HNSW index that cannot be recomputed: rebuilding
/// from the same vectors in the same order reproduces it only if no vector was
/// ever deleted, because a tombstone leaves edges behind that a rebuild does
/// not create. A snapshot that stored only the vectors would therefore come
/// back as a *different* index that happens to hold the same data, and its
/// search results would differ. So the edges go on disk.
///
/// Borrowed rather than owned because it is read once, straight into a writer,
/// and the base layer alone is `slot_count * 2M * 4` bytes — 7.5 MiB for 50 000
/// vectors at `M = 16`, which is not worth copying to hand over.
#[derive(Debug, Clone, Copy)]
pub struct HnswGraph<'a> {
    /// Level of each slot, indexed by slot number.
    pub levels: &'a [u8],
    /// Base-layer adjacency, `slot_count * m0` entries padded with `u32::MAX`.
    pub level0: &'a [u32],
    /// Upper-layer adjacency, one `level * m` entry list per slot.
    pub upper: &'a [Vec<u32>],
    /// Entry point, or `None` for an empty index.
    pub entry: Option<Slot>,
    /// Level of the entry point.
    pub entry_level: usize,
    /// Draws taken from the level generator so far.
    pub levels_drawn: u64,
}

/// The owned form of [`HnswGraph`], for [`HnswIndex::from_parts`].
#[derive(Debug, Clone)]
pub struct HnswGraphParts {
    /// See [`HnswGraph::levels`].
    pub levels: Vec<u8>,
    /// See [`HnswGraph::level0`].
    pub level0: Vec<u32>,
    /// See [`HnswGraph::upper`].
    pub upper: Vec<Vec<u32>>,
    /// See [`HnswGraph::entry`].
    pub entry: Option<Slot>,
    /// See [`HnswGraph::entry_level`].
    pub entry_level: usize,
    /// See [`HnswGraph::levels_drawn`].
    pub levels_drawn: u64,
}

impl HnswIndex {
    /// An empty index.
    ///
    /// Validates `config` immediately; a bad parameter is a build-time error,
    /// never a silently degraded search.
    pub fn new(space: VectorSpace, config: HnswConfig) -> Result<Self> {
        config.validate()?;
        Ok(HnswIndex {
            store: VectorStore::new(space),
            levels: Vec::new(),
            level0: Vec::new(),
            upper: Vec::new(),
            entry: None,
            entry_level: 0,
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            draws: 0,
            config,
        })
    }

    /// An empty index with room for `capacity` vectors reserved.
    pub fn with_capacity(space: VectorSpace, config: HnswConfig, capacity: usize) -> Result<Self> {
        config.validate()?;
        Ok(HnswIndex {
            store: VectorStore::with_capacity(space, capacity),
            levels: Vec::with_capacity(capacity),
            level0: Vec::with_capacity(capacity * config.m0()),
            upper: Vec::with_capacity(capacity),
            entry: None,
            entry_level: 0,
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            draws: 0,
            config,
        })
    }

    /// The configuration in force.
    pub const fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// The space this index is over.
    pub fn space(&self) -> &VectorSpace {
        self.store.space()
    }

    /// The stored (prepared) vector for `id`.
    pub fn vector(&self, id: VectorId) -> Option<&[f32]> {
        self.store.get(id)
    }

    /// Metadata for `id`.
    pub fn metadata(&self, id: VectorId) -> Option<&Metadata> {
        self.store.metadata_of(id)
    }

    /// Replace the metadata for `id` without touching its vector or its edges.
    ///
    /// Cheap, and the reason filters read metadata rather than closing over an
    /// external map: re-tagging a corpus does not require rebuilding the graph.
    pub fn set_metadata(&mut self, id: VectorId, metadata: Metadata) -> Result<()> {
        self.store.set_metadata(id, metadata)
    }

    /// Live ids, in slot order.
    pub fn ids(&self) -> impl Iterator<Item = VectorId> + '_ {
        self.store.iter().map(|(id, _)| id)
    }

    /// Tombstoned slots: removed vectors still occupying memory and still
    /// traversed through.
    pub fn tombstones(&self) -> usize {
        self.store.slot_count() - self.store.len()
    }

    /// Search with an explicit `ef`, overriding [`HnswConfig::ef_search`].
    ///
    /// The knob that draws a recall/latency curve without rebuilding anything.
    /// `ef` below `k` is raised to `k` — returning fewer results than asked for
    /// because a parameter was too small is exactly the silent failure this
    /// crate is trying to avoid.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: &dyn Filter,
    ) -> Result<Vec<SearchResult>> {
        check_k(k)?;
        let prepared = self.store.space().prepare(query)?;
        Ok(self.search_prepared(&prepared, k, ef.max(k).max(1), filter))
    }

    /// Rebuild the graph from the live vectors, reclaiming tombstoned slots.
    ///
    /// Returns the number of slots reclaimed. `O(n log n)` distance
    /// evaluations — it is a full rebuild, because there is no cheaper correct
    /// answer: repairing the graph around one removed node means re-running
    /// neighbour selection for everything that pointed at it, and once a
    /// meaningful fraction of nodes are tombstoned that is most of the graph.
    ///
    /// Ids, vectors and metadata are preserved exactly. The *graph* is not: see
    /// the type-level docs.
    pub fn compact(&mut self) -> Result<usize> {
        let reclaimed = self.tombstones();
        if reclaimed == 0 {
            return Ok(0);
        }

        // Snapshot the live set in slot order, so the rebuild is deterministic.
        let live: Vec<(VectorId, Vec<f32>, Metadata)> = self
            .store
            .iter()
            .map(|(id, slot)| {
                (
                    id,
                    self.store.vector(slot).to_vec(),
                    self.store.metadata(slot).clone(),
                )
            })
            .collect();

        self.store.clear();
        self.levels.clear();
        self.level0.clear();
        self.upper.clear();
        self.entry = None;
        self.entry_level = 0;
        // Reset the generator too: compaction is a rebuild, and a rebuild from
        // the same seed and the same order must give the same graph.
        self.rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        self.draws = 0;

        for (id, vector, metadata) in live {
            // The vectors are already prepared, so this cannot renormalise
            // them; `insert_prepared` skips the redundant pass.
            self.insert_prepared(id, &vector, metadata)?;
        }
        Ok(reclaimed)
    }

    /// The underlying store.
    pub fn store(&self) -> &VectorStore {
        &self.store
    }

    /// A borrowed view of the graph, for serialisation.
    ///
    /// Together with [`HnswIndex::store`] and [`HnswIndex::config`] this is the
    /// complete state of the index: reconstructing from these three reproduces
    /// it exactly, which is what [`HnswIndex::from_parts`] asserts.
    pub fn graph(&self) -> HnswGraph<'_> {
        HnswGraph {
            levels: &self.levels,
            level0: &self.level0,
            upper: &self.upper,
            entry: self.entry,
            entry_level: self.entry_level,
            levels_drawn: self.draws,
        }
    }

    /// Rebuild an index from a store and a graph.
    ///
    /// The deserialisation constructor. It validates the graph against the
    /// store rather than trusting it, because a snapshot's checksum proves only
    /// that the bytes are the ones that were written — it says nothing about
    /// whether the writer was a version of Proxima that laid `m0` out
    /// differently, and an out-of-range neighbour would otherwise be an index
    /// panic in the middle of a query.
    ///
    /// The generator is fast-forwarded by `levels_drawn` draws so that inserts
    /// after a restore land on the same levels they would have without one.
    /// That is `O(levels_drawn)`, about 20 ns per draw, so restoring a
    /// million-vector index spends roughly 20 ms replaying the level sequence.
    /// Skipping it would be free and would make the determinism guarantee stop
    /// at the process boundary.
    pub fn from_parts(
        store: VectorStore,
        config: HnswConfig,
        graph: HnswGraphParts,
    ) -> Result<Self> {
        config.validate()?;
        let slot_count = store.slot_count();
        let m0 = config.m0();

        if graph.levels.len() != slot_count {
            return Err(ProximaError::corrupt(
                "hnsw levels",
                format!("{} entries for {slot_count} slots", graph.levels.len()),
            ));
        }
        if graph.level0.len() != slot_count * m0 {
            return Err(ProximaError::corrupt(
                "hnsw layer 0",
                format!(
                    "{} links for {slot_count} slots at m0 = {m0}, expected {}",
                    graph.level0.len(),
                    slot_count * m0
                ),
            ));
        }
        if graph.upper.len() != slot_count {
            return Err(ProximaError::corrupt(
                "hnsw upper layers",
                format!("{} entries for {slot_count} slots", graph.upper.len()),
            ));
        }
        for (index, links) in graph.upper.iter().enumerate() {
            let expected = graph.levels[index] as usize * config.m;
            if links.len() != expected {
                return Err(ProximaError::corrupt(
                    "hnsw upper layers",
                    format!(
                        "slot {index} at level {} holds {} links, expected {expected}",
                        graph.levels[index],
                        links.len()
                    ),
                ));
            }
        }
        // One pass over every link. `O(slot_count * 2M)` — 1.6 M comparisons for
        // 50 000 vectors at M = 16, which is under a millisecond and buys the
        // guarantee that no later query can index out of bounds.
        let in_range = |link: &u32| *link == SENTINEL || (*link as usize) < slot_count;
        if !graph.level0.iter().all(in_range) || !graph.upper.iter().flatten().all(in_range) {
            return Err(ProximaError::corrupt(
                "hnsw adjacency",
                format!("a neighbour slot is outside 0..{slot_count}"),
            ));
        }
        match graph.entry {
            Some(entry) if entry.index() >= slot_count => {
                return Err(ProximaError::corrupt(
                    "hnsw entry point",
                    format!("slot {entry} outside 0..{slot_count}"),
                ));
            }
            Some(entry) if graph.levels[entry.index()] as usize != graph.entry_level => {
                return Err(ProximaError::corrupt(
                    "hnsw entry point",
                    format!(
                        "slot {entry} is at level {} but entry_level is {}",
                        graph.levels[entry.index()],
                        graph.entry_level
                    ),
                ));
            }
            Some(_) | None => {}
        }

        let mut index = HnswIndex {
            store,
            levels: graph.levels,
            level0: graph.level0,
            upper: graph.upper,
            entry: graph.entry,
            entry_level: graph.entry_level,
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            draws: 0,
            config,
        };
        for _ in 0..graph.levels_drawn {
            index.random_level();
        }
        Ok(index)
    }

    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Draw a level from the exponential distribution the paper prescribes.
    ///
    /// `floor(-ln(U) * m_L)` with `m_L = 1 / ln(m)`. The result is geometric
    /// with ratio `1/m`, which is what makes each layer about `m` times smaller
    /// than the one below and the descent logarithmic.
    fn random_level(&mut self) -> usize {
        // `gen` yields `[0, 1)`; `ln(0)` is `-inf`, so clamp away from zero.
        self.draws += 1;
        let uniform: f64 = self.rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let level = (-uniform.ln() * self.config.level_scale()).floor();
        (level as usize).min(self.config.max_level)
    }

    /// Insert a vector that has already been through [`VectorSpace::prepare`].
    fn insert_prepared(&mut self, id: VectorId, vector: &[f32], metadata: Metadata) -> Result<()> {
        if self.store.contains(id) {
            // Checked before drawing a level, so a rejected insert does not
            // advance the generator and break reproducibility.
            return Err(ProximaError::DuplicateId { id });
        }
        let level = self.random_level();
        let slot = self.store.insert(id, vector, metadata)?;
        self.allocate_links(slot, level);

        let Some(entry) = self.entry else {
            self.entry = Some(slot);
            self.entry_level = level;
            return Ok(());
        };

        // Phase 1: greedy descent through the layers above the new node's own,
        // with ef = 1. This is pure navigation; nothing is linked.
        let mut ep = entry;
        for layer in ((level + 1)..=self.entry_level).rev() {
            ep = self.greedy_step(vector, ep, layer);
        }

        // Phase 2: on each layer the new node belongs to, find candidates,
        // select neighbours by the diversity heuristic, and link both ways.
        let mut visited: VisitedSet = VisitedSet::default();
        let mut scratch = Vec::with_capacity(self.store.dim());
        for layer in (0..=level.min(self.entry_level)).rev() {
            visited.clear();
            let candidates = self.search_layer(
                vector,
                &[ep],
                self.config.ef_construction,
                layer,
                &mut visited,
                &mut |_, _| {},
            );
            let selected = self.select_neighbours(&candidates, self.config.m);
            self.set_links(slot, layer, &selected);

            let degree_cap = if layer == 0 {
                self.config.m0()
            } else {
                self.config.m
            };
            for neighbour in &selected {
                self.link_back(Slot::new(*neighbour), slot, layer, degree_cap, &mut scratch);
            }

            if let Some(nearest) = candidates.first() {
                ep = nearest.slot;
            }
        }

        if level > self.entry_level {
            self.entry = Some(slot);
            self.entry_level = level;
        }
        Ok(())
    }

    /// Make room in the adjacency structures for a newly allocated slot.
    fn allocate_links(&mut self, slot: Slot, level: usize) {
        let index = slot.index();
        if index >= self.levels.len() {
            self.levels.resize(index + 1, 0);
            self.level0.resize((index + 1) * self.config.m0(), SENTINEL);
            self.upper.resize(index + 1, Vec::new());
        }
        self.levels[index] = level as u8;
        let base = index * self.config.m0();
        self.level0[base..base + self.config.m0()].fill(SENTINEL);
        self.upper[index] = vec![SENTINEL; level * self.config.m];
    }

    /// Add `node` to `neighbour`'s list on `layer`, re-selecting if it
    /// overflows.
    ///
    /// The re-selection is the important half. Truncating to the nearest
    /// `degree_cap` would be simpler and would systematically delete exactly
    /// the long-range edges the diversity heuristic added, converting the graph
    /// back into disconnected clusters over the course of a build.
    fn link_back(
        &mut self,
        neighbour: Slot,
        node: Slot,
        layer: usize,
        degree_cap: usize,
        scratch: &mut Vec<f32>,
    ) {
        let mut links: Vec<u32> = self.links(neighbour, layer).to_vec();
        if links.contains(&node.get()) {
            return;
        }
        if links.len() < degree_cap {
            links.push(node.get());
            self.set_links(neighbour, layer, &links);
            return;
        }

        links.push(node.get());
        // Copy the base vector out: `select_neighbours` borrows the store
        // immutably and `set_links` needs the index mutably, and one reused
        // buffer is cheaper than splitting the borrow by hand.
        scratch.clear();
        scratch.extend_from_slice(self.store.vector(neighbour));
        let space = self.store.space();
        let mut candidates: Vec<Candidate> = links
            .iter()
            .map(|link| {
                let slot = Slot::new(*link);
                Candidate {
                    distance: space.distance(scratch, self.store.vector(slot)),
                    slot,
                }
            })
            .collect();
        candidates.sort_unstable();
        let selected = self.select_neighbours(&candidates, degree_cap);
        self.set_links(neighbour, layer, &selected);
    }

    /// The neighbour selection heuristic (Malkov & Yashunin, algorithm 4).
    ///
    /// `candidates` must be sorted nearest-first, and each one's `distance` is
    /// its distance to the base node — which is why the base vector itself is
    /// not a parameter. Returns at most `m` slots.
    ///
    /// The rule is one line: keep `c` only if no already-kept `r` is closer to
    /// `c` than `c` is to the base node. Everything interesting about it is in
    /// the type-level docs; what is worth noting here is the cost, `O(m * |c|)`
    /// distance evaluations, which is why `ef_construction` and `m` both show
    /// up in build time.
    fn select_neighbours(&self, candidates: &[Candidate], m: usize) -> Vec<u32> {
        let space = self.store.space();
        let mut selected: Vec<u32> = Vec::with_capacity(m);
        let mut pruned: Vec<u32> = Vec::new();

        for candidate in candidates {
            if selected.len() >= m {
                break;
            }
            if candidate.distance == 0.0 && selected.contains(&candidate.slot.get()) {
                continue;
            }
            let candidate_vector = self.store.vector(candidate.slot);
            let dominated = selected.iter().any(|kept| {
                space.distance(candidate_vector, self.store.vector(Slot::new(*kept)))
                    < candidate.distance
            });
            if dominated {
                pruned.push(candidate.slot.get());
            } else {
                selected.push(candidate.slot.get());
            }
        }

        if self.config.keep_pruned_connections {
            // Top up in distance order: the pruned list is already sorted,
            // because `candidates` was.
            for slot in pruned {
                if selected.len() >= m {
                    break;
                }
                selected.push(slot);
            }
        }
        selected
    }

    // -----------------------------------------------------------------
    // Traversal
    // -----------------------------------------------------------------

    /// Neighbours of `slot` on `layer`, without the sentinel padding.
    #[inline]
    fn links(&self, slot: Slot, layer: usize) -> &[u32] {
        let raw = if layer == 0 {
            let base = slot.index() * self.config.m0();
            &self.level0[base..base + self.config.m0()]
        } else {
            let node = &self.upper[slot.index()];
            let base = (layer - 1) * self.config.m;
            let end = base + self.config.m;
            if end > node.len() {
                return &[];
            }
            &node[base..end]
        };
        let len = raw.iter().position(|x| *x == SENTINEL).unwrap_or(raw.len());
        &raw[..len]
    }

    /// Overwrite `slot`'s neighbour list on `layer`, padding with sentinels.
    fn set_links(&mut self, slot: Slot, layer: usize, links: &[u32]) {
        let (target, cap) = if layer == 0 {
            let base = slot.index() * self.config.m0();
            let cap = self.config.m0();
            (&mut self.level0[base..base + cap], cap)
        } else {
            let base = (layer - 1) * self.config.m;
            let cap = self.config.m;
            (&mut self.upper[slot.index()][base..base + cap], cap)
        };
        let take = links.len().min(cap);
        target[..take].copy_from_slice(&links[..take]);
        target[take..].fill(SENTINEL);
    }

    /// One layer of greedy descent: walk downhill until no neighbour is closer.
    ///
    /// `ef = 1`, so there is no candidate list and no heap — just a pointer that
    /// moves to the best neighbour until it cannot. This is the cheap part of a
    /// search and it is what makes the whole structure `O(log n)`.
    fn greedy_step(&self, query: &[f32], entry: Slot, layer: usize) -> Slot {
        let space = self.store.space();
        let mut current = entry;
        let mut current_distance = space.distance(query, self.store.vector(current));
        loop {
            let mut improved = false;
            for link in self.links(current, layer) {
                let slot = Slot::new(*link);
                let distance = space.distance(query, self.store.vector(slot));
                // `<` and not `<=`, plus the slot tiebreak below, so that a
                // plateau of equidistant nodes cannot cycle.
                if distance < current_distance {
                    current_distance = distance;
                    current = slot;
                    improved = true;
                }
            }
            if !improved {
                return current;
            }
        }
    }

    /// The `ef` nearest nodes to `query` on `layer`, nearest first.
    ///
    /// `observe` is called once for every node whose distance is computed,
    /// including ones that never make the returned list. That is how filtered
    /// search collects accepted results *during* the traversal rather than from
    /// the survivors: a node the filter accepts is recorded the moment it is
    /// examined, even if a hundred nearer rejected nodes later push it out of
    /// the `ef` list.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[Slot],
        ef: usize,
        layer: usize,
        visited: &mut VisitedSet,
        observe: &mut dyn FnMut(Slot, f32),
    ) -> Vec<Candidate> {
        let space = self.store.space();
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef + 1);

        for entry in entry_points {
            if !visited.insert(entry.get()) {
                continue;
            }
            let distance = space.distance(query, self.store.vector(*entry));
            observe(*entry, distance);
            let candidate = Candidate {
                distance,
                slot: *entry,
            };
            candidates.push(Reverse(candidate));
            results.push(candidate);
        }

        while let Some(Reverse(nearest)) = candidates.pop() {
            // Stop once the closest unexplored candidate is further away than
            // the worst node already kept: nothing reachable through it can
            // improve the list. This is the entire cost control of HNSW, and
            // it is why `ef` and not `n` bounds the work.
            if let Some(worst) = results.peek() {
                if results.len() >= ef && nearest.distance > worst.distance {
                    break;
                }
            }

            for link in self.links(nearest.slot, layer) {
                if !visited.insert(*link) {
                    continue;
                }
                let slot = Slot::new(*link);
                let distance = space.distance(query, self.store.vector(slot));
                observe(slot, distance);

                let worst = results.peek().map_or(f32::INFINITY, |c| c.distance);
                if results.len() < ef || distance < worst {
                    let candidate = Candidate { distance, slot };
                    candidates.push(Reverse(candidate));
                    results.push(candidate);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results.into_sorted_vec()
    }

    /// The search proper, on an already-prepared query.
    fn search_prepared(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: &dyn Filter,
    ) -> Vec<SearchResult> {
        if k == 0 || self.store.is_empty() {
            return Vec::new();
        }
        let Some(entry) = self.entry else {
            return Vec::new();
        };

        // The most results any search over this index could possibly return.
        // Escalating past this point cannot help and would burn a full scan.
        let target = k.min(self.store.len());
        // ...and the size the *accepted* candidate pool has to reach before the
        // top `k` of it can be trusted.
        //
        // Filling `k` results is not enough. Under a 1-in-40 filter an
        // `ef = 64` traversal examines a few hundred nodes and about ten of
        // them pass, so it can return ten results that are merely the ten
        // accepted nodes it happened to walk past -- not the ten nearest. The
        // fix is to require that the traversal has *seen* as many accepted
        // nodes as an unfiltered search would have had candidates, which makes
        // the filtered search as selective among accepted vectors as the
        // unfiltered one is among all of them.
        let pool_target = ef.min(self.store.len());

        let mut ef = ef;
        let (mut best, mut pool) = self.run_search(query, k, ef, entry, filter);
        while (best.len() < target || pool < pool_target) && ef < self.config.filter_ef_limit {
            // Geometric escalation, so the total work is at most twice the
            // final attempt's. A linear ramp would be cheaper per step and
            // would need many more steps on a highly selective filter.
            ef = (ef * 2).min(self.config.filter_ef_limit);
            (best, pool) = self.run_search(query, k, ef, entry, filter);
        }

        if (best.len() < target || pool < pool_target) && self.config.exhaustive_filter_fallback {
            // The escalation ran out. Either the accepted vectors sit in a
            // region the greedy walk never reaches, or the filter is simply so
            // selective that no bounded traversal gathers enough of them. Below
            // roughly `1 / ef` selectivity a scan is the *right* algorithm and
            // not merely a safety net: the graph's whole advantage is that it
            // examines a small fraction of the corpus, and if the filter
            // rejects almost all of that fraction the advantage is gone.
            //
            // `O(n)` and correct beats sub-linear and quietly wrong. Turning
            // this off (see `with_exhaustive_filter_fallback`) chooses the
            // other side of that trade explicitly.
            return self.exhaustive(query, k, filter);
        }
        best
    }

    /// One attempt at a given `ef`.
    ///
    /// Returns the results and the number of *accepted* nodes the traversal
    /// examined — the second is what tells the caller whether `ef` was large
    /// enough for the filter's selectivity.
    fn run_search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        entry: Slot,
        filter: &dyn Filter,
    ) -> (Vec<SearchResult>, usize) {
        let mut point = entry;
        for layer in (1..=self.entry_level).rev() {
            point = self.greedy_step(query, point, layer);
        }

        let mut accepted = TopK::new(k);
        let mut pool = 0usize;
        let mut visited: VisitedSet =
            VisitedSet::with_capacity_and_hasher(ef * 4, BuildHasherDefault::default());
        {
            let store = &self.store;
            let mut observe = |slot: Slot, distance: f32| {
                // A tombstoned slot has no id, which is exactly the liveness
                // test: it is still traversed through, it is never returned.
                if let Some(id) = store.id_of(slot) {
                    if filter.accepts(id, store.metadata(slot)) {
                        pool += 1;
                        accepted.push(id, distance);
                    }
                }
            };
            self.search_layer(query, &[point], ef, 0, &mut visited, &mut observe);
        }
        (accepted.into_sorted_vec(), pool)
    }

    /// Filtered exhaustive scan, identical in behaviour to [`FlatIndex`].
    ///
    /// [`FlatIndex`]: crate::FlatIndex
    fn exhaustive(&self, query: &[f32], k: usize, filter: &dyn Filter) -> Vec<SearchResult> {
        let space = self.store.space();
        let mut top = TopK::new(k);
        for (id, slot) in self.store.iter() {
            if filter.accepts(id, self.store.metadata(slot)) {
                top.push(id, space.distance(query, self.store.vector(slot)));
            }
        }
        top.into_sorted_vec()
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&mut self, id: VectorId, vector: &[f32], metadata: Metadata) -> Result<()> {
        let prepared = self.store.space().prepare(vector)?;
        self.insert_prepared(id, &prepared, metadata)
    }

    fn remove(&mut self, id: VectorId) -> Result<()> {
        // `detach`, not `remove`: the slot stays reserved because the graph
        // still holds edges pointing at it. See the type-level docs.
        self.store.detach(id)?;
        Ok(())
    }

    /// The `k` nearest live vectors accepted by `filter`.
    ///
    /// # The filtering recall trap
    ///
    /// There are two obvious ways to combine a filter with a graph search and
    /// both are wrong.
    ///
    /// **Filter afterwards.** Search for `k`, then drop the results the filter
    /// rejects. You now have fewer than `k` results and no way to know how many
    /// more you needed; searching for `10k` instead just moves the threshold at
    /// which it breaks, and for a filter that matches 1 in 10 000 it breaks
    /// always.
    ///
    /// **Filter during expansion.** Refuse to traverse *through* rejected
    /// nodes. This keeps the result count right and destroys the graph: HNSW's
    /// connectivity depends on every node's edges, and removing 90 % of the
    /// nodes from the walk leaves the accepted ones scattered across
    /// disconnected fragments. The search finds whichever fragment it starts
    /// in and reports its contents as the nearest neighbours.
    ///
    /// **What this implementation does.** Traversal is unfiltered — every node
    /// is expanded, so the graph stays exactly as connected as it was built to
    /// be — and the filter is applied when a node is *examined*, with accepted
    /// nodes accumulating in a separate bounded heap of size `k`. A node that
    /// passes the filter is recorded even if it never survives in the `ef`-sized
    /// candidate list.
    ///
    /// **Filling `k` is not the stopping condition.** Under a 1-in-40 filter,
    /// an `ef = 64` traversal examines a few hundred nodes and about ten of
    /// them pass — so it *can* return ten results, and they are the ten
    /// accepted nodes it happened to walk past rather than the ten nearest.
    /// Measured on the integration fixture, stopping there gives recall 0.39.
    /// The search therefore also requires that the traversal has *seen* at
    /// least `ef` accepted nodes, which restores the same degree of selectivity
    /// among accepted vectors that an unfiltered search has among all of them,
    /// and takes that same case to recall 1.00.
    ///
    /// If either condition is unmet, `ef` is doubled and the search is
    /// repeated, up to [`HnswConfig::filter_ef_limit`]. If it still
    /// falls short, and
    /// [`with_exhaustive_filter_fallback`](HnswConfig::with_exhaustive_filter_fallback)
    /// is on, the query degrades to a filtered exhaustive scan. That is `O(n)`
    /// and it is deliberate: a selective filter genuinely can put every
    /// accepted vector behind a wall of rejected ones, and there is no bounded
    /// algorithm that finds them. Being slow is a problem you can measure;
    /// silently returning three of ten results is not.
    fn search(&self, query: &[f32], k: usize, filter: &dyn Filter) -> Result<Vec<SearchResult>> {
        check_k(k)?;
        let prepared = self.store.space().prepare(query)?;
        let ef = self.config.ef_search.max(k).max(1);
        Ok(self.search_prepared(&prepared, k, ef, filter))
    }

    fn contains(&self, id: VectorId) -> bool {
        self.store.contains(id)
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn stats(&self) -> IndexStats {
        let slots = self.store.slot_count();
        let mut nodes_per_layer = vec![0usize; self.entry_level + 1];
        let mut edges = 0usize;
        for index in 0..slots {
            let slot = Slot::new(index as u32);
            let level = self.levels[index] as usize;
            for count in nodes_per_layer
                .iter_mut()
                .take(level.min(self.entry_level) + 1)
            {
                *count += 1;
            }
            edges += self.links(slot, 0).len();
        }
        let graph_bytes = self.level0.capacity() * std::mem::size_of::<u32>()
            + self.levels.capacity()
            + self.upper.capacity() * std::mem::size_of::<Vec<u32>>()
            + self
                .upper
                .iter()
                .map(|links| links.capacity() * std::mem::size_of::<u32>())
                .sum::<usize>();

        IndexStats {
            len: self.store.len(),
            deleted: self.tombstones(),
            dim: self.store.dim(),
            metric: self.store.space().metric(),
            layers: self.entry_level + 1,
            nodes_per_layer,
            edges,
            vector_bytes: self.store.vector_bytes(),
            graph_bytes,
        }
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

    fn space(dim: usize) -> VectorSpace {
        VectorSpace::new(dim, Metric::Euclidean).unwrap()
    }

    fn index(dim: usize) -> HnswIndex {
        HnswIndex::new(space(dim), HnswConfig::new().with_seed(1)).unwrap()
    }

    #[test]
    fn config_validation_catches_every_bad_parameter() {
        assert!(HnswConfig::new().with_m(1).validate().is_err());
        assert!(HnswConfig::new().with_m(1024).validate().is_err());
        assert!(HnswConfig::new()
            .with_m(64)
            .with_ef_construction(8)
            .validate()
            .is_err());
        assert!(HnswConfig::new().with_ef_search(0).validate().is_err());
        assert!(HnswConfig::new()
            .with_ef_search(2048)
            .with_filter_ef_limit(16)
            .validate()
            .is_err());
        assert!(HnswConfig::new().validate().is_ok());
    }

    #[test]
    fn level_distribution_decays_by_roughly_m() {
        // The generator is the reason the descent is logarithmic; if it drifts,
        // recall drifts with it and nothing else would notice.
        let mut index = index(2);
        let mut counts = [0usize; 8];
        for _ in 0..20_000 {
            let level = index.random_level().min(7);
            counts[level] += 1;
        }
        assert!(counts[0] > 18_000, "level 0 share {}", counts[0]);
        for level in 0..3 {
            let ratio = counts[level] as f64 / counts[level + 1].max(1) as f64;
            assert!(
                (8.0..32.0).contains(&ratio),
                "level {level} to {} ratio {ratio}, expected about m = 16",
                level + 1
            );
        }
    }

    #[test]
    fn empty_and_single_vector_indexes() {
        let mut index = index(3);
        assert!(index.search_all(&[1.0, 2.0, 3.0], 5).unwrap().is_empty());
        index
            .insert(VectorId::new(1), &[1.0, 2.0, 3.0], Metadata::new())
            .unwrap();
        let hits = index.search_all(&[1.0, 2.0, 3.0], 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].distance, 0.0);
    }

    #[test]
    fn all_identical_vectors_do_not_hang_or_lose_results() {
        // A degenerate graph: every distance is zero, so the greedy walk has no
        // gradient at all. The `<` in `greedy_step` is what stops it cycling.
        let mut index = index(4);
        for i in 0..200u64 {
            index
                .insert(VectorId::new(i), &[1.0; 4], Metadata::new())
                .unwrap();
        }
        let hits = index.search_all(&[1.0; 4], 10).unwrap();
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|h| h.distance == 0.0));
    }

    #[test]
    fn duplicate_insert_does_not_advance_the_level_generator() {
        // Reproducibility depends on exactly one draw per successful insert.
        let mut a = index(2);
        let mut b = index(2);
        for i in 0..50u64 {
            a.insert(VectorId::new(i), &[i as f32, 0.0], Metadata::new())
                .unwrap();
            b.insert(VectorId::new(i), &[i as f32, 0.0], Metadata::new())
                .unwrap();
            assert!(b
                .insert(VectorId::new(i), &[0.0, 0.0], Metadata::new())
                .is_err());
        }
        assert_eq!(a.levels, b.levels);
        assert_eq!(a.level0, b.level0);
    }

    #[test]
    fn remove_tombstones_and_compaction_reclaims() {
        let mut index = index(2);
        for i in 0..200u64 {
            index
                .insert(
                    VectorId::new(i),
                    &[i as f32, (i % 7) as f32],
                    Metadata::new(),
                )
                .unwrap();
        }
        for i in 0..100u64 {
            index.remove(VectorId::new(i)).unwrap();
        }
        assert_eq!(index.len(), 100);
        assert_eq!(index.tombstones(), 100);
        // Removed ids are gone from results even though their nodes remain.
        let hits = index.search_all(&[0.0, 0.0], 10).unwrap();
        assert!(hits.iter().all(|h| h.id.get() >= 100));

        let reclaimed = index.compact().unwrap();
        assert_eq!(reclaimed, 100);
        assert_eq!(index.tombstones(), 0);
        assert_eq!(index.len(), 100);
        let after = index.search_all(&[0.0, 0.0], 10).unwrap();
        assert_eq!(after.len(), 10);
        assert!(after.iter().all(|h| h.id.get() >= 100));
    }

    #[test]
    fn compaction_of_a_clean_index_is_free() {
        let mut index = index(2);
        for i in 0..20u64 {
            index
                .insert(VectorId::new(i), &[i as f32, 0.0], Metadata::new())
                .unwrap();
        }
        let before = index.stats();
        assert_eq!(index.compact().unwrap(), 0);
        assert_eq!(index.stats(), before);
    }

    #[test]
    fn removing_the_entry_point_leaves_the_index_searchable() {
        let mut index = index(2);
        for i in 0..300u64 {
            index
                .insert(VectorId::new(i), &[i as f32 * 0.1, 0.0], Metadata::new())
                .unwrap();
        }
        let entry_slot = index.entry.unwrap();
        let entry_id = index.store.id_of(entry_slot).unwrap();
        index.remove(entry_id).unwrap();
        // The entry point is now a tombstone that is still navigated through.
        let hits = index.search_all(&[15.0, 0.0], 5).unwrap();
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|h| h.id != entry_id));
    }

    #[test]
    fn stats_layer_histogram_is_consistent() {
        let mut index = index(4);
        for i in 0..1000u64 {
            let t = i as f32 * 0.01;
            index
                .insert(VectorId::new(i), &[t, t * t, -t, 1.0], Metadata::new())
                .unwrap();
        }
        let stats = index.stats();
        assert_eq!(stats.len, 1000);
        assert_eq!(stats.nodes_per_layer[0], 1000);
        assert_eq!(stats.layers, stats.nodes_per_layer.len());
        for pair in stats.nodes_per_layer.windows(2) {
            assert!(pair[0] >= pair[1], "layer sizes must not grow upwards");
        }
        // Mean degree should be in the ballpark of m; well below would mean the
        // heuristic is over-pruning.
        assert!(
            stats.mean_degree() > 8.0,
            "mean degree {}",
            stats.mean_degree()
        );
        assert!(stats.graph_bytes > 0);
    }

    #[test]
    fn search_with_ef_below_k_is_raised_rather_than_truncating() {
        let mut index = index(2);
        for i in 0..100u64 {
            index
                .insert(VectorId::new(i), &[i as f32, 0.0], Metadata::new())
                .unwrap();
        }
        assert_eq!(
            index
                .search_with_ef(&[50.0, 0.0], 20, 1, &crate::AcceptAll)
                .unwrap()
                .len(),
            20
        );
    }

    #[test]
    fn a_filter_matching_nothing_returns_nothing_without_hanging() {
        let mut index = index(2);
        for i in 0..500u64 {
            index
                .insert(VectorId::new(i), &[i as f32, 0.0], Metadata::new())
                .unwrap();
        }
        let none = index
            .search(&[10.0, 0.0], 10, &|_id, _m: &Metadata| false)
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn a_highly_selective_filter_still_returns_k() {
        // One in 250 accepted, and the accepted vectors are clustered at the
        // far end of the space from the query. This is the case that
        // post-filtering gets wrong and that the escalation plus fallback is
        // there to handle.
        let mut index = index(2);
        for i in 0..1000u64 {
            let keep = i >= 990;
            index
                .insert(
                    VectorId::new(i),
                    &[i as f32, 0.0],
                    Metadata::new().with("keep", keep),
                )
                .unwrap();
        }
        let hits = index
            .search(&[0.0, 0.0], 10, &|_id, m: &Metadata| {
                m.bool("keep") == Some(true)
            })
            .unwrap();
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|h| h.id.get() >= 990));
        assert_eq!(hits[0].id, VectorId::new(990));
    }

    #[test]
    fn disabling_the_fallback_can_return_fewer_results() {
        // Pinning the documented consequence of the flag, so that turning it
        // off is understood to be a real trade rather than a free win.
        let config = HnswConfig::new()
            .with_seed(3)
            .with_ef_search(8)
            .with_filter_ef_limit(8)
            .with_exhaustive_filter_fallback(false);
        let mut index = HnswIndex::new(space(2), config).unwrap();
        for i in 0..2000u64 {
            index
                .insert(
                    VectorId::new(i),
                    &[i as f32, 0.0],
                    Metadata::new().with("keep", i >= 1990),
                )
                .unwrap();
        }
        let hits = index
            .search(&[0.0, 0.0], 10, &|_id, m: &Metadata| {
                m.bool("keep") == Some(true)
            })
            .unwrap();
        assert!(
            hits.len() < 10,
            "expected a short result set, got {}",
            hits.len()
        );
    }

    #[test]
    fn same_seed_same_graph() {
        let build = |seed: u64| {
            let mut index = HnswIndex::new(space(3), HnswConfig::new().with_seed(seed)).unwrap();
            for i in 0..500u64 {
                let t = i as f32 * 0.03;
                index
                    .insert(VectorId::new(i), &[t.sin(), t.cos(), t], Metadata::new())
                    .unwrap();
            }
            index
        };
        let a = build(99);
        let b = build(99);
        assert_eq!(a.levels, b.levels);
        assert_eq!(a.level0, b.level0);
        assert_eq!(a.upper, b.upper);
        assert_eq!(a.entry, b.entry);

        let c = build(100);
        assert_ne!(a.levels, c.levels, "different seeds produced one graph");
    }

    #[test]
    fn k_larger_than_the_index() {
        let mut index = index(2);
        for i in 0..5u64 {
            index
                .insert(VectorId::new(i), &[i as f32, 0.0], Metadata::new())
                .unwrap();
        }
        assert_eq!(index.search_all(&[0.0, 0.0], 50).unwrap().len(), 5);
        assert!(index.search_all(&[0.0, 0.0], 0).unwrap().is_empty());
    }

    #[test]
    fn dimension_and_finiteness_are_validated() {
        let mut index = index(3);
        assert!(matches!(
            index.insert(VectorId::new(1), &[1.0, 2.0], Metadata::new()),
            Err(ProximaError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            index.insert(VectorId::new(1), &[1.0, 2.0, f32::NAN], Metadata::new()),
            Err(ProximaError::NonFinite { .. })
        ));
        assert!(matches!(
            index.search_all(&[1.0], 1),
            Err(ProximaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn upsert_replaces_and_leaves_a_tombstone() {
        let mut index = index(2);
        for i in 0..50u64 {
            index
                .insert(VectorId::new(i), &[i as f32, 0.0], Metadata::new())
                .unwrap();
        }
        index
            .upsert(VectorId::new(10), &[999.0, 0.0], Metadata::new())
            .unwrap();
        assert_eq!(index.len(), 50);
        assert_eq!(index.tombstones(), 1);
        assert_eq!(index.vector(VectorId::new(10)), Some(&[999.0, 0.0][..]));
    }
}
