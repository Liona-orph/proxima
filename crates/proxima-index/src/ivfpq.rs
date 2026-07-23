//! [`IvfPqIndex`]: an inverted file over product-quantized residuals.
//!
//! The index that exists to hold vectors in a tenth of the memory, and the
//! answer to "what are the quantizers for". [`HnswIndex`](crate::HnswIndex)
//! stores every vector at full precision and adds a graph on top, so it costs
//! *more* than the corpus; this one stores a `subspaces`-byte code per vector
//! and can throw the corpus away.
//!
//! # How a query is answered
//!
//! 1. **Coarse quantization.** `nlist` centroids are fitted by k-means over a
//!    sample of the corpus, and every vector is assigned to its nearest. The
//!    assignment is the inverted file: list `i` holds the slots assigned to
//!    centroid `i`.
//! 2. **Residual product quantization.** What is encoded is not the vector but
//!    `x - c(x)`, its offset from its own centroid. Residuals are an order of
//!    magnitude smaller than the vectors and, crucially, they are distributed
//!    around the origin *the same way in every list*, so one codebook fits all
//!    of them. Encoding the raw vectors instead would ask 256 centroids per
//!    subspace to cover the whole corpus rather than the spread within one
//!    cell, and measurably costs recall.
//! 3. **Probing.** At search time the query is scored against all `nlist`
//!    centroids and the nearest `nprobe` lists are scanned. This is the recall
//!    dial: `nprobe = nlist` is an exhaustive scan of the codes, `nprobe = 1`
//!    looks in one cell and misses anything near a boundary.
//! 4. **Asymmetric distance.** The query stays at full precision; only the
//!    stored vectors are quantized. One `256 * dim` table is computed per probed
//!    list and each candidate then costs `subspaces` table lookups and adds,
//!    independent of the dimension. See
//!    [`ProductQuantizer::distance_table`](proxima_quantize::ProductQuantizer::distance_table).
//! 5. **Optional exact reranking.** The approximate scores order a shortlist;
//!    the top [`IvfPqConfig::rerank`] of it is then rescored against the stored
//!    full-precision vectors. This is where most of the accuracy lost to
//!    quantization comes back — and it only works if the vectors were kept, so
//!    it is the direct trade against the memory saving.
//!
//! # What it costs, measured
//!
//! The numbers are in `BENCHMARKS.md` and are produced by `proxima bench`.
//! The short version on the 50 000-vector, 128-dimensional fixture: IVF-PQ at
//! `m = 32` holds vectors in **48 bytes** each against flat's 512 and HNSW's
//! 663, and that is the entire point of it. What it buys costs recall, and the
//! honest figure on this fixture is low; `BENCHMARKS.md` says how low and why.
//!
//! # Restrictions
//!
//! Only [`Metric::Euclidean`] and [`Metric::Cosine`]. The residual construction
//! is an L2 argument — `||q - (c + r)|| = ||(q - c) - r||` — and cosine rides on
//! it because prepared cosine operands are unit vectors and the scoring
//! identity `1 - <q, c> - <q, r>` decomposes the same way. Inner product does
//! not: maximum inner product search is not nearest-neighbour search under any
//! metric, the nearest centroid is not the most promising list, and pretending
//! otherwise would return plausible wrong answers. Manhattan has no
//! decomposition through a distance table at all. Both are rejected at
//! construction.
//!
//! [`Metric::Euclidean`]: proxima_core::Metric::Euclidean
//! [`Metric::Cosine`]: proxima_core::Metric::Cosine

use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use proxima_core::{
    Metadata, Metric, ProximaError, Result, SearchResult, Slot, SlotAllocator, TopK, VectorId,
    VectorSpace,
};
use proxima_quantize::{kmeans, DistanceTable, PqConfig, ProductQuantizer};

use crate::index::{check_k, Filter, IndexStats, VectorIndex};

/// Marks a slot that has not been assigned to a list yet.
const UNASSIGNED: u32 = u32::MAX;

/// Build and search parameters for [`IvfPqIndex`].
///
/// # Choosing `nlist` and `nprobe`
///
/// `nlist` is conventionally around `sqrt(n)`: it balances the cost of scoring
/// every centroid (`O(nlist * dim)` per query) against the cost of scanning a
/// list (`O(n / nlist * subspaces)` per probe). `nprobe` is then the recall
/// dial, and the useful range is 1 % to 20 % of `nlist`.
///
/// `subspaces` is the compression: a vector of dimension `d` becomes
/// `subspaces` bytes, so `d * 4 / subspaces` is the ratio. It is also the
/// dominant term in recall loss, far more than `nprobe`; see the crate
/// benchmarks.
///
/// ```
/// use proxima_index::IvfPqConfig;
///
/// let config = IvfPqConfig::new(64, 16).with_nprobe(8).with_rerank(100);
/// assert!(config.validate(128).is_ok());
///
/// // More subspaces than dimensions cannot be satisfied.
/// assert!(IvfPqConfig::new(4, 200).validate(128).is_err());
/// // Reranking needs the full-precision vectors it reranks against.
/// assert!(IvfPqConfig::new(4, 8)
///     .with_rerank(10)
///     .with_retain_vectors(false)
///     .validate(128)
///     .is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfPqConfig {
    nlist: usize,
    subspaces: usize,
    nprobe: usize,
    rerank: usize,
    seed: u64,
    max_iterations: usize,
    training_sample: usize,
    train_threshold: usize,
    retain_vectors: bool,
}

impl IvfPqConfig {
    /// `nlist` coarse centroids and `subspaces` bytes per code.
    ///
    /// The two parameters with no sensible default: they depend on the corpus
    /// size and on how much memory the caller is trying to save, and guessing
    /// either produces an index that is quietly bad.
    pub fn new(nlist: usize, subspaces: usize) -> Self {
        IvfPqConfig {
            nlist,
            subspaces,
            // A sixteenth of the lists, which is the region where the recall
            // curve is still climbing steeply on every fixture measured here.
            nprobe: (nlist / 16).max(1),
            // Off. Reranking is a large recall gain for a hundred extra
            // distance evaluations, but it is also the thing that stops the
            // index from being able to drop its vectors, so it is a decision
            // rather than a default.
            rerank: 0,
            seed: 0x1FF_9C0D,
            // 25 Lloyd iterations is where the coarse quantizer's inertia has
            // flattened on the benchmark fixtures; the PQ codebooks use the
            // same figure for the same reason.
            max_iterations: 25,
            // 100 000 vectors is enough to fit 4 096 centroids well and bounds
            // training at about a minute single-threaded. Beyond it, k-means
            // sees diminishing returns per point.
            training_sample: 100_000,
            // `39 * k` is the rule of thumb for "enough points per centroid to
            // fit it"; 100 is that rounded up to a number worth defending.
            train_threshold: nlist.saturating_mul(100),
            retain_vectors: true,
        }
    }

    /// Lists probed per query. The recall dial.
    pub const fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Rescore the top `rerank` approximate candidates with exact distances.
    ///
    /// Zero disables it. Requires [`IvfPqConfig::with_retain_vectors`] to be
    /// on, which is the whole trade: exact reranking needs the exact vectors.
    pub const fn with_rerank(mut self, rerank: usize) -> Self {
        self.rerank = rerank;
        self
    }

    /// Seed for k-means initialisation, in both the coarse and the PQ stage.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Cap on Lloyd iterations.
    pub const fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Cap on how many vectors k-means is fitted to.
    pub const fn with_training_sample(mut self, vectors: usize) -> Self {
        self.training_sample = vectors;
        self
    }

    /// Live vectors at which [`IvfPqIndex::train`] runs by itself.
    ///
    /// An index cannot encode anything before it has codebooks, so inserts
    /// before this point are held at full precision and the whole corpus is
    /// encoded in one pass when the threshold is crossed. That makes exactly
    /// one insert enormously more expensive than its neighbours — seconds
    /// rather than microseconds — which is worth knowing about and is why the
    /// threshold is explicit rather than hidden.
    ///
    /// Set it to `usize::MAX` to disable auto-training entirely and call
    /// [`IvfPqIndex::train`] yourself.
    pub const fn with_train_threshold(mut self, vectors: usize) -> Self {
        self.train_threshold = vectors;
        self
    }

    /// Whether the full-precision vectors are kept after training.
    ///
    /// On by default, because dropping them is irreversible: without them the
    /// index cannot rerank, cannot be retrained with different parameters and
    /// cannot answer "what was the vector for id 7". Turning it off is what
    /// makes the memory figure in `BENCHMARKS.md` achievable, and the benchmark
    /// measures both.
    pub const fn with_retain_vectors(mut self, retain: bool) -> Self {
        self.retain_vectors = retain;
        self
    }

    /// Number of coarse centroids.
    pub const fn nlist(&self) -> usize {
        self.nlist
    }

    /// Bytes per code, i.e. PQ subspaces.
    pub const fn subspaces(&self) -> usize {
        self.subspaces
    }

    /// Lists probed per query.
    pub const fn nprobe(&self) -> usize {
        self.nprobe
    }

    /// Candidates rescored exactly, or zero.
    pub const fn rerank(&self) -> usize {
        self.rerank
    }

    /// k-means seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Lloyd iteration cap.
    pub const fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Training sample cap.
    pub const fn training_sample(&self) -> usize {
        self.training_sample
    }

    /// Auto-training threshold.
    pub const fn train_threshold(&self) -> usize {
        self.train_threshold
    }

    /// Whether full-precision vectors are retained.
    pub const fn retain_vectors(&self) -> bool {
        self.retain_vectors
    }

    /// Check every parameter against the space it will be used in.
    pub fn validate(&self, dim: usize) -> Result<()> {
        if self.nlist == 0 {
            return Err(ProximaError::config(
                "nlist",
                self.nlist,
                "must be at least 1",
            ));
        }
        if self.subspaces == 0 || self.subspaces > dim {
            return Err(ProximaError::config(
                "subspaces",
                self.subspaces,
                "must be between 1 and the space dimension",
            ));
        }
        if self.nprobe == 0 || self.nprobe > self.nlist {
            return Err(ProximaError::config(
                "nprobe",
                self.nprobe,
                "must be between 1 and nlist",
            ));
        }
        if self.max_iterations == 0 {
            return Err(ProximaError::config(
                "max_iterations",
                self.max_iterations,
                "must be at least 1",
            ));
        }
        if self.training_sample == 0 {
            return Err(ProximaError::config(
                "training_sample",
                self.training_sample,
                "must be at least 1",
            ));
        }
        if self.rerank > 0 && !self.retain_vectors {
            return Err(ProximaError::config(
                "rerank",
                self.rerank,
                "requires retain_vectors, because reranking is against the exact vectors",
            ));
        }
        Ok(())
    }
}

/// The state that only exists once the index has been trained.
#[derive(Debug, Clone)]
struct Codebooks {
    /// `nlist * dim` coarse centroids.
    centroids: Vec<f32>,
    /// The residual quantizer.
    pq: ProductQuantizer,
}

/// What training produced.
///
/// Both inertias are reported because they answer different questions.
/// `coarse_inertia` says whether `nlist` is large enough to partition the
/// corpus; `residual_inertia` says whether `subspaces` is large enough to
/// describe what is left. A high coarse inertia and a low residual one means
/// raising `nlist`; the reverse means raising `subspaces`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingReport {
    /// Vectors k-means was fitted to, after the [`IvfPqConfig::training_sample`]
    /// cap.
    pub sampled: usize,
    /// Sum of squared distances from each sampled vector to its coarse
    /// centroid.
    pub coarse_inertia: f64,
    /// Lloyd iterations run by the coarse quantizer.
    pub coarse_iterations: usize,
    /// Total PQ reconstruction error over the residuals.
    pub residual_inertia: f64,
    /// Non-empty inverted lists.
    pub populated_lists: usize,
    /// Vectors in the largest list.
    ///
    /// The number that says whether the partition is balanced. A single list
    /// holding a tenth of the corpus means one query in `nlist` scans a tenth
    /// of everything, and the latency distribution has a long tail nothing in
    /// the mean shows.
    pub largest_list: usize,
}

impl std::fmt::Display for TrainingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "trained on {} vectors: coarse inertia {:.1} in {} iterations, residual inertia {:.1}, {} lists populated, largest {}",
            self.sampled,
            self.coarse_inertia,
            self.coarse_iterations,
            self.residual_inertia,
            self.populated_lists,
            self.largest_list
        )
    }
}

/// A borrowed view of an [`IvfPqIndex`]'s state, for serialisation.
///
/// Every array is in slot order and is either empty or exactly `slot_count`
/// long, which is what [`IvfPqIndex::from_parts`] checks. `vectors` is empty
/// when [`IvfPqConfig::retain_vectors`] is off, and `codes`, `centroids` and
/// `codebooks` are empty before training.
#[derive(Debug, Clone, Copy)]
pub struct IvfPqParts<'a> {
    /// Slot to id, `None` for a free slot.
    pub ids: &'a [Option<VectorId>],
    /// Metadata in slot order.
    pub metadata: &'a [Metadata],
    /// `slot_count * dim` prepared vectors, or empty.
    pub vectors: &'a [f32],
    /// `slot_count * subspaces` PQ codes, or empty.
    pub codes: &'a [u8],
    /// Slot to inverted list, or empty.
    pub list_of: &'a [u32],
    /// `nlist * dim` coarse centroids, or empty.
    pub centroids: &'a [f32],
    /// The residual PQ codebooks, or empty.
    pub codebooks: &'a [f32],
    /// The last training report, if the index has been trained.
    pub report: Option<&'a TrainingReport>,
}

/// The owned form of [`IvfPqParts`], for [`IvfPqIndex::from_parts`].
#[derive(Debug, Clone)]
pub struct IvfPqPartsOwned {
    /// See [`IvfPqParts::ids`].
    pub ids: Vec<Option<VectorId>>,
    /// See [`IvfPqParts::metadata`].
    pub metadata: Vec<Metadata>,
    /// See [`IvfPqParts::vectors`].
    pub vectors: Vec<f32>,
    /// See [`IvfPqParts::codes`].
    pub codes: Vec<u8>,
    /// See [`IvfPqParts::list_of`].
    pub list_of: Vec<u32>,
    /// See [`IvfPqParts::centroids`].
    pub centroids: Vec<f32>,
    /// See [`IvfPqParts::codebooks`].
    pub codebooks: Vec<f32>,
    /// See [`IvfPqParts::report`].
    pub report: Option<TrainingReport>,
}

/// An inverted file index over product-quantized residuals.
///
/// # Lifecycle
///
/// An `IvfPqIndex` is **untrained** until it has codebooks. Untrained, it holds
/// full-precision vectors and answers searches by exhaustive scan, so it is
/// correct but has none of the properties it exists for. Training happens
/// either explicitly through [`IvfPqIndex::train`] or automatically when the
/// live count reaches [`IvfPqConfig::train_threshold`], and encodes the whole
/// corpus in one pass.
///
/// Inserts after training are encoded immediately against the existing
/// codebooks. They do **not** retrain, and they do not move the centroids: an
/// index whose distribution drifts a long way from its training set will lose
/// recall silently, and the fix is to rebuild it. That is the standard IVF-PQ
/// bargain and it is why [`IvfPqIndex::training_report`] is worth exporting to
/// a dashboard.
///
/// ```
/// use proxima_core::{Metadata, Metric, VectorId, VectorSpace};
/// use proxima_index::{IvfPqConfig, IvfPqIndex, VectorIndex};
///
/// let space = VectorSpace::new(16, Metric::Euclidean)?;
/// let config = IvfPqConfig::new(16, 8).with_nprobe(4).with_seed(5);
/// let mut index = IvfPqIndex::new(space, config)?;
///
/// for i in 0..2000u64 {
///     let cluster = (i % 20) as f32;
///     let vector: Vec<f32> = (0..16)
///         .map(|d| (cluster * 1.3 + d as f32 * 0.2).sin() * 4.0 + (i as f32 * 0.0005))
///         .collect();
///     index.insert(VectorId::new(i), &vector, Metadata::new())?;
/// }
/// let report = index.train()?;
/// assert!(report.populated_lists > 1, "{report}");
///
/// let query: Vec<f32> = (0..16).map(|d| (1.3 + d as f32 * 0.2).sin() * 4.0).collect();
/// let hits = index.search_all(&query, 10)?;
/// assert_eq!(hits.len(), 10);
///
/// // 16 dimensions of f32 is 64 bytes; the code is 8.
/// assert_eq!(index.code_bytes_per_vector(), 8);
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
#[derive(Debug, Clone)]
pub struct IvfPqIndex {
    space: VectorSpace,
    config: IvfPqConfig,
    codebooks: Option<Codebooks>,
    report: Option<TrainingReport>,

    /// Slot-to-id map, `None` for a free slot.
    ids: Vec<Option<VectorId>>,
    by_id: HashMap<VectorId, Slot>,
    metadata: Vec<Metadata>,
    slots: SlotAllocator,

    /// `slot_count * dim` prepared vectors, or empty once dropped.
    ///
    /// This index does not use `VectorStore`, unlike every other one here, for
    /// exactly one reason: it has to be able to *not* have the vectors. A store
    /// whose `vector(slot)` sometimes returns nothing would be a worse type for
    /// the two indexes that always have them.
    vectors: Vec<f32>,
    /// `slot_count * subspaces` codes, empty until trained.
    codes: Vec<u8>,
    /// Slot to list, [`UNASSIGNED`] until trained.
    list_of: Vec<u32>,
    /// List to slots, in ascending slot order.
    lists: Vec<Vec<u32>>,
}

impl IvfPqIndex {
    /// An empty, untrained index.
    pub fn new(space: VectorSpace, config: IvfPqConfig) -> Result<Self> {
        match space.metric() {
            Metric::Euclidean | Metric::Cosine => {}
            other => {
                return Err(ProximaError::config(
                    "metric",
                    other,
                    "IVF-PQ supports euclidean and cosine only; see the module docs",
                ))
            }
        }
        config.validate(space.dim())?;
        let nlist = config.nlist;
        Ok(IvfPqIndex {
            space,
            config,
            codebooks: None,
            report: None,
            ids: Vec::new(),
            by_id: HashMap::new(),
            metadata: Vec::new(),
            slots: SlotAllocator::new(),
            vectors: Vec::new(),
            codes: Vec::new(),
            list_of: Vec::new(),
            lists: vec![Vec::new(); nlist],
        })
    }

    /// The configuration in force.
    pub fn config(&self) -> &IvfPqConfig {
        &self.config
    }

    /// The space this index is over.
    pub fn space(&self) -> &VectorSpace {
        &self.space
    }

    /// `true` once codebooks exist.
    pub fn is_trained(&self) -> bool {
        self.codebooks.is_some()
    }

    /// What the last training run produced, if any.
    pub fn training_report(&self) -> Option<&TrainingReport> {
        self.report.as_ref()
    }

    /// Metadata for `id`.
    pub fn metadata(&self, id: VectorId) -> Option<&Metadata> {
        self.by_id.get(&id).map(|slot| &self.metadata[slot.index()])
    }

    /// Replace the metadata for `id`.
    pub fn set_metadata(&mut self, id: VectorId, metadata: Metadata) -> Result<()> {
        let slot = *self.by_id.get(&id).ok_or(ProximaError::UnknownId { id })?;
        self.metadata[slot.index()] = metadata;
        Ok(())
    }

    /// The stored (prepared) vector for `id`, or `None` if vectors were
    /// dropped.
    pub fn vector(&self, id: VectorId) -> Option<&[f32]> {
        if self.vectors.is_empty() {
            return None;
        }
        let slot = self.by_id.get(&id)?;
        let dim = self.space.dim();
        Some(&self.vectors[slot.index() * dim..(slot.index() + 1) * dim])
    }

    /// Live ids, in slot order.
    pub fn ids(&self) -> impl Iterator<Item = VectorId> + '_ {
        self.ids.iter().filter_map(|id| *id)
    }

    /// Bytes of PQ code per vector — the number this index exists to make small.
    pub fn code_bytes_per_vector(&self) -> usize {
        self.config.subspaces
    }

    /// Length of every inverted list, in list order.
    ///
    /// The distribution that decides the latency tail; [`TrainingReport`] has
    /// the argument.
    pub fn list_lengths(&self) -> Vec<usize> {
        self.lists.iter().map(Vec::len).collect()
    }

    /// Heap bytes held by the codes, the inverted lists and the codebooks.
    ///
    /// Everything except the full-precision vectors, which is what a deployment
    /// running with [`IvfPqConfig::with_retain_vectors`] off actually pays.
    pub fn compressed_bytes(&self) -> usize {
        let codes = self.codes.capacity();
        let lists: usize = self
            .lists
            .iter()
            .map(|list| list.capacity() * std::mem::size_of::<u32>())
            .sum();
        let list_of = self.list_of.capacity() * std::mem::size_of::<u32>();
        let books = self.codebooks.as_ref().map_or(0, |books| {
            books.centroids.capacity() * std::mem::size_of::<f32>() + books.pq.codebook_bytes()
        });
        codes + lists + list_of + books
    }

    /// Bytes held by the retained full-precision vectors, or zero.
    pub fn vector_bytes(&self) -> usize {
        self.vectors.capacity() * std::mem::size_of::<f32>()
    }

    // -----------------------------------------------------------------
    // Serialisation surface
    // -----------------------------------------------------------------

    /// A borrowed view of everything that has to be persisted.
    ///
    /// Unlike [`HnswIndex`](crate::HnswIndex), this index cannot be rebuilt
    /// from its vectors: `k-means++` seeding is deterministic given the seed and
    /// the *sample*, and the sample is a stride over the live slots, so a
    /// rebuild after any deletion would fit different centroids and assign
    /// vectors to different lists. And when the vectors have been dropped there
    /// is nothing to rebuild from at all. So the codebooks go on disk.
    pub fn parts(&self) -> IvfPqParts<'_> {
        IvfPqParts {
            ids: &self.ids,
            metadata: &self.metadata,
            vectors: &self.vectors,
            codes: &self.codes,
            list_of: &self.list_of,
            centroids: self.codebooks.as_ref().map_or(&[], |c| &c.centroids),
            codebooks: self.codebooks.as_ref().map_or(&[], |c| c.pq.codebooks()),
            report: self.report.as_ref(),
        }
    }

    /// Rebuild an index from [`IvfPqIndex::parts`].
    ///
    /// Validates the array lengths against each other; the inverted lists are
    /// reconstructed from `list_of` rather than stored, because storing both
    /// would be two copies of one fact that a corrupt file could make disagree.
    pub fn from_parts(
        space: VectorSpace,
        config: IvfPqConfig,
        parts: IvfPqPartsOwned,
    ) -> Result<Self> {
        let mut index = IvfPqIndex::new(space, config)?;
        let dim = index.space.dim();
        let code_len = index.config.subspaces;
        let slot_count = parts.ids.len();

        let check = |what: &'static str, actual: usize, expected: usize| -> Result<()> {
            if actual != expected {
                return Err(ProximaError::corrupt(
                    "ivfpq index",
                    format!("{what} holds {actual} entries, expected {expected}"),
                ));
            }
            Ok(())
        };
        check("metadata", parts.metadata.len(), slot_count)?;
        if !parts.vectors.is_empty() {
            check("vectors", parts.vectors.len(), slot_count * dim)?;
        }
        if !parts.codes.is_empty() {
            check("codes", parts.codes.len(), slot_count * code_len)?;
            check("list assignments", parts.list_of.len(), slot_count)?;
        }

        let trained = !parts.centroids.is_empty();
        if trained {
            if parts.centroids.len() % dim != 0 {
                return Err(ProximaError::corrupt(
                    "ivfpq centroids",
                    format!(
                        "{} floats is not a multiple of dimension {dim}",
                        parts.centroids.len()
                    ),
                ));
            }
            let produced = parts.centroids.len() / dim;
            let pq = ProductQuantizer::from_parts(
                dim,
                code_len,
                parts.codebooks,
                parts.report.as_ref().map_or(0.0, |r| r.residual_inertia),
                0,
            )?;
            index.codebooks = Some(Codebooks {
                centroids: parts.centroids,
                pq,
            });
            index.lists = vec![Vec::new(); produced];
        }

        for _ in 0..slot_count {
            index.slots.allocate()?;
        }
        for (slot, id) in parts.ids.iter().enumerate() {
            match id {
                Some(id) => {
                    if index.by_id.insert(*id, Slot::new(slot as u32)).is_some() {
                        return Err(ProximaError::corrupt(
                            "ivfpq index",
                            format!("id {id} appears at two slots"),
                        ));
                    }
                }
                // Free rather than tombstoned: this index releases a slot the
                // moment its vector is removed, exactly as `FlatIndex` does.
                None => index.slots.release(Slot::new(slot as u32)),
            }
        }
        for (slot, list) in parts.list_of.iter().enumerate() {
            if *list == UNASSIGNED || parts.ids[slot].is_none() {
                continue;
            }
            let list = *list as usize;
            if list >= index.lists.len() {
                return Err(ProximaError::corrupt(
                    "ivfpq index",
                    format!("slot {slot} names list {list} of {}", index.lists.len()),
                ));
            }
            index.lists[list].push(slot as u32);
        }

        index.ids = parts.ids;
        index.metadata = parts.metadata;
        index.vectors = parts.vectors;
        index.codes = parts.codes;
        index.list_of = parts.list_of;
        index.report = parts.report;
        Ok(index)
    }

    // -----------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------

    /// Fit the coarse centroids and the residual codebooks, and encode
    /// everything.
    ///
    /// Cost is `O(iterations * sample * nlist * dim)` for the coarse stage plus
    /// the PQ training, then one encode per live vector. On the benchmark
    /// fixture — 50 000 vectors of dimension 128, `nlist = 256`, `m = 32` —
    /// that is about 12 s single-threaded, and it is the number in
    /// `BENCHMARKS.md`'s build column.
    ///
    /// Retraining an already-trained index is legal and re-encodes everything
    /// from the retained vectors; it fails with
    /// [`ProximaError::Configuration`] if they were dropped.
    pub fn train(&mut self) -> Result<TrainingReport> {
        if self.vectors.is_empty() && !self.by_id.is_empty() {
            return Err(ProximaError::config(
                "retain_vectors",
                false,
                "training needs the full-precision vectors, which were dropped",
            ));
        }
        let dim = self.space.dim();
        let live: Vec<Slot> = self
            .ids
            .iter()
            .enumerate()
            .filter_map(|(index, id)| id.map(|_| Slot::new(index as u32)))
            .collect();
        if live.is_empty() {
            return Err(ProximaError::config(
                "len",
                0,
                "an empty index has nothing to fit centroids to",
            ));
        }

        // A deterministic stride, not a random sample: the corpus is already in
        // an arbitrary order and a stride is reproducible without consuming the
        // generator, which keeps the k-means seeding the only thing the seed
        // controls.
        let stride = live.len().div_ceil(self.config.training_sample).max(1);
        let sample: Vec<f32> = live
            .iter()
            .step_by(stride)
            .flat_map(|slot| self.vectors[slot.index() * dim..(slot.index() + 1) * dim].to_vec())
            .collect();
        let sampled = sample.len() / dim;

        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        let coarse = kmeans(
            &sample,
            dim,
            self.config.nlist,
            self.config.max_iterations,
            &mut rng,
        );
        // k-means clamps `k` to the sample size, so a tiny corpus produces
        // fewer centroids than asked for. Pad the list array to match what was
        // actually produced rather than to `nlist`, so that a list index is
        // always a valid centroid.
        let produced = coarse.centroids.len() / dim;

        // Residuals of the *sample*, not of the corpus: the PQ codebooks are
        // fitted to the same points the coarse stage saw, which keeps training
        // cost proportional to `training_sample` rather than to `n`.
        let mut residuals = Vec::with_capacity(sample.len());
        for vector in sample.chunks_exact(dim) {
            let list = nearest_centroid(vector, &coarse.centroids, dim, produced);
            let centroid = &coarse.centroids[list * dim..(list + 1) * dim];
            residuals.extend(vector.iter().zip(centroid).map(|(x, c)| x - c));
        }

        let pq = ProductQuantizer::train(
            &self.space,
            &residuals,
            PqConfig::new(self.config.subspaces)
                .with_seed(self.config.seed)
                .with_max_iterations(self.config.max_iterations),
        )?;
        let residual_inertia = pq.training_inertia();

        self.codebooks = Some(Codebooks {
            centroids: coarse.centroids,
            pq,
        });
        self.lists = vec![Vec::new(); produced];
        self.encode_all()?;

        if !self.config.retain_vectors {
            self.vectors = Vec::new();
            self.vectors.shrink_to_fit();
        }

        let report = TrainingReport {
            sampled,
            coarse_inertia: coarse.inertia,
            coarse_iterations: coarse.iterations,
            residual_inertia,
            populated_lists: self.lists.iter().filter(|list| !list.is_empty()).count(),
            largest_list: self.lists.iter().map(Vec::len).max().unwrap_or(0),
        };
        self.report = Some(report.clone());
        Ok(report)
    }

    /// Assign and encode every live vector from scratch.
    fn encode_all(&mut self) -> Result<()> {
        let dim = self.space.dim();
        let code_len = self.config.subspaces;
        let slot_count = self.slots.high_water();
        self.codes = vec![0u8; slot_count * code_len];
        self.list_of = vec![UNASSIGNED; slot_count];
        for list in &mut self.lists {
            list.clear();
        }

        let Some(books) = self.codebooks.as_ref() else {
            return Err(ProximaError::Internal {
                location: "ivfpq::encode_all",
                detail: "called before the codebooks were fitted".into(),
            });
        };
        let centroid_count = self.lists.len();
        let mut residual = vec![0.0f32; dim];

        for index in 0..slot_count {
            if self.ids[index].is_none() {
                continue;
            }
            let vector = &self.vectors[index * dim..(index + 1) * dim];
            let list = nearest_centroid(vector, &books.centroids, dim, centroid_count);
            let centroid = &books.centroids[list * dim..(list + 1) * dim];
            for ((slot, x), c) in residual.iter_mut().zip(vector).zip(centroid) {
                *slot = x - c;
            }
            books.pq.encode_into(
                &residual,
                &mut self.codes[index * code_len..(index + 1) * code_len],
            )?;
            self.list_of[index] = list as u32;
            self.lists[list].push(index as u32);
        }
        Ok(())
    }

    /// Encode one already-prepared vector at `slot`, appending it to its list.
    fn encode_slot(&mut self, slot: Slot, vector: &[f32]) -> Result<()> {
        let Some(books) = self.codebooks.as_ref() else {
            return Ok(());
        };
        let dim = self.space.dim();
        let code_len = self.config.subspaces;
        let index = slot.index();
        let list = nearest_centroid(vector, &books.centroids, dim, self.lists.len());
        let centroid = &books.centroids[list * dim..(list + 1) * dim];
        let residual: Vec<f32> = vector.iter().zip(centroid).map(|(x, c)| x - c).collect();

        if self.codes.len() < (index + 1) * code_len {
            self.codes.resize((index + 1) * code_len, 0);
        }
        if self.list_of.len() < index + 1 {
            self.list_of.resize(index + 1, UNASSIGNED);
        }
        let code = books.pq.encode(&residual)?;
        self.codes[index * code_len..(index + 1) * code_len].copy_from_slice(&code);
        self.list_of[index] = list as u32;

        // Ascending slot order, so that a scan of the list visits slots in the
        // same order a scan of the whole store would and ties break the same
        // way. `insert` into a sorted `Vec` is `O(len)`, which is fine because
        // a well-sized list holds a few hundred entries.
        let position = self.lists[list]
            .binary_search(&(index as u32))
            .unwrap_or_else(|position| position);
        self.lists[list].insert(position, index as u32);
        Ok(())
    }

    /// Drop tombstoned slots and renumber, shrinking every array.
    ///
    /// Unlike a graph index, IVF-PQ has no edges to repair: a removed slot is
    /// simply taken out of its list. Compaction is therefore only about giving
    /// memory back, and it does not touch the codebooks — the centroids stay
    /// exactly where they were, so search results before and after a compaction
    /// are identical. Returns the number of slots reclaimed.
    pub fn compact(&mut self) -> Result<usize> {
        let reclaimed = self.slots.high_water() - self.by_id.len();
        if reclaimed == 0 {
            return Ok(0);
        }
        let dim = self.space.dim();
        let code_len = self.config.subspaces;
        let keep: Vec<usize> = (0..self.ids.len())
            .filter(|index| self.ids[*index].is_some())
            .collect();

        let mut vectors = Vec::with_capacity(if self.vectors.is_empty() {
            0
        } else {
            keep.len() * dim
        });
        let mut codes = Vec::with_capacity(if self.codes.is_empty() {
            0
        } else {
            keep.len() * code_len
        });
        let mut metadata = Vec::with_capacity(keep.len());
        let mut ids = Vec::with_capacity(keep.len());
        let mut list_of = Vec::with_capacity(keep.len());

        for old in &keep {
            if !self.vectors.is_empty() {
                vectors.extend_from_slice(&self.vectors[old * dim..(old + 1) * dim]);
            }
            if !self.codes.is_empty() {
                codes.extend_from_slice(&self.codes[old * code_len..(old + 1) * code_len]);
            }
            metadata.push(std::mem::take(&mut self.metadata[*old]));
            ids.push(self.ids[*old]);
            if !self.list_of.is_empty() {
                list_of.push(self.list_of[*old]);
            }
        }

        self.by_id.clear();
        for (new, id) in ids.iter().enumerate() {
            if let Some(id) = id {
                self.by_id.insert(*id, Slot::new(new as u32));
            }
        }
        self.slots.clear();
        for _ in 0..keep.len() {
            self.slots.allocate()?;
        }
        for list in &mut self.lists {
            list.clear();
        }
        for (new, list) in list_of.iter().enumerate() {
            if *list != UNASSIGNED {
                self.lists[*list as usize].push(new as u32);
            }
        }

        self.vectors = vectors;
        self.codes = codes;
        self.metadata = metadata;
        self.ids = ids;
        self.list_of = list_of;
        Ok(reclaimed)
    }

    // -----------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------

    /// Search with an explicit `nprobe`, overriding [`IvfPqConfig::nprobe`].
    ///
    /// The runtime dial, the counterpart of
    /// [`HnswIndex::search_with_ef`](crate::HnswIndex::search_with_ef).
    pub fn search_with_nprobe(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        filter: &dyn Filter,
    ) -> Result<Vec<SearchResult>> {
        check_k(k)?;
        let prepared = self.space.prepare(query)?;
        Ok(self.search_prepared(
            &prepared,
            k,
            nprobe.clamp(1, self.lists.len().max(1)),
            filter,
        ))
    }

    fn search_prepared(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        filter: &dyn Filter,
    ) -> Vec<SearchResult> {
        if k == 0 || self.by_id.is_empty() {
            return Vec::new();
        }
        let Some(books) = self.codebooks.as_ref() else {
            // Untrained: there are no codes to scan, and the vectors are all
            // still here, so the only correct answer is the exact one.
            return self.exhaustive(query, k, filter);
        };

        let dim = self.space.dim();
        // Score every centroid and keep the nearest `nprobe`. `O(nlist * dim)`,
        // which at nlist = 256 and dim = 128 is 32 768 multiply-adds — the same
        // order as scanning one list, and the reason `nlist` much above
        // `sqrt(n)` stops paying.
        let mut coarse: Vec<(f32, usize)> = (0..self.lists.len())
            .map(|list| {
                let centroid = &books.centroids[list * dim..(list + 1) * dim];
                (squared_l2(query, centroid), list)
            })
            .collect();
        coarse.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        coarse.truncate(nprobe);

        // How many candidates to carry into reranking. Reranking needs a
        // shortlist longer than `k`, or it has nothing to reorder.
        let shortlist = if self.config.rerank > 0 {
            self.config.rerank.max(k)
        } else {
            k
        };
        let mut top = TopK::new(shortlist);
        let code_len = self.config.subspaces;

        // Cosine's table is query-only and therefore built once; Euclidean's
        // depends on the residual `q - c` and is rebuilt per probed list. That
        // asymmetry is worth the branch: at nprobe = 16 it is fifteen fewer
        // `256 * dim` table builds per query.
        let shared_table: Option<DistanceTable> = match self.space.metric() {
            Metric::Cosine => books.pq.distance_table(query).ok(),
            _ => None,
        };
        let mut residual = vec![0.0f32; dim];
        let mut per_list_table: Option<DistanceTable> = None;

        for (_, list) in &coarse {
            let centroid = &books.centroids[list * dim..(list + 1) * dim];
            if shared_table.is_none() {
                for ((slot, q), c) in residual.iter_mut().zip(query).zip(centroid) {
                    *slot = q - c;
                }
                match books.pq.distance_table(&residual) {
                    Ok(table) => per_list_table = Some(table),
                    // Only reachable on a dimension mismatch, which the space
                    // has already ruled out; skipping the list is still better
                    // than returning nothing at all.
                    Err(_) => continue,
                }
            }
            let Some(table) = shared_table.as_ref().or(per_list_table.as_ref()) else {
                continue;
            };
            // `q . c`, needed by the cosine decomposition and unused by the
            // Euclidean one, which folds the centroid into the residual table.
            let query_dot_centroid = if shared_table.is_some() {
                dot(query, centroid)
            } else {
                0.0
            };

            for slot in &self.lists[*list] {
                let index = *slot as usize;
                let Some(id) = self.ids[index] else { continue };
                if !filter.accepts(id, &self.metadata[index]) {
                    continue;
                }
                let code = &self.codes[index * code_len..(index + 1) * code_len];
                let score = match self.space.metric() {
                    // `1 - <q, c + r>` split into the exact centroid term and
                    // the approximate residual term the table supplies.
                    Metric::Cosine => 1.0 - query_dot_centroid + table.inner_product(code),
                    // `||q - (c + r)||^2 = ||(q - c) - r||^2`, and the table was
                    // built on `q - c`, so this is the whole distance.
                    _ => table.squared_l2(code),
                };
                top.push(id, score);
            }
        }

        let candidates = top.into_sorted_vec();
        if self.config.rerank == 0 || self.vectors.is_empty() {
            return candidates.into_iter().take(k).collect();
        }

        // Exact rescoring of the shortlist. `rerank * dim` multiply-adds, which
        // at rerank = 100 and dim = 128 is about the cost of scanning two more
        // lists — and buys far more recall than two more probes do.
        let mut exact = TopK::new(k);
        for hit in candidates {
            let Some(slot) = self.by_id.get(&hit.id) else {
                continue;
            };
            let vector = &self.vectors[slot.index() * dim..(slot.index() + 1) * dim];
            exact.push(hit.id, self.space.distance(query, vector));
        }
        exact.into_sorted_vec()
    }

    /// Exhaustive scan, used before training and by nothing else.
    fn exhaustive(&self, query: &[f32], k: usize, filter: &dyn Filter) -> Vec<SearchResult> {
        let dim = self.space.dim();
        let mut top = TopK::new(k);
        for (index, id) in self.ids.iter().enumerate() {
            let Some(id) = id else { continue };
            if !filter.accepts(*id, &self.metadata[index]) {
                continue;
            }
            let vector = &self.vectors[index * dim..(index + 1) * dim];
            top.push(*id, self.space.distance(query, vector));
        }
        top.into_sorted_vec()
    }
}

impl VectorIndex for IvfPqIndex {
    fn insert(&mut self, id: VectorId, vector: &[f32], metadata: Metadata) -> Result<()> {
        let prepared = self.space.prepare(vector)?.into_owned();
        if self.by_id.contains_key(&id) {
            return Err(ProximaError::DuplicateId { id });
        }
        if self.codebooks.is_some() && !self.config.retain_vectors {
            // Nothing to store the vector in, and nothing that needs it beyond
            // this call: it is encoded and discarded.
            let slot = self.slots.allocate()?;
            self.place(slot, id, metadata);
            self.encode_slot(slot, &prepared)?;
            return Ok(());
        }

        let dim = self.space.dim();
        let slot = self.slots.allocate()?;
        let index = slot.index();
        if index == self.ids.len() {
            self.vectors.extend_from_slice(&prepared);
        } else {
            self.vectors[index * dim..(index + 1) * dim].copy_from_slice(&prepared);
        }
        self.place(slot, id, metadata);
        self.encode_slot(slot, &prepared)?;

        if self.codebooks.is_none() && self.by_id.len() >= self.config.train_threshold {
            self.train()?;
        }
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<()> {
        let slot = self
            .by_id
            .remove(&id)
            .ok_or(ProximaError::UnknownId { id })?;
        let index = slot.index();
        self.ids[index] = None;
        self.metadata[index] = Metadata::new();
        if let Some(list) = self.list_of.get_mut(index) {
            if *list != UNASSIGNED {
                let target = *list as usize;
                *list = UNASSIGNED;
                self.lists[target].retain(|entry| *entry != index as u32);
            }
        }
        // No edges point at this slot, so it goes straight back on the free
        // list — the same reasoning `FlatIndex` uses, and the reason IVF-PQ
        // needs no tombstones.
        self.slots.release(slot);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize, filter: &dyn Filter) -> Result<Vec<SearchResult>> {
        self.search_with_nprobe(query, k, self.config.nprobe, filter)
    }

    fn contains(&self, id: VectorId) -> bool {
        self.by_id.contains_key(&id)
    }

    fn len(&self) -> usize {
        self.by_id.len()
    }

    fn stats(&self) -> IndexStats {
        IndexStats {
            len: self.by_id.len(),
            // Slots go straight back on the free list, so there are no
            // tombstones to report; `compact` reclaims the array holes instead.
            deleted: 0,
            dim: self.space.dim(),
            metric: self.space.metric(),
            layers: 1,
            nodes_per_layer: vec![self.by_id.len()],
            edges: 0,
            vector_bytes: self.vector_bytes(),
            // Codes, lists and codebooks all land under `graph_bytes`, which is
            // the field's real meaning: everything the index adds on top of the
            // vectors. For this index it is the *only* thing, once the vectors
            // are dropped.
            graph_bytes: self.compressed_bytes(),
        }
    }
}

impl IvfPqIndex {
    /// Record an id and its metadata at `slot`, growing the arrays if needed.
    fn place(&mut self, slot: Slot, id: VectorId, metadata: Metadata) {
        let index = slot.index();
        if index == self.ids.len() {
            self.ids.push(Some(id));
            self.metadata.push(metadata);
        } else {
            self.ids[index] = Some(id);
            self.metadata[index] = metadata;
        }
        self.by_id.insert(id, slot);
    }
}

/// Index of the centroid nearest to `vector`, ties to the lower index.
fn nearest_centroid(vector: &[f32], centroids: &[f32], dim: usize, count: usize) -> usize {
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for list in 0..count {
        let centroid = &centroids[list * dim..(list + 1) * dim];
        let distance = squared_l2(vector, centroid);
        if distance < best_distance {
            best_distance = distance;
            best = list;
        }
    }
    best
}

/// Squared L2, local so that the coarse stage does not go through `Metric`.
#[inline]
fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    proxima_core::distance::squared_euclidean(a, b)
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    proxima_core::distance::dot(a, b)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::FlatIndex;

    /// Clustered fixture, the same shape the benchmarks use.
    fn corpus(count: u64, dim: usize, clusters: u64) -> Vec<Vec<f32>> {
        (0..count)
            .map(|i| {
                let cluster = (i % clusters) as f32;
                (0..dim)
                    .map(|d| {
                        (cluster * 1.7 + d as f32 * 0.11).sin() * 5.0
                            + ((i / clusters) as f32 * 0.013 * (d as f32 + 1.0)).cos()
                    })
                    .collect()
            })
            .collect()
    }

    fn build(
        count: u64,
        dim: usize,
        config: IvfPqConfig,
    ) -> (IvfPqIndex, FlatIndex, Vec<Vec<f32>>) {
        let space = VectorSpace::new(dim, Metric::Euclidean).unwrap();
        let mut index = IvfPqIndex::new(space.clone(), config).unwrap();
        let mut flat = FlatIndex::new(space);
        let data = corpus(count, dim, 24);
        for (i, vector) in data.iter().enumerate() {
            index
                .insert(VectorId::new(i as u64), vector, Metadata::new())
                .unwrap();
            flat.insert(VectorId::new(i as u64), vector, Metadata::new())
                .unwrap();
        }
        (index, flat, data)
    }

    #[test]
    fn rejects_metrics_it_cannot_serve() {
        for metric in [Metric::InnerProduct, Metric::Manhattan] {
            let space = VectorSpace::new(8, metric).unwrap();
            let err = IvfPqIndex::new(space, IvfPqConfig::new(4, 4)).unwrap_err();
            assert!(
                matches!(
                    err,
                    ProximaError::Configuration {
                        parameter: "metric",
                        ..
                    }
                ),
                "{err}"
            );
        }
    }

    #[test]
    fn an_untrained_index_is_exact() {
        // Before training there are no codes, so the only honest answer is the
        // exhaustive one — and it must be exactly the flat index's.
        let (index, flat, data) = build(
            200,
            16,
            IvfPqConfig::new(8, 4).with_train_threshold(usize::MAX),
        );
        assert!(!index.is_trained());
        for query in data.iter().take(10) {
            assert_eq!(
                index.search_all(query, 5).unwrap(),
                flat.search_all(query, 5).unwrap()
            );
        }
    }

    #[test]
    fn training_populates_lists_and_encodes_everything() {
        let (mut index, _, _) = build(
            1000,
            16,
            IvfPqConfig::new(16, 8)
                .with_seed(3)
                .with_train_threshold(usize::MAX),
        );
        let report = index.train().unwrap();
        assert!(index.is_trained());
        assert_eq!(report.sampled, 1000);
        assert!(report.populated_lists > 1, "{report}");
        assert_eq!(index.list_lengths().iter().sum::<usize>(), 1000);
        assert_eq!(index.code_bytes_per_vector(), 8);
        assert!(!report.to_string().is_empty());
    }

    #[test]
    fn auto_training_fires_at_the_threshold() {
        let space = VectorSpace::new(8, Metric::Euclidean).unwrap();
        let mut index =
            IvfPqIndex::new(space, IvfPqConfig::new(4, 4).with_train_threshold(50)).unwrap();
        for (i, vector) in corpus(60, 8, 6).iter().enumerate() {
            index
                .insert(VectorId::new(i as u64), vector, Metadata::new())
                .unwrap();
            assert_eq!(index.is_trained(), i + 1 >= 50, "after {} inserts", i + 1);
        }
        // Everything inserted before *and* after the threshold is encoded.
        assert_eq!(index.list_lengths().iter().sum::<usize>(), 60);
    }

    #[test]
    fn probing_more_lists_never_lowers_recall_much_and_usually_raises_it() {
        let (mut index, flat, data) = build(
            2000,
            32,
            IvfPqConfig::new(32, 16)
                .with_seed(11)
                .with_train_threshold(usize::MAX),
        );
        index.train().unwrap();

        let queries: Vec<f32> = data.iter().step_by(40).flatten().copied().collect();
        let mut previous = 0.0f32;
        for nprobe in [1usize, 4, 16, 32] {
            let mut hits = 0usize;
            let mut total = 0usize;
            for query in queries.chunks_exact(32) {
                let truth: Vec<VectorId> = flat
                    .search_all(query, 10)
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.id)
                    .collect();
                let got = index
                    .search_with_nprobe(query, 10, nprobe, &crate::AcceptAll)
                    .unwrap();
                hits += got.iter().filter(|hit| truth.contains(&hit.id)).count();
                total += truth.len();
            }
            let recall = hits as f32 / total as f32;
            assert!(
                recall + 0.02 >= previous,
                "recall fell from {previous} to {recall} at nprobe {nprobe}"
            );
            previous = recall;
        }
        // Scanning every list is the ceiling the codes allow.
        assert!(previous > 0.4, "recall at nprobe = nlist was {previous}");
    }

    #[test]
    fn reranking_beats_the_same_index_without_it() {
        // The claim the module docs make, asserted rather than asserted-about.
        let dim = 32;
        let base = IvfPqConfig::new(32, 8)
            .with_seed(7)
            .with_nprobe(8)
            .with_train_threshold(usize::MAX);
        let (mut plain, flat, data) = build(2000, dim, base.clone());
        plain.train().unwrap();
        let (mut reranked, _, _) = build(2000, dim, base.with_rerank(100));
        reranked.train().unwrap();

        let measure = |index: &IvfPqIndex| {
            let mut hits = 0usize;
            let mut total = 0usize;
            for query in data.iter().step_by(40) {
                let truth: Vec<VectorId> = flat
                    .search_all(query, 10)
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.id)
                    .collect();
                let got = index.search_all(query, 10).unwrap();
                hits += got.iter().filter(|hit| truth.contains(&hit.id)).count();
                total += truth.len();
            }
            hits as f32 / total as f32
        };
        let plain_recall = measure(&plain);
        let reranked_recall = measure(&reranked);
        assert!(
            reranked_recall > plain_recall,
            "reranking must help: {reranked_recall} vs {plain_recall}"
        );
    }

    #[test]
    fn dropping_the_vectors_leaves_only_the_codes() {
        let (mut index, _, _) = build(
            1000,
            64,
            IvfPqConfig::new(16, 8)
                .with_seed(2)
                .with_retain_vectors(false)
                .with_train_threshold(usize::MAX),
        );
        index.train().unwrap();
        assert_eq!(index.vector_bytes(), 0);
        assert!(index.vector(VectorId::new(0)).is_none());
        // 64 dimensions of f32 is 256 bytes; the whole compressed structure is
        // a fraction of that per vector.
        let per_vector = index.compressed_bytes() as f32 / index.len() as f32;
        assert!(per_vector < 256.0, "{per_vector} bytes per vector");
        // ...and it still answers queries.
        assert_eq!(index.search_all(&[0.0; 64], 5).unwrap().len(), 5);
        // A later insert is encoded and its vector discarded.
        index
            .insert(VectorId::new(9999), &vec![1.0; 64], Metadata::new())
            .unwrap();
        assert_eq!(index.vector_bytes(), 0);
        assert!(index.contains(VectorId::new(9999)));
    }

    #[test]
    fn retraining_without_vectors_is_refused_rather_than_silently_wrong() {
        let (mut index, _, _) = build(
            300,
            16,
            IvfPqConfig::new(8, 4)
                .with_retain_vectors(false)
                .with_train_threshold(usize::MAX),
        );
        index.train().unwrap();
        let err = index.train().unwrap_err();
        assert!(
            matches!(
                err,
                ProximaError::Configuration {
                    parameter: "retain_vectors",
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn remove_takes_the_slot_out_of_its_list() {
        let (mut index, _, _) = build(
            400,
            16,
            IvfPqConfig::new(8, 4)
                .with_seed(1)
                .with_train_threshold(usize::MAX),
        );
        index.train().unwrap();
        let before: usize = index.list_lengths().iter().sum();
        index.remove(VectorId::new(5)).unwrap();
        assert_eq!(index.list_lengths().iter().sum::<usize>(), before - 1);
        assert!(!index.contains(VectorId::new(5)));
        assert!(index
            .search_all(&[0.0; 16], 400)
            .unwrap()
            .iter()
            .all(|hit| hit.id != VectorId::new(5)));
        assert!(matches!(
            index.remove(VectorId::new(5)),
            Err(ProximaError::UnknownId { .. })
        ));
    }

    #[test]
    fn compaction_does_not_change_results() {
        // The claim in `compact`'s docs: no edges, so nothing moves logically.
        let (mut index, _, data) = build(
            600,
            16,
            IvfPqConfig::new(8, 4)
                .with_seed(4)
                .with_train_threshold(usize::MAX),
        );
        index.train().unwrap();
        for id in (0..600).step_by(3) {
            index.remove(VectorId::new(id)).unwrap();
        }
        let before: Vec<Vec<SearchResult>> = data
            .iter()
            .step_by(50)
            .map(|q| index.search_all(q, 10).unwrap())
            .collect();
        let reclaimed = index.compact().unwrap();
        assert_eq!(reclaimed, 200);
        let after: Vec<Vec<SearchResult>> = data
            .iter()
            .step_by(50)
            .map(|q| index.search_all(q, 10).unwrap())
            .collect();
        assert_eq!(before, after);
        assert_eq!(index.compact().unwrap(), 0);
    }

    #[test]
    fn filters_are_applied_during_the_scan() {
        let (mut index, _, data) = build(
            800,
            16,
            IvfPqConfig::new(16, 8)
                .with_nprobe(16)
                .with_seed(6)
                .with_train_threshold(usize::MAX),
        );
        index.train().unwrap();
        for id in 0..800u64 {
            index
                .set_metadata(VectorId::new(id), Metadata::new().with("even", id % 2 == 0))
                .unwrap();
        }
        let hits = index
            .search(&data[0], 10, &|_id, m: &Metadata| {
                m.bool("even") == Some(true)
            })
            .unwrap();
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|hit| hit.id.get() % 2 == 0));
    }

    #[test]
    fn cosine_scores_agree_with_the_exact_ones_within_quantization_error() {
        // The cosine decomposition `1 - q.c - q.r` is the one piece of algebra
        // in this module that is not obviously right, so it is checked against
        // the exact distance of the vectors it claims to approximate.
        let dim = 32;
        let space = VectorSpace::new(dim, Metric::Cosine).unwrap();
        let config = IvfPqConfig::new(8, 16)
            .with_nprobe(8)
            .with_seed(9)
            .with_train_threshold(usize::MAX);
        let mut index = IvfPqIndex::new(space.clone(), config).unwrap();
        let data = corpus(1000, dim, 20);
        for (i, vector) in data.iter().enumerate() {
            index
                .insert(VectorId::new(i as u64), vector, Metadata::new())
                .unwrap();
        }
        index.train().unwrap();

        let query = space.prepare(&data[3]).unwrap();
        for hit in index.search_all(&data[3], 20).unwrap() {
            let stored = index.vector(hit.id).unwrap();
            let exact = space.distance(&query, stored);
            assert!(
                (hit.distance - exact).abs() < 0.25,
                "approximate {} vs exact {exact} for id {}",
                hit.distance,
                hit.id
            );
        }
    }

    #[test]
    fn same_seed_same_index() {
        let config = IvfPqConfig::new(16, 8)
            .with_seed(42)
            .with_train_threshold(usize::MAX);
        let (mut a, _, data) = build(800, 16, config.clone());
        let (mut b, _, _) = build(800, 16, config);
        a.train().unwrap();
        b.train().unwrap();
        assert_eq!(a.list_lengths(), b.list_lengths());
        for query in data.iter().step_by(80) {
            assert_eq!(
                a.search_all(query, 10).unwrap(),
                b.search_all(query, 10).unwrap()
            );
        }
    }

    #[test]
    fn duplicate_ids_are_refused() {
        let (mut index, _, data) = build(
            100,
            16,
            IvfPqConfig::new(4, 4).with_train_threshold(usize::MAX),
        );
        assert!(matches!(
            index.insert(VectorId::new(0), &data[0], Metadata::new()),
            Err(ProximaError::DuplicateId { .. })
        ));
    }

    #[test]
    fn stats_report_the_compressed_footprint() {
        let (mut index, _, _) = build(
            1000,
            64,
            IvfPqConfig::new(16, 8)
                .with_seed(8)
                .with_train_threshold(usize::MAX),
        );
        index.train().unwrap();
        let stats = index.stats();
        assert_eq!(stats.len, 1000);
        assert_eq!(stats.dim, 64);
        assert_eq!(stats.deleted, 0);
        assert!(stats.graph_bytes >= 1000 * 8, "codes alone are 8 KiB");
        assert!(stats.vector_bytes >= 1000 * 64 * 4);
    }
}
