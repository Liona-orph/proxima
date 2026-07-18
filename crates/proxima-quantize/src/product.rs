//! Product quantization: `m` bytes per vector, whatever the dimension.
//!
//! # The idea
//!
//! A codebook that could represent a 768-dimensional vector to any useful
//! accuracy would need astronomically many entries. PQ sidesteps that by
//! splitting the vector into `m` contiguous **subspaces** and quantizing each
//! one independently against its own 256-entry codebook. The effective codebook
//! is the Cartesian product of the `m` small ones — `256^m` points — while the
//! storage is `m` bytes and the training is `m` independent k-means runs over
//! `dim/m` dimensions.
//!
//! A 768-dimensional `f32` vector is 3072 bytes. At `m = 96` it is 96 bytes:
//! **32x**. At `m = 16`, 16 bytes: **192x**. That is the entire reason PQ
//! exists, and the accuracy cost is real and measured in
//! `tests/recall.rs` rather than asserted here.
//!
//! # Why the distance table is the whole trick
//!
//! Comparing a query against a code could be done by reconstructing the vector
//! and running a normal kernel: `dim` multiply-adds per candidate, plus the
//! reconstruction. The distance table replaces that with `m` loads and `m`
//! adds — for `dim = 768, m = 96`, eight times fewer operations, and none of
//! them multiplies.
//!
//! It works because the subspaces are independent under a decomposable metric:
//!
//! ```text
//! ||q - x||^2 = sum over subspaces s of ||q_s - centroid[s][code_s]||^2
//! ```
//!
//! Each term on the right depends only on the query (fixed for the whole
//! search) and on one byte of the code. So before scanning, compute
//! `table[s][c] = ||q_s - centroid[s][c]||^2` for all `m * 256` pairs — one
//! `m * 256 * (dim/m)` = `256 * dim` operation, amortised over every candidate
//! in the scan — and then each candidate costs `m` table lookups and `m - 1`
//! additions. The table is `m * 256 * 4` bytes, 96 KiB at `m = 96`, which sits
//! in L2; at the more usual `m = 16` it is 16 KiB and sits in L1.
//!
//! This is an **asymmetric** computation: the query is never quantized, so the
//! only error is the stored vector's. See [`DistanceTable`].
//!
//! # Out of scope: OPQ
//!
//! An *Optimized* Product Quantizer learns a rotation `R` and quantizes `Rx`
//! instead of `x`, so that variance is spread evenly across subspaces rather
//! than concentrated in whichever contiguous slice of dimensions happens to
//! carry it. It reliably buys a few points of recall on real embeddings.
//!
//! It is deliberately **not implemented here**. It needs an eigendecomposition
//! (or an iterative SVD) inside the training loop, which means either a linear
//! algebra dependency — against this project's whole premise — or several
//! hundred lines of numerical code whose correctness is far harder to establish
//! than anything else in this repository. A caller who needs OPQ can apply
//! their own rotation to the vectors before they reach Proxima and get exactly
//! the same result; the API does not stand in the way.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use proxima_core::{ProximaError, Result, VectorSpace};

use crate::kmeans::kmeans;

/// Entries per subspace codebook.
///
/// 256, so that a code fits in exactly one byte. Smaller codebooks (16 entries,
/// two codes per byte) exist in the literature and trade accuracy for a bit
/// more compression; they also require bit-unpacking in the scan's inner loop,
/// which costs more than the memory saves at these sizes.
pub const CENTROIDS_PER_SUBSPACE: usize = 256;

/// Training configuration.
///
/// ```
/// use proxima_quantize::PqConfig;
///
/// // 16 subspaces, the usual starting point for a 768-dimensional model.
/// let config = PqConfig::new(16).with_seed(42).with_max_iterations(25);
/// assert_eq!(config.subspaces(), 16);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PqConfig {
    subspaces: usize,
    max_iterations: usize,
    seed: u64,
}

impl PqConfig {
    /// `m` subspaces, with default training settings.
    pub const fn new(subspaces: usize) -> Self {
        PqConfig {
            subspaces,
            // 25 is where the inertia curve flattens on every corpus this was
            // tried against; the marginal iteration past ~20 moves centroids by
            // less than the quantization step they produce.
            max_iterations: 25,
            seed: 0x5EED_1EAF,
        }
    }

    /// Set the k-means seed.
    ///
    /// The same seed and the same training data give the same codebooks, byte
    /// for byte. That is a hard guarantee, not a tendency: the generator is
    /// ChaCha8, whose stream is specified.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Cap on Lloyd iterations per subspace.
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Number of subspaces, which is also the code length in bytes.
    pub const fn subspaces(&self) -> usize {
        self.subspaces
    }

    /// The k-means seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Cap on Lloyd iterations per subspace.
    pub const fn max_iterations(&self) -> usize {
        self.max_iterations
    }
}

/// A trained product quantizer.
///
/// ```
/// use proxima_core::{Metric, VectorSpace};
/// use proxima_quantize::{PqConfig, ProductQuantizer};
///
/// // 8-dimensional vectors in two clusters, split into 4 subspaces of 2.
/// let space = VectorSpace::new(8, Metric::Euclidean)?;
/// let mut training = Vec::new();
/// for i in 0..256 {
///     let t = i as f32 / 255.0;
///     training.extend_from_slice(&[t, -t, t * 2.0, 1.0 - t, t, t, -t, t]);
/// }
///
/// let pq = ProductQuantizer::train(&space, &training, PqConfig::new(4))?;
/// assert_eq!(pq.code_len(), 4);
/// assert_eq!(pq.compression_ratio(), 8.0);   // 32 bytes -> 4
///
/// let query = [0.5, -0.5, 1.0, 0.5, 0.5, 0.5, -0.5, 0.5];
/// let code = pq.encode(&query)?;
///
/// // The table is built once per query and then reused for every candidate.
/// let table = pq.distance_table(&query)?;
/// let approximate = table.squared_l2(&code);
/// assert!(approximate < 0.05, "{approximate}");
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    dim: usize,
    subspaces: usize,
    /// Dimension of each subspace. `subspace_dims[s]` differs by at most one
    /// across `s` when `dim` is not divisible by `subspaces`.
    subspace_dims: Vec<usize>,
    /// Start offset of each subspace within a vector; `subspaces + 1` entries.
    offsets: Vec<usize>,
    /// Codebooks, concatenated: subspace `s` occupies
    /// `codebook_offsets[s] .. codebook_offsets[s+1]`, laid out as
    /// `CENTROIDS_PER_SUBSPACE` centroids of `subspace_dims[s]` floats.
    codebooks: Vec<f32>,
    codebook_offsets: Vec<usize>,
    /// Total k-means inertia over all subspaces, i.e. the training
    /// reconstruction error. Reported rather than hidden because it is the one
    /// number that says whether `m` was chosen well.
    training_inertia: f64,
    /// Lloyd iterations summed over the subspaces. Equal to
    /// `subspaces * max_iterations` exactly when every subspace hit the cap,
    /// which is the signal that `max_iterations` was too low.
    training_iterations: usize,
}

impl ProductQuantizer {
    /// Train `m` codebooks on a flat `n * dim` training array.
    ///
    /// Subspaces are contiguous slices of the dimension. When `dim` is not
    /// divisible by `m` the remainder is spread one dimension at a time over
    /// the leading subspaces, so widths differ by at most one — the alternative
    /// (one fat trailing subspace) would make that subspace's codebook
    /// systematically worse than the others.
    ///
    /// Cost is `O(iterations * n * 256 * dim)` — the `m` cancels, because each
    /// of the `m` k-means runs sees `dim/m` dimensions. Training 100k vectors
    /// of dimension 768 for 25 iterations is on the order of a minute
    /// single-threaded; it is a build-time cost paid once.
    pub fn train(space: &VectorSpace, training: &[f32], config: PqConfig) -> Result<Self> {
        let dim = space.dim();
        if config.subspaces == 0 || config.subspaces > dim {
            return Err(ProximaError::config(
                "subspaces",
                config.subspaces,
                "must be between 1 and the space dimension",
            ));
        }
        if config.max_iterations == 0 {
            return Err(ProximaError::config(
                "max_iterations",
                config.max_iterations,
                "must be at least 1",
            ));
        }
        if training.is_empty() || training.len() % dim != 0 {
            return Err(ProximaError::config(
                "training",
                training.len(),
                "must be a non-empty multiple of the space dimension",
            ));
        }
        let count = training.len() / dim;
        for vector in training.chunks_exact(dim) {
            space.validate(vector)?;
        }

        let subspaces = config.subspaces;
        let base = dim / subspaces;
        let remainder = dim % subspaces;
        let subspace_dims: Vec<usize> = (0..subspaces)
            .map(|s| base + usize::from(s < remainder))
            .collect();
        let mut offsets = Vec::with_capacity(subspaces + 1);
        let mut cursor = 0;
        for width in &subspace_dims {
            offsets.push(cursor);
            cursor += width;
        }
        offsets.push(cursor);

        let mut codebooks = Vec::new();
        let mut codebook_offsets = Vec::with_capacity(subspaces + 1);
        let mut training_inertia = 0.0;
        let mut training_iterations = 0;
        // One generator, advanced across subspaces rather than reseeded per
        // subspace: reseeding from `seed + s` would correlate the streams, and
        // ChaCha's guarantee is about one stream, not about a family of them.
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

        // Scratch buffer for the projection of the training set onto one
        // subspace. Reused across subspaces so training allocates once.
        let mut projected = Vec::with_capacity(count * subspace_dims[0]);

        for subspace in 0..subspaces {
            let (start, end) = (offsets[subspace], offsets[subspace + 1]);
            let width = end - start;
            projected.clear();
            for vector in training.chunks_exact(dim) {
                projected.extend_from_slice(&vector[start..end]);
            }

            let clustering = kmeans(
                &projected,
                width,
                CENTROIDS_PER_SUBSPACE,
                config.max_iterations,
                &mut rng,
            );
            training_inertia += clustering.inertia;
            training_iterations += clustering.iterations;
            codebook_offsets.push(codebooks.len());

            // k-means clamps k to the number of training points. Pad the
            // codebook out to a full 256 entries by repeating the last centroid
            // so that every byte value decodes to *something*; the padded
            // entries are unreachable from `encode` because they are duplicates
            // and ties go to the lower index.
            let produced = clustering.centroids.len() / width;
            codebooks.extend_from_slice(&clustering.centroids);
            for _ in produced..CENTROIDS_PER_SUBSPACE {
                let last = clustering.centroids.len() - width;
                codebooks.extend_from_slice(&clustering.centroids[last..]);
            }
        }
        codebook_offsets.push(codebooks.len());

        Ok(ProductQuantizer {
            dim,
            subspaces,
            subspace_dims,
            offsets,
            codebooks,
            codebook_offsets,
            training_inertia,
            training_iterations,
        })
    }

    /// The dimension this quantizer was trained for.
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Number of subspaces, `m`.
    pub const fn subspaces(&self) -> usize {
        self.subspaces
    }

    /// Bytes per code: one per subspace.
    pub const fn code_len(&self) -> usize {
        self.subspaces
    }

    /// Uncompressed bytes divided by compressed bytes.
    ///
    /// Excludes the codebooks, which are `m * 256 * (dim/m) * 4` = `1024 * dim`
    /// bytes total — 786 KiB at dimension 768, independent of `m` and amortised
    /// over the whole corpus. Use [`ProductQuantizer::codebook_bytes`] if the
    /// corpus is small enough for that to matter.
    pub fn compression_ratio(&self) -> f32 {
        (self.dim * std::mem::size_of::<f32>()) as f32 / self.code_len() as f32
    }

    /// Bytes of trained codebook.
    pub fn codebook_bytes(&self) -> usize {
        self.codebooks.len() * std::mem::size_of::<f32>()
    }

    /// Total k-means inertia across subspaces after training.
    ///
    /// Sum of squared reconstruction error over the training set. Compare it
    /// between two values of `m` on the same data: it is the cheapest available
    /// proxy for the recall each will deliver, and it costs nothing to look at.
    pub const fn training_inertia(&self) -> f64 {
        self.training_inertia
    }

    /// Lloyd iterations summed over all subspaces.
    ///
    /// If this equals `subspaces() * config.max_iterations()`, every subspace
    /// stopped because it ran out of iterations rather than because it
    /// converged, and the codebooks are probably worth retraining with a higher
    /// cap. Anything below that means at least one subspace converged.
    pub const fn training_iterations(&self) -> usize {
        self.training_iterations
    }

    /// Quantize one vector to `m` bytes.
    ///
    /// `O(dim * 256)`: each subspace is compared against all 256 of its
    /// centroids. That is 200k operations for a 768-dimensional vector, roughly
    /// 60x the cost of a single distance evaluation — an insert-time cost, and
    /// the reason a bulk index build wants [`ProductQuantizer::encode_into`]
    /// and a reused buffer.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.subspaces];
        self.encode_into(vector, &mut out)?;
        Ok(out)
    }

    /// [`ProductQuantizer::encode`] into a caller-owned buffer.
    pub fn encode_into(&self, vector: &[f32], out: &mut [u8]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(ProximaError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        if out.len() != self.subspaces {
            return Err(ProximaError::config(
                "out",
                out.len(),
                "output buffer must be code_len() bytes",
            ));
        }
        for subspace in 0..self.subspaces {
            let slice = &vector[self.offsets[subspace]..self.offsets[subspace + 1]];
            out[subspace] = self.nearest_centroid(subspace, slice);
        }
        Ok(())
    }

    /// Reconstruct an approximation of the original vector from its code.
    ///
    /// The reconstruction is the concatenation of the chosen centroids. Useful
    /// for reranking and for measuring quantization error; it is *not* how
    /// distances are computed — see [`ProductQuantizer::distance_table`].
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>> {
        if code.len() != self.subspaces {
            return Err(ProximaError::DimensionMismatch {
                expected: self.subspaces,
                actual: code.len(),
            });
        }
        let mut out = Vec::with_capacity(self.dim);
        for (subspace, byte) in code.iter().enumerate() {
            out.extend_from_slice(self.centroid(subspace, *byte));
        }
        Ok(out)
    }

    /// Precompute query-to-centroid distances for every subspace.
    ///
    /// This is the operation the module docs are about. Cost is
    /// `O(256 * dim)` once; every subsequent candidate then costs `m` lookups.
    /// Build one per query, before the scan, and hold it for the scan's
    /// duration.
    pub fn distance_table(&self, query: &[f32]) -> Result<DistanceTable> {
        if query.len() != self.dim {
            return Err(ProximaError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let mut squared = vec![0.0f32; self.subspaces * CENTROIDS_PER_SUBSPACE];
        let mut dot = vec![0.0f32; self.subspaces * CENTROIDS_PER_SUBSPACE];
        for subspace in 0..self.subspaces {
            let slice = &query[self.offsets[subspace]..self.offsets[subspace + 1]];
            let width = self.subspace_dims[subspace];
            let book = &self.codebooks
                [self.codebook_offsets[subspace]..self.codebook_offsets[subspace + 1]];
            let row = subspace * CENTROIDS_PER_SUBSPACE;
            for (centroid_index, centroid) in book.chunks_exact(width).enumerate() {
                let mut l2 = 0.0f32;
                let mut ip = 0.0f32;
                for (q, c) in slice.iter().zip(centroid) {
                    let d = q - c;
                    l2 += d * d;
                    ip += q * c;
                }
                squared[row + centroid_index] = l2;
                dot[row + centroid_index] = ip;
            }
        }
        Ok(DistanceTable {
            subspaces: self.subspaces,
            squared,
            dot,
        })
    }

    /// The concatenated codebooks, for serialisation.
    ///
    /// Layout is subspace-major: subspace `s` occupies a contiguous run of
    /// `256 * subspace_dims[s]` floats, and the widths are derived from `dim`
    /// and `subspaces` by the same rule [`ProductQuantizer::train`] uses. That
    /// derivation is why the persisted form needs only these three numbers and
    /// not the offset tables.
    pub fn codebooks(&self) -> &[f32] {
        &self.codebooks
    }

    /// Rebuild a quantizer from [`ProductQuantizer::codebooks`].
    ///
    /// The deserialisation constructor. It recomputes the subspace widths and
    /// offsets rather than trusting stored ones, so a snapshot cannot describe
    /// a layout `encode` would not produce; the only thing it has to trust is
    /// the float array, and the snapshot's checksum covers that.
    ///
    /// `training_inertia` and `training_iterations` are carried through because
    /// they are the only evidence of how well the codebook was fitted, and
    /// losing them on every restart would mean the number that says whether `m`
    /// was chosen well is available exactly once.
    pub fn from_parts(
        dim: usize,
        subspaces: usize,
        codebooks: Vec<f32>,
        training_inertia: f64,
        training_iterations: usize,
    ) -> Result<Self> {
        if subspaces == 0 || subspaces > dim {
            return Err(ProximaError::config(
                "subspaces",
                subspaces,
                "must be between 1 and the space dimension",
            ));
        }
        let base = dim / subspaces;
        let remainder = dim % subspaces;
        let subspace_dims: Vec<usize> = (0..subspaces)
            .map(|s| base + usize::from(s < remainder))
            .collect();
        let mut offsets = Vec::with_capacity(subspaces + 1);
        let mut codebook_offsets = Vec::with_capacity(subspaces + 1);
        let mut cursor = 0;
        let mut codebook_cursor = 0;
        for width in &subspace_dims {
            offsets.push(cursor);
            codebook_offsets.push(codebook_cursor);
            cursor += width;
            codebook_cursor += width * CENTROIDS_PER_SUBSPACE;
        }
        offsets.push(cursor);
        codebook_offsets.push(codebook_cursor);

        if codebooks.len() != codebook_cursor {
            return Err(ProximaError::corrupt(
                "product quantizer",
                format!(
                    "{} codebook floats for {subspaces} subspaces of dimension {dim}, expected {codebook_cursor}",
                    codebooks.len()
                ),
            ));
        }

        Ok(ProductQuantizer {
            dim,
            subspaces,
            subspace_dims,
            offsets,
            codebooks,
            codebook_offsets,
            training_inertia,
            training_iterations,
        })
    }

    /// The centroid a code byte names, in a given subspace.
    #[inline]
    fn centroid(&self, subspace: usize, byte: u8) -> &[f32] {
        let width = self.subspace_dims[subspace];
        let base = self.codebook_offsets[subspace] + byte as usize * width;
        &self.codebooks[base..base + width]
    }

    /// Index of the nearest centroid to `slice` within one subspace.
    ///
    /// Ties go to the lower index, which is what makes `encode` deterministic
    /// in the presence of the padded duplicate entries.
    fn nearest_centroid(&self, subspace: usize, slice: &[f32]) -> u8 {
        let width = self.subspace_dims[subspace];
        let book =
            &self.codebooks[self.codebook_offsets[subspace]..self.codebook_offsets[subspace + 1]];
        let mut best = 0usize;
        let mut best_distance = f32::INFINITY;
        for (index, centroid) in book.chunks_exact(width).enumerate() {
            let mut sum = 0.0f32;
            for (x, c) in slice.iter().zip(centroid) {
                let d = x - c;
                sum += d * d;
            }
            if sum < best_distance {
                best_distance = sum;
                best = index;
            }
        }
        best as u8
    }
}

/// Precomputed query-to-centroid distances for one query.
///
/// Built by [`ProductQuantizer::distance_table`] and then consulted once per
/// candidate. Holds `m * 256` entries for each supported metric family:
/// squared L2 and dot product. Both are computed in the same pass because the
/// pass is dominated by loading the codebooks, and computing the second one
/// costs one extra multiply-add per centroid component.
///
/// The scoring methods take a code and are the hottest loop in a quantized
/// scan; they do not validate the code's length, for the same reason
/// [`ScalarQuantizer::asymmetric_squared_l2`](crate::ScalarQuantizer::asymmetric_squared_l2)
/// does not.
#[derive(Debug, Clone)]
pub struct DistanceTable {
    subspaces: usize,
    /// `m * 256` squared L2 distances, subspace-major.
    squared: Vec<f32>,
    /// `m * 256` dot products, subspace-major.
    dot: Vec<f32>,
}

impl DistanceTable {
    /// Approximate squared L2 between the query and the vector `code` encodes.
    ///
    /// `m` loads and `m` adds, independent of the dimension.
    ///
    /// # Panics
    ///
    /// If `code` is longer than the table's subspace count.
    #[inline]
    pub fn squared_l2(&self, code: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (subspace, byte) in code.iter().enumerate() {
            sum += self.squared[subspace * CENTROIDS_PER_SUBSPACE + *byte as usize];
        }
        sum
    }

    /// Approximate *negated* dot product, ranking min-first like everything
    /// else in Proxima.
    ///
    /// # Panics
    ///
    /// If `code` is longer than the table's subspace count.
    #[inline]
    pub fn inner_product(&self, code: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (subspace, byte) in code.iter().enumerate() {
            sum += self.dot[subspace * CENTROIDS_PER_SUBSPACE + *byte as usize];
        }
        -sum
    }

    /// Number of subspaces the table covers.
    pub const fn subspaces(&self) -> usize {
        self.subspaces
    }

    /// Bytes the table occupies.
    ///
    /// The number to check against your L1 and L2 sizes: a table that spills to
    /// L3 turns each of the `m` lookups into a cache miss and undoes the whole
    /// optimisation.
    pub fn bytes(&self) -> usize {
        (self.squared.len() + self.dot.len()) * std::mem::size_of::<f32>()
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

    /// A training set with genuine cluster structure in every subspace, so that
    /// k-means has something to find.
    fn clustered(dim: usize, count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let centres: Vec<Vec<f32>> = (0..8)
            .map(|c| (0..dim).map(|d| ((c * 7 + d * 3) % 11) as f32).collect())
            .collect();
        let mut out = Vec::with_capacity(count * dim);
        for i in 0..count {
            let centre = &centres[i % centres.len()];
            for value in centre {
                out.push(value + next() * 0.4);
            }
        }
        out
    }

    #[test]
    fn code_length_and_compression_are_what_they_claim() {
        let training = clustered(64, 512, 1);
        let pq = ProductQuantizer::train(&space(64), &training, PqConfig::new(8)).unwrap();
        assert_eq!(pq.code_len(), 8);
        assert_eq!(pq.encode(&training[..64]).unwrap().len(), 8);
        // 64 f32 = 256 bytes, down to 8.
        assert_eq!(pq.compression_ratio(), 32.0);
    }

    #[test]
    fn uneven_split_spreads_the_remainder() {
        // dim 10, m 4 -> widths 3, 3, 2, 2 rather than 2, 2, 2, 4.
        let training = clustered(10, 300, 2);
        let pq = ProductQuantizer::train(&space(10), &training, PqConfig::new(4)).unwrap();
        assert_eq!(pq.subspace_dims, vec![3, 3, 2, 2]);
        assert_eq!(
            pq.decode(&pq.encode(&training[..10]).unwrap())
                .unwrap()
                .len(),
            10
        );
    }

    #[test]
    fn the_table_agrees_with_reconstruct_and_measure() {
        // The identity the whole optimisation rests on: summing per-subspace
        // table entries equals the squared distance to the reconstruction.
        let training = clustered(32, 512, 3);
        let pq = ProductQuantizer::train(&space(32), &training, PqConfig::new(8)).unwrap();
        let query = &training[100 * 32..101 * 32];
        let table = pq.distance_table(query).unwrap();

        for vector in training.chunks_exact(32).take(64) {
            let code = pq.encode(vector).unwrap();
            let via_table = table.squared_l2(&code);
            let reconstructed = pq.decode(&code).unwrap();
            let direct = proxima_core::distance::squared_euclidean(query, &reconstructed);
            assert!(
                (via_table - direct).abs() <= 1e-3 * direct.max(1.0),
                "{via_table} vs {direct}"
            );

            let via_table = table.inner_product(&code);
            let direct = proxima_core::distance::inner_product(query, &reconstructed);
            assert!(
                (via_table - direct).abs() <= 1e-3 * direct.abs().max(1.0),
                "{via_table} vs {direct}"
            );
        }
    }

    #[test]
    fn training_is_reproducible_from_the_seed() {
        let training = clustered(32, 512, 4);
        let a =
            ProductQuantizer::train(&space(32), &training, PqConfig::new(8).with_seed(7)).unwrap();
        let b =
            ProductQuantizer::train(&space(32), &training, PqConfig::new(8).with_seed(7)).unwrap();
        assert_eq!(a.codebooks, b.codebooks);
        assert_eq!(a.training_inertia(), b.training_inertia());
        assert_eq!(a.training_iterations(), b.training_iterations());
        assert!(
            a.training_iterations() < a.subspaces() * PqConfig::new(8).max_iterations(),
            "every subspace hit the iteration cap; the default is too low"
        );

        let c =
            ProductQuantizer::train(&space(32), &training, PqConfig::new(8).with_seed(8)).unwrap();
        assert_ne!(
            a.codebooks, c.codebooks,
            "different seeds gave one codebook"
        );
    }

    #[test]
    fn more_subspaces_means_less_error() {
        // The central trade-off, asserted rather than described.
        let training = clustered(32, 1024, 5);
        let coarse = ProductQuantizer::train(&space(32), &training, PqConfig::new(2)).unwrap();
        let fine = ProductQuantizer::train(&space(32), &training, PqConfig::new(16)).unwrap();
        assert!(
            fine.training_inertia() < coarse.training_inertia(),
            "m=16 inertia {} not below m=2 inertia {}",
            fine.training_inertia(),
            coarse.training_inertia()
        );
        assert!(fine.compression_ratio() < coarse.compression_ratio());
    }

    #[test]
    fn m_equal_to_dim_is_exact_scalar_clustering() {
        // With one dimension per subspace and 256 centroids over a training set
        // of 200 distinct values, k-means can place a centroid on every point,
        // so reconstruction should be essentially lossless.
        let training: Vec<f32> = (0..200).map(|i| i as f32 * 0.25).collect();
        let pq = ProductQuantizer::train(&space(1), &training, PqConfig::new(1)).unwrap();
        for value in &training {
            let decoded = pq.decode(&pq.encode(&[*value]).unwrap()).unwrap();
            assert!(
                (decoded[0] - value).abs() < 1e-3,
                "{} vs {}",
                decoded[0],
                value
            );
        }
    }

    #[test]
    fn tiny_training_set_still_produces_a_full_codebook() {
        // k-means clamps k to n; the padding keeps every byte value decodable.
        let training = clustered(8, 3, 6);
        let pq = ProductQuantizer::train(&space(8), &training, PqConfig::new(2)).unwrap();
        for byte in [0u8, 1, 2, 200, 255] {
            assert_eq!(pq.decode(&[byte, byte]).unwrap().len(), 8);
        }
    }

    #[test]
    fn configuration_is_validated_at_training_time() {
        let training = clustered(8, 32, 7);
        assert!(matches!(
            ProductQuantizer::train(&space(8), &training, PqConfig::new(0)),
            Err(ProximaError::Configuration {
                parameter: "subspaces",
                ..
            })
        ));
        assert!(matches!(
            ProductQuantizer::train(&space(8), &training, PqConfig::new(9)),
            Err(ProximaError::Configuration {
                parameter: "subspaces",
                ..
            })
        ));
        assert!(matches!(
            ProductQuantizer::train(
                &space(8),
                &training,
                PqConfig::new(2).with_max_iterations(0)
            ),
            Err(ProximaError::Configuration {
                parameter: "max_iterations",
                ..
            })
        ));
        assert!(matches!(
            ProductQuantizer::train(&space(8), &[1.0, 2.0], PqConfig::new(2)),
            Err(ProximaError::Configuration {
                parameter: "training",
                ..
            })
        ));
    }

    #[test]
    fn table_size_is_reported_honestly() {
        let training = clustered(32, 256, 8);
        let pq = ProductQuantizer::train(&space(32), &training, PqConfig::new(8)).unwrap();
        let table = pq.distance_table(&training[..32]).unwrap();
        assert_eq!(table.subspaces(), 8);
        assert_eq!(table.bytes(), 8 * 256 * 4 * 2);
    }
}
