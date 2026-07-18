//! Measured recall loss and compression ratio for both quantizers.
//!
//! Quantization is lossy, so a quantizer without a recall number attached is a
//! claim with no content. This test builds a fixed synthetic corpus, runs exact
//! search to get ground truth, then runs the same queries through each
//! quantizer and asserts recall@10 stays above a stated floor.
//!
//! The floors are set just below what the run actually produces, so they will
//! catch a regression and will not fail spuriously; the run is deterministic
//! (seeded generator, seeded k-means) so "just below" is meaningful.
//!
//! The measured numbers, printed by this test with `--nocapture`, on 5 000
//! vectors of dimension 128 drawn from 32 isotropic Gaussian clusters, 200
//! queries, `cargo test --release`:
//!
//! ```text
//! scalar u8 per-dimension    4.0x   recall@10 0.972
//! scalar u8 global           4.0x   recall@10 0.960
//! product m=32              16.0x   recall@10 0.473
//! product m=16              32.0x   recall@10 0.308
//! product m=8               64.0x   recall@10 0.219
//! product m=8 + rerank 10x  64.0x   recall@10 0.833
//! ```
//!
//! **This fixture is close to the worst case for product quantization, and the
//! numbers should be read that way.** Isotropic Gaussian noise in 128
//! dimensions has no low-dimensional structure for a subspace codebook to
//! exploit: the within-cluster variance is spread evenly over every dimension,
//! so each 4-dimensional subspace sees an almost spherical blob and 256
//! centroids buy relatively little. Real embeddings concentrate near a much
//! lower-dimensional manifold, which is exactly the structure PQ was designed
//! for, and published figures on SIFT-like data are correspondingly higher.
//!
//! Two things follow, and both are the point of running this at all:
//!
//! * The *ordering* is what generalises. More subspaces means more accuracy and
//!   less compression, monotonically, and the exchange rate is steep at the
//!   aggressive end.
//! * Aggressive PQ is not used bare. Reranking the top `10k` PQ candidates with
//!   exact distances takes `m = 8` from 0.219 to 0.833 for 100 extra distance
//!   evaluations per query, which is why 64x compression is deployable at all.
//!
//! Raising `PqConfig::max_iterations` from 25 to 80 changes none of these
//! figures, so the codebooks are converged; the loss is representational, not a
//! training artefact.

#![allow(clippy::unwrap_used)]

use proxima_core::{Metric, VectorSpace};
use proxima_quantize::{CodeType, PqConfig, ProductQuantizer, RangeMode, ScalarQuantizer};

const DIM: usize = 128;
const CORPUS: usize = 5_000;
const QUERIES: usize = 200;
const K: usize = 10;

/// splitmix64, so the fixture is reproducible without a dependency and without
/// depending on `rand`'s version-to-version behaviour.
struct Prng(u64);

impl Prng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / f32::from(1u16 << 12) - 1.0
    }

    /// Approximately standard normal, via the central limit theorem on six
    /// uniforms. Box-Muller would be more correct and needs a `ln` and a `cos`
    /// per pair; the fixture only needs "clustered blobs", not exact Gaussians.
    fn next_normal(&mut self) -> f32 {
        (0..6).map(|_| self.next_f32()).sum::<f32>() / 2.0
    }
}

/// `count` vectors drawn from 32 Gaussian clusters.
///
/// Clustered rather than uniform because uniform high-dimensional data has no
/// structure for a quantizer to exploit and would understate every method
/// equally — real embeddings are clustered, and the ranking between `m` values
/// is what this test is for.
fn corpus(count: usize, seed: u64) -> Vec<f32> {
    let mut rng = Prng(seed);
    let centres: Vec<Vec<f32>> = (0..32)
        .map(|_| (0..DIM).map(|_| rng.next_normal() * 3.0).collect())
        .collect();
    let mut out = Vec::with_capacity(count * DIM);
    for i in 0..count {
        let centre = &centres[i % centres.len()];
        for value in centre {
            out.push(value + rng.next_normal() * 0.6);
        }
    }
    out
}

/// Exact top-`K` neighbour ids for one query, by brute force.
fn exact_top_k(corpus: &[f32], query: &[f32]) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = corpus
        .chunks_exact(DIM)
        .enumerate()
        .map(|(i, v)| (proxima_core::distance::squared_euclidean(query, v), i))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().take(K).map(|(_, i)| i).collect()
}

/// Fraction of the exact top-`K` that appears in `approximate`.
fn recall(exact: &[usize], approximate: &[usize]) -> f32 {
    let hits = approximate.iter().filter(|i| exact.contains(i)).count();
    hits as f32 / exact.len() as f32
}

/// Top-`K` by a caller-supplied scoring function.
fn top_k_by(count: usize, mut score: impl FnMut(usize) -> f32) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = (0..count).map(|i| (score(i), i)).collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().take(K).map(|(_, i)| i).collect()
}

#[test]
fn measured_recall_and_compression() {
    let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
    let data = corpus(CORPUS, 0x1234_5678);
    let queries = corpus(QUERIES, 0x8765_4321);
    let ground_truth: Vec<Vec<usize>> = queries
        .chunks_exact(DIM)
        .map(|q| exact_top_k(&data, q))
        .collect();

    // ---- scalar quantization ------------------------------------------
    for (label, mode, floor) in [
        ("scalar u8 per-dimension", RangeMode::PerDimension, 0.95),
        ("scalar u8 global", RangeMode::Global, 0.94),
    ] {
        let sq = ScalarQuantizer::train(&space, &data, mode, CodeType::U8).unwrap();
        let codes: Vec<Vec<u8>> = data
            .chunks_exact(DIM)
            .map(|v| sq.encode(v).unwrap())
            .collect();

        let mut total = 0.0f32;
        for (query, exact) in queries.chunks_exact(DIM).zip(&ground_truth) {
            let approximate = top_k_by(CORPUS, |i| sq.asymmetric_squared_l2(query, &codes[i]));
            total += recall(exact, &approximate);
        }
        let mean = total / QUERIES as f32;
        println!(
            "{label:26} {:5.1}x   recall@{K} {mean:.3}",
            sq.compression_ratio()
        );
        assert!(
            (sq.compression_ratio() - 4.0).abs() < 1e-6,
            "scalar quantization is always exactly 4x"
        );
        assert!(mean >= floor, "{label}: recall {mean} below floor {floor}");
    }

    // ---- product quantization -----------------------------------------
    // Floors sit ~0.02 below the measured value: enough headroom that a
    // different libm or a reordered sum does not fail the build, tight enough
    // that a real regression does. They are low in absolute terms because the
    // fixture is hard; see the module docs.
    for (m, expected_ratio, floor) in [(32usize, 16.0f32, 0.45), (16, 32.0, 0.29), (8, 64.0, 0.20)]
    {
        let pq =
            ProductQuantizer::train(&space, &data, PqConfig::new(m).with_seed(0xA5A5)).unwrap();
        let codes: Vec<Vec<u8>> = data
            .chunks_exact(DIM)
            .map(|v| pq.encode(v).unwrap())
            .collect();

        let mut total = 0.0f32;
        for (query, exact) in queries.chunks_exact(DIM).zip(&ground_truth) {
            let table = pq.distance_table(query).unwrap();
            let approximate = top_k_by(CORPUS, |i| table.squared_l2(&codes[i]));
            total += recall(exact, &approximate);
        }
        let mean = total / QUERIES as f32;
        println!(
            "product m={m:<20} {:5.1}x   recall@{K} {mean:.3}",
            pq.compression_ratio()
        );
        assert!((pq.compression_ratio() - expected_ratio).abs() < 1e-6);
        assert!(mean >= floor, "m={m}: recall {mean} below floor {floor}");
    }

    // ---- the standard remedy: rerank the PQ shortlist exactly ----------
    // Aggressive PQ plus a full-precision rerank of 10x the requested k is how
    // 64x compression is actually deployed. It costs 100 exact distance
    // evaluations per query against 5 000 approximate ones, and it recovers
    // most of the loss -- which is the point worth making, because the 0.219
    // figure above reads like a disqualification and is not one.
    let pq = ProductQuantizer::train(&space, &data, PqConfig::new(8).with_seed(0xA5A5)).unwrap();
    let codes: Vec<Vec<u8>> = data
        .chunks_exact(DIM)
        .map(|v| pq.encode(v).unwrap())
        .collect();

    let mut total = 0.0f32;
    for (query, exact) in queries.chunks_exact(DIM).zip(&ground_truth) {
        let table = pq.distance_table(query).unwrap();
        let mut shortlist: Vec<(f32, usize)> = (0..CORPUS)
            .map(|i| (table.squared_l2(&codes[i]), i))
            .collect();
        shortlist.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        shortlist.truncate(10 * K);

        let mut reranked: Vec<(f32, usize)> = shortlist
            .into_iter()
            .map(|(_, i)| {
                (
                    proxima_core::distance::squared_euclidean(query, &data[i * DIM..(i + 1) * DIM]),
                    i,
                )
            })
            .collect();
        reranked.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let approximate: Vec<usize> = reranked.into_iter().take(K).map(|(_, i)| i).collect();
        total += recall(exact, &approximate);
    }
    let mean = total / QUERIES as f32;
    println!("product m=8 + rerank 10x    64.0x   recall@{K} {mean:.3}");
    assert!(
        mean >= 0.80,
        "reranking should recover most of the loss, got {mean}"
    );
}

#[test]
fn quantization_error_shrinks_monotonically_with_code_length() {
    // A weaker but much cheaper claim than recall, checked over more settings:
    // more subspaces cannot make reconstruction worse.
    let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
    let data = corpus(1_000, 0xBEEF);

    let mut previous = f64::INFINITY;
    for m in [4usize, 8, 16, 32, 64] {
        let pq = ProductQuantizer::train(&space, &data, PqConfig::new(m).with_seed(11)).unwrap();
        let error: f64 = data
            .chunks_exact(DIM)
            .map(|v| {
                let decoded = pq.decode(&pq.encode(v).unwrap()).unwrap();
                f64::from(proxima_core::distance::squared_euclidean(v, &decoded))
            })
            .sum();
        assert!(
            error < previous,
            "m={m} reconstruction error {error} did not improve on {previous}"
        );
        previous = error;
    }
}
