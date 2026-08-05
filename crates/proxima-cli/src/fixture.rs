//! The generated dataset every benchmark in this repository is measured on.
//!
//! # The fixture, and the one methodological choice worth arguing about
//!
//! `n` vectors of dimension `d`, drawn from `clusters` isotropic Gaussian
//! clusters with unit within-cluster standard deviation and cluster centres at
//! scale 4. Queries are **perturbed corpus points**: a stored vector plus
//! Gaussian noise at 0.25.
//!
//! That query construction is the choice. Sampling queries from *fresh* cluster
//! centres puts every query far outside the corpus, where the hundreds of
//! members of the nearest cluster all sit within a few percent of the same
//! distance and the "true" top ten is decided by noise in the fifth decimal
//! place. Recall measured that way comes out around 0.7 for a *correct*
//! implementation and is a measurement of the fixture, not the index. A real
//! query is an embedding of something similar to something stored, which is
//! what this models. `crates/proxima-index/BENCHMARKS.md` says the same thing
//! at greater length, because it is the number everything else rests on.
//!
//! # It is generated, not downloaded
//!
//! `sift-1m` or `glove-100` would be more convincing and would make
//! `proxima bench` a command that needs the network, a cache directory and a
//! checksum. Everything here is one seeded `ChaCha8Rng` away from reproducible,
//! and the seed is a flag. The cost is that a generated Gaussian mixture is
//! *kind* to a graph index and *hostile* to product quantization — isotropic
//! clusters have no low-dimensional structure for a subspace codebook to
//! exploit — so the absolute IVF-PQ recall figures here are pessimistic. That
//! is stated wherever they are reported.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A generated corpus and a matching query set.
#[derive(Debug, Clone)]
pub(crate) struct Fixture {
    /// `count * dim` corpus vectors.
    pub(crate) vectors: Vec<f32>,
    /// `queries * dim` query vectors.
    pub(crate) queries: Vec<f32>,
    /// Dimension.
    pub(crate) dim: usize,
    /// Corpus size.
    pub(crate) count: usize,
}

impl Fixture {
    /// Corpus vectors, one at a time.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.vectors.chunks_exact(self.dim)
    }

    /// Query vectors, one at a time.
    pub(crate) fn queries(&self) -> impl Iterator<Item = &[f32]> {
        self.queries.chunks_exact(self.dim)
    }

    /// Bytes the raw corpus occupies as `f32`.
    ///
    /// The denominator for every "bytes per vector" figure.
    pub(crate) fn corpus_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
    }
}

/// Generate a clustered corpus and a perturbed query set.
///
/// Deterministic given `seed`: the same seed produces the same bytes, on every
/// machine and every release, because the generator is ChaCha8 and its stream is
/// specified.
pub(crate) fn generate(
    count: usize,
    dim: usize,
    clusters: usize,
    queries: usize,
    seed: u64,
) -> Fixture {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Cluster centres at scale 4: far enough apart that a nearest-neighbour
    // query has a meaningful answer, close enough that the clusters overlap at
    // the edges and the graph has to do real work.
    let centres: Vec<f32> = (0..clusters * dim)
        .map(|_| gaussian(&mut rng) * 4.0)
        .collect();

    let mut vectors = Vec::with_capacity(count * dim);
    for index in 0..count {
        let cluster = index % clusters;
        for d in 0..dim {
            vectors.push(centres[cluster * dim + d] + gaussian(&mut rng));
        }
    }

    // Queries are corpus points plus noise, taken at a stride so they are
    // spread across the clusters rather than concentrated in the first few.
    let stride = (count / queries.max(1)).max(1);
    let mut query_vectors = Vec::with_capacity(queries * dim);
    for q in 0..queries {
        let source = (q * stride) % count;
        for d in 0..dim {
            query_vectors.push(vectors[source * dim + d] + gaussian(&mut rng) * 0.25);
        }
    }

    Fixture {
        vectors,
        queries: query_vectors,
        dim,
        count,
    }
}

/// One standard normal sample, by Box-Muller.
///
/// Four lines rather than a `rand_distr` dependency, which is the workspace's
/// stated position on distributions it needs exactly one of.
fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
    // `gen` yields `[0, 1)`; `ln(0)` is `-inf`, so clamp away from zero.
    let u1: f32 = rng.gen::<f32>().max(f32::MIN_POSITIVE);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_bytes() {
        let a = generate(200, 8, 4, 10, 7);
        let b = generate(200, 8, 4, 10, 7);
        assert_eq!(a.vectors, b.vectors);
        assert_eq!(a.queries, b.queries);

        let c = generate(200, 8, 4, 10, 8);
        assert_ne!(a.vectors, c.vectors);
    }

    #[test]
    fn the_shape_is_what_was_asked_for() {
        let fixture = generate(500, 16, 10, 25, 1);
        assert_eq!(fixture.vectors.len(), 500 * 16);
        assert_eq!(fixture.queries.len(), 25 * 16);
        assert_eq!(fixture.iter().count(), 500);
        assert_eq!(fixture.queries().count(), 25);
        assert_eq!(fixture.corpus_bytes(), 500 * 16 * 4);
    }

    #[test]
    fn everything_generated_is_finite() {
        // A single NaN in a fixture would silently remove a vector from every
        // search, which is the failure `ProximaError::NonFinite` exists to
        // catch — but a benchmark that never inserts would just report a lower
        // number, so it is checked here too.
        let fixture = generate(2000, 32, 16, 50, 3);
        assert!(fixture.vectors.iter().all(|v| v.is_finite()));
        assert!(fixture.queries.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_query_is_nearer_to_its_source_than_to_a_random_point() {
        // The property the fixture design rests on: the true nearest neighbour
        // of a query is a specific corpus point, not a coin flip among a
        // cluster.
        let fixture = generate(1000, 32, 20, 20, 5);
        let stride = 1000 / 20;
        let at = |index: usize| &fixture.vectors[index * 32..(index + 1) * 32];
        for (q, query) in fixture.queries().enumerate() {
            let source_index = (q * stride) % 1000;
            if source_index == 0 {
                continue;
            }
            let near = proxima_core::distance::squared_euclidean(query, at(source_index));
            let far = proxima_core::distance::squared_euclidean(query, at(0));
            assert!(near < far, "query {q}: {near} vs {far}");
        }
    }
}
