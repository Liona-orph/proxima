//! Lloyd's algorithm with k-means++ seeding, over a flat vector array.
//!
//! [`ProductQuantizer`](crate::ProductQuantizer) runs it once per subspace on a
//! few-dimensional slice of the training set, and `IvfPqIndex` (in
//! `proxima-index`) runs it once over whole vectors to build its coarse
//! quantizer.
//!
//! It was private until the second of those existed, on the argument that a
//! general k-means nothing general uses is dead weight. The coarse quantizer
//! falsified that: it wants exactly this interface — a flat array in, centroids
//! and an inertia out, seeded and deterministic — and a second copy of Lloyd's
//! algorithm in another crate would be two implementations to keep agreeing
//! about tie-breaking, which is the one detail that decides whether two builds
//! from the same seed produce the same index. So it is public, with the
//! interface unchanged.
//!
//! # Why k-means++ and not random initialisation
//!
//! Lloyd's algorithm converges to a local optimum, and which one depends
//! entirely on where it starts. Uniformly random initial centroids routinely
//! place several of them inside one dense cluster and none in another, which
//! costs a whole codebook entry and shows up directly as recall loss. k-means++
//! picks each new centre with probability proportional to its squared distance
//! from the nearest centre already chosen — so the far-away cluster is
//! *likely*, not merely possible — and comes with an `O(log k)` expected
//! approximation guarantee that uniform seeding has nothing comparable to.
//!
//! The cost is `k` extra passes over the training set during seeding, which is
//! the same order as one Lloyd iteration; measured against 25 iterations of
//! Lloyd it is under 5 % of training time.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// Result of clustering one subspace.
#[derive(Debug, Clone)]
pub struct Clustering {
    /// `k * dim` floats, centroid `c` at `c*dim .. (c+1)*dim`.
    pub centroids: Vec<f32>,
    /// Sum of squared distances from each point to its assigned centroid.
    ///
    /// The objective Lloyd's algorithm minimises, kept because it is the only
    /// honest way to compare two runs and because it is what the quantizer
    /// reports as its training error.
    pub inertia: f64,
    /// Iterations actually run, which is less than the cap when it converged.
    pub iterations: usize,
}

/// Cluster `points` (a flat `n * dim` array) into `k` centroids.
///
/// `k` is clamped to `n`: asking for more centroids than points is not an
/// error, it just means some codes will never be used, and returning an error
/// would make every caller special-case a tiny training set.
///
/// Deterministic given `rng`. Ties in the assignment step break towards the
/// lower centroid index, which matters because otherwise two runs with the same
/// seed could disagree on a point equidistant from two centres.
pub fn kmeans(
    points: &[f32],
    dim: usize,
    k: usize,
    max_iterations: usize,
    rng: &mut ChaCha8Rng,
) -> Clustering {
    let n = if dim == 0 { 0 } else { points.len() / dim };
    let k = k.min(n).max(1);
    if n == 0 {
        return Clustering {
            centroids: vec![0.0; k * dim],
            inertia: 0.0,
            iterations: 0,
        };
    }

    let mut centroids = kmeans_plus_plus(points, dim, k, rng);
    let mut assignment = vec![0u32; n];
    let mut counts = vec![0u32; k];
    // Accumulate in f64: a subspace of 100k points summed in f32 loses the low
    // bits of the mean, and the loss compounds over 25 iterations.
    let mut sums = vec![0.0f64; k * dim];
    let mut inertia = 0.0f64;
    let mut iterations = 0;

    for iteration in 0..max_iterations {
        iterations = iteration + 1;
        let mut changed = false;
        inertia = 0.0;

        for (point_index, point) in points.chunks_exact(dim).enumerate() {
            let (best, best_distance) = nearest(point, &centroids, dim, k);
            inertia += f64::from(best_distance);
            if assignment[point_index] != best as u32 {
                assignment[point_index] = best as u32;
                changed = true;
            }
        }

        if !changed && iteration > 0 {
            // Assignments are stable, so the centroids would not move either.
            break;
        }

        counts.iter_mut().for_each(|c| *c = 0);
        sums.iter_mut().for_each(|s| *s = 0.0);
        for (point_index, point) in points.chunks_exact(dim).enumerate() {
            let cluster = assignment[point_index] as usize;
            counts[cluster] += 1;
            let base = cluster * dim;
            for (sum, value) in sums[base..base + dim].iter_mut().zip(point) {
                *sum += f64::from(*value);
            }
        }

        for (cluster, count) in counts.iter().enumerate() {
            if *count == 0 {
                // An empty cluster wastes a code, so it is re-seeded onto the
                // point currently worst served by its own centroid. That is the
                // standard fix and it is strictly better than leaving the
                // centroid where it is: the objective cannot increase, and the
                // wasted code is recovered.
                let victim = farthest_point(points, dim, &centroids, k).to_vec();
                let base = cluster * dim;
                centroids[base..base + dim].copy_from_slice(&victim);
                continue;
            }
            let base = cluster * dim;
            let inverse = 1.0 / f64::from(*count);
            for (centroid, sum) in centroids[base..base + dim]
                .iter_mut()
                .zip(&sums[base..base + dim])
            {
                *centroid = (sum * inverse) as f32;
            }
        }
    }

    Clustering {
        centroids,
        inertia,
        iterations,
    }
}

/// k-means++ seeding: `D^2` sampling from a seeded PRNG.
fn kmeans_plus_plus(points: &[f32], dim: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let n = points.len() / dim;
    let mut centroids = Vec::with_capacity(k * dim);

    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(&points[first * dim..(first + 1) * dim]);

    // Distance to the nearest chosen centre, maintained incrementally: adding a
    // centre can only ever *decrease* these, so each round is one pass rather
    // than a full recomputation against all chosen centres.
    let mut nearest_squared: Vec<f32> = points
        .chunks_exact(dim)
        .map(|p| squared(p, &centroids[..dim]))
        .collect();

    for chosen in 1..k {
        let total: f64 = nearest_squared.iter().map(|d| f64::from(*d)).sum();
        let picked = if total <= 0.0 {
            // Every remaining point coincides with a chosen centre. Fall back to
            // uniform sampling so that duplicate-heavy training sets (a real
            // case: a corpus with many identical embeddings) still terminate.
            rng.gen_range(0..n)
        } else {
            let target = rng.gen_range(0.0..total);
            let mut cumulative = 0.0f64;
            let mut index = n - 1;
            for (i, d) in nearest_squared.iter().enumerate() {
                cumulative += f64::from(*d);
                if cumulative >= target {
                    index = i;
                    break;
                }
            }
            index
        };

        let centre = &points[picked * dim..(picked + 1) * dim];
        centroids.extend_from_slice(centre);
        let base = chosen * dim;
        for (point, nearest) in points.chunks_exact(dim).zip(nearest_squared.iter_mut()) {
            let d = squared(point, &centroids[base..base + dim]);
            if d < *nearest {
                *nearest = d;
            }
        }
    }

    centroids
}

/// Index of, and squared distance to, the nearest centroid.
///
/// Ties go to the lower index; see [`kmeans`] on why that is load-bearing.
#[inline]
fn nearest(point: &[f32], centroids: &[f32], dim: usize, k: usize) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for cluster in 0..k {
        let base = cluster * dim;
        let d = squared(point, &centroids[base..base + dim]);
        if d < best_distance {
            best_distance = d;
            best = cluster;
        }
    }
    (best, best_distance)
}

/// The point with the largest distance to its nearest centroid.
fn farthest_point<'a>(points: &'a [f32], dim: usize, centroids: &[f32], k: usize) -> &'a [f32] {
    let mut best = &points[..dim];
    let mut best_distance = -1.0f32;
    for point in points.chunks_exact(dim) {
        let (_, d) = nearest(point, centroids, dim, k);
        if d > best_distance {
            best_distance = d;
            best = point;
        }
    }
    best
}

/// Squared L2 over a subspace.
///
/// Deliberately not `proxima_core::distance::squared_euclidean`: subspace
/// dimensions here are typically 2 to 16, well under the 8-lane unrolled
/// kernel's break-even point, where the chunk setup costs more than it saves.
#[inline]
fn squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    // Exact float comparison is deliberate here: every constant below is
    // exactly representable and the point is that the kernel returns it
    // bit-for-bit.
    #![allow(clippy::float_cmp)]

    use super::*;
    use rand::SeedableRng;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0x00C0_FFEE)
    }

    /// Three tight, well-separated blobs in 2-D.
    fn blobs() -> Vec<f32> {
        let mut points = Vec::new();
        for (cx, cy) in [(0.0f32, 0.0f32), (100.0, 0.0), (0.0, 100.0)] {
            for i in 0..30 {
                let jitter = (i as f32) * 0.01;
                points.extend_from_slice(&[cx + jitter, cy - jitter]);
            }
        }
        points
    }

    #[test]
    fn recovers_well_separated_clusters() {
        let points = blobs();
        let result = kmeans(&points, 2, 3, 25, &mut rng());
        let mut found: Vec<(i32, i32)> = result
            .centroids
            .chunks_exact(2)
            .map(|c| (c[0].round() as i32, c[1].round() as i32))
            .collect();
        found.sort_unstable();
        assert_eq!(found, vec![(0, 0), (0, 100), (100, 0)]);
        // Each blob is 30 points spread evenly over a 0.29-long diagonal, whose
        // within-cluster sum of squares is 0.45; three of them is 1.35. Anything
        // meaningfully above that means a centroid landed in the wrong blob.
        assert!(result.inertia < 1.4, "inertia {}", result.inertia);
    }

    #[test]
    fn is_deterministic_for_a_given_seed() {
        let points = blobs();
        let a = kmeans(&points, 2, 3, 25, &mut rng());
        let b = kmeans(&points, 2, 3, 25, &mut rng());
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.iterations, b.iterations);
    }

    #[test]
    fn more_centroids_than_points_is_clamped_not_rejected() {
        let points = vec![1.0, 2.0, 3.0, 4.0];
        let result = kmeans(&points, 2, 16, 25, &mut rng());
        assert_eq!(result.centroids.len(), 2 * 2);
    }

    #[test]
    fn identical_points_terminate() {
        // The `total <= 0.0` branch in the seeder: every candidate is at
        // distance zero from an already-chosen centre.
        let points = vec![7.0f32; 64];
        let result = kmeans(&points, 4, 8, 25, &mut rng());
        assert_eq!(result.inertia, 0.0);
        assert!(result.centroids.iter().all(|c| *c == 7.0));
    }

    #[test]
    fn empty_training_set_yields_zero_centroids() {
        let result = kmeans(&[], 4, 8, 25, &mut rng());
        assert_eq!(result.centroids.len(), 4);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn inertia_does_not_increase_across_iterations() {
        // Lloyd's guarantee, and the thing that catches a broken update step.
        let points = blobs();
        let mut previous = f64::INFINITY;
        for cap in 1..8 {
            let result = kmeans(&points, 2, 3, cap, &mut rng());
            assert!(
                result.inertia <= previous + 1e-9,
                "inertia rose from {} to {} at cap {}",
                previous,
                result.inertia,
                cap
            );
            previous = result.inertia;
        }
    }
}
