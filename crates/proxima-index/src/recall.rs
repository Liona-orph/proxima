//! Recall measurement, as a function rather than a paragraph.
//!
//! An approximate index without a recall number is a claim with no content.
//! This module is the crate's headline feature: [`recall_at_k`] takes any
//! [`VectorIndex`] and a [`FlatIndex`] over the same data and returns what
//! fraction of the true nearest neighbours the approximate index actually
//! found.
//!
//! Everything here is deliberately simple and deliberately in the library
//! rather than in a benchmark, so that a user can measure *their* corpus with
//! *their* parameters in three lines, in their own test suite, and turn "we
//! think recall is fine" into an assertion that fails when it stops being true.

use proxima_core::{Result, VectorId};

use crate::flat::FlatIndex;
use crate::index::{AcceptAll, Filter, VectorIndex};

/// The outcome of a recall measurement.
///
/// Reports the distribution and not just the mean, because the mean hides the
/// failure that matters. An index with mean recall 0.95 might miss half a
/// neighbour on every query, or return perfect results for 90 % of queries and
/// nothing at all for the other 10 %. The second is a much worse product and
/// only [`RecallReport::min`] and [`RecallReport::below`] tell them apart.
#[derive(Debug, Clone, PartialEq)]
pub struct RecallReport {
    /// Number of queries measured.
    pub queries: usize,
    /// The `k` that was requested.
    pub k: usize,
    /// Mean recall across queries, in `[0, 1]`.
    pub mean: f32,
    /// Worst single-query recall.
    pub min: f32,
    /// Per-query recall, in query order, for callers that want a histogram.
    pub per_query: Vec<f32>,
    /// Queries where the approximate index returned fewer results than the
    /// exact one.
    ///
    /// Non-zero here means something other than ranking error: either the
    /// filter's escalation gave up, or `k` exceeded the live set. Worth
    /// separating from recall loss, because the fixes are different.
    pub short_results: usize,
}

impl RecallReport {
    /// Fraction of queries whose recall fell below `threshold`.
    pub fn below(&self, threshold: f32) -> f32 {
        if self.per_query.is_empty() {
            return 0.0;
        }
        let count = self.per_query.iter().filter(|r| **r < threshold).count();
        count as f32 / self.per_query.len() as f32
    }
}

impl std::fmt::Display for RecallReport {
    /// One line, formatted for a log or a benchmark table.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "recall@{}: mean {:.4}, min {:.4}, below 0.9 {:.1}% ({} queries)",
            self.k,
            self.mean,
            self.min,
            self.below(0.9) * 100.0,
            self.queries
        )
    }
}

/// Measure `candidate`'s recall@k against exhaustive search.
///
/// `truth` must be a [`FlatIndex`] holding the *same vectors under the same
/// ids* as `candidate`; nothing checks that, because there is no cheap way to,
/// and a mismatch shows up immediately as implausibly low recall.
///
/// `queries` is a flat `q * dim` array. Cost is one exhaustive scan plus one
/// approximate search per query, so measuring 100 queries against a million
/// vectors is 100 million distance evaluations — this is a test-time tool, not
/// a monitoring one.
///
/// ```
/// use proxima_core::{Metadata, Metric, VectorId, VectorSpace};
/// use proxima_index::{recall_at_k, FlatIndex, HnswConfig, HnswIndex, VectorIndex};
///
/// let space = VectorSpace::new(8, Metric::Euclidean)?;
/// let mut flat = FlatIndex::new(space.clone());
/// let mut hnsw = HnswIndex::new(space, HnswConfig::new().with_seed(3))?;
///
/// let mut data = Vec::new();
/// for i in 0..2000u64 {
///     let t = i as f32 * 0.005;
///     let v = [t, t.sin(), t.cos(), (2.0 * t).sin(), -t, t * t, 1.0, t.tan().clamp(-9.0, 9.0)];
///     flat.insert(VectorId::new(i), &v, Metadata::new())?;
///     hnsw.insert(VectorId::new(i), &v, Metadata::new())?;
///     if i % 20 == 0 {
///         data.extend_from_slice(&v);
///     }
/// }
///
/// let report = recall_at_k(&hnsw, &flat, &data, 10)?;
/// assert_eq!(report.queries, 100);
/// assert!(report.mean > 0.95, "{report}");
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
pub fn recall_at_k(
    candidate: &dyn VectorIndex,
    truth: &FlatIndex,
    queries: &[f32],
    k: usize,
) -> Result<RecallReport> {
    recall_at_k_filtered(candidate, truth, queries, k, &AcceptAll)
}

/// [`recall_at_k`] with a filter applied to both sides.
///
/// Filtered recall is the measurement people forget to take, and it is the one
/// that exposes the failure mode described in
/// [`HnswIndex::search`](crate::HnswIndex::search): an index can have 0.99
/// unfiltered recall and 0.3 recall under a 1-in-1000 filter.
pub fn recall_at_k_filtered(
    candidate: &dyn VectorIndex,
    truth: &FlatIndex,
    queries: &[f32],
    k: usize,
    filter: &dyn Filter,
) -> Result<RecallReport> {
    let dim = truth.space().dim();
    if dim == 0 || queries.len() % dim != 0 {
        return Err(proxima_core::ProximaError::config(
            "queries",
            queries.len(),
            "must be a multiple of the index dimension",
        ));
    }

    let mut per_query = Vec::with_capacity(queries.len() / dim.max(1));
    let mut short_results = 0usize;

    for query in queries.chunks_exact(dim) {
        let exact = truth.exact_ids(query, k, filter)?;
        let approximate = candidate.search(query, k, filter)?;
        if approximate.len() < exact.len() {
            short_results += 1;
        }
        per_query.push(overlap(&exact, &approximate));
    }

    let queries_measured = per_query.len();
    let mean = if queries_measured == 0 {
        // No queries means no evidence, and 1.0 would be a lie that reads as a
        // pass. Zero is also a lie, so the report says so via `queries`.
        0.0
    } else {
        per_query.iter().sum::<f32>() / queries_measured as f32
    };
    let min = per_query.iter().copied().fold(f32::INFINITY, f32::min);

    Ok(RecallReport {
        queries: queries_measured,
        k,
        mean,
        min: if min.is_finite() { min } else { 0.0 },
        per_query,
        short_results,
    })
}

/// Fraction of `exact` that appears in `approximate`.
///
/// Denominator is `exact.len()`, not `k`: when the index holds fewer than `k`
/// live vectors, or a filter accepts fewer, the best possible answer is shorter
/// than `k` and dividing by `k` would report a failure that is not one.
fn overlap(exact: &[VectorId], approximate: &[proxima_core::SearchResult]) -> f32 {
    if exact.is_empty() {
        return 1.0;
    }
    // Linear scan rather than a `HashSet`: `k` is a handful, and building a set
    // per query costs more than the scan for any `k` under a few hundred.
    let hits = approximate
        .iter()
        .filter(|hit| exact.contains(&hit.id))
        .count();
    hits as f32 / exact.len() as f32
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    // Exact float comparison is deliberate here: every constant below is
    // exactly representable and the point is that the harness returns it
    // bit-for-bit.
    #![allow(clippy::float_cmp)]

    use super::*;
    use proxima_core::{Metadata, Metric, VectorSpace};

    fn fixture(count: u64) -> (FlatIndex, Vec<f32>) {
        let space = VectorSpace::new(4, Metric::Euclidean).unwrap();
        let mut flat = FlatIndex::new(space);
        let mut queries = Vec::new();
        for i in 0..count {
            let t = i as f32 * 0.1;
            let v = [t, t.sin(), -t, 1.0];
            flat.insert(VectorId::new(i), &v, Metadata::new()).unwrap();
            if i % 10 == 0 {
                queries.extend_from_slice(&v);
            }
        }
        (flat, queries)
    }

    #[test]
    fn an_index_measured_against_itself_has_perfect_recall() {
        let (flat, queries) = fixture(200);
        let report = recall_at_k(&flat, &flat, &queries, 10).unwrap();
        assert_eq!(report.mean, 1.0);
        assert_eq!(report.min, 1.0);
        assert_eq!(report.short_results, 0);
        assert_eq!(report.below(1.0), 0.0);
    }

    #[test]
    fn a_deliberately_broken_index_scores_badly() {
        // Guards against the harness reporting 1.0 for everything, which is the
        // only way a recall function can be worse than useless.
        let (flat, queries) = fixture(200);
        let space = VectorSpace::new(4, Metric::Euclidean).unwrap();
        let mut half = FlatIndex::new(space);
        for (i, id) in flat.ids().enumerate() {
            if i % 2 == 0 {
                let v = flat.vector(id).unwrap().to_vec();
                half.insert(id, &v, Metadata::new()).unwrap();
            }
        }
        let report = recall_at_k(&half, &flat, &queries, 10).unwrap();
        assert!(report.mean < 0.75, "{report}");
        assert!(report.mean > 0.25, "{report}");
    }

    #[test]
    fn short_results_are_counted_separately_from_ranking_error() {
        let (flat, queries) = fixture(5);
        let report = recall_at_k(&flat, &flat, &queries, 10).unwrap();
        // Both sides return 5, so recall is 1.0 and nothing is "short".
        assert_eq!(report.mean, 1.0);
        assert_eq!(report.short_results, 0);
    }

    #[test]
    fn query_array_must_match_the_dimension() {
        let (flat, _) = fixture(10);
        assert!(recall_at_k(&flat, &flat, &[1.0, 2.0, 3.0], 5).is_err());
    }

    #[test]
    fn no_queries_reports_zero_rather_than_a_flattering_one() {
        let (flat, _) = fixture(10);
        let report = recall_at_k(&flat, &flat, &[], 5).unwrap();
        assert_eq!(report.queries, 0);
        assert_eq!(report.mean, 0.0);
    }

    #[test]
    fn display_is_one_line() {
        let (flat, queries) = fixture(50);
        let report = recall_at_k(&flat, &flat, &queries, 5).unwrap();
        let rendered = report.to_string();
        assert!(rendered.starts_with("recall@5: mean 1.0000"));
        assert!(!rendered.contains('\n'));
    }
}
