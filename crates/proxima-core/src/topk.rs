//! [`SearchResult`] and the bounded max-heap that keeps the best `k` of them.
//!
//! # The mistake this module exists to prevent
//!
//! The naive way to take the nearest `k` of `n` candidates is to score all `n`,
//! push them into a `Vec`, sort it, and truncate. That is `O(n log n)` time and,
//! worse, `O(n)` memory — an exhaustive scan of a million vectors materialises a
//! million `(id, distance)` pairs to hand back ten of them.
//!
//! A bounded max-heap of size `k` is `O(n log k)` time and `O(k)` memory, and
//! the constant factor is better than the asymptotics suggest: once the heap is
//! full, the common case is a single `f32` comparison against the current worst
//! result, which fails, and the candidate is discarded without touching the heap
//! at all. On a corpus where candidates arrive in no particular order, the
//! fraction of candidates that ever cause a heap operation is `O(k log n / n)`.
//!
//! *Max*-heap, when the caller wants the *smallest* distances, is the part that
//! trips people up: the element you need constant-time access to is the one you
//! are about to evict, which is the worst of the current best.

use std::collections::BinaryHeap;

use crate::id::VectorId;

/// One scored hit.
///
/// `distance` is whatever the index's [`Metric`](crate::Metric) produces, which
/// for Euclidean is the *squared* distance and for inner product is *negated*.
/// [`Metric::to_true_distance`](crate::Metric::to_true_distance) converts it for
/// display. Keeping the raw score here means a caller can compare results from
/// two searches over the same index without a rounding trip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    /// The vector's id.
    pub id: VectorId,
    /// Ranking score; smaller is nearer.
    pub distance: f32,
}

impl SearchResult {
    /// Build a result.
    pub const fn new(id: VectorId, distance: f32) -> Self {
        SearchResult { id, distance }
    }
}

impl Eq for SearchResult {}

impl Ord for SearchResult {
    /// Orders by `distance` using [`f32::total_cmp`], then by `id`.
    ///
    /// `total_cmp` rather than `partial_cmp` for two reasons. It is a *total*
    /// order, so `Ord` is not a lie and the heap cannot be corrupted by a NaN
    /// (which `partial_cmp().unwrap()` would panic on and
    /// `partial_cmp().unwrap_or(Equal)` would silently break the heap
    /// invariant with). And it places NaN above every finite value, so a
    /// NaN-scored candidate is the first thing a bounded max-heap evicts —
    /// which is the behaviour the [`crate::distance`] NaN policy promises.
    ///
    /// The `id` tiebreak is what makes results reproducible: without it, two
    /// vectors at identical distance would be ordered by whatever arbitrary
    /// sequence the heap happened to see them in.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A bounded max-heap holding the `k` nearest results seen so far.
///
/// ```
/// use proxima_core::{TopK, VectorId};
///
/// let mut top = TopK::new(2);
/// top.push(VectorId::new(1), 5.0);
/// top.push(VectorId::new(2), 1.0);
/// top.push(VectorId::new(3), 9.0);   // rejected without a heap operation
/// top.push(VectorId::new(4), 3.0);   // evicts id 1
///
/// let hits = top.into_sorted_vec();
/// assert_eq!(hits.len(), 2);
/// assert_eq!(hits[0].id, VectorId::new(2));
/// assert_eq!(hits[1].id, VectorId::new(4));
/// ```
#[derive(Debug, Clone)]
pub struct TopK {
    k: usize,
    heap: BinaryHeap<SearchResult>,
}

impl TopK {
    /// A heap that will keep the best `k`.
    ///
    /// `k == 0` is legal and yields a structure that accepts nothing; callers
    /// that treat `k = 0` as an error should say so themselves, because "give
    /// me zero results" is a perfectly reasonable degenerate query and
    /// returning an error for it forces every caller to special-case it.
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            // Reserve k+1: `push` can transiently hold one more than k, and
            // this makes the whole structure exactly one allocation.
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    /// Offer a candidate.
    ///
    /// Returns `true` if it was kept. The early-out comparison against the
    /// current worst is the fast path and the reason this is not just a sort.
    #[inline]
    pub fn push(&mut self, id: VectorId, distance: f32) -> bool {
        if self.k == 0 {
            return false;
        }
        let candidate = SearchResult::new(id, distance);
        if self.heap.len() < self.k {
            self.heap.push(candidate);
            return true;
        }
        // `peek` is the current worst kept result: the only one that can be
        // displaced, and the only comparison most candidates ever cause.
        match self.heap.peek() {
            Some(worst) if candidate < *worst => {
                self.heap.pop();
                self.heap.push(candidate);
                true
            }
            _ => false,
        }
    }

    /// The worst result currently kept, or `None` while the heap is under-full.
    ///
    /// `None` rather than `f32::INFINITY` because "not yet full" and "full, and
    /// the worst is infinitely far" are different states, and a graph search
    /// stops on the first but not the second.
    #[inline]
    pub fn worst(&self) -> Option<f32> {
        self.heap.peek().map(|r| r.distance)
    }

    /// The score a candidate must beat to be kept.
    ///
    /// `f32::INFINITY` while under-full. This is the form the search loops
    /// actually want, and it is separate from [`TopK::worst`] so that the
    /// "not full" case cannot be lost by accident.
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            self.heap.peek().map_or(f32::INFINITY, |r| r.distance)
        }
    }

    /// `true` once `k` results are held.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// Number of results held.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// `true` if nothing has been kept.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// The bound.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Consume the heap, returning results nearest-first.
    ///
    /// `O(k log k)`. `BinaryHeap::into_sorted_vec` yields ascending order under
    /// our `Ord`, which is already nearest-first — no reverse pass.
    pub fn into_sorted_vec(self) -> Vec<SearchResult> {
        self.heap.into_sorted_vec()
    }

    /// Drop every result, keeping the allocation and the bound.
    ///
    /// For a query loop that reuses one heap across many queries.
    pub fn clear(&mut self) {
        self.heap.clear();
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

    fn ids(results: &[SearchResult]) -> Vec<u64> {
        results.iter().map(|r| r.id.get()).collect()
    }

    #[test]
    fn keeps_only_the_best_k() {
        let mut top = TopK::new(3);
        for (i, d) in [(1u64, 8.0), (2, 2.0), (3, 5.0), (4, 1.0), (5, 9.0)] {
            top.push(VectorId::new(i), d);
        }
        assert_eq!(top.len(), 3);
        assert_eq!(ids(&top.into_sorted_vec()), vec![4, 2, 3]);
    }

    #[test]
    fn threshold_is_infinite_until_full() {
        let mut top = TopK::new(2);
        assert!(top.threshold().is_infinite());
        top.push(VectorId::new(1), 4.0);
        assert!(top.threshold().is_infinite());
        assert_eq!(top.worst(), Some(4.0));
        top.push(VectorId::new(2), 1.0);
        assert_eq!(top.threshold(), 4.0);
    }

    #[test]
    fn fewer_candidates_than_k_is_not_an_error() {
        let mut top = TopK::new(10);
        top.push(VectorId::new(1), 1.0);
        let out = top.into_sorted_vec();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn k_zero_accepts_nothing() {
        let mut top = TopK::new(0);
        assert!(!top.push(VectorId::new(1), 0.0));
        assert!(top.is_full());
        assert!(top.into_sorted_vec().is_empty());
    }

    #[test]
    fn ties_break_on_id_so_results_are_reproducible() {
        // Insert the same distances in two different orders; the output must
        // not depend on arrival order.
        let mut a = TopK::new(2);
        for i in [7u64, 3, 5] {
            a.push(VectorId::new(i), 1.0);
        }
        let mut b = TopK::new(2);
        for i in [5u64, 7, 3] {
            b.push(VectorId::new(i), 1.0);
        }
        assert_eq!(ids(&a.into_sorted_vec()), ids(&b.into_sorted_vec()));
    }

    #[test]
    fn nan_scores_are_evicted_rather_than_corrupting_the_heap() {
        // The documented consequence of ordering with `total_cmp`.
        let mut top = TopK::new(2);
        top.push(VectorId::new(1), f32::NAN);
        top.push(VectorId::new(2), f32::NAN);
        assert!(top.is_full());
        assert!(top.push(VectorId::new(3), 100.0));
        assert!(top.push(VectorId::new(4), 100.0));
        assert_eq!(ids(&top.into_sorted_vec()), vec![3, 4]);
    }

    #[test]
    fn infinity_orders_normally() {
        let mut top = TopK::new(2);
        top.push(VectorId::new(1), f32::INFINITY);
        top.push(VectorId::new(2), 0.0);
        top.push(VectorId::new(3), 1.0);
        assert_eq!(ids(&top.into_sorted_vec()), vec![2, 3]);
    }

    #[test]
    fn clear_keeps_the_bound() {
        let mut top = TopK::new(2);
        top.push(VectorId::new(1), 1.0);
        top.clear();
        assert!(top.is_empty());
        assert_eq!(top.k(), 2);
    }
}
