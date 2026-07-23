//! Properties that must hold for *any* sequence of operations.
//!
//! The example-based tests check the cases someone thought of. These check the
//! invariants that hold regardless: that a graph index and an exhaustive one
//! agree about what they contain after an arbitrary mix of inserts, removes and
//! upserts, and that a search never returns something it should not.
//!
//! The operation sequence is generated, so the shrinker will hand back the
//! shortest sequence that breaks an invariant — which for a stateful structure
//! is worth considerably more than the property itself.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;

use proptest::prelude::*;
use proxima_core::{Metadata, Metric, VectorId, VectorSpace};
use proxima_index::{FlatIndex, HnswConfig, HnswIndex, VectorIndex};

const DIM: usize = 8;

/// One mutation, over a deliberately small id space so that duplicates,
/// removals of absent ids and reinsertions after removal all occur often.
#[derive(Debug, Clone)]
enum Op {
    Insert(u64, Vec<f32>),
    Upsert(u64, Vec<f32>),
    Remove(u64),
}

fn vector() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0, DIM)
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        3 => (0u64..24, vector()).prop_map(|(id, v)| Op::Insert(id, v)),
        1 => (0u64..24, vector()).prop_map(|(id, v)| Op::Upsert(id, v)),
        1 => (0u64..24).prop_map(Op::Remove),
    ]
}

/// Apply the same sequence to both indexes, ignoring errors identically.
fn apply(ops: &[Op], flat: &mut FlatIndex, hnsw: &mut HnswIndex) -> BTreeSet<u64> {
    let mut expected = BTreeSet::new();
    for operation in ops {
        match operation {
            Op::Insert(id, vector) => {
                let id = VectorId::new(*id);
                let a = flat.insert(id, vector, Metadata::new());
                let b = hnsw.insert(id, vector, Metadata::new());
                // Both must agree on whether the operation was legal.
                assert_eq!(a.is_ok(), b.is_ok());
                if a.is_ok() {
                    expected.insert(id.get());
                }
            }
            Op::Upsert(id, vector) => {
                let id = VectorId::new(*id);
                flat.upsert(id, vector, Metadata::new()).unwrap();
                hnsw.upsert(id, vector, Metadata::new()).unwrap();
                expected.insert(id.get());
            }
            Op::Remove(id) => {
                let id = VectorId::new(*id);
                let a = flat.remove(id);
                let b = hnsw.remove(id);
                assert_eq!(a.is_ok(), b.is_ok());
                if a.is_ok() {
                    expected.remove(&id.get());
                }
            }
        }
    }
    expected
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// After any sequence, both indexes hold exactly the ids the sequence says
    /// they should, and a search returns only live ids, without duplicates, in
    /// non-decreasing distance order.
    #[test]
    fn arbitrary_operation_sequences_leave_both_indexes_consistent(
        ops in prop::collection::vec(op(), 0..80),
        query in vector(),
        k in 0usize..12,
    ) {
        let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
        let mut flat = FlatIndex::new(space.clone());
        let mut hnsw = HnswIndex::new(space, HnswConfig::new().with_seed(4)).unwrap();
        let expected = apply(&ops, &mut flat, &mut hnsw);

        prop_assert_eq!(flat.len(), expected.len());
        prop_assert_eq!(hnsw.len(), expected.len());
        for id in &expected {
            prop_assert!(flat.contains(VectorId::new(*id)));
            prop_assert!(hnsw.contains(VectorId::new(*id)));
        }

        for hits in [flat.search_all(&query, k).unwrap(), hnsw.search_all(&query, k).unwrap()] {
            prop_assert!(hits.len() <= k.min(expected.len()));
            let mut seen = BTreeSet::new();
            let mut previous = f32::NEG_INFINITY;
            for hit in &hits {
                prop_assert!(expected.contains(&hit.id.get()), "returned a dead id {}", hit.id);
                prop_assert!(seen.insert(hit.id.get()), "returned {} twice", hit.id);
                prop_assert!(hit.distance >= previous, "results are not sorted");
                previous = hit.distance;
            }
        }
    }

    /// A tombstoned graph and the same graph compacted answer the same
    /// questions about membership, and compaction leaves no tombstones.
    #[test]
    fn compaction_preserves_the_live_set(
        ops in prop::collection::vec(op(), 0..80),
    ) {
        let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
        let mut flat = FlatIndex::new(space.clone());
        let mut hnsw = HnswIndex::new(space, HnswConfig::new().with_seed(6)).unwrap();
        let expected = apply(&ops, &mut flat, &mut hnsw);

        hnsw.compact().unwrap();
        prop_assert_eq!(hnsw.tombstones(), 0);
        prop_assert_eq!(hnsw.len(), expected.len());
        for id in &expected {
            prop_assert!(hnsw.contains(VectorId::new(*id)));
        }
        // A second compaction of an already-clean index changes nothing.
        prop_assert_eq!(hnsw.compact().unwrap(), 0);
    }

    /// A filter can only ever remove results, never add or reorder them.
    #[test]
    fn filtering_is_a_subsequence_of_the_unfiltered_answer(
        ops in prop::collection::vec(op(), 4..80),
        query in vector(),
    ) {
        let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
        let mut flat = FlatIndex::new(space.clone());
        let mut hnsw = HnswIndex::new(space, HnswConfig::new().with_seed(9)).unwrap();
        let expected = apply(&ops, &mut flat, &mut hnsw);
        prop_assume!(!expected.is_empty());

        let even = |id: VectorId, _m: &Metadata| id.get() % 2 == 0;
        // Ask for everything, so the comparison is not confused by the two
        // searches drawing their k from different-sized populations.
        let k = expected.len();
        let all = flat.search_all(&query, k).unwrap();
        let filtered = flat.search(&query, k, &even).unwrap();

        let mut remaining = all.iter().filter(|hit| hit.id.get() % 2 == 0);
        for hit in &filtered {
            prop_assert_eq!(remaining.next().map(|h| h.id), Some(hit.id));
        }
    }
}
