//! The property that matters: a reloaded index answers identically.
//!
//! Every assertion in this file compares full `Vec<SearchResult>` values — ids,
//! distances and order — for exact equality, not for approximate agreement and
//! not for set equality. A snapshot that returned the same ids in a different
//! order, or the same order with distances one ulp apart, would be a snapshot
//! that had quietly rebuilt something rather than restored it.

#![allow(clippy::unwrap_used)]

use proxima_core::{Metadata, Metric, SearchResult, VectorId, VectorSpace};
use proxima_index::{HnswConfig, IvfPqConfig, VectorIndex};
use proxima_store::{snapshot, AnyIndex, IndexSpec};

const DIM: usize = 24;

fn vector(i: u64) -> Vec<f32> {
    (0..DIM)
        .map(|d| ((i % 23) as f32 * 0.9 + d as f32 * 0.21).sin() * 3.5 + (i as f32) * 0.0017)
        .collect()
}

fn queries(count: u64) -> Vec<Vec<f32>> {
    (0..count).map(|q| vector(q * 37 + 11)).collect()
}

fn results(index: &AnyIndex) -> Vec<Vec<SearchResult>> {
    queries(100)
        .iter()
        .map(|query| index.search_all(query, 10).unwrap())
        .collect()
}

/// A collection with a history: inserts, deletes and overwrites, so the index
/// is in a state no rebuild from its contents could reproduce.
fn lived_in(spec: &IndexSpec, count: u64) -> AnyIndex {
    let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
    let mut index = spec.build(space).unwrap();
    for i in 0..count {
        index
            .insert(
                VectorId::new(i),
                &vector(i),
                Metadata::new()
                    .with("bucket", (i % 9) as i64)
                    .with("tag", if i % 2 == 0 { "even" } else { "odd" }),
            )
            .unwrap();
    }
    for i in (0..count).step_by(6) {
        index.remove(VectorId::new(i)).unwrap();
    }
    for i in (1..count).step_by(13) {
        index
            .upsert(VectorId::new(i), &vector(i + count), Metadata::new())
            .unwrap();
    }
    index
}

#[test]
fn a_flat_index_reloads_to_identical_results() {
    let index = lived_in(&IndexSpec::Flat, 800);
    let restored = snapshot::decode(&snapshot::encode(&index)).unwrap();
    assert_eq!(results(&index), results(&restored));
}

#[test]
fn a_graph_reloads_to_identical_results() {
    let index = lived_in(&IndexSpec::Hnsw(HnswConfig::new().with_seed(23)), 1500);
    let restored = snapshot::decode(&snapshot::encode(&index)).unwrap();
    assert_eq!(results(&index), results(&restored));
}

#[test]
fn a_graph_reloads_to_identical_filtered_results() {
    // Filtered search escalates `ef` and can fall back to a scan, so it takes a
    // different path through the graph. It has to agree too.
    let index = lived_in(&IndexSpec::Hnsw(HnswConfig::new().with_seed(29)), 1500);
    let restored = snapshot::decode(&snapshot::encode(&index)).unwrap();
    let selective = |_id: VectorId, meta: &Metadata| meta.int("bucket") == Some(3);
    for query in queries(50) {
        assert_eq!(
            index.search(&query, 10, &selective).unwrap(),
            restored.search(&query, 10, &selective).unwrap()
        );
    }
}

#[test]
fn an_ivfpq_index_reloads_to_identical_results() {
    let index = lived_in(
        &IndexSpec::IvfPq(
            IvfPqConfig::new(16, 12)
                .with_seed(37)
                .with_nprobe(6)
                .with_rerank(50)
                .with_train_threshold(400),
        ),
        1200,
    );
    let restored = snapshot::decode(&snapshot::encode(&index)).unwrap();
    assert_eq!(results(&index), results(&restored));
}

#[test]
fn an_ivfpq_index_that_dropped_its_vectors_reloads() {
    let index = lived_in(
        &IndexSpec::IvfPq(
            IvfPqConfig::new(16, 12)
                .with_seed(41)
                .with_nprobe(6)
                .with_retain_vectors(false)
                .with_train_threshold(400),
        ),
        1200,
    );
    match &index {
        AnyIndex::IvfPq(ivf) => assert_eq!(ivf.vector_bytes(), 0),
        _ => unreachable!(),
    }
    let bytes = snapshot::encode(&index);
    let restored = snapshot::decode(&bytes).unwrap();
    assert_eq!(results(&index), results(&restored));

    // Smaller than the corpus it describes — 1 200 vectors of 24 dimensions is
    // 115 KiB of f32 — but not dramatically so at this size, because the PQ
    // codebooks are a fixed 256 * dim * 4 bytes per subspace regardless of how
    // many vectors they encode. The compression only pays once the corpus is
    // large against the codebook, which is the honest shape of the trade.
    let flat = snapshot::encode(&lived_in(&IndexSpec::Flat, 1200));
    assert!(
        bytes.len() < flat.len(),
        "{} bytes against flat's {}",
        bytes.len(),
        flat.len()
    );
}

#[test]
fn a_reloaded_graph_keeps_accepting_writes_identically() {
    // The subtler half of the guarantee: not just that the reload answers the
    // same, but that it *continues* the same. This is what the level
    // generator's fast-forward is for.
    let spec = IndexSpec::Hnsw(HnswConfig::new().with_seed(43));
    let mut original = lived_in(&spec, 900);
    let mut restored = snapshot::decode(&snapshot::encode(&original)).unwrap();

    for i in 10_000..10_200u64 {
        original
            .insert(VectorId::new(i), &vector(i), Metadata::new())
            .unwrap();
        restored
            .insert(VectorId::new(i), &vector(i), Metadata::new())
            .unwrap();
    }
    assert_eq!(results(&original), results(&restored));
    assert_eq!(original.stats().edges, restored.stats().edges);
    assert_eq!(
        original.stats().nodes_per_layer,
        restored.stats().nodes_per_layer
    );
}

#[test]
fn a_rebuild_from_the_same_vectors_is_a_different_index() {
    // The control for the tests above. If a rebuild happened to reproduce the
    // graph, none of them would be evidence of anything.
    let index = lived_in(&IndexSpec::Hnsw(HnswConfig::new().with_seed(23)), 1500);
    let AnyIndex::Hnsw(hnsw) = &index else {
        unreachable!()
    };

    let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
    let mut rebuilt = IndexSpec::Hnsw(HnswConfig::new().with_seed(23))
        .build(space)
        .unwrap();
    for id in hnsw.ids().collect::<Vec<_>>() {
        let vector = hnsw.vector(id).unwrap().to_vec();
        rebuilt
            .insert(id, &vector, hnsw.metadata(id).cloned().unwrap_or_default())
            .unwrap();
    }
    assert_eq!(rebuilt.len(), index.len());
    // The graph, not the answers: a different graph can still agree on the top
    // ten for a well-separated fixture, and it is the graph the snapshot has to
    // carry.
    assert_ne!(
        (rebuilt.stats().edges, rebuilt.stats().nodes_per_layer),
        (index.stats().edges, index.stats().nodes_per_layer),
        "a rebuild reproduced the graph, so the snapshot tests prove nothing"
    );
}

#[test]
fn the_encoding_is_stable_across_processes() {
    // Two encodes of one index are byte-identical. `Metadata` is a `HashMap`
    // with a per-process random seed, so this only holds because the codec
    // sorts the keys — and it is what lets a snapshot be diffed or content
    // addressed.
    let index = lived_in(&IndexSpec::Flat, 400);
    let first = snapshot::encode(&index);
    for _ in 0..16 {
        assert_eq!(snapshot::encode(&index), first);
    }
    // ...and a reload re-encodes to the same bytes, which is the round-trip
    // stated as a fixed point.
    let restored = snapshot::decode(&first).unwrap();
    assert_eq!(snapshot::encode(&restored), first);
}

#[test]
fn every_single_byte_of_corruption_is_detected() {
    let index = lived_in(&IndexSpec::Flat, 60);
    let bytes = snapshot::encode(&index);
    let mut checked = 0;
    for offset in 0..bytes.len() {
        let mut corrupt = bytes.clone();
        corrupt[offset] ^= 0x01;
        assert!(
            snapshot::decode(&corrupt).is_err(),
            "a flipped bit at offset {offset} was not detected"
        );
        checked += 1;
    }
    assert!(checked > 1000, "the fixture must be big enough to matter");
}
