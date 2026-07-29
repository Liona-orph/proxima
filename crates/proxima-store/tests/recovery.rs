//! Crash recovery, simulated by truncating a real collection's files.
//!
//! A crash is not something a test can arrange politely, so it is arranged
//! rudely: run a workload against a real collection, close it, then truncate
//! the log to every plausible length — including lengths that land in the
//! middle of a record's payload and in the middle of its length prefix — and
//! check what comes back.
//!
//! The property being asserted is **prefix consistency**: whatever a recovered
//! collection contains, it is the state some prefix of the acknowledged
//! operations would have produced. Never a suffix, never a mixture, never an
//! error.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use proxima_core::{Metadata, Metric, SearchResult, VectorId};
use proxima_index::{AcceptAll, HnswConfig, VectorIndex};
use proxima_store::{Collection, CollectionOptions, IndexSpec, SyncPolicy};

const DIM: usize = 16;

fn temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "proxima-recovery-{name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn vector(i: u64) -> Vec<f32> {
    (0..DIM)
        .map(|d| ((i % 19) as f32 * 1.3 + d as f32 * 0.29).sin() * 4.0 + i as f32 * 0.0031)
        .collect()
}

fn open(dir: &Path, spec: IndexSpec) -> Collection {
    Collection::open(
        dir,
        "recovery",
        DIM,
        Metric::Euclidean,
        spec,
        CollectionOptions {
            sync_policy: SyncPolicy::Always,
            // Off, so that the whole history stays in the log and truncating it
            // is a meaningful thing to do.
            snapshot_after_records: 0,
        },
    )
    .unwrap()
    .0
}

/// The workload every test in this file replays: interleaved upserts, deletes
/// and overwrites, so the log exercises both record types and the index ends up
/// with tombstones.
fn workload(collection: &Collection, count: u64) {
    for i in 0..count {
        collection
            .upsert(
                VectorId::new(i),
                &vector(i),
                Metadata::new().with("bucket", (i % 7) as i64),
            )
            .unwrap();
        if i % 5 == 4 {
            collection.delete(VectorId::new(i / 5)).unwrap();
        }
        if i % 11 == 10 {
            collection
                .upsert(VectorId::new(i / 11), &vector(i * 3), Metadata::new())
                .unwrap();
        }
    }
}

#[test]
fn truncating_the_log_at_every_byte_recovers_a_prefix() {
    let source = temp_dir("prefix-source");
    let collection = open(&source, IndexSpec::Flat);
    workload(&collection, 120);
    let expected_len = collection.len();
    drop(collection);

    let log = std::fs::read(source.join("wal")).unwrap();
    let manifest = std::fs::read(source.join("manifest")).unwrap();
    assert!(log.len() > 1000, "the fixture must be big enough to matter");

    // Every byte offset, not a sample: the interesting ones are inside the
    // 8-byte frame header, and a stride would step over them.
    let mut lengths_seen = Vec::new();
    for keep in 0..=log.len() {
        let victim = temp_dir(&format!("prefix-{keep}"));
        std::fs::create_dir_all(&victim).unwrap();
        std::fs::write(victim.join("manifest"), &manifest).unwrap();
        std::fs::write(victim.join("wal"), &log[..keep]).unwrap();

        let (collection, report) = Collection::open(
            &victim,
            "recovery",
            DIM,
            Metric::Euclidean,
            IndexSpec::Flat,
            CollectionOptions {
                sync_policy: SyncPolicy::Always,
                snapshot_after_records: 0,
            },
        )
        .unwrap();
        assert_eq!(report.replayed, report.recovery.records, "keep {keep}");
        lengths_seen.push(collection.len());

        // A recovered collection is immediately writable, and the write lands
        // after whatever survived.
        collection
            .upsert(VectorId::new(9999), &vector(1), Metadata::new())
            .unwrap();
        assert!(collection.contains(VectorId::new(9999)));
        drop(collection);
        std::fs::remove_dir_all(&victim).unwrap();
    }

    // The recovered sizes must be monotone in how much of the log survived:
    // that is what "a prefix of history" means when the history is a mix of
    // inserts and deletes replayed in order.
    let full = *lengths_seen.last().unwrap();
    assert_eq!(full, expected_len);
    assert_eq!(lengths_seen[0], 0);
    std::fs::remove_dir_all(source).unwrap();
}

#[test]
fn a_torn_record_is_discarded_and_the_log_is_repaired_in_place() {
    // Not just ignored on read: the bytes are gone from the file afterwards,
    // so the next append does not write after garbage.
    let dir = temp_dir("repair");
    let collection = open(&dir, IndexSpec::Flat);
    workload(&collection, 40);
    drop(collection);

    let path = dir.join("wal");
    let full = std::fs::read(&path).unwrap();
    // Cut nine bytes off the end, which is inside the final record's payload.
    std::fs::write(&path, &full[..full.len() - 9]).unwrap();

    let (collection, report) = Collection::open(
        &dir,
        "recovery",
        DIM,
        Metric::Euclidean,
        IndexSpec::Flat,
        CollectionOptions {
            sync_policy: SyncPolicy::Always,
            snapshot_after_records: 0,
        },
    )
    .unwrap();
    assert!(report.recovery.torn_tail);
    assert!(report.recovery.bytes_discarded > 0);

    let repaired = std::fs::metadata(&path).unwrap().len();
    assert_eq!(repaired, report.recovery.bytes_kept);
    assert!(repaired < (full.len() - 9) as u64);

    collection
        .upsert(VectorId::new(7777), &vector(2), Metadata::new())
        .unwrap();
    drop(collection);

    // ...and reopening after the repair finds no fault at all.
    let (collection, report) = Collection::open(
        &dir,
        "recovery",
        DIM,
        Metric::Euclidean,
        IndexSpec::Flat,
        CollectionOptions {
            sync_policy: SyncPolicy::Always,
            snapshot_after_records: 0,
        },
    )
    .unwrap();
    assert!(!report.recovery.torn_tail);
    assert!(collection.contains(VectorId::new(7777)));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn recovery_over_a_snapshot_reproduces_the_same_search_results() {
    // Snapshot in the middle, keep writing, then "crash" by dropping the
    // handle. The graph the log replays onto is the snapshot's graph, not a
    // rebuild, which is what makes the results identical rather than merely
    // similar.
    let dir = temp_dir("snapshot-tail");
    let spec = IndexSpec::Hnsw(HnswConfig::new().with_seed(31));
    let queries: Vec<Vec<f32>> = (0..64).map(|q| vector(q * 13 + 5)).collect();

    let collection = open(&dir, spec.clone());
    workload(&collection, 200);
    collection.snapshot().unwrap();
    workload_range(&collection, 200, 320);
    let before: Vec<Vec<SearchResult>> = queries
        .iter()
        .map(|q| collection.search(q, 10, &AcceptAll).unwrap())
        .collect();
    let stats_before = collection.with_index(VectorIndex::stats);
    drop(collection);

    let collection = open(&dir, spec);
    let after: Vec<Vec<SearchResult>> = queries
        .iter()
        .map(|q| collection.search(q, 10, &AcceptAll).unwrap())
        .collect();
    assert_eq!(before, after);

    let stats_after = collection.with_index(VectorIndex::stats);
    assert_eq!(stats_before.len, stats_after.len);
    assert_eq!(stats_before.deleted, stats_after.deleted);
    assert_eq!(stats_before.edges, stats_after.edges);
    assert_eq!(stats_before.nodes_per_layer, stats_after.nodes_per_layer);
    std::fs::remove_dir_all(dir).unwrap();
}

fn workload_range(collection: &Collection, from: u64, to: u64) {
    for i in from..to {
        collection
            .upsert(VectorId::new(i), &vector(i), Metadata::new())
            .unwrap();
        if i % 4 == 0 {
            collection.delete(VectorId::new(i - from)).unwrap();
        }
    }
}

#[test]
fn a_kill_at_every_hundredth_write_leaves_a_consistent_collection() {
    // The "kill and reopen at many points" test. Each round writes a few more
    // records and then abandons the handle without a flush, which is what a
    // `SIGKILL` looks like to the filesystem: the page cache holds everything
    // `append` wrote, and `SyncPolicy::Always` has already forced it to disk.
    let dir = temp_dir("kill-loop");
    let spec = IndexSpec::Hnsw(HnswConfig::new().with_seed(17));
    let mut written = 0u64;

    for round in 0..12u64 {
        let collection = open(&dir, spec.clone());
        assert_eq!(
            collection.len() as u64,
            written,
            "round {round} opened with the wrong size"
        );
        for _ in 0..25 {
            collection
                .upsert(VectorId::new(written), &vector(written), Metadata::new())
                .unwrap();
            written += 1;
        }
        if round % 3 == 2 {
            collection.snapshot().unwrap();
        }
        // No flush, no close: just let it go.
        std::mem::forget(collection);
    }

    let collection = open(&dir, spec);
    assert_eq!(collection.len() as u64, written);
    for id in 0..written {
        assert!(collection.contains(VectorId::new(id)), "lost id {id}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_weaker_sync_policy_still_survives_a_process_crash() {
    // `SyncPolicy::Never` gives up on power loss, not on process death: the
    // bytes are in the kernel's page cache the moment `append` returns, and a
    // process that dies without flushing loses nothing.
    let dir = temp_dir("never-sync");
    let (collection, _) = Collection::open(
        &dir,
        "recovery",
        DIM,
        Metric::Euclidean,
        IndexSpec::Flat,
        CollectionOptions {
            sync_policy: SyncPolicy::Never,
            snapshot_after_records: 0,
        },
    )
    .unwrap();
    for i in 0..100u64 {
        collection
            .upsert(VectorId::new(i), &vector(i), Metadata::new())
            .unwrap();
    }
    std::mem::forget(collection);

    let (collection, report) = Collection::open(
        &dir,
        "recovery",
        DIM,
        Metric::Euclidean,
        IndexSpec::Flat,
        CollectionOptions {
            sync_policy: SyncPolicy::Never,
            snapshot_after_records: 0,
        },
    )
    .unwrap();
    assert_eq!(report.replayed, 100);
    assert!(!report.recovery.torn_tail);
    assert_eq!(collection.len(), 100);
    std::fs::remove_dir_all(dir).unwrap();
}
