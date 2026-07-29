//! What `fsync` costs, on whatever machine you run it on.
//!
//! The numbers in `proxima_store::wal`'s documentation come from this file, and
//! it is here rather than in `benches/` because the quantity being measured is a
//! syscall's interaction with a filesystem, not a function's throughput —
//! criterion's statistics would be measuring the disk's queue depth and calling
//! it variance.
//!
//! ```text
//! cargo run --release -p proxima-store --example fsync_cost
//! ```
//!
//! Run it on the storage you actually intend to deploy on. The ratio between
//! the policies is stable; the absolute numbers are a property of the device.

#![allow(clippy::unwrap_used)]

use std::time::Instant;

use proxima_core::{Metadata, VectorId};
use proxima_store::{SyncPolicy, Wal, WalRecord};

/// Records per policy. `Always` gets fewer because it is three orders of
/// magnitude slower and the point is a per-record figure, not a long wait.
const SYNCED_RECORDS: u32 = 500;
const UNSYNCED_RECORDS: u32 = 20_000;
const BATCH: u64 = 1_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::temp_dir().join(format!("proxima-fsync-cost-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory)?;

    let record = WalRecord::Upsert {
        id: VectorId::new(1),
        vector: vec![0.5f32; 128],
        metadata: Metadata::new().with("k", "v"),
    };

    println!(
        "{:<12} {:>8}  {:>12}  {:>12}",
        "policy", "records", "total", "per record"
    );
    for (name, policy, count) in [
        ("always", SyncPolicy::Always, SYNCED_RECORDS),
        ("every_64", SyncPolicy::EveryN(64), UNSYNCED_RECORDS),
        ("never", SyncPolicy::Never, UNSYNCED_RECORDS),
    ] {
        let (mut wal, _, _) = Wal::open(directory.join(name), policy)?;
        // Warm the file up so that the first extent allocation is not counted.
        for _ in 0..50 {
            wal.append(&record)?;
        }
        let start = Instant::now();
        for _ in 0..count {
            wal.append(&record)?;
        }
        wal.sync()?;
        let elapsed = start.elapsed();
        println!(
            "{name:<12} {count:>8}  {:>12.1?}  {:>12.1?}",
            elapsed,
            elapsed / count
        );
    }

    // The row that matters: one fsync for a whole batch, at full durability.
    let (mut wal, _, _) = Wal::open(directory.join("batch"), SyncPolicy::Always)?;
    let batch: Vec<WalRecord> = (0..BATCH)
        .map(|i| WalRecord::Upsert {
            id: VectorId::new(i),
            vector: vec![0.5f32; 128],
            metadata: Metadata::new(),
        })
        .collect();
    let start = Instant::now();
    wal.append_batch(&batch)?;
    let elapsed = start.elapsed();
    println!(
        "{:<12} {BATCH:>8}  {:>12.1?}  {:>12.1?}   (one fsync, still `always`)",
        "batch",
        elapsed,
        elapsed / BATCH as u32
    );

    std::fs::remove_dir_all(&directory)?;
    Ok(())
}
