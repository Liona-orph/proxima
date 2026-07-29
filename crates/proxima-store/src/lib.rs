//! Durability for Proxima: a write-ahead log, snapshots, and the
//! [`Collection`] that ties them to an index.
//!
//! Without this crate Proxima is a library — a good one, but one whose entire
//! contents vanish when the process does. This is what makes it a system.
//!
//! # The guarantee, stated precisely
//!
//! Under the default [`SyncPolicy::Always`]:
//!
//! > When [`Collection::upsert`] or [`Collection::delete`] returns `Ok`, the
//! > mutation is on stable storage. If the machine loses power in the next
//! > instant, reopening the collection produces an index that answers every
//! > query with **exactly** the same results, distances included.
//!
//! Both halves are tested. `tests/recovery.rs` truncates a real log at every
//! byte offset and checks that recovery lands on a prefix of history;
//! `tests/snapshot.rs` compares full result vectors across a save and load.
//!
//! What is *not* guaranteed, and is documented rather than glossed:
//!
//! * **No transactions.** A batch is one `fsync`, not one atomic unit. A crash
//!   mid-batch leaves a prefix durable.
//! * **Single node.** There is no replication and no consensus. The durability
//!   claim is about one disk.
//! * **One writer at a time.** See [`Collection`]'s concurrency model.
//!
//! # How the pieces fit
//!
//! ```text
//!   upsert / delete
//!         |
//!         v
//!   [ Wal::append ]  ---- length + CRC32C framed record, then fsync
//!         |
//!         v
//!   [ index.upsert ] ---- only after the log write succeeded
//!
//!   open()
//!     read manifest -> load snapshot -> replay wal tail -> truncate torn bytes
//! ```
//!
//! The log is the source of truth between snapshots and the snapshot is a
//! checkpoint, which is the WAL-plus-snapshot design rather than an LSM: see
//! `docs/adr/0006-wal-plus-snapshot-over-lsm.md` for why, given that the whole
//! index has to be in memory to be searched anyway.
//!
//! ```no_run
//! use proxima_core::{Metadata, Metric, VectorId};
//! use proxima_index::{AcceptAll, HnswConfig};
//! use proxima_store::{Collection, CollectionOptions, IndexSpec};
//!
//! let (collection, report) = Collection::open(
//!     "/var/lib/proxima/docs",
//!     "docs",
//!     8,
//!     Metric::Euclidean,
//!     IndexSpec::Hnsw(HnswConfig::new()),
//!     CollectionOptions::default(),
//! )?;
//! if report.recovery.torn_tail {
//!     eprintln!("discarded {} torn bytes", report.recovery.bytes_discarded);
//! }
//!
//! collection.upsert(VectorId::new(1), &[1.0; 8], Metadata::new().with("lang", "en"))?;
//! let hits = collection.search_with_metadata(&[1.0; 8], 5, &AcceptAll)?;
//! assert_eq!(hits[0].1.text("lang"), Some("en"));
//! # Ok::<(), proxima_core::ProximaError>(())
//! ```
#![forbid(unsafe_code)]

pub mod codec;
pub mod collection;
pub mod crc;
pub mod index;
pub mod snapshot;
pub mod wal;

pub use collection::{
    Collection, CollectionDescription, CollectionOptions, Counters, Manifest, OpenReport,
};
pub use crc::{crc32c, crc32c_append};
pub use index::{AnyIndex, IndexSpec};
pub use snapshot::{
    read_snapshot, read_snapshot_header, write_snapshot, SnapshotHeader, FORMAT_VERSION,
};
pub use wal::{Recovery, SyncPolicy, Wal, WalRecord};
