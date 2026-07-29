//! The write-ahead log: every mutation on disk before it is acknowledged.
//!
//! # The record format
//!
//! ```text
//! file   := header record*
//! header := "PROXWAL\x01"                    8 bytes, magic and format version
//! record := u32 payload_len | u32 crc32c(payload) | payload[payload_len]
//! payload := u8 op | body
//!   op 1, upsert := u64 id | u32 dim | f32[dim] | metadata
//!   op 2, delete := u64 id
//! ```
//!
//! The length comes before the checksum so that a reader can decide how much to
//! read in one syscall, and the checksum covers the payload but not the length.
//! That is deliberate: a torn write leaves a *shorter* file than the length
//! claims, which [`Wal::open`] detects from the file size alone without needing
//! to trust the checksum, and a length corrupted in place then fails the
//! checksum on the payload it selects. Covering the length as well would be
//! marginally stronger and would cost a second checksum region on every record.
//!
//! # The torn tail
//!
//! A log is opened after a crash, so the last record is the one that matters.
//! Three things can be true of it:
//!
//! * It is complete and its checksum matches. Replay it.
//! * It is short — the file ends inside the header or inside the payload.
//!   The write never completed, so the operation was never acknowledged, and
//!   the bytes are discarded and the file truncated to the last good record.
//! * It is the right length and the checksum does not match. The same
//!   conclusion: the kernel wrote some pages of the record and not others.
//!
//! [`Wal::open`] stops at the **first** bad record and truncates there. It does
//! not scan forward looking for a valid record after a bad one, because this is
//! an append-only file: a good record after a bad one would mean the corruption
//! is not a torn tail but damage in the middle, and silently skipping it would
//! replay a suffix of history without its prefix. That produces an index that
//! looks fine and is wrong, which is worse than refusing — so the discarded
//! bytes are counted and reported in [`Recovery`], and a caller that sees a
//! large `bytes_discarded` knows to look.
//!
//! The residual risk is a torn record whose truncated payload happens to
//! checksum to the value in its header. That is a `2^-32` event per crash; see
//! [`crate::crc`].
//!
//! # fsync, and what it costs
//!
//! [`SyncPolicy`] is the honest part of this module. `fsync` is the only thing
//! that makes a write survive a power cut, and it is expensive. Measured by
//! `cargo run --release -p proxima-store --example fsync_cost` on the benchmark
//! machine, appending a 128-dimensional upsert record:
//!
//! | policy | per record | relative |
//! |---|--:|--:|
//! | [`SyncPolicy::Always`] | 570-690 µs | 1x |
//! | [`SyncPolicy::EveryN`]`(64)` | 23-25 µs | ~26x faster |
//! | [`SyncPolicy::Never`] | 1.8 µs | ~350x faster |
//! | [`Wal::append_batch`] of 1 000, under `Always` | 2.8-3.0 µs | ~200x faster |
//!
//! The ranges are the spread across runs, which is what an `fsync` benchmark
//! looks like — the number is a property of the device's queue, not of this
//! code. These are on the benchmark container's filesystem; the absolutes move
//! by an order of magnitude between a consumer SSD and a battery-backed array,
//! but the *ratios* are the shape of the trade everywhere.
//!
//! The last row is the important one. **Batching is the way out that costs
//! nothing.** A thousand-record batch is one `fsync`, and it is acknowledged as
//! a whole, so it is exactly as durable as a thousand separate appends and 245
//! times cheaper. Weakening the policy is the other way out, and it costs
//! something real.
//!
//! What each policy survives:
//!
//! | policy | process crash | machine power loss |
//! |---|---|---|
//! | [`SyncPolicy::Always`] | everything acknowledged | everything acknowledged |
//! | [`SyncPolicy::EveryN`] | everything acknowledged | all but the last `n - 1` records |
//! | [`SyncPolicy::Never`] | everything acknowledged | nothing is promised |
//!
//! Every policy survives a *process* crash, including a panic or a `SIGKILL`,
//! because the bytes are already in the kernel's page cache by the time
//! `append` returns. Only power loss and kernel panics distinguish them.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use proxima_core::{Metadata, ProximaError, Result, VectorId};

use crate::codec::{Reader, Writer};
use crate::crc::crc32c;

/// Magic and format version at the head of every log file.
const MAGIC: &[u8; 8] = b"PROXWAL\x01";

/// Largest payload a single record may claim.
///
/// A vector of 1 048 576 dimensions is 4 MiB, and metadata is bounded by
/// nothing in particular, so 64 MiB is generous by a wide margin. Its real job
/// is to stop a corrupt length prefix from becoming a multi-gigabyte
/// allocation before the checksum has had a chance to reject it.
const MAX_RECORD_BYTES: usize = 64 << 20;

const OP_UPSERT: u8 = 1;
const OP_DELETE: u8 = 2;

/// When the log calls `fsync`.
///
/// See the module docs for the measured cost and for what each policy actually
/// survives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncPolicy {
    /// `fsync` before every `append` returns.
    ///
    /// The default, and the only policy under which "durable before it is
    /// acknowledged" is true. Everything else is a considered weakening.
    #[default]
    Always,
    /// `fsync` every `n` records.
    ///
    /// Bounds the loss window by a record count rather than by time, which is
    /// the version a caller can reason about: "at most 63 writes" is a
    /// statement about the workload, "at most 5 ms" is a statement about the
    /// disk. `n = 1` is [`SyncPolicy::Always`] and is normalised to it.
    EveryN(u32),
    /// Never `fsync`; let the kernel flush when it likes.
    ///
    /// Appropriate when the log is a rebuildable cache of a source of truth
    /// somewhere else, and never otherwise.
    Never,
}

impl std::fmt::Display for SyncPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyncPolicy::Always => f.write_str("always"),
            SyncPolicy::EveryN(n) => write!(f, "every_{n}"),
            SyncPolicy::Never => f.write_str("never"),
        }
    }
}

impl SyncPolicy {
    /// Parse the name [`Display`](std::fmt::Display) produces.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "always" => Some(SyncPolicy::Always),
            "never" => Some(SyncPolicy::Never),
            other => other
                .strip_prefix("every_")
                .and_then(|n| n.parse().ok())
                .map(SyncPolicy::EveryN),
        }
    }
}

/// One logged mutation.
///
/// Only two operations, and neither is "insert": the log records the *effect*
/// of a request, not the request. An insert that failed because the id already
/// existed writes nothing, and an insert that succeeded is indistinguishable
/// from an upsert once it is on disk. Keeping the distinction would mean replay
/// could fail with `DuplicateId` on a log that is perfectly valid — which is
/// exactly what happens if a snapshot is taken between the two writes.
#[derive(Debug, Clone, PartialEq)]
pub enum WalRecord {
    /// Store `vector` under `id`, replacing anything already there.
    Upsert {
        /// The id written.
        id: VectorId,
        /// The vector **as the caller supplied it**, before normalisation.
        ///
        /// Raw rather than prepared, so that replay goes through exactly the
        /// same `VectorSpace::prepare` path a live insert does. Logging the
        /// prepared form would make the log's meaning depend on the metric
        /// recorded elsewhere, and a snapshot restored under a changed metric
        /// would then replay vectors normalised for the old one.
        vector: Vec<f32>,
        /// The metadata written.
        metadata: Metadata,
    },
    /// Remove `id`.
    Delete {
        /// The id removed.
        id: VectorId,
    },
}

impl WalRecord {
    /// Encode the payload, without the length and checksum framing.
    fn encode(&self, writer: &mut Writer) {
        match self {
            WalRecord::Upsert {
                id,
                vector,
                metadata,
            } => {
                writer.u8(OP_UPSERT).u64(id.get()).f32_slice(vector);
                writer.metadata(metadata);
            }
            WalRecord::Delete { id } => {
                writer.u8(OP_DELETE).u64(id.get());
            }
        }
    }

    /// Decode a payload produced by [`WalRecord::encode`].
    fn decode(payload: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(payload, "wal record");
        let op = reader.u8()?;
        let record = match op {
            OP_UPSERT => {
                let id = VectorId::new(reader.u64()?);
                let vector = reader.f32_vec()?;
                let metadata = reader.metadata()?;
                WalRecord::Upsert {
                    id,
                    vector,
                    metadata,
                }
            }
            OP_DELETE => WalRecord::Delete {
                id: VectorId::new(reader.u64()?),
            },
            other => {
                return Err(ProximaError::corrupt(
                    "wal record",
                    format!("op {other} is neither 1 (upsert) nor 2 (delete)"),
                ))
            }
        };
        reader.expect_end()?;
        Ok(record)
    }
}

/// What [`Wal::open`] found and what it did about it.
///
/// Returned rather than logged, because "the log had a torn tail" is
/// information the operator needs and a library has no business deciding how to
/// print. A non-zero `bytes_discarded` after a clean shutdown is a bug or a
/// failing disk, and it is worth alerting on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Recovery {
    /// Records successfully read and returned.
    pub records: usize,
    /// Bytes of valid log kept, header included.
    pub bytes_kept: u64,
    /// Bytes after the last good record that were discarded and truncated away.
    pub bytes_discarded: u64,
    /// `true` if anything was discarded.
    pub torn_tail: bool,
}

/// An append-only log of mutations.
///
/// Holds one open file descriptor and does its own framing. It knows nothing
/// about indexes: [`crate::Collection`] is what pairs a log with the structure
/// it describes, and the split is what makes this file testable by writing
/// bytes and reading them back.
#[derive(Debug)]
pub struct Wal {
    file: File,
    path: PathBuf,
    policy: SyncPolicy,
    /// Current file length, tracked rather than `stat`-ed: it is consulted on
    /// every append to decide whether a snapshot is due.
    length: u64,
    records: u64,
    records_since_sync: u32,
    /// `true` when bytes have been written that no `fsync` has covered.
    dirty: bool,
}

impl Wal {
    /// Open (or create) the log at `path` and replay it.
    ///
    /// Returns the log, every record that was intact, and a [`Recovery`]
    /// describing what happened. A torn tail is **truncated on disk** here, not
    /// merely skipped: leaving it in place would mean the next append writes
    /// after garbage, and the record after it would then be unreachable to the
    /// next recovery for the same reason this one stops at the first fault.
    pub fn open(
        path: impl AsRef<Path>,
        policy: SyncPolicy,
    ) -> Result<(Self, Vec<WalRecord>, Recovery)> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|err| ProximaError::io("open wal", path.display(), &err))?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|err| ProximaError::io("read wal", path.display(), &err))?;

        let policy = match policy {
            SyncPolicy::EveryN(1) => SyncPolicy::Always,
            other => other,
        };

        let mut wal = Wal {
            file,
            path,
            policy,
            length: 0,
            records: 0,
            records_since_sync: 0,
            dirty: false,
        };

        // A file shorter than the magic, whose bytes are a prefix of it, is a
        // header that was interrupted mid-write — the very first append to a
        // brand new log. It is a torn tail like any other and is rewritten
        // rather than refused. A short file that is *not* a prefix is a
        // different file entirely and falls through to the error below.
        if bytes.len() < MAGIC.len() && MAGIC.starts_with(&bytes[..]) {
            let discarded = bytes.len() as u64;
            wal.file
                .set_len(0)
                .map_err(|err| ProximaError::io("truncate wal", wal.path.display(), &err))?;
            wal.file
                .seek(SeekFrom::Start(0))
                .map_err(|err| ProximaError::io("seek wal", wal.path.display(), &err))?;
            wal.write_all(MAGIC)?;
            wal.sync()?;
            return Ok((
                wal,
                Vec::new(),
                Recovery {
                    records: 0,
                    bytes_kept: MAGIC.len() as u64,
                    bytes_discarded: discarded,
                    torn_tail: discarded > 0,
                },
            ));
        }
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            return Err(ProximaError::corrupt(
                "wal header",
                format!(
                    "expected magic {:?}, found {:?}",
                    String::from_utf8_lossy(MAGIC),
                    String::from_utf8_lossy(&bytes[..bytes.len().min(MAGIC.len())])
                ),
            ));
        }

        let (records, good_end) = scan(&bytes)?;
        let recovery = Recovery {
            records: records.len(),
            bytes_kept: good_end as u64,
            bytes_discarded: (bytes.len() - good_end) as u64,
            torn_tail: good_end != bytes.len(),
        };

        if recovery.torn_tail {
            wal.file
                .set_len(good_end as u64)
                .map_err(|err| ProximaError::io("truncate wal", wal.path.display(), &err))?;
            wal.file
                .sync_all()
                .map_err(|err| ProximaError::io("sync wal", wal.path.display(), &err))?;
        }
        wal.file
            .seek(SeekFrom::Start(good_end as u64))
            .map_err(|err| ProximaError::io("seek wal", wal.path.display(), &err))?;
        wal.length = good_end as u64;
        wal.records = records.len() as u64;

        Ok((wal, records, recovery))
    }

    /// Append one record, honouring the sync policy.
    pub fn append(&mut self, record: &WalRecord) -> Result<()> {
        self.append_unsynced(record)?;
        self.maybe_sync()
    }

    /// Append many records with **one** sync at the end.
    ///
    /// The reason batch upsert exists as an API rather than as a loop in the
    /// caller: at [`SyncPolicy::Always`] the fsync dominates by two orders of
    /// magnitude, so a thousand-vector batch costs one fsync instead of a
    /// thousand and the whole request is roughly a thousand times cheaper. The
    /// durability guarantee is unchanged — the batch is acknowledged as a
    /// whole, and either all of it is on disk or a torn tail removes the
    /// suffix that is not.
    ///
    /// It is *not* atomic. A crash mid-batch leaves a prefix of it durable, and
    /// replay applies that prefix. Proxima has no transactions; see the README.
    pub fn append_batch(&mut self, records: &[WalRecord]) -> Result<()> {
        for record in records {
            self.append_unsynced(record)?;
        }
        self.maybe_sync()
    }

    fn append_unsynced(&mut self, record: &WalRecord) -> Result<()> {
        let mut payload = Writer::with_capacity(64);
        record.encode(&mut payload);
        let payload = payload.into_bytes();
        if payload.len() > MAX_RECORD_BYTES {
            return Err(ProximaError::Capacity {
                what: "wal record",
                limit: MAX_RECORD_BYTES,
            });
        }

        // One buffer, one `write_all`: two calls would let a signal or a short
        // write split the frame from its payload, which is a torn record we
        // caused ourselves rather than one the machine did.
        let mut framed = Vec::with_capacity(8 + payload.len());
        framed.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        framed.extend_from_slice(&crc32c(&payload).to_le_bytes());
        framed.extend_from_slice(&payload);
        self.write_all(&framed)?;
        self.records += 1;
        self.records_since_sync = self.records_since_sync.saturating_add(1);
        Ok(())
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        self.file
            .write_all(bytes)
            .map_err(|err| ProximaError::io("append to wal", self.path.display(), &err))?;
        self.length += bytes.len() as u64;
        self.dirty = true;
        Ok(())
    }

    fn maybe_sync(&mut self) -> Result<()> {
        match self.policy {
            SyncPolicy::Always => self.sync(),
            SyncPolicy::EveryN(n) if self.records_since_sync >= n => self.sync(),
            SyncPolicy::EveryN(_) | SyncPolicy::Never => Ok(()),
        }
    }

    /// Flush and `fsync` unconditionally.
    ///
    /// `sync_all` and not `sync_data`: the log's length is metadata, and a
    /// `sync_data` that leaves the new length in the inode cache can lose the
    /// tail of the file even though every data page reached the platter.
    pub fn sync(&mut self) -> Result<()> {
        self.file
            .flush()
            .map_err(|err| ProximaError::io("flush wal", self.path.display(), &err))?;
        self.file
            .sync_all()
            .map_err(|err| ProximaError::io("sync wal", self.path.display(), &err))?;
        self.records_since_sync = 0;
        self.dirty = false;
        Ok(())
    }

    /// Discard every record, keeping the header.
    ///
    /// Called immediately after a snapshot has been written and `fsync`-ed: the
    /// snapshot now covers everything the log described, so the log's contents
    /// are redundant. The order matters and is enforced by
    /// [`Collection::snapshot`](crate::Collection::snapshot) — truncate first
    /// and the window between the two loses data.
    pub fn truncate(&mut self) -> Result<()> {
        self.file
            .set_len(MAGIC.len() as u64)
            .map_err(|err| ProximaError::io("truncate wal", self.path.display(), &err))?;
        self.file
            .seek(SeekFrom::Start(MAGIC.len() as u64))
            .map_err(|err| ProximaError::io("seek wal", self.path.display(), &err))?;
        self.length = MAGIC.len() as u64;
        self.records = 0;
        self.records_since_sync = 0;
        self.dirty = true;
        self.sync()
    }

    /// Bytes the log occupies on disk.
    pub fn size_bytes(&self) -> u64 {
        self.length
    }

    /// Records appended since the last [`Wal::truncate`].
    pub fn records(&self) -> u64 {
        self.records
    }

    /// The sync policy in force.
    pub fn policy(&self) -> SyncPolicy {
        self.policy
    }

    /// `true` if bytes have been written that no `fsync` has covered.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// The file this log is backed by.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Read every intact record, returning them and the offset of the first fault.
///
/// Separated from [`Wal::open`] and taking a plain slice so that the recovery
/// logic can be tested by handing it bytes — every truncation of a known-good
/// log, which is what `tests/recovery.rs` does — rather than by arranging real
/// crashes.
fn scan(bytes: &[u8]) -> Result<(Vec<WalRecord>, usize)> {
    let mut records = Vec::new();
    let mut offset = MAGIC.len();

    loop {
        // A frame is eight bytes. Fewer than that left means the append was
        // interrupted before even the length reached the disk.
        if bytes.len() - offset < 8 {
            return Ok((records, offset));
        }
        let length = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        let expected = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);

        // Checked before the slice, so a corrupt length cannot index out of
        // bounds and cannot reserve 4 GiB.
        if length > MAX_RECORD_BYTES || bytes.len() - offset - 8 < length {
            return Ok((records, offset));
        }
        let payload = &bytes[offset + 8..offset + 8 + length];
        if crc32c(payload) != expected {
            return Ok((records, offset));
        }

        // The checksum passed, so the bytes are the ones that were written; a
        // decode failure here therefore means a format mismatch rather than a
        // torn write, and it is an error rather than a truncation point.
        records.push(WalRecord::decode(payload)?);
        offset += 8 + length;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::io::Write as _;

    use super::*;

    fn upsert(id: u64) -> WalRecord {
        WalRecord::Upsert {
            id: VectorId::new(id),
            vector: vec![id as f32, 0.5, -1.25],
            metadata: Metadata::new().with("n", id as i64),
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "proxima-wal-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn an_empty_log_gets_a_header_and_replays_nothing() {
        let dir = temp_dir("empty");
        let path = dir.join("wal");
        let (wal, records, recovery) = Wal::open(&path, SyncPolicy::Always).unwrap();
        assert!(records.is_empty());
        assert_eq!(
            recovery,
            Recovery {
                records: 0,
                bytes_kept: MAGIC.len() as u64,
                bytes_discarded: 0,
                torn_tail: false
            }
        );
        assert_eq!(wal.size_bytes(), MAGIC.len() as u64);
        assert!(!wal.is_dirty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn records_survive_a_close_and_reopen() {
        let dir = temp_dir("reopen");
        let path = dir.join("wal");
        {
            let (mut wal, _, _) = Wal::open(&path, SyncPolicy::Always).unwrap();
            wal.append(&upsert(1)).unwrap();
            wal.append(&WalRecord::Delete {
                id: VectorId::new(1),
            })
            .unwrap();
            wal.append(&upsert(2)).unwrap();
        }
        let (_, records, recovery) = Wal::open(&path, SyncPolicy::Always).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0], upsert(1));
        assert_eq!(
            records[1],
            WalRecord::Delete {
                id: VectorId::new(1)
            }
        );
        assert!(!recovery.torn_tail);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn every_truncation_point_is_recovered_to_a_prefix() {
        // The crash test, run at every possible byte: truncating the file after
        // `keep` bytes must yield exactly the records whose frames fit entirely
        // within `keep`, and must never error.
        let dir = temp_dir("torn");
        let path = dir.join("wal");
        {
            let (mut wal, _, _) = Wal::open(&path, SyncPolicy::Always).unwrap();
            for id in 0..8u64 {
                wal.append(&upsert(id)).unwrap();
            }
        }
        let full = std::fs::read(&path).unwrap();

        for keep in 0..=full.len() {
            let victim = dir.join(format!("torn-{keep}"));
            std::fs::write(&victim, &full[..keep]).unwrap();

            let (wal, records, recovery) = Wal::open(&victim, SyncPolicy::Always).unwrap();
            // Whatever came back is a prefix of the original history.
            for (index, record) in records.iter().enumerate() {
                assert_eq!(*record, upsert(index as u64), "keep {keep}");
            }
            assert_eq!(recovery.records, records.len());
            assert_eq!(recovery.torn_tail, recovery.bytes_discarded > 0);
            assert_eq!(recovery.bytes_kept, wal.size_bytes());
            // The torn bytes are gone from the file, not merely ignored.
            assert_eq!(
                std::fs::metadata(&victim).unwrap().len(),
                wal.size_bytes(),
                "keep {keep}"
            );

            // ...and the truncated log is immediately usable again.
            drop(wal);
            let (mut wal, _, _) = Wal::open(&victim, SyncPolicy::Always).unwrap();
            wal.append(&upsert(99)).unwrap();
            drop(wal);
            let (_, again, _) = Wal::open(&victim, SyncPolicy::Always).unwrap();
            assert_eq!(again.len(), records.len() + 1, "keep {keep}");
            assert_eq!(*again.last().unwrap(), upsert(99));
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_corrupt_payload_truncates_at_that_record() {
        let dir = temp_dir("corrupt");
        let path = dir.join("wal");
        {
            let (mut wal, _, _) = Wal::open(&path, SyncPolicy::Always).unwrap();
            for id in 0..4u64 {
                wal.append(&upsert(id)).unwrap();
            }
        }
        let mut bytes = std::fs::read(&path).unwrap();
        // Flip a bit in the middle, which lands inside the second or third
        // record's payload.
        let victim_byte = bytes.len() / 2;
        bytes[victim_byte] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();

        let (_, records, recovery) = Wal::open(&path, SyncPolicy::Always).unwrap();
        assert!(records.len() < 4, "the damaged record must not be replayed");
        assert!(recovery.torn_tail);
        assert!(recovery.bytes_discarded > 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_half_written_header_is_repaired_rather_than_refused() {
        // The very first append can be interrupted inside the eight-byte magic,
        // and the file that leaves behind must not look like somebody else's
        // file.
        let dir = temp_dir("half-header");
        for keep in 0..MAGIC.len() {
            let path = dir.join(format!("wal-{keep}"));
            std::fs::write(&path, &MAGIC[..keep]).unwrap();
            let (mut wal, records, recovery) = Wal::open(&path, SyncPolicy::Always).unwrap();
            assert!(records.is_empty());
            assert_eq!(recovery.bytes_discarded, keep as u64);
            assert_eq!(recovery.torn_tail, keep > 0);
            wal.append(&upsert(1)).unwrap();
            drop(wal);
            let (_, records, _) = Wal::open(&path, SyncPolicy::Always).unwrap();
            assert_eq!(records, vec![upsert(1)]);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_wrong_magic_is_an_error_and_not_a_truncation() {
        // Pointing the server at the wrong file must not silently erase it.
        let dir = temp_dir("magic");
        let path = dir.join("wal");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"not a wal at all").unwrap();
        drop(file);

        let err = Wal::open(&path, SyncPolicy::Always).unwrap_err();
        assert!(matches!(err, ProximaError::Corruption { .. }), "{err}");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 16);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn truncate_keeps_the_header_and_the_log_still_works() {
        let dir = temp_dir("truncate");
        let path = dir.join("wal");
        let (mut wal, _, _) = Wal::open(&path, SyncPolicy::Always).unwrap();
        wal.append(&upsert(1)).unwrap();
        wal.truncate().unwrap();
        assert_eq!(wal.records(), 0);
        assert_eq!(wal.size_bytes(), MAGIC.len() as u64);
        wal.append(&upsert(2)).unwrap();
        drop(wal);

        let (_, records, _) = Wal::open(&path, SyncPolicy::Always).unwrap();
        assert_eq!(records, vec![upsert(2)]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn every_n_syncs_on_the_nth_record() {
        let dir = temp_dir("everyn");
        let path = dir.join("wal");
        let (mut wal, _, _) = Wal::open(&path, SyncPolicy::EveryN(3)).unwrap();
        wal.append(&upsert(1)).unwrap();
        assert!(wal.is_dirty());
        wal.append(&upsert(2)).unwrap();
        assert!(wal.is_dirty());
        wal.append(&upsert(3)).unwrap();
        assert!(!wal.is_dirty(), "the third record triggers the sync");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn every_one_is_normalised_to_always() {
        let dir = temp_dir("normalise");
        let path = dir.join("wal");
        let (wal, _, _) = Wal::open(&path, SyncPolicy::EveryN(1)).unwrap();
        assert_eq!(wal.policy(), SyncPolicy::Always);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sync_policy_names_round_trip() {
        for policy in [
            SyncPolicy::Always,
            SyncPolicy::Never,
            SyncPolicy::EveryN(64),
        ] {
            assert_eq!(SyncPolicy::from_name(&policy.to_string()), Some(policy));
        }
        assert_eq!(SyncPolicy::from_name("sometimes"), None);
    }

    #[test]
    fn records_round_trip_through_the_codec() {
        for record in [
            upsert(0),
            upsert(u64::MAX),
            WalRecord::Delete {
                id: VectorId::new(7),
            },
            WalRecord::Upsert {
                id: VectorId::new(3),
                vector: Vec::new(),
                metadata: Metadata::new(),
            },
        ] {
            let mut writer = Writer::new();
            record.encode(&mut writer);
            assert_eq!(WalRecord::decode(&writer.into_bytes()).unwrap(), record);
        }
    }

    #[test]
    fn an_unknown_op_is_named_in_the_error() {
        let err = WalRecord::decode(&[9, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap_err();
        assert!(err.to_string().contains('9'), "{err}");
    }
}
