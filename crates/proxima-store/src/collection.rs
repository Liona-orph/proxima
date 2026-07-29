//! [`Collection`]: an index, a log, a snapshot and a concurrency model.
//!
//! # The concurrency model
//!
//! One `RwLock` around the index and the log together. Readers share it,
//! writers take it exclusively, and there is nothing finer-grained anywhere.
//!
//! That is a real limitation and it is chosen rather than defaulted into, so
//! here is the argument.
//!
//! **Searches are read-only and concurrent.** [`Collection::search`] takes the
//! read lock, and any number of them run at once on as many cores as the caller
//! has. This is the operation a vector database spends its life doing, and it
//! scales.
//!
//! **Writes serialise, and the lock is not what makes them serial.** A write is
//! an `fsync` — 570 to 690 µs on the benchmark machine, measured — then a graph
//! insert of about 100 µs. Even with a lock-free index the `fsync`s would queue
//! on the device. Making the index concurrently mutable would buy the 100 µs and
//! cost a rewrite of `HnswIndex` around atomics or fine-grained locks, plus the
//! loss of the determinism guarantee: concurrent inserts finish in a
//! nondeterministic order, and "same seed, same insertion order, same graph"
//! would become "same seed, same graph, if you inserted single-threaded". That
//! guarantee is load-bearing for the snapshot tests and for reproducing a
//! benchmark, and it was judged worth more than 100 µs per write.
//!
//! **Writers block readers.** This is the honest cost. A batch upsert of ten
//! thousand vectors holds the write lock for about a second, and every search
//! arriving in that second waits. The mitigation available today is to size
//! batches for the latency you can accept; the mitigation that is not
//! implemented is a copy-on-write index behind an `ArcSwap`, which would double
//! the memory. See the README's "What this is not".
//!
//! **The lock is not poisoned-aware.** A panic while holding the write lock
//! would poison it and every later call would fail. Rather than `unwrap` the
//! `LockResult` — which the project's no-`unwrap` rule forbids anyway —
//! [`Collection`] recovers the guard from the poison and reports the panic
//! through [`Collection::is_poisoned`]. The alternative, propagating an error
//! from every method forever, turns one bug into a permanently broken process.
//!
//! # The on-disk layout
//!
//! ```text
//! <dir>/
//!   manifest      the collection's name, space and index spec
//!   snapshot      the last snapshot, or absent
//!   wal           mutations since that snapshot
//! ```
//!
//! Recovery is: read the manifest, load the snapshot if there is one, replay
//! every intact WAL record on top, truncate the torn tail. The manifest exists
//! so that an empty collection — one created and not yet written to — still
//! knows its dimension and metric after a restart, which a snapshot-only design
//! would forget.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use proxima_core::{Metadata, Metric, ProximaError, Result, SearchResult, VectorId, VectorSpace};
use proxima_index::{Filter, IndexStats, VectorIndex};

use crate::codec::{Reader, Writer};
use crate::crc::crc32c;
use crate::index::{AnyIndex, IndexSpec};
use crate::snapshot::{self, SnapshotHeader};
use crate::wal::{Recovery, SyncPolicy, Wal, WalRecord};

/// File names inside a collection directory.
const MANIFEST: &str = "manifest";
const SNAPSHOT: &str = "snapshot";
const WAL: &str = "wal";

const MANIFEST_MAGIC: &[u8; 8] = b"PROXMANI";

/// How a collection is opened.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionOptions {
    /// When the log calls `fsync`.
    pub sync_policy: SyncPolicy,
    /// Take a snapshot automatically once the log holds this many records.
    ///
    /// Zero disables it. The trade is bounded recovery time against write
    /// amplification: a snapshot rewrites the whole index, so a threshold of
    /// 1 000 on a million-vector index rewrites four gigabytes every thousand
    /// writes. The default of 100 000 is chosen so that recovery replays at
    /// most a hundred thousand records — a few seconds — while the snapshot
    /// cost is amortised over a hundred thousand writes.
    pub snapshot_after_records: u64,
}

impl Default for CollectionOptions {
    fn default() -> Self {
        CollectionOptions {
            sync_policy: SyncPolicy::Always,
            snapshot_after_records: 100_000,
        }
    }
}

/// What opening a collection did.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpenReport {
    /// `true` if a snapshot was found and loaded.
    pub loaded_snapshot: bool,
    /// Live vectors in the snapshot, before replay.
    pub snapshot_len: usize,
    /// What the log replay found.
    pub recovery: Recovery,
    /// Records that replay applied. Differs from `recovery.records` only if a
    /// record was a delete of something already gone, which replay tolerates.
    pub replayed: usize,
}

/// A named index with durable storage.
///
/// ```no_run
/// use proxima_core::{Metadata, Metric, VectorId};
/// use proxima_store::{Collection, CollectionOptions, IndexSpec};
/// use proxima_index::HnswConfig;
///
/// let (collection, report) = Collection::open(
///     "/tmp/proxima/docs",
///     "docs",
///     384,
///     Metric::Cosine,
///     IndexSpec::Hnsw(HnswConfig::new()),
///     CollectionOptions::default(),
/// )?;
/// println!("replayed {} records", report.replayed);
///
/// collection.upsert(VectorId::new(1), &vec![0.1; 384], Metadata::new().with("lang", "en"))?;
/// let hits = collection.search(&vec![0.1; 384], 10, &proxima_index::AcceptAll)?;
/// assert_eq!(hits[0].id, VectorId::new(1));
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
#[derive(Debug)]
pub struct Collection {
    name: String,
    directory: PathBuf,
    space: VectorSpace,
    options: CollectionOptions,
    inner: RwLock<Inner>,
    /// Counters for the metrics endpoint.
    ///
    /// Atomics outside the lock so that scraping metrics never blocks a write,
    /// which is the whole reason a metrics endpoint is allowed to exist on a
    /// hot path.
    searches: AtomicU64,
    upserts: AtomicU64,
    deletes: AtomicU64,
    snapshots: AtomicU64,
}

#[derive(Debug)]
struct Inner {
    index: AnyIndex,
    wal: Wal,
}

impl Collection {
    /// Open the collection in `directory`, creating it if it is not there.
    ///
    /// If a manifest already exists, `dim`, `metric` and `spec` must agree with
    /// it; a mismatch is a [`ProximaError::Configuration`] rather than a
    /// silent reconfiguration, because reopening a 768-dimensional collection
    /// as a 384-dimensional one would make every replayed record fail and leave
    /// an empty index where the data used to be.
    pub fn open(
        directory: impl AsRef<Path>,
        name: impl Into<String>,
        dim: usize,
        metric: Metric,
        spec: IndexSpec,
        options: CollectionOptions,
    ) -> Result<(Self, OpenReport)> {
        let directory = directory.as_ref().to_path_buf();
        let name = name.into();
        std::fs::create_dir_all(&directory).map_err(|err| {
            ProximaError::io("create collection directory", directory.display(), &err)
        })?;

        let space = VectorSpace::new(dim, metric)?;
        let manifest = Manifest { name, space, spec };
        let manifest_path = directory.join(MANIFEST);
        if manifest_path.exists() {
            let existing = read_manifest(&manifest_path)?;
            if existing != manifest {
                return Err(ProximaError::config(
                    "manifest",
                    format!(
                        "{}/{}/{}",
                        existing.name,
                        existing.space,
                        existing.spec.name()
                    ),
                    "does not match the collection being opened",
                ));
            }
        } else {
            write_manifest(&manifest_path, &manifest)?;
        }
        let Manifest { name, space, spec } = manifest;

        let snapshot_path = directory.join(SNAPSHOT);
        let mut report = OpenReport::default();
        let mut index = if snapshot_path.exists() {
            let index = snapshot::read_snapshot(&snapshot_path)?;
            if index.space() != &space {
                return Err(ProximaError::corrupt(
                    "snapshot",
                    format!("holds a {} index, the manifest says {space}", index.space()),
                ));
            }
            report.loaded_snapshot = true;
            report.snapshot_len = index.len();
            index
        } else {
            spec.build(space.clone())?
        };

        let (wal, records, recovery) = Wal::open(directory.join(WAL), options.sync_policy)?;
        report.recovery = recovery;
        for record in &records {
            apply(&mut index, record)?;
            report.replayed += 1;
        }

        Ok((
            Collection {
                name,
                directory,
                space,
                options,
                inner: RwLock::new(Inner { index, wal }),
                searches: AtomicU64::new(0),
                upserts: AtomicU64::new(0),
                deletes: AtomicU64::new(0),
                snapshots: AtomicU64::new(0),
            },
            report,
        ))
    }

    /// The collection's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The directory backing it.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// The space its vectors live in.
    pub fn space(&self) -> &VectorSpace {
        &self.space
    }

    /// The options it was opened with.
    pub fn options(&self) -> &CollectionOptions {
        &self.options
    }

    /// `true` if a panic poisoned the lock at some point.
    ///
    /// The collection keeps working — see the module docs — but a `true` here
    /// means a bug was hit and the index may not be in the state the log
    /// describes. Worth exporting to a dashboard, and worth restarting for.
    pub fn is_poisoned(&self) -> bool {
        self.inner.is_poisoned()
    }

    fn read(&self) -> RwLockReadGuard<'_, Inner> {
        // A poisoned lock means a previous writer panicked. The index is a
        // plain data structure with no `unsafe`, so the worst case is a
        // half-applied insert, not undefined behaviour; refusing every
        // subsequent read would turn that into an outage.
        self.inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, Inner> {
        self.inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    // -----------------------------------------------------------------
    // Writes
    // -----------------------------------------------------------------

    /// Store one vector, replacing anything already under `id`.
    ///
    /// Durable before it returns, under [`SyncPolicy::Always`]. The order is
    /// log-then-apply, never the reverse: applying first and crashing before
    /// the log write would leave a mutation that recovery cannot reproduce, and
    /// the index would silently differ from the log every time.
    pub fn upsert(&self, id: VectorId, vector: &[f32], metadata: Metadata) -> Result<()> {
        self.upsert_batch(&[(id, vector.to_vec(), metadata)])
    }

    /// Store many vectors under one `fsync`.
    ///
    /// The reason the HTTP API has a batch endpoint. Every vector is validated
    /// *before* anything is logged, so a batch with one bad vector in it writes
    /// nothing at all — which is as close to atomic as a design with no
    /// transactions gets, and the closest thing to a useful guarantee here.
    pub fn upsert_batch(&self, batch: &[(VectorId, Vec<f32>, Metadata)]) -> Result<()> {
        for (_, vector, _) in batch {
            self.space.validate(vector)?;
        }
        let records: Vec<WalRecord> = batch
            .iter()
            .map(|(id, vector, metadata)| WalRecord::Upsert {
                id: *id,
                vector: vector.clone(),
                metadata: metadata.clone(),
            })
            .collect();

        let mut inner = self.write();
        inner.wal.append_batch(&records)?;
        for record in &records {
            apply(&mut inner.index, record)?;
        }
        self.upserts
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        self.maybe_snapshot(&mut inner)
    }

    /// Remove `id`.
    ///
    /// Removing something that is not there is **not** an error: it is logged
    /// and applied as a no-op. Deletes have to be idempotent or replay breaks,
    /// because a log can contain two deletes of the same id and the second one
    /// must not fail the whole recovery.
    pub fn delete(&self, id: VectorId) -> Result<bool> {
        let mut inner = self.write();
        let present = inner.index.contains(id);
        inner.wal.append(&WalRecord::Delete { id })?;
        if present {
            inner.index.remove(id)?;
            self.deletes.fetch_add(1, Ordering::Relaxed);
        }
        self.maybe_snapshot(&mut inner)?;
        Ok(present)
    }

    /// Remove many ids under one `fsync`.
    pub fn delete_batch(&self, ids: &[VectorId]) -> Result<usize> {
        let records: Vec<WalRecord> = ids.iter().map(|id| WalRecord::Delete { id: *id }).collect();
        let mut inner = self.write();
        inner.wal.append_batch(&records)?;
        let mut removed = 0;
        for id in ids {
            if inner.index.contains(*id) {
                inner.index.remove(*id)?;
                removed += 1;
            }
        }
        self.deletes.fetch_add(removed as u64, Ordering::Relaxed);
        self.maybe_snapshot(&mut inner)?;
        Ok(removed)
    }

    // -----------------------------------------------------------------
    // Reads
    // -----------------------------------------------------------------

    /// The `k` nearest vectors accepted by `filter`.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: &dyn Filter,
    ) -> Result<Vec<SearchResult>> {
        let inner = self.read();
        let hits = inner.index.search(query, k, filter)?;
        self.searches.fetch_add(1, Ordering::Relaxed);
        Ok(hits)
    }

    /// The `k` nearest vectors, with each one's metadata attached.
    ///
    /// One lock acquisition rather than a `search` followed by `k` metadata
    /// lookups: between the two, a writer could delete a hit and the response
    /// would carry a distance with no payload.
    pub fn search_with_metadata(
        &self,
        query: &[f32],
        k: usize,
        filter: &dyn Filter,
    ) -> Result<Vec<(SearchResult, Metadata)>> {
        let inner = self.read();
        let hits = inner.index.search(query, k, filter)?;
        let with_metadata = hits
            .into_iter()
            .map(|hit| {
                let metadata = inner.index.metadata(hit.id).cloned().unwrap_or_default();
                (hit, metadata)
            })
            .collect();
        self.searches.fetch_add(1, Ordering::Relaxed);
        Ok(with_metadata)
    }

    /// `true` if `id` is present.
    pub fn contains(&self, id: VectorId) -> bool {
        self.read().index.contains(id)
    }

    /// Live vectors.
    pub fn len(&self) -> usize {
        self.read().index.len()
    }

    /// `true` if there are none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A description of the collection, for the API's describe endpoint.
    pub fn describe(&self) -> CollectionDescription {
        let inner = self.read();
        CollectionDescription {
            name: self.name.clone(),
            dim: self.space.dim(),
            metric: self.space.metric(),
            index: inner.index.kind(),
            stats: inner.index.stats(),
            metadata_keys: snapshot::metadata_keys(&inner.index),
            wal_records: inner.wal.records(),
            wal_bytes: inner.wal.size_bytes(),
            sync_policy: inner.wal.policy(),
            counters: self.counters(),
        }
    }

    /// The operation counters, for the metrics endpoint.
    pub fn counters(&self) -> Counters {
        Counters {
            searches: self.searches.load(Ordering::Relaxed),
            upserts: self.upserts.load(Ordering::Relaxed),
            deletes: self.deletes.load(Ordering::Relaxed),
            snapshots: self.snapshots.load(Ordering::Relaxed),
        }
    }

    // -----------------------------------------------------------------
    // Durability
    // -----------------------------------------------------------------

    /// Write a snapshot and truncate the log.
    ///
    /// The order is: write the snapshot to a temporary, `fsync`, rename,
    /// `fsync` the directory, and only then truncate the log. Doing it the
    /// other way round leaves a window in which neither file describes the
    /// data.
    pub fn snapshot(&self) -> Result<u64> {
        let mut inner = self.write();
        self.snapshot_locked(&mut inner)
    }

    fn snapshot_locked(&self, inner: &mut Inner) -> Result<u64> {
        let bytes = snapshot::write_snapshot(self.directory.join(SNAPSHOT), &inner.index)?;
        inner.wal.truncate()?;
        self.snapshots.fetch_add(1, Ordering::Relaxed);
        Ok(bytes)
    }

    fn maybe_snapshot(&self, inner: &mut Inner) -> Result<()> {
        if self.options.snapshot_after_records > 0
            && inner.wal.records() >= self.options.snapshot_after_records
        {
            self.snapshot_locked(inner)?;
        }
        Ok(())
    }

    /// `fsync` the log without taking a snapshot.
    ///
    /// What graceful shutdown calls. Under [`SyncPolicy::Always`] it is a
    /// no-op; under the weaker policies it is the difference between losing the
    /// last few hundred writes and losing none.
    pub fn flush(&self) -> Result<()> {
        self.write().wal.sync()
    }

    /// Reclaim tombstoned slots, then snapshot.
    ///
    /// The snapshot is not optional: compaction renumbers slots, so the
    /// existing snapshot plus the existing log no longer describe the index in
    /// memory, and a crash before the next snapshot would recover a *different*
    /// index. Returns the number of slots reclaimed.
    pub fn compact(&self) -> Result<usize> {
        let mut inner = self.write();
        let reclaimed = inner.index.compact()?;
        if reclaimed > 0 {
            self.snapshot_locked(&mut inner)?;
        }
        Ok(reclaimed)
    }

    /// The header of the on-disk snapshot, if there is one.
    pub fn snapshot_header(&self) -> Result<Option<SnapshotHeader>> {
        let path = self.directory.join(SNAPSHOT);
        if !path.exists() {
            return Ok(None);
        }
        snapshot::read_snapshot_header(path).map(Some)
    }

    /// Delete every file this collection owns.
    ///
    /// Consumes the collection, because the alternative is a handle onto a
    /// directory that no longer exists.
    pub fn drop_collection(self) -> Result<()> {
        let directory = self.directory.clone();
        drop(self);
        std::fs::remove_dir_all(&directory).map_err(|err| {
            ProximaError::io("remove collection directory", directory.display(), &err)
        })
    }

    /// Run `f` against the index itself.
    ///
    /// The escape hatch for the operations that are specific to one
    /// implementation — `IvfPqIndex::train`, `HnswIndex::search_with_ef` — which
    /// a collection has no business enumerating. Holds the write lock for the
    /// duration, so `f` should not block.
    pub fn with_index_mut<T>(&self, f: impl FnOnce(&mut AnyIndex) -> T) -> T {
        f(&mut self.write().index)
    }

    /// Run `f` against the index for reading.
    pub fn with_index<T>(&self, f: impl FnOnce(&AnyIndex) -> T) -> T {
        f(&self.read().index)
    }
}

/// Operation counters, monotonically increasing since the process started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counters {
    /// Searches served.
    pub searches: u64,
    /// Vectors upserted.
    pub upserts: u64,
    /// Vectors actually removed, not delete requests.
    pub deletes: u64,
    /// Snapshots written.
    pub snapshots: u64,
}

/// Everything the describe endpoint reports.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionDescription {
    /// The collection's name.
    pub name: String,
    /// Vector dimension.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Index kind: `"flat"`, `"hnsw"` or `"ivfpq"`.
    pub index: &'static str,
    /// Size and shape of the index.
    pub stats: IndexStats,
    /// Every metadata key present, sorted.
    pub metadata_keys: Vec<String>,
    /// Records in the log since the last snapshot.
    pub wal_records: u64,
    /// Bytes the log occupies.
    pub wal_bytes: u64,
    /// The log's sync policy.
    pub sync_policy: SyncPolicy,
    /// Operation counters.
    pub counters: Counters,
}

/// Apply one logged record to an index.
///
/// The single place replay and the live write path share, which is what makes
/// "recovery reproduces the index" a structural property rather than two
/// implementations that have to agree.
fn apply(index: &mut AnyIndex, record: &WalRecord) -> Result<()> {
    match record {
        WalRecord::Upsert {
            id,
            vector,
            metadata,
        } => index.upsert(*id, vector, metadata.clone()),
        WalRecord::Delete { id } => {
            if index.contains(*id) {
                index.remove(*id)
            } else {
                // Idempotent by design; see `Collection::delete`.
                Ok(())
            }
        }
    }
}

/// What a collection directory says about itself, before it is opened.
///
/// [`Collection::open`] needs the dimension, the metric and the index spec as
/// arguments — it has to, because it also creates collections — so a process
/// that is *reopening* what it finds on disk needs a way to ask first. That is
/// this. It is also the reason the manifest exists as a separate file rather
/// than only inside the snapshot: a collection that was created and never
/// written to has no snapshot and would otherwise be unopenable after a
/// restart.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    /// The name the collection was created under.
    pub name: String,
    /// Its vector space.
    pub space: VectorSpace,
    /// Its index kind and configuration.
    pub spec: IndexSpec,
}

impl Manifest {
    /// Read the manifest in `directory`.
    ///
    /// Returns [`ProximaError::Io`] if there is no manifest there, which is how
    /// a caller scanning a data directory tells a collection from an unrelated
    /// folder.
    pub fn read(directory: impl AsRef<Path>) -> Result<Self> {
        read_manifest(&directory.as_ref().join(MANIFEST))
    }
}

fn write_manifest(path: &Path, manifest: &Manifest) -> Result<()> {
    let mut body = Writer::new();
    body.str(&manifest.name);
    body.space(&manifest.space);
    match &manifest.spec {
        IndexSpec::Flat => {
            body.u8(1);
        }
        IndexSpec::Hnsw(config) => {
            body.u8(2)
                .len_prefix(config.m())
                .len_prefix(config.ef_construction())
                .len_prefix(config.ef_search())
                .u64(config.seed())
                .len_prefix(config.filter_ef_limit())
                .u8(u8::from(config.exhaustive_filter_fallback()))
                .u8(u8::from(config.keep_pruned_connections()));
        }
        IndexSpec::IvfPq(config) => {
            body.u8(3)
                .len_prefix(config.nlist())
                .len_prefix(config.subspaces())
                .len_prefix(config.nprobe())
                .len_prefix(config.rerank())
                .u64(config.seed())
                .len_prefix(config.max_iterations())
                .len_prefix(config.training_sample())
                .u64(config.train_threshold() as u64)
                .u8(u8::from(config.retain_vectors()));
        }
    }
    let body = body.into_bytes();

    let mut file = Writer::new();
    file.raw(MANIFEST_MAGIC).u32(crc32c(&body)).raw(&body);
    let bytes = file.into_bytes();

    let temporary = path.with_extension("tmp");
    std::fs::write(&temporary, &bytes)
        .map_err(|err| ProximaError::io("write manifest", temporary.display(), &err))?;
    std::fs::rename(&temporary, path)
        .map_err(|err| ProximaError::io("rename manifest", path.display(), &err))
}

fn read_manifest(path: &Path) -> Result<Manifest> {
    let bytes = std::fs::read(path)
        .map_err(|err| ProximaError::io("read manifest", path.display(), &err))?;
    if bytes.len() < 12 || &bytes[..8] != MANIFEST_MAGIC {
        return Err(ProximaError::corrupt(
            "manifest",
            "missing or wrong magic; this is not a Proxima collection directory",
        ));
    }
    let expected = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let body = &bytes[12..];
    if crc32c(body) != expected {
        return Err(ProximaError::corrupt(
            "manifest",
            format!(
                "checksum {:#010x} does not match {expected:#010x}",
                crc32c(body)
            ),
        ));
    }

    let mut reader = Reader::new(body, "manifest");
    let name = reader.str()?.to_owned();
    let space = reader.space()?;
    let spec = match reader.u8()? {
        1 => IndexSpec::Flat,
        2 => {
            let m = reader.u32()? as usize;
            let ef_construction = reader.u32()? as usize;
            let ef_search = reader.u32()? as usize;
            let seed = reader.u64()?;
            let filter_ef_limit = reader.u32()? as usize;
            let fallback = reader.u8()? != 0;
            let keep_pruned = reader.u8()? != 0;
            IndexSpec::Hnsw(
                proxima_index::HnswConfig::new()
                    .with_m(m)
                    .with_ef_construction(ef_construction)
                    .with_ef_search(ef_search)
                    .with_seed(seed)
                    .with_filter_ef_limit(filter_ef_limit)
                    .with_exhaustive_filter_fallback(fallback)
                    .with_keep_pruned_connections(keep_pruned),
            )
        }
        3 => {
            let nlist = reader.u32()? as usize;
            let subspaces = reader.u32()? as usize;
            let nprobe = reader.u32()? as usize;
            let rerank = reader.u32()? as usize;
            let seed = reader.u64()?;
            let max_iterations = reader.u32()? as usize;
            let training_sample = reader.u32()? as usize;
            let train_threshold = reader.u64()? as usize;
            let retain_vectors = reader.u8()? != 0;
            IndexSpec::IvfPq(
                proxima_index::IvfPqConfig::new(nlist, subspaces)
                    .with_nprobe(nprobe)
                    .with_rerank(rerank)
                    .with_seed(seed)
                    .with_max_iterations(max_iterations)
                    .with_training_sample(training_sample)
                    .with_train_threshold(train_threshold)
                    .with_retain_vectors(retain_vectors),
            )
        }
        other => {
            return Err(ProximaError::corrupt(
                "manifest",
                format!("index kind {other} is not one of 1..=3"),
            ))
        }
    };
    reader.expect_end()?;
    Ok(Manifest { name, space, spec })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use proxima_index::{AcceptAll, HnswConfig};

    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "proxima-collection-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    fn vector(i: u64, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|d| ((i % 13) as f32 * 1.7 + d as f32 * 0.3).sin() * 4.0 + i as f32 * 0.002)
            .collect()
    }

    fn open(dir: &Path, spec: IndexSpec, options: CollectionOptions) -> (Collection, OpenReport) {
        Collection::open(dir, "t", 12, Metric::Euclidean, spec, options).unwrap()
    }

    #[test]
    fn writes_survive_a_reopen_without_a_snapshot() {
        let dir = temp_dir("replay");
        {
            let (collection, report) = open(
                &dir,
                IndexSpec::Hnsw(HnswConfig::new().with_seed(2)),
                CollectionOptions::default(),
            );
            assert!(!report.loaded_snapshot);
            for i in 0..200u64 {
                collection
                    .upsert(
                        VectorId::new(i),
                        &vector(i, 12),
                        Metadata::new().with("i", i as i64),
                    )
                    .unwrap();
            }
            collection.delete(VectorId::new(5)).unwrap();
            assert_eq!(collection.len(), 199);
        }
        let (collection, report) = open(
            &dir,
            IndexSpec::Hnsw(HnswConfig::new().with_seed(2)),
            CollectionOptions::default(),
        );
        assert!(!report.loaded_snapshot);
        assert_eq!(report.replayed, 201);
        assert_eq!(collection.len(), 199);
        assert!(!collection.contains(VectorId::new(5)));
        assert_eq!(
            collection
                .search_with_metadata(&vector(7, 12), 1, &AcceptAll)
                .unwrap()[0]
                .1
                .int("i"),
            Some(7)
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_snapshot_plus_a_tail_reproduces_the_same_results() {
        // The property the whole crate exists for, stated as an assertion.
        let dir = temp_dir("snapshot-tail");
        let spec = IndexSpec::Hnsw(HnswConfig::new().with_seed(9));
        let queries: Vec<Vec<f32>> = (0..40).map(|q| vector(q * 7 + 3, 12)).collect();

        let before: Vec<Vec<SearchResult>> = {
            let (collection, _) = open(&dir, spec.clone(), CollectionOptions::default());
            for i in 0..300u64 {
                collection
                    .upsert(VectorId::new(i), &vector(i, 12), Metadata::new())
                    .unwrap();
            }
            collection.snapshot().unwrap();
            // ...and then more writes, which live only in the log.
            for i in 300..400u64 {
                collection
                    .upsert(VectorId::new(i), &vector(i, 12), Metadata::new())
                    .unwrap();
            }
            for i in (0..50).step_by(3) {
                collection.delete(VectorId::new(i)).unwrap();
            }
            queries
                .iter()
                .map(|q| collection.search(q, 10, &AcceptAll).unwrap())
                .collect()
        };

        let (collection, report) = open(&dir, spec, CollectionOptions::default());
        assert!(report.loaded_snapshot);
        assert_eq!(report.snapshot_len, 300);
        assert_eq!(report.replayed, 117);
        let after: Vec<Vec<SearchResult>> = queries
            .iter()
            .map(|q| collection.search(q, 10, &AcceptAll).unwrap())
            .collect();
        assert_eq!(before, after);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_batch_with_one_bad_vector_writes_nothing() {
        let dir = temp_dir("batch-validate");
        let (collection, _) = open(&dir, IndexSpec::Flat, CollectionOptions::default());
        let batch = vec![
            (VectorId::new(1), vector(1, 12), Metadata::new()),
            (VectorId::new(2), vec![0.0; 3], Metadata::new()),
        ];
        assert!(collection.upsert_batch(&batch).is_err());
        assert_eq!(collection.len(), 0);
        assert_eq!(collection.describe().wal_records, 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn deleting_something_absent_is_not_an_error() {
        let dir = temp_dir("delete-absent");
        let (collection, _) = open(&dir, IndexSpec::Flat, CollectionOptions::default());
        assert!(!collection.delete(VectorId::new(42)).unwrap());
        // ...and replaying that log does not fail either.
        drop(collection);
        let (collection, report) = open(&dir, IndexSpec::Flat, CollectionOptions::default());
        assert_eq!(report.replayed, 1);
        assert!(collection.is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reopening_with_a_different_shape_is_refused() {
        let dir = temp_dir("manifest-mismatch");
        let (collection, _) = open(&dir, IndexSpec::Flat, CollectionOptions::default());
        drop(collection);
        let err = Collection::open(
            &dir,
            "t",
            24,
            Metric::Euclidean,
            IndexSpec::Flat,
            CollectionOptions::default(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                ProximaError::Configuration {
                    parameter: "manifest",
                    ..
                }
            ),
            "{err}"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn automatic_snapshots_truncate_the_log() {
        let dir = temp_dir("auto-snapshot");
        let options = CollectionOptions {
            snapshot_after_records: 50,
            ..CollectionOptions::default()
        };
        let (collection, _) = open(&dir, IndexSpec::Flat, options);
        for i in 0..120u64 {
            collection
                .upsert(VectorId::new(i), &vector(i, 12), Metadata::new())
                .unwrap();
        }
        let description = collection.describe();
        assert!(description.wal_records < 50, "{description:?}");
        assert!(collection.counters().snapshots >= 2);
        assert_eq!(collection.snapshot_header().unwrap().unwrap().kind, "flat");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn compaction_snapshots_so_that_recovery_agrees() {
        let dir = temp_dir("compact");
        let spec = IndexSpec::Hnsw(HnswConfig::new().with_seed(4));
        let (collection, _) = open(&dir, spec.clone(), CollectionOptions::default());
        for i in 0..300u64 {
            collection
                .upsert(VectorId::new(i), &vector(i, 12), Metadata::new())
                .unwrap();
        }
        for i in (0..300).step_by(2) {
            collection.delete(VectorId::new(i)).unwrap();
        }
        assert_eq!(collection.compact().unwrap(), 150);
        let before: Vec<SearchResult> = collection.search(&vector(11, 12), 10, &AcceptAll).unwrap();
        drop(collection);

        let (collection, _) = open(&dir, spec, CollectionOptions::default());
        assert_eq!(collection.len(), 150);
        assert_eq!(
            collection.search(&vector(11, 12), 10, &AcceptAll).unwrap(),
            before
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn describe_reports_the_metadata_keys_and_the_counters() {
        let dir = temp_dir("describe");
        let (collection, _) = open(&dir, IndexSpec::Flat, CollectionOptions::default());
        collection
            .upsert(
                VectorId::new(1),
                &vector(1, 12),
                Metadata::new().with("lang", "en").with("year", 2026_i64),
            )
            .unwrap();
        collection.search(&vector(1, 12), 5, &AcceptAll).unwrap();
        let description = collection.describe();
        assert_eq!(description.metadata_keys, vec!["lang", "year"]);
        assert_eq!(description.counters.searches, 1);
        assert_eq!(description.counters.upserts, 1);
        assert_eq!(description.index, "flat");
        assert_eq!(description.sync_policy, SyncPolicy::Always);
        assert!(!collection.is_poisoned());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn concurrent_searches_do_not_block_each_other() {
        // Not a timing assertion — those are flaky — but a structural one: many
        // threads hold the read guard at once and all of them make progress.
        let dir = temp_dir("concurrent");
        let (collection, _) = open(&dir, IndexSpec::Flat, CollectionOptions::default());
        for i in 0..200u64 {
            collection
                .upsert(VectorId::new(i), &vector(i, 12), Metadata::new())
                .unwrap();
        }
        let collection = std::sync::Arc::new(collection);
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let collection = collection.clone();
                std::thread::spawn(move || {
                    for i in 0..50u64 {
                        let hits = collection
                            .search(&vector(i + t, 12), 10, &AcceptAll)
                            .unwrap();
                        assert_eq!(hits.len(), 10);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(collection.counters().searches, 400);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn dropping_a_collection_removes_its_files() {
        let dir = temp_dir("drop");
        let (collection, _) = open(&dir, IndexSpec::Flat, CollectionOptions::default());
        collection
            .upsert(VectorId::new(1), &vector(1, 12), Metadata::new())
            .unwrap();
        assert!(dir.join(WAL).exists());
        collection.drop_collection().unwrap();
        assert!(!dir.exists());
    }

    #[test]
    fn a_corrupt_manifest_is_refused_rather_than_overwritten() {
        let dir = temp_dir("bad-manifest");
        let (collection, _) = open(&dir, IndexSpec::Flat, CollectionOptions::default());
        drop(collection);
        let path = dir.join(MANIFEST);
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let err = Collection::open(
            &dir,
            "t",
            12,
            Metric::Euclidean,
            IndexSpec::Flat,
            CollectionOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, ProximaError::Corruption { .. }), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_escape_hatch_reaches_the_concrete_index() {
        let dir = temp_dir("escape");
        let (collection, _) = open(&dir, IndexSpec::Flat, CollectionOptions::default());
        collection
            .upsert(VectorId::new(1), &vector(1, 12), Metadata::new())
            .unwrap();
        let kind = collection.with_index(AnyIndex::kind);
        assert_eq!(kind, "flat");
        let reclaimed = collection.with_index_mut(|index| index.compact().unwrap());
        assert_eq!(reclaimed, 0);
        assert_eq!(collection.space().dim(), 12);
        assert_eq!(collection.name(), "t");
        assert_eq!(collection.directory(), dir.as_path());
        assert_eq!(collection.options().sync_policy, SyncPolicy::Always);
        collection.flush().unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }
}
