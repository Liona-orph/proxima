//! Snapshots: a whole index in one checksummed file.
//!
//! # The file
//!
//! ```text
//! file    := magic | version | byte_order | reserved | body_len | body | crc32c(body)
//! magic   := "PROXSNAP"                      8 bytes
//! version := u32                             format version, currently 1
//! byte_order := u32 = 0x0102_0304            little-endian marker
//! reserved   := u32 = 0                      padding to an 8-byte body offset,
//!                                            and checked, so that every byte of
//!                                            the file is covered by something
//! body_len   := u64                          length of body in bytes
//! body       := space | kind | index-specific
//! ```
//!
//! The byte-order marker earns its four bytes: the codec is little-endian and a
//! big-endian reader would otherwise decode a `dim` of 4 096 as 65 536 and
//! produce a confidently wrong index rather than an error.
//!
//! "Identical" is about *answers*: ids, distances, order, and the graph they
//! came out of. It is not about the process's memory footprint — the arrays a
//! reload allocates are sized exactly, while a grown index carries the
//! allocator's slack, so `IndexStats::vector_bytes` can legitimately shrink
//! across a save and load.
//!
//! The checksum is over the body only and sits at the end, so writing is a
//! single forward pass — the body is built in memory, checksummed, and written
//! with the header in front. Putting it in the header would mean seeking back
//! over a file that may be gigabytes.
//!
//! # What "reload reproduces the same results" means, and why it needs the
//! graph
//!
//! The property this module exists to provide is that an index and its reload
//! answer every query identically, to the bit. That is stronger than "holds the
//! same vectors", and the difference is the whole reason a snapshot is not just
//! a dump of the corpus:
//!
//! * **Slot numbers must be preserved.** Ties in distance are broken by id, but
//!   the order candidates are *examined* in is slot order, and on a graph the
//!   examination order decides which local optimum a greedy walk finds. A
//!   snapshot that renumbered slots would give a different — still good, but
//!   different — answer.
//! * **Tombstones must be preserved.** A removed vector still has edges
//!   pointing at it and is still traversed through. Dropping the tombstones
//!   would change every path that ran through one.
//! * **The graph itself must be preserved.** Rebuilding from the vectors in
//!   slot order reproduces the graph only if nothing was ever deleted. After a
//!   single deletion it does not, because the rebuild never creates the edges
//!   the tombstone left behind.
//!
//! `tests/snapshot.rs` asserts the property directly: build, delete, upsert,
//! snapshot, reload, and compare a hundred queries' full result vectors —
//! distances included — for exact equality.
//!
//! # Atomicity
//!
//! [`write_snapshot`] writes to `path.tmp`, `fsync`s it, renames it over
//! `path`, and then `fsync`s the *directory*. All four steps are needed: the
//! rename is atomic on every POSIX filesystem, but without the first `fsync` the
//! rename can be durable while the contents are not, and without the last one
//! the rename itself can be lost. A crash at any point leaves either the
//! previous snapshot or the new one, never a half-written file.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use proxima_core::{ProximaError, Result, UnmappedSlots, VectorId, VectorSpace, VectorStore};
use proxima_index::{
    FlatIndex, HnswConfig, HnswGraphParts, HnswIndex, IvfPqConfig, IvfPqIndex, IvfPqPartsOwned,
    TrainingReport, VectorIndex,
};

use crate::codec::{Reader, Writer};
use crate::crc::crc32c;
use crate::index::AnyIndex;

/// Magic at the head of every snapshot.
const MAGIC: &[u8; 8] = b"PROXSNAP";

/// Format version. Bumped whenever the body layout changes in a way an older
/// reader would misinterpret; a reader refuses anything it does not know.
pub const FORMAT_VERSION: u32 = 1;

/// Written as `u32` little-endian, so a big-endian reader sees `0x04030201`.
const BYTE_ORDER_MARK: u32 = 0x0102_0304;

const HEADER_BYTES: usize = 8 + 4 + 4 + 4 + 8;

const KIND_FLAT: u8 = 1;
const KIND_HNSW: u8 = 2;
const KIND_IVFPQ: u8 = 3;

/// What a snapshot file says about itself, without decoding the body.
///
/// Read by `proxima describe` and by the recovery path, which wants to know the
/// dimension and the index kind before committing to loading a multi-gigabyte
/// body.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotHeader {
    /// Format version the file was written with.
    pub format_version: u32,
    /// Body length in bytes, checksum excluded.
    pub body_bytes: u64,
    /// The space the index is over.
    pub space: VectorSpace,
    /// `"flat"`, `"hnsw"` or `"ivfpq"`.
    pub kind: &'static str,
    /// Live vectors.
    pub len: usize,
}

/// Serialise `index` into a byte vector.
///
/// Separated from the file writing so that the format can be tested without
/// touching a disk, and so that a caller who wants to put a snapshot somewhere
/// other than a file — an object store, a test fixture — can.
pub fn encode(index: &AnyIndex) -> Vec<u8> {
    let mut body = Writer::with_capacity(1 << 16);
    body.space(index.space());
    match index {
        AnyIndex::Flat(flat) => {
            body.u8(KIND_FLAT);
            encode_store(&mut body, flat.store());
        }
        AnyIndex::Hnsw(hnsw) => {
            body.u8(KIND_HNSW);
            encode_hnsw_config(&mut body, hnsw.config());
            encode_store(&mut body, hnsw.store());
            let graph = hnsw.graph();
            body.u8_slice(graph.levels);
            body.u32_slice(graph.level0);
            body.len_prefix(graph.upper.len());
            for links in graph.upper {
                body.u32_slice(links);
            }
            match graph.entry {
                Some(entry) => {
                    body.u8(1).u32(entry.get());
                }
                None => {
                    body.u8(0).u32(0);
                }
            }
            body.u32(graph.entry_level as u32);
            body.u64(graph.levels_drawn);
        }
        AnyIndex::IvfPq(ivf) => {
            body.u8(KIND_IVFPQ);
            encode_ivfpq_config(&mut body, ivf.config());
            let parts = ivf.parts();
            body.len_prefix(parts.ids.len());
            for id in parts.ids {
                match id {
                    Some(id) => {
                        body.u8(1).u64(id.get());
                    }
                    None => {
                        body.u8(0).u64(0);
                    }
                }
            }
            body.len_prefix(parts.metadata.len());
            for metadata in parts.metadata {
                body.metadata(metadata);
            }
            body.f32_slice(parts.vectors);
            body.u8_slice(parts.codes);
            body.u32_slice(parts.list_of);
            body.f32_slice(parts.centroids);
            body.f32_slice(parts.codebooks);
            match parts.report {
                Some(report) => {
                    body.u8(1)
                        .len_prefix(report.sampled)
                        .f64(report.coarse_inertia)
                        .len_prefix(report.coarse_iterations)
                        .f64(report.residual_inertia)
                        .len_prefix(report.populated_lists)
                        .len_prefix(report.largest_list);
                }
                None => {
                    body.u8(0);
                }
            }
        }
    }
    let body = body.into_bytes();

    let mut file = Writer::with_capacity(HEADER_BYTES + body.len() + 4);
    file.raw(MAGIC);
    file.u32(FORMAT_VERSION);
    file.u32(BYTE_ORDER_MARK);
    file.u32(0);
    file.u64(body.len() as u64);
    file.raw(&body);
    file.u32(crc32c(&body));
    file.into_bytes()
}

/// The inverse of [`encode`].
pub fn decode(bytes: &[u8]) -> Result<AnyIndex> {
    let body = verify(bytes)?;
    let mut reader = Reader::new(body, "snapshot body");
    let space = reader.space()?;
    let kind = reader.u8()?;
    let index = match kind {
        KIND_FLAT => {
            let store = decode_store(&mut reader, space, UnmappedSlots::Free)?;
            AnyIndex::Flat(FlatIndex::from_store(store))
        }
        KIND_HNSW => {
            let config = decode_hnsw_config(&mut reader)?;
            // Tombstoned, not free: a graph still has edges pointing at every
            // unmapped slot, and handing one back to the allocator would let the
            // next insert inherit them.
            let store = decode_store(&mut reader, space, UnmappedSlots::Tombstoned)?;
            let levels = reader.u8_vec()?;
            let level0 = reader.u32_vec()?;
            let upper_len = reader.len_prefix(4)?;
            let mut upper = Vec::with_capacity(upper_len);
            for _ in 0..upper_len {
                upper.push(reader.u32_vec()?);
            }
            let has_entry = reader.u8()? != 0;
            let entry_slot = reader.u32()?;
            let entry = has_entry.then(|| proxima_core::Slot::new(entry_slot));
            let entry_level = reader.u32()? as usize;
            let levels_drawn = reader.u64()?;
            AnyIndex::Hnsw(HnswIndex::from_parts(
                store,
                config,
                HnswGraphParts {
                    levels,
                    level0,
                    upper,
                    entry,
                    entry_level,
                    levels_drawn,
                },
            )?)
        }
        KIND_IVFPQ => {
            let config = decode_ivfpq_config(&mut reader)?;
            let id_count = reader.len_prefix(9)?;
            let mut ids = Vec::with_capacity(id_count);
            for _ in 0..id_count {
                let present = reader.u8()? != 0;
                let raw = reader.u64()?;
                ids.push(present.then(|| VectorId::new(raw)));
            }
            let metadata_count = reader.len_prefix(4)?;
            let mut metadata = Vec::with_capacity(metadata_count);
            for _ in 0..metadata_count {
                metadata.push(reader.metadata()?);
            }
            let parts = IvfPqPartsOwned {
                ids,
                metadata,
                vectors: reader.f32_vec()?,
                codes: reader.u8_vec()?,
                list_of: reader.u32_vec()?,
                centroids: reader.f32_vec()?,
                codebooks: reader.f32_vec()?,
                report: decode_training_report(&mut reader)?,
            };
            AnyIndex::IvfPq(IvfPqIndex::from_parts(space, config, parts)?)
        }
        other => {
            return Err(ProximaError::corrupt(
                "snapshot body",
                format!("index kind {other} is not one of 1..=3"),
            ))
        }
    };
    reader.expect_end()?;
    Ok(index)
}

/// Read the header without decoding the body.
pub fn peek(bytes: &[u8]) -> Result<SnapshotHeader> {
    let body = verify(bytes)?;
    let mut reader = Reader::new(body, "snapshot header");
    let space = reader.space()?;
    let kind = match reader.u8()? {
        KIND_FLAT => "flat",
        KIND_HNSW => "hnsw",
        KIND_IVFPQ => "ivfpq",
        other => {
            return Err(ProximaError::corrupt(
                "snapshot header",
                format!("index kind {other} is not one of 1..=3"),
            ))
        }
    };
    // `len` is not in the header, so it costs a full decode. That is a
    // deliberate omission: a length in the header is a second copy of a fact
    // the body already contains, and the two can disagree.
    let len = decode(bytes)?.len();
    Ok(SnapshotHeader {
        format_version: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        body_bytes: body.len() as u64,
        space,
        kind,
        len,
    })
}

/// Check the magic, version, byte order and checksum, and return the body.
fn verify(bytes: &[u8]) -> Result<&[u8]> {
    if bytes.len() < HEADER_BYTES + 4 {
        return Err(ProximaError::corrupt(
            "snapshot header",
            format!(
                "{} bytes is shorter than the {HEADER_BYTES}-byte header plus checksum",
                bytes.len()
            ),
        ));
    }
    if &bytes[..8] != MAGIC {
        return Err(ProximaError::corrupt(
            "snapshot header",
            format!(
                "expected magic {:?}, found {:?}",
                String::from_utf8_lossy(MAGIC),
                String::from_utf8_lossy(&bytes[..8])
            ),
        ));
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != FORMAT_VERSION {
        return Err(ProximaError::corrupt(
            "snapshot header",
            format!("format version {version}, this build reads {FORMAT_VERSION}"),
        ));
    }
    let reserved = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    if reserved != 0 {
        return Err(ProximaError::corrupt(
            "snapshot header",
            format!("reserved word is {reserved:#010x}, not zero"),
        ));
    }
    let order = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    if order != BYTE_ORDER_MARK {
        return Err(ProximaError::corrupt(
            "snapshot header",
            format!("byte-order mark {order:#010x} means the file was written on a machine of the opposite endianness"),
        ));
    }
    let body_len = u64::from_le_bytes([
        bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27],
    ]) as usize;
    if bytes.len() != HEADER_BYTES + body_len + 4 {
        return Err(ProximaError::corrupt(
            "snapshot",
            format!(
                "header claims a {body_len}-byte body, file holds {}",
                bytes.len().saturating_sub(HEADER_BYTES + 4)
            ),
        ));
    }
    let body = &bytes[HEADER_BYTES..HEADER_BYTES + body_len];
    let expected = u32::from_le_bytes([
        bytes[HEADER_BYTES + body_len],
        bytes[HEADER_BYTES + body_len + 1],
        bytes[HEADER_BYTES + body_len + 2],
        bytes[HEADER_BYTES + body_len + 3],
    ]);
    let actual = crc32c(body);
    if actual != expected {
        return Err(ProximaError::corrupt(
            "snapshot",
            format!("checksum {actual:#010x} does not match the recorded {expected:#010x}"),
        ));
    }
    Ok(body)
}

fn encode_store(body: &mut Writer, store: &VectorStore) {
    let ids = store.raw_ids();
    body.len_prefix(ids.len());
    for id in ids {
        match id {
            Some(id) => {
                body.u8(1).u64(id.get());
            }
            None => {
                body.u8(0).u64(0);
            }
        }
    }
    body.f32_slice(store.raw_data());
    let metadata = store.raw_metadata();
    body.len_prefix(metadata.len());
    for entry in metadata {
        body.metadata(entry);
    }
}

fn decode_store(
    reader: &mut Reader<'_>,
    space: VectorSpace,
    unmapped: UnmappedSlots,
) -> Result<VectorStore> {
    // Nine bytes per id entry: a presence byte and a u64.
    let count = reader.len_prefix(9)?;
    let mut ids: Vec<Option<VectorId>> = Vec::with_capacity(count);
    for _ in 0..count {
        let present = reader.u8()? != 0;
        let raw = reader.u64()?;
        ids.push(present.then(|| VectorId::new(raw)));
    }
    let data = reader.f32_vec()?;
    let metadata_count = reader.len_prefix(4)?;
    let mut metadata = Vec::with_capacity(metadata_count);
    for _ in 0..metadata_count {
        metadata.push(reader.metadata()?);
    }
    VectorStore::from_parts(space, data, metadata, ids, unmapped)
}

fn encode_hnsw_config(body: &mut Writer, config: &HnswConfig) {
    body.len_prefix(config.m())
        .len_prefix(config.ef_construction())
        .len_prefix(config.ef_search())
        .len_prefix(config.max_level())
        .u64(config.seed())
        .len_prefix(config.filter_ef_limit())
        .u8(u8::from(config.exhaustive_filter_fallback()))
        .u8(u8::from(config.keep_pruned_connections()));
}

fn decode_hnsw_config(reader: &mut Reader<'_>) -> Result<HnswConfig> {
    let m = reader.u32()? as usize;
    let ef_construction = reader.u32()? as usize;
    let ef_search = reader.u32()? as usize;
    let max_level = reader.u32()? as usize;
    let seed = reader.u64()?;
    let filter_ef_limit = reader.u32()? as usize;
    let fallback = reader.u8()? != 0;
    let keep_pruned = reader.u8()? != 0;
    let config = HnswConfig::new()
        .with_m(m)
        .with_ef_construction(ef_construction)
        .with_ef_search(ef_search)
        .with_seed(seed)
        .with_filter_ef_limit(filter_ef_limit)
        .with_exhaustive_filter_fallback(fallback)
        .with_keep_pruned_connections(keep_pruned);
    // `max_level` has no builder, because nothing but a snapshot has a reason
    // to set it; restoring one that differs from the default would otherwise
    // silently change the level cap.
    if max_level != config.max_level() {
        return Err(ProximaError::corrupt(
            "snapshot hnsw config",
            format!(
                "max_level {max_level} differs from this build's {}",
                config.max_level()
            ),
        ));
    }
    config.validate()?;
    Ok(config)
}

fn encode_ivfpq_config(body: &mut Writer, config: &IvfPqConfig) {
    body.len_prefix(config.nlist())
        .len_prefix(config.subspaces())
        .len_prefix(config.nprobe())
        .len_prefix(config.rerank())
        .u64(config.seed())
        .len_prefix(config.max_iterations())
        .len_prefix(config.training_sample())
        .u64(config.train_threshold() as u64)
        .u8(u8::from(config.retain_vectors()));
}

fn decode_ivfpq_config(reader: &mut Reader<'_>) -> Result<IvfPqConfig> {
    let nlist = reader.u32()? as usize;
    let subspaces = reader.u32()? as usize;
    let nprobe = reader.u32()? as usize;
    let rerank = reader.u32()? as usize;
    let seed = reader.u64()?;
    let max_iterations = reader.u32()? as usize;
    let training_sample = reader.u32()? as usize;
    let train_threshold = reader.u64()? as usize;
    let retain_vectors = reader.u8()? != 0;
    Ok(IvfPqConfig::new(nlist, subspaces)
        .with_nprobe(nprobe)
        .with_rerank(rerank)
        .with_seed(seed)
        .with_max_iterations(max_iterations)
        .with_training_sample(training_sample)
        .with_train_threshold(train_threshold)
        .with_retain_vectors(retain_vectors))
}

fn decode_training_report(reader: &mut Reader<'_>) -> Result<Option<TrainingReport>> {
    if reader.u8()? == 0 {
        return Ok(None);
    }
    Ok(Some(TrainingReport {
        sampled: reader.u32()? as usize,
        coarse_inertia: reader.f64()?,
        coarse_iterations: reader.u32()? as usize,
        residual_inertia: reader.f64()?,
        populated_lists: reader.u32()? as usize,
        largest_list: reader.u32()? as usize,
    }))
}

/// Write a snapshot to `path`, atomically.
///
/// See the module docs for why this is four syscalls rather than one `write`.
pub fn write_snapshot(path: impl AsRef<Path>, index: &AnyIndex) -> Result<u64> {
    let path = path.as_ref();
    let bytes = encode(index);
    let temporary = path.with_extension("tmp");

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&temporary)
        .map_err(|err| ProximaError::io("create snapshot", temporary.display(), &err))?;
    file.write_all(&bytes)
        .map_err(|err| ProximaError::io("write snapshot", temporary.display(), &err))?;
    file.sync_all()
        .map_err(|err| ProximaError::io("sync snapshot", temporary.display(), &err))?;
    drop(file);

    std::fs::rename(&temporary, path)
        .map_err(|err| ProximaError::io("rename snapshot", path.display(), &err))?;

    // The rename is a directory update, and a directory update is not durable
    // until the directory itself is synced. Skipping this is the classic
    // "the file is there but empty after a power cut" bug.
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            dir.sync_all().map_err(|err| {
                ProximaError::io("sync snapshot directory", parent.display(), &err)
            })?;
        }
    }
    Ok(bytes.len() as u64)
}

/// Read a snapshot from `path`.
pub fn read_snapshot(path: impl AsRef<Path>) -> Result<AnyIndex> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|err| ProximaError::io("read snapshot", path.display(), &err))?;
    decode(&bytes)
}

/// Read only the header of the snapshot at `path`.
pub fn read_snapshot_header(path: impl AsRef<Path>) -> Result<SnapshotHeader> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|err| ProximaError::io("read snapshot", path.display(), &err))?;
    peek(&bytes)
}

/// Every metadata key present in the index, sorted and deduplicated.
///
/// `Collection::describe` reports the key set so that a caller can tell what a
/// filter may name. `O(n * fields)` and it clones nothing but the keys, which
/// is the reason it is a function here rather than a `HashSet` maintained on
/// the write path — the write path is the one that has to be fast.
pub(crate) fn metadata_keys(index: &AnyIndex) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    let ids: Vec<VectorId> = match index {
        AnyIndex::Flat(flat) => flat.ids().collect(),
        AnyIndex::Hnsw(hnsw) => hnsw.ids().collect(),
        AnyIndex::IvfPq(ivf) => ivf.ids().collect(),
    };
    for id in ids {
        if let Some(metadata) = index.metadata(id) {
            for (key, _) in metadata.iter() {
                if !keys.iter().any(|existing| existing == key) {
                    keys.push(key.to_owned());
                }
            }
        }
    }
    keys.sort();
    keys
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use proxima_core::{Metadata, Metric, SearchResult};
    use proxima_index::{HnswConfig, IvfPqConfig};

    use super::*;
    use crate::index::IndexSpec;

    fn fixture(spec: &IndexSpec, dim: usize, count: u64) -> AnyIndex {
        let space = VectorSpace::new(dim, Metric::Euclidean).unwrap();
        let mut index = spec.build(space).unwrap();
        for i in 0..count {
            let cluster = (i % 17) as f32;
            let vector: Vec<f32> = (0..dim)
                .map(|d| (cluster * 1.9 + d as f32 * 0.17).sin() * 3.0 + i as f32 * 0.001)
                .collect();
            index
                .insert(
                    VectorId::new(i),
                    &vector,
                    Metadata::new().with("group", (i % 5) as i64).with("k", "v"),
                )
                .unwrap();
        }
        index
    }

    fn queries(dim: usize, count: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|q| {
                (0..dim)
                    .map(|d| ((q * 7 + d) as f32 * 0.31).sin() * 3.0)
                    .collect()
            })
            .collect()
    }

    fn results(index: &AnyIndex, dim: usize) -> Vec<Vec<SearchResult>> {
        queries(dim, 50)
            .iter()
            .map(|query| index.search_all(query, 10).unwrap())
            .collect()
    }

    #[test]
    fn a_flat_index_round_trips_bit_for_bit() {
        let index = fixture(&IndexSpec::Flat, 12, 400);
        let restored = decode(&encode(&index)).unwrap();
        assert_eq!(results(&index, 12), results(&restored, 12));
        assert_eq!(restored.len(), index.len());
    }

    #[test]
    fn a_graph_with_tombstones_round_trips_bit_for_bit() {
        // The case that a vectors-only snapshot would get wrong: after a delete
        // the graph cannot be rebuilt from its contents.
        let mut index = fixture(&IndexSpec::Hnsw(HnswConfig::new().with_seed(5)), 12, 600);
        for id in (0..600).step_by(7) {
            index.remove(VectorId::new(id)).unwrap();
        }
        index
            .upsert(VectorId::new(3), &[0.25; 12], Metadata::new())
            .unwrap();

        let restored = decode(&encode(&index)).unwrap();
        assert_eq!(results(&index, 12), results(&restored, 12));
        // Structure, not footprint: `vector_bytes` is derived from `Vec`
        // *capacity*, and a restored index allocates exactly what it needs
        // while a grown one carries the allocator's slack. The reload is
        // indistinguishable in every way that affects an answer, and smaller.
        let (before, after) = (index.stats(), restored.stats());
        assert_eq!(
            (
                before.len,
                before.deleted,
                before.edges,
                before.nodes_per_layer
            ),
            (after.len, after.deleted, after.edges, after.nodes_per_layer)
        );
        assert!(after.vector_bytes <= before.vector_bytes);

        // A rebuild from the same vectors in the same order is *not* the same
        // index, which is why the graph is on disk.
        let mut rebuilt = IndexSpec::Hnsw(HnswConfig::new().with_seed(5))
            .build(VectorSpace::new(12, Metric::Euclidean).unwrap())
            .unwrap();
        match &index {
            AnyIndex::Hnsw(hnsw) => {
                for id in hnsw.ids().collect::<Vec<_>>() {
                    let vector = hnsw.vector(id).unwrap().to_vec();
                    rebuilt
                        .insert(id, &vector, hnsw.metadata(id).cloned().unwrap_or_default())
                        .unwrap();
                }
            }
            _ => unreachable!(),
        }
        assert_ne!(
            rebuilt.stats().edges,
            index.stats().edges,
            "if a rebuild happened to match, this test proves nothing"
        );
    }

    #[test]
    fn an_ivfpq_index_round_trips_bit_for_bit() {
        let spec = IndexSpec::IvfPq(
            IvfPqConfig::new(8, 6)
                .with_seed(3)
                .with_nprobe(4)
                .with_train_threshold(200),
        );
        let index = fixture(&spec, 12, 500);
        match &index {
            AnyIndex::IvfPq(ivf) => assert!(ivf.is_trained()),
            _ => unreachable!(),
        }
        let restored = decode(&encode(&index)).unwrap();
        assert_eq!(results(&index, 12), results(&restored, 12));
        match (&index, &restored) {
            (AnyIndex::IvfPq(a), AnyIndex::IvfPq(b)) => {
                assert_eq!(a.list_lengths(), b.list_lengths());
                assert_eq!(a.training_report(), b.training_report());
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn an_untrained_ivfpq_index_round_trips() {
        let spec = IndexSpec::IvfPq(IvfPqConfig::new(8, 6).with_train_threshold(usize::MAX));
        let index = fixture(&spec, 12, 60);
        let restored = decode(&encode(&index)).unwrap();
        assert_eq!(results(&index, 12), results(&restored, 12));
    }

    #[test]
    fn an_empty_index_round_trips() {
        for spec in [
            IndexSpec::Flat,
            IndexSpec::Hnsw(HnswConfig::new()),
            IndexSpec::IvfPq(IvfPqConfig::new(4, 4)),
        ] {
            let index = fixture(&spec, 8, 0);
            let restored = decode(&encode(&index)).unwrap();
            assert!(restored.is_empty());
            assert_eq!(restored.search_all(&[0.0; 8], 5).unwrap(), Vec::new());
        }
    }

    #[test]
    fn a_flipped_bit_anywhere_in_the_body_is_caught() {
        let index = fixture(&IndexSpec::Flat, 8, 40);
        let bytes = encode(&index);
        for offset in (HEADER_BYTES..bytes.len() - 4).step_by(37) {
            let mut corrupt = bytes.clone();
            corrupt[offset] ^= 0x40;
            assert!(decode(&corrupt).is_err(), "offset {offset}");
        }
    }

    #[test]
    fn truncation_is_caught_by_the_length_check() {
        let index = fixture(&IndexSpec::Flat, 8, 40);
        let bytes = encode(&index);
        for keep in (0..bytes.len()).step_by(11) {
            assert!(decode(&bytes[..keep]).is_err(), "kept {keep}");
        }
    }

    #[test]
    fn a_wrong_magic_version_or_byte_order_is_named_in_the_error() {
        let index = fixture(&IndexSpec::Flat, 8, 4);
        let bytes = encode(&index);

        let mut wrong_magic = bytes.clone();
        wrong_magic[0] = b'X';
        assert!(decode(&wrong_magic)
            .unwrap_err()
            .to_string()
            .contains("magic"));

        let mut wrong_version = bytes.clone();
        wrong_version[8] = 99;
        assert!(decode(&wrong_version)
            .unwrap_err()
            .to_string()
            .contains("version"));

        // Byte-swap the marker, which is exactly what a big-endian writer
        // would have produced.
        let mut wrong_order = bytes;
        wrong_order.swap(12, 15);
        wrong_order.swap(13, 14);
        assert!(decode(&wrong_order)
            .unwrap_err()
            .to_string()
            .contains("endianness"));
    }

    #[test]
    fn the_header_can_be_read_without_trusting_the_body() {
        let index = fixture(&IndexSpec::Hnsw(HnswConfig::new()), 16, 100);
        let header = peek(&encode(&index)).unwrap();
        assert_eq!(header.format_version, FORMAT_VERSION);
        assert_eq!(header.kind, "hnsw");
        assert_eq!(header.len, 100);
        assert_eq!(header.space.dim(), 16);
    }

    #[test]
    fn encoding_is_deterministic() {
        // Two encodes of one index must be byte-identical, or the format
        // cannot be diffed, deduplicated or golden-tested. The metadata key
        // sort in the codec is what makes this true.
        let index = fixture(&IndexSpec::Flat, 8, 200);
        let first = encode(&index);
        for _ in 0..8 {
            assert_eq!(encode(&index), first);
        }
    }

    #[test]
    fn files_are_written_atomically_and_read_back() {
        let mut dir = std::env::temp_dir();
        dir.push(format!("proxima-snap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.snap");

        let index = fixture(&IndexSpec::Hnsw(HnswConfig::new().with_seed(1)), 10, 300);
        let written = write_snapshot(&path, &index).unwrap();
        assert_eq!(written, std::fs::metadata(&path).unwrap().len());
        // The temporary file is gone, not left behind.
        assert!(!path.with_extension("tmp").exists());

        let restored = read_snapshot(&path).unwrap();
        assert_eq!(results(&index, 10), results(&restored, 10));
        assert_eq!(read_snapshot_header(&path).unwrap().kind, "hnsw");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn metadata_keys_are_sorted_and_deduplicated() {
        let index = fixture(&IndexSpec::Flat, 8, 20);
        assert_eq!(
            metadata_keys(&index),
            vec!["group".to_owned(), "k".to_owned()]
        );
    }
}
