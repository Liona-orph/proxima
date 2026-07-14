//! [`VectorStore`]: the flat array every index is built on.
//!
//! # Why one `Vec<f32>` and not a `Vec<Vec<f32>>`
//!
//! The obvious representation of "n vectors of dimension d" is `Vec<Vec<f32>>`.
//! It is also the single worst decision available, and it is worth being
//! explicit about why, because the cost is invisible in a microbenchmark that
//! iterates them in order.
//!
//! A graph search does not iterate in order. It hops: examine a node, read its
//! neighbour list, evaluate the distance to each neighbour. Each of those
//! neighbours is at an unpredictable slot. With `Vec<Vec<f32>>` every one of
//! those evaluations is **two** dependent cache misses — load the `Vec` header
//! (24 bytes: pointer, capacity, length) to learn the address, then load the
//! data at that address — and the second miss cannot even be issued until the
//! first returns, so the ~100 ns latencies serialise instead of overlapping.
//! For a 768-dimensional vector the payload itself is 3 KiB, i.e. 48 cache
//! lines, but the prefetcher cannot start on it until the header arrives.
//!
//! With a flat array, the address is `slot * dim` — arithmetic, not a load. The
//! pointer chase is gone, the hardware prefetcher sees a contiguous 3 KiB run
//! the moment the first line is touched, and the per-vector allocation
//! overhead (a `Vec` header plus the allocator's own bookkeeping, easily 40
//! bytes on a 12-byte vector) disappears. At 3-dimensional vectors the
//! `Vec<Vec<f32>>` form spends more memory on headers than on data.
//!
//! The price is that the store is dimension-homogeneous and that growing it
//! reallocates and copies everything. Both are fine: a vector index has a fixed
//! dimension by construction, and the copy amortises to O(1) per insert.

use std::collections::HashMap;

use crate::error::{ProximaError, Result};
use crate::id::{Slot, SlotAllocator, VectorId};
use crate::metadata::Metadata;
use crate::space::VectorSpace;

/// What an unmapped slot means to [`VectorStore::from_parts`].
///
/// A snapshot records which slots hold an id and which do not. It cannot record
/// *why* a slot holds no id, because the store itself does not distinguish the
/// two cases in its arrays — the difference lives in the allocator's free list.
/// The index being restored does know, and this is how it says so.
///
/// Choosing wrongly is not caught by any check: restoring a graph index with
/// [`UnmappedSlots::Free`] lets the next insert land on a tombstoned slot and
/// silently inherit the edges that still point at it, which is precisely the
/// failure the `detach`/`release` split exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnmappedSlots {
    /// Reusable. What [`FlatIndex`](https://docs.rs/proxima-index) means: it
    /// releases a slot the moment the vector is deleted.
    Free,
    /// Tombstones: still traversed through, never handed out again until a
    /// compaction rebuilds the structure. What a graph index means.
    Tombstoned,
}

/// Slot-addressed storage for vectors and their metadata.
///
/// Owns the mapping between the caller's [`VectorId`]s and internal [`Slot`]s,
/// so that `FlatIndex` and `HnswIndex` (in `proxima-index`) do not each
/// reimplement it and disagree in the corners.
///
/// # Deletion: `remove` versus `detach` + `release`
///
/// A flat index can free a slot the instant a vector is deleted. A graph index
/// cannot: other nodes still hold edges pointing at that slot, and handing it
/// straight back to the allocator would let a later insert land there and
/// silently inherit those edges. So deletion is split:
///
/// * [`VectorStore::detach`] unmaps the id and returns the slot, leaving the
///   data in place and the slot **not** on the free list. This is a tombstone.
/// * [`VectorStore::release`] puts a detached slot back on the free list. A
///   graph index calls it only from compaction, once the edges are gone.
/// * [`VectorStore::remove`] is `detach` followed by `release`, which is what a
///   flat index wants.
///
/// ```
/// use proxima_core::{Metadata, Metric, VectorId, VectorSpace, VectorStore};
///
/// let space = VectorSpace::new(2, Metric::Euclidean)?;
/// let mut store = VectorStore::new(space);
///
/// let slot = store.insert(VectorId::new(1), &[1.0, 0.0], Metadata::new())?;
/// assert_eq!(store.vector(slot), &[1.0, 0.0]);
/// assert_eq!(store.len(), 1);
///
/// // Detached: the id is gone, the slot is still reserved.
/// store.detach(VectorId::new(1))?;
/// assert_eq!(store.len(), 0);
/// assert_eq!(store.slot_count(), 1);
///
/// // Released: the next insert reuses it.
/// store.release(slot);
/// assert_eq!(store.insert(VectorId::new(2), &[0.0, 1.0], Metadata::new())?, slot);
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
#[derive(Debug, Clone)]
pub struct VectorStore {
    space: VectorSpace,
    /// `slot_count * dim` floats. Slot `s` occupies `s*dim .. (s+1)*dim`.
    data: Vec<f32>,
    /// Parallel to `data`, one entry per slot.
    metadata: Vec<Metadata>,
    /// Reverse mapping, `None` for a slot that is free or detached.
    ids: Vec<Option<VectorId>>,
    by_id: HashMap<VectorId, Slot>,
    slots: SlotAllocator,
    /// Returned by [`VectorStore::metadata`] for an out-of-range slot. A field
    /// rather than a `const` because `HashMap::new` is not a `const fn`, and a
    /// `&'static` fallback is the only way to keep that accessor branch-cheap
    /// and non-allocating.
    absent: Metadata,
}

impl VectorStore {
    /// An empty store over `space`.
    pub fn new(space: VectorSpace) -> Self {
        VectorStore {
            space,
            data: Vec::new(),
            metadata: Vec::new(),
            ids: Vec::new(),
            by_id: HashMap::new(),
            slots: SlotAllocator::new(),
            absent: Metadata::new(),
        }
    }

    /// An empty store with room for `capacity` vectors already reserved.
    ///
    /// Worth using whenever the corpus size is known: the flat array is one
    /// allocation, so growing it copies every float stored so far.
    pub fn with_capacity(space: VectorSpace, capacity: usize) -> Self {
        let dim = space.dim();
        VectorStore {
            space,
            data: Vec::with_capacity(capacity * dim),
            metadata: Vec::with_capacity(capacity),
            ids: Vec::with_capacity(capacity),
            by_id: HashMap::with_capacity(capacity),
            slots: SlotAllocator::with_capacity(0),
            absent: Metadata::new(),
        }
    }

    /// The space this store is over.
    pub fn space(&self) -> &VectorSpace {
        &self.space
    }

    /// The dimension of every vector here.
    pub fn dim(&self) -> usize {
        self.space.dim()
    }

    /// Number of live (mapped) vectors.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// `true` if nothing is mapped.
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Number of slots the backing arrays cover, live or not.
    ///
    /// This, not [`VectorStore::len`], bounds every slot-indexed array a caller
    /// keeps alongside the store.
    pub fn slot_count(&self) -> usize {
        self.slots.high_water()
    }

    /// Slots that are detached-and-released and awaiting reuse.
    pub fn free_slots(&self) -> usize {
        self.slots.free_count()
    }

    /// Store a vector under `id`.
    ///
    /// The vector must already be [prepared](VectorSpace::prepare); the store
    /// re-validates its length and finiteness but does not normalise, because
    /// doing so here would mean an index could not tell whether the copy it
    /// holds matches the one the caller passed.
    ///
    /// Returns [`ProximaError::DuplicateId`] rather than replacing. Amortised
    /// O(dim).
    pub fn insert(&mut self, id: VectorId, vector: &[f32], metadata: Metadata) -> Result<Slot> {
        self.space.validate(vector)?;
        if self.by_id.contains_key(&id) {
            return Err(ProximaError::DuplicateId { id });
        }
        let slot = self.slots.allocate()?;
        let index = slot.index();
        let dim = self.space.dim();

        if index == self.ids.len() {
            // A fresh slot at the high-water mark: extend all three arrays.
            self.data.extend_from_slice(vector);
            self.metadata.push(metadata);
            self.ids.push(Some(id));
        } else {
            // A reused slot: overwrite in place, no growth. This is the branch
            // that makes churn free.
            let start = index * dim;
            self.data[start..start + dim].copy_from_slice(vector);
            self.metadata[index] = metadata;
            self.ids[index] = Some(id);
        }
        self.by_id.insert(id, slot);
        Ok(slot)
    }

    /// Unmap `id`, keeping its slot reserved and its data intact.
    ///
    /// The tombstone half of deletion; see the type-level docs. Metadata is
    /// dropped immediately (it can be arbitrarily large and nothing reads it
    /// once the id is gone) but the vector data is not, because a graph index
    /// still traverses *through* tombstoned nodes and needs their coordinates
    /// to do so.
    pub fn detach(&mut self, id: VectorId) -> Result<Slot> {
        let slot = self
            .by_id
            .remove(&id)
            .ok_or(ProximaError::UnknownId { id })?;
        self.ids[slot.index()] = None;
        self.metadata[slot.index()] = Metadata::new();
        Ok(slot)
    }

    /// Return a detached slot to the free list.
    ///
    /// # Panics
    ///
    /// In debug builds, if the slot is still mapped to an id or is already
    /// free. Releasing a live slot would let two ids resolve to one vector.
    pub fn release(&mut self, slot: Slot) {
        debug_assert!(
            slot.index() < self.ids.len() && self.ids[slot.index()].is_none(),
            "released slot {slot} while it was still mapped"
        );
        self.slots.release(slot);
    }

    /// Unmap `id` and free its slot in one step.
    pub fn remove(&mut self, id: VectorId) -> Result<Slot> {
        let slot = self.detach(id)?;
        self.release(slot);
        Ok(slot)
    }

    /// The vector at `slot`.
    ///
    /// The hot accessor: one multiply and a slice, no branch. Used by the
    /// search loops, which have already obtained the slot from the graph and so
    /// know it is valid.
    ///
    /// # Panics
    ///
    /// If `slot` is beyond [`VectorStore::slot_count`]. That is an internal
    /// invariant violation — a graph edge pointing outside the store — and
    /// panicking is the right response to it; the recoverable form is
    /// [`VectorStore::get_slot`].
    #[inline]
    pub fn vector(&self, slot: Slot) -> &[f32] {
        let dim = self.space.dim();
        let start = slot.index() * dim;
        &self.data[start..start + dim]
    }

    /// The vector at `slot`, or `None` if the slot is out of range.
    ///
    /// Note that a *detached* slot still yields its data: tombstoned vectors
    /// remain readable until compaction, by design.
    pub fn get_slot(&self, slot: Slot) -> Option<&[f32]> {
        if slot.index() >= self.slot_count() {
            return None;
        }
        Some(self.vector(slot))
    }

    /// The vector stored under `id`.
    pub fn get(&self, id: VectorId) -> Option<&[f32]> {
        self.by_id.get(&id).map(|slot| self.vector(*slot))
    }

    /// The slot `id` occupies.
    pub fn slot_of(&self, id: VectorId) -> Option<Slot> {
        self.by_id.get(&id).copied()
    }

    /// The id occupying `slot`, or `None` if it is free or detached.
    #[inline]
    pub fn id_of(&self, slot: Slot) -> Option<VectorId> {
        self.ids.get(slot.index()).copied().flatten()
    }

    /// `true` if `id` is mapped.
    pub fn contains(&self, id: VectorId) -> bool {
        self.by_id.contains_key(&id)
    }

    /// Metadata for `slot`.
    ///
    /// Returns empty metadata for a free or detached slot rather than `None`,
    /// so that a filter running during traversal does not need a second branch;
    /// liveness is [`VectorStore::id_of`]'s job.
    #[inline]
    pub fn metadata(&self, slot: Slot) -> &Metadata {
        self.metadata.get(slot.index()).unwrap_or(&self.absent)
    }

    /// Metadata for `id`.
    pub fn metadata_of(&self, id: VectorId) -> Option<&Metadata> {
        self.by_id.get(&id).map(|slot| self.metadata(*slot))
    }

    /// Replace the metadata for `id`, leaving the vector alone.
    pub fn set_metadata(&mut self, id: VectorId, metadata: Metadata) -> Result<()> {
        let slot = *self.by_id.get(&id).ok_or(ProximaError::UnknownId { id })?;
        self.metadata[slot.index()] = metadata;
        Ok(())
    }

    /// Iterate over live `(id, slot)` pairs in **slot order**.
    ///
    /// Slot order, not hash order: every result Proxima produces has to be
    /// reproducible, and iterating the `HashMap` would make a tie between two
    /// equidistant vectors resolve differently between runs.
    pub fn iter(&self) -> impl Iterator<Item = (VectorId, Slot)> + '_ {
        self.ids
            .iter()
            .enumerate()
            .filter_map(|(index, id)| id.map(|id| (id, Slot::new(index as u32))))
    }

    /// Iterate over live slots only.
    pub fn slots(&self) -> impl Iterator<Item = Slot> + '_ {
        self.iter().map(|(_, slot)| slot)
    }

    /// Drop everything, including the high-water mark.
    pub fn clear(&mut self) {
        self.data.clear();
        self.metadata.clear();
        self.ids.clear();
        self.by_id.clear();
        self.slots.clear();
    }

    /// Bytes held by the vector array, capacity included.
    ///
    /// The number that dominates an index's footprint: at 768 dimensions this
    /// is 3 KiB per vector against roughly 128 bytes of graph.
    pub fn vector_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<f32>()
    }

    /// Approximate total heap footprint: vectors, metadata and both id maps.
    pub fn heap_bytes(&self) -> usize {
        let metadata: usize = self.metadata.iter().map(Metadata::heap_bytes).sum();
        let id_maps = self.ids.capacity() * std::mem::size_of::<Option<VectorId>>()
            + self.by_id.capacity()
                * (std::mem::size_of::<VectorId>() + std::mem::size_of::<Slot>() + 1);
        self.vector_bytes() + metadata + id_maps
    }

    // -----------------------------------------------------------------
    // Serialisation surface
    // -----------------------------------------------------------------
    //
    // Persistence lives in `proxima-store`, which is a crate above this one, so
    // it needs a way to read the three parallel arrays out and put them back
    // *at the same slot numbers*. Rebuilding by replaying `insert` would not
    // do: a store with tombstoned slots would come back compacted, every slot
    // number would shift, and a graph index's adjacency lists would then point
    // at the wrong vectors. These four methods are the whole of the contract,
    // and they are deliberately narrow — no mutable access to the arrays.

    /// The raw vector array: `slot_count * dim` floats in slot order.
    pub fn raw_data(&self) -> &[f32] {
        &self.data
    }

    /// Metadata in slot order, including entries for detached slots.
    pub fn raw_metadata(&self) -> &[Metadata] {
        &self.metadata
    }

    /// The slot-to-id map, `None` for a slot that is free or detached.
    pub fn raw_ids(&self) -> &[Option<VectorId>] {
        &self.ids
    }

    /// Rebuild a store from the arrays [`VectorStore::raw_data`] and friends
    /// returned, preserving every slot number.
    ///
    /// `unmapped` says what the `None` entries in `ids` mean, and getting it
    /// wrong is a silent corruption rather than an error, which is why it is a
    /// required argument and not a default: see [`UnmappedSlots`].
    ///
    /// When they are treated as free, the list is rebuilt in descending slot
    /// order so the allocator hands the lowest one out first. That is *not*
    /// necessarily the order the original store would have used — the free list
    /// is a stack whose order depends on the sequence of deletes, which no
    /// snapshot records — so a reloaded store can place a subsequent insert in
    /// a different slot from the one the original would have chosen. Nothing
    /// about search depends on it; it is written down because "a reload is
    /// indistinguishable from the original" is otherwise a claim with an
    /// exception nobody stated.
    pub fn from_parts(
        space: VectorSpace,
        data: Vec<f32>,
        metadata: Vec<Metadata>,
        ids: Vec<Option<VectorId>>,
        unmapped: UnmappedSlots,
    ) -> Result<Self> {
        let dim = space.dim();
        let slot_count = ids.len();
        if data.len() != slot_count * dim {
            return Err(ProximaError::corrupt(
                "vector store",
                format!(
                    "{} floats for {slot_count} slots of dimension {dim}, expected {}",
                    data.len(),
                    slot_count * dim
                ),
            ));
        }
        if metadata.len() != slot_count {
            return Err(ProximaError::corrupt(
                "vector store",
                format!("{} metadata entries for {slot_count} slots", metadata.len()),
            ));
        }

        let mut slots = SlotAllocator::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.allocate()?;
        }
        let mut by_id = HashMap::with_capacity(slot_count);
        for (index, entry) in ids.iter().enumerate() {
            let slot = Slot::new(index as u32);
            if let Some(id) = entry {
                if by_id.insert(*id, slot).is_some() {
                    return Err(ProximaError::corrupt(
                        "vector store",
                        format!("id {id} appears at two slots"),
                    ));
                }
            }
        }
        if unmapped == UnmappedSlots::Free {
            for index in (0..slot_count).rev() {
                if ids[index].is_none() {
                    slots.release(Slot::new(index as u32));
                }
            }
        }

        Ok(VectorStore {
            space,
            data,
            metadata,
            ids,
            by_id,
            slots,
            absent: Metadata::new(),
        })
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
    use crate::distance::Metric;

    fn store(dim: usize) -> VectorStore {
        VectorStore::new(VectorSpace::new(dim, Metric::Euclidean).unwrap())
    }

    #[test]
    fn insert_get_remove_round_trip() {
        let mut s = store(3);
        let slot = s
            .insert(VectorId::new(10), &[1.0, 2.0, 3.0], Metadata::new())
            .unwrap();
        assert_eq!(s.get(VectorId::new(10)), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(s.id_of(slot), Some(VectorId::new(10)));
        assert_eq!(s.remove(VectorId::new(10)).unwrap(), slot);
        assert_eq!(s.get(VectorId::new(10)), None);
        assert_eq!(s.id_of(slot), None);
        assert!(matches!(
            s.remove(VectorId::new(10)),
            Err(ProximaError::UnknownId { .. })
        ));
    }

    #[test]
    fn duplicate_insert_is_rejected_not_merged() {
        let mut s = store(2);
        s.insert(VectorId::new(1), &[0.0, 0.0], Metadata::new())
            .unwrap();
        let err = s
            .insert(VectorId::new(1), &[9.0, 9.0], Metadata::new())
            .unwrap_err();
        assert!(matches!(err, ProximaError::DuplicateId { .. }));
        // The original survives untouched.
        assert_eq!(s.get(VectorId::new(1)), Some(&[0.0, 0.0][..]));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn dimension_and_finiteness_are_checked_on_insert() {
        let mut s = store(2);
        assert!(matches!(
            s.insert(VectorId::new(1), &[1.0], Metadata::new()),
            Err(ProximaError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            s.insert(VectorId::new(1), &[1.0, f32::NAN], Metadata::new()),
            Err(ProximaError::NonFinite { index: 1, .. })
        ));
        // A rejected insert must not have consumed a slot.
        assert_eq!(s.slot_count(), 0);
    }

    #[test]
    fn churn_reuses_slots_and_the_array_stops_growing() {
        let mut s = store(4);
        for i in 0..16u64 {
            s.insert(VectorId::new(i), &[i as f32; 4], Metadata::new())
                .unwrap();
        }
        let bytes_after_fill = s.vector_bytes();
        for round in 0..500u64 {
            let victim = VectorId::new(round % 16);
            s.remove(victim).unwrap();
            s.insert(victim, &[round as f32; 4], Metadata::new())
                .unwrap();
        }
        assert_eq!(s.len(), 16);
        assert_eq!(s.slot_count(), 16);
        assert_eq!(s.vector_bytes(), bytes_after_fill);
    }

    #[test]
    fn detached_slot_keeps_its_data_but_loses_its_metadata() {
        let mut s = store(2);
        let slot = s
            .insert(
                VectorId::new(3),
                &[5.0, 6.0],
                Metadata::new().with("k", "v"),
            )
            .unwrap();
        s.detach(VectorId::new(3)).unwrap();
        // Traversal through a tombstone still needs the coordinates.
        assert_eq!(s.vector(slot), &[5.0, 6.0]);
        assert!(s.metadata(slot).is_empty());
        assert_eq!(s.slot_count(), 1);
        assert_eq!(s.free_slots(), 0);
    }

    #[test]
    fn iteration_is_in_slot_order() {
        let mut s = store(1);
        for i in [900u64, 3, 41, 7] {
            s.insert(VectorId::new(i), &[i as f32], Metadata::new())
                .unwrap();
        }
        let ids: Vec<u64> = s.iter().map(|(id, _)| id.get()).collect();
        assert_eq!(ids, vec![900, 3, 41, 7]);
    }

    #[test]
    fn metadata_of_unmapped_slot_is_empty_not_absent() {
        let s = store(1);
        assert!(s.metadata(Slot::new(999)).is_empty());
    }

    #[test]
    fn set_metadata_leaves_the_vector_alone() {
        let mut s = store(2);
        s.insert(VectorId::new(1), &[1.0, 2.0], Metadata::new())
            .unwrap();
        s.set_metadata(VectorId::new(1), Metadata::new().with("a", 1_i64))
            .unwrap();
        assert_eq!(s.get(VectorId::new(1)), Some(&[1.0, 2.0][..]));
        assert_eq!(
            s.metadata_of(VectorId::new(1)).and_then(|m| m.int("a")),
            Some(1)
        );
    }
}
