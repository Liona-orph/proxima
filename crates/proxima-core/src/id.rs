//! Identifiers, and the slot allocator that keeps a churning index from growing
//! without bound.
//!
//! There are two numbering schemes here and keeping them apart is load-bearing:
//!
//! * [`VectorId`] is the caller's name for a vector. It is `u64`, it is never
//!   invented by Proxima, and it never changes.
//! * [`Slot`] is Proxima's *address* for a vector: the row it occupies in the
//!   flat backing array and the node number it has in the graph. It is `u32`,
//!   it is dense, and it is reused after deletion.
//!
//! The indirection exists because the graph wants small, dense, contiguous node
//! numbers (adjacency lists are `Vec<u32>`, the visited set is an array indexed
//! by slot) while callers want to bring their own sparse 64-bit keys.

use crate::error::{ProximaError, Result};

/// A caller-supplied identifier for a vector.
///
/// A newtype rather than a bare `u64` because the two numbering schemes in this
/// module have the same shape and swapping them is the kind of bug that
/// produces plausible-looking wrong answers rather than a crash.
///
/// ```
/// use proxima_core::VectorId;
///
/// let id = VectorId::new(4_815_162_342);
/// assert_eq!(id.get(), 4_815_162_342);
/// assert_eq!(id.to_string(), "4815162342");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VectorId(u64);

impl VectorId {
    /// Wrap a raw `u64`.
    pub const fn new(raw: u64) -> Self {
        VectorId(raw)
    }

    /// The underlying `u64`.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for VectorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for VectorId {
    fn from(raw: u64) -> Self {
        VectorId(raw)
    }
}

impl From<VectorId> for u64 {
    fn from(id: VectorId) -> Self {
        id.0
    }
}

/// A dense internal address for a stored vector.
///
/// Slots index the flat vector array and the graph adjacency lists directly, so
/// they must stay small and dense. `u32` caps an index at just under 4.3 billion
/// live vectors, which at 768 dimensions is 13 TiB of `f32` — the machine runs
/// out long before the address space does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slot(u32);

impl Slot {
    /// Wrap a raw slot number.
    ///
    /// Public because the graph layer builds slots from stored adjacency data;
    /// it is not a way to conjure a valid address out of nothing, and passing a
    /// slot that was never allocated to a store method will panic or return
    /// `None` depending on the method.
    pub const fn new(raw: u32) -> Self {
        Slot(raw)
    }

    /// The raw slot number.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The slot as an array index.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Hands out dense [`Slot`]s and takes them back.
///
/// The whole point is the free list. Without it, an index that inserts and
/// deletes a million vectors a day grows its backing array by a million rows a
/// day even though its live set is constant — the array is addressed by slot,
/// so an abandoned slot is a permanent hole. With it, the array's high-water
/// mark tracks the *peak* live count instead of the *cumulative* insert count.
///
/// The free list is a stack, not a queue: the most recently released slot is
/// the first reused, so a delete-then-insert pair touches memory that is still
/// in cache. A queue would spread the reuse evenly across the array, which
/// sounds fairer and measurably is not.
///
/// The allocator does not remember which slots it has handed out — that would
/// duplicate state the caller already keeps. Releasing a slot twice corrupts
/// the free list; [`SlotAllocator::release`] carries a `debug_assert` that
/// catches it in tests, and [`crate::VectorStore`] is the intended caller and
/// makes it structurally impossible.
///
/// ```
/// use proxima_core::SlotAllocator;
///
/// let mut alloc = SlotAllocator::new();
/// let a = alloc.allocate()?;
/// let b = alloc.allocate()?;
/// assert_eq!(alloc.high_water(), 2);
///
/// alloc.release(a);
/// assert_eq!(alloc.live(), 1);
/// // The released slot comes back before any fresh one is minted.
/// assert_eq!(alloc.allocate()?, a);
/// assert_eq!(alloc.high_water(), 2);
/// # let _ = b;
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct SlotAllocator {
    /// Next never-used slot number; also the high-water mark.
    next: u32,
    /// Released slots, most recent last.
    free: Vec<u32>,
}

/// One slot number is reserved so that `u32::MAX` can act as a sentinel in
/// adjacency data without needing an `Option<Slot>` (four bytes to eight).
pub const MAX_SLOTS: usize = u32::MAX as usize;

impl SlotAllocator {
    /// An allocator that has never handed out a slot.
    pub fn new() -> Self {
        SlotAllocator::default()
    }

    /// An allocator whose free list is preallocated for `capacity` releases.
    ///
    /// Only affects the free list; the caller's backing arrays are its own
    /// business.
    pub fn with_capacity(capacity: usize) -> Self {
        SlotAllocator {
            next: 0,
            free: Vec::with_capacity(capacity),
        }
    }

    /// Take a slot, reusing a released one if there is one.
    ///
    /// O(1), and allocates only when the free list is empty and the caller then
    /// grows its own arrays.
    pub fn allocate(&mut self) -> Result<Slot> {
        if let Some(slot) = self.free.pop() {
            return Ok(Slot(slot));
        }
        if self.next as usize >= MAX_SLOTS {
            return Err(ProximaError::Capacity {
                what: "slot allocator",
                limit: MAX_SLOTS,
            });
        }
        let slot = Slot(self.next);
        self.next += 1;
        Ok(slot)
    }

    /// Return a slot for reuse.
    ///
    /// # Panics
    ///
    /// In debug builds, if `slot` was never allocated or is already free. In
    /// release builds a double release silently corrupts the mapping, which is
    /// why the only caller in this workspace is [`crate::VectorStore`].
    pub fn release(&mut self, slot: Slot) {
        debug_assert!(
            slot.0 < self.next,
            "released a slot that was never allocated"
        );
        debug_assert!(
            !self.free.contains(&slot.0),
            "released slot {slot} twice; the free list is now corrupt"
        );
        self.free.push(slot.0);
    }

    /// Number of slots currently handed out.
    pub fn live(&self) -> usize {
        self.next as usize - self.free.len()
    }

    /// One past the largest slot number ever handed out.
    ///
    /// This — not [`SlotAllocator::live`] — is the length the caller's backing
    /// arrays must have.
    pub fn high_water(&self) -> usize {
        self.next as usize
    }

    /// Number of slots waiting on the free list.
    ///
    /// The gap between this and zero is wasted memory in the backing arrays and
    /// is what a compaction pass exists to close.
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Forget everything, including the high-water mark.
    pub fn clear(&mut self) {
        self.next = 0;
        self.free.clear();
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
    use std::collections::HashSet;

    #[test]
    fn ids_round_trip_through_u64() {
        for raw in [0, 1, u64::MAX / 2, u64::MAX] {
            assert_eq!(u64::from(VectorId::from(raw)), raw);
        }
    }

    #[test]
    fn churn_does_not_grow_the_high_water_mark() {
        // This is the property the free list exists for: a workload whose live
        // set is bounded must have a bounded backing array, no matter how many
        // inserts it performs in total.
        let mut alloc = SlotAllocator::new();
        let mut live: Vec<Slot> = (0..64).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(alloc.high_water(), 64);

        for round in 0..1000 {
            let victim = live.swap_remove(round % live.len());
            alloc.release(victim);
            live.push(alloc.allocate().unwrap());
        }

        assert_eq!(alloc.high_water(), 64, "1000 replacements grew the array");
        assert_eq!(alloc.live(), 64);
        let distinct: HashSet<u32> = live.iter().map(|s| s.get()).collect();
        assert_eq!(distinct.len(), 64, "a slot was handed out twice");
    }

    #[test]
    fn released_slots_come_back_in_lifo_order() {
        let mut alloc = SlotAllocator::new();
        let slots: Vec<Slot> = (0..4).map(|_| alloc.allocate().unwrap()).collect();
        for slot in &slots {
            alloc.release(*slot);
        }
        let reused: Vec<u32> = (0..4).map(|_| alloc.allocate().unwrap().get()).collect();
        assert_eq!(reused, vec![3, 2, 1, 0]);
    }

    #[test]
    fn clear_resets_the_high_water_mark() {
        let mut alloc = SlotAllocator::with_capacity(8);
        let _ = alloc.allocate().unwrap();
        let _ = alloc.allocate().unwrap();
        alloc.clear();
        assert_eq!(alloc.high_water(), 0);
        assert_eq!(alloc.allocate().unwrap(), Slot::new(0));
    }
}
