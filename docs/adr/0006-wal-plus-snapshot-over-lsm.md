# 0006. Write-ahead log plus snapshot, not an LSM tree

* **Status:** Accepted
* **Date:** 2026-02
* **Visible in:** `crates/proxima-store/src/wal.rs`,
  `crates/proxima-store/src/snapshot.rs`, `crates/proxima-store/src/collection.rs`

## Context

The index needed durability. The options considered:

**Embed an LSM store** (RocksDB, sled, redb) and keep vectors and metadata as
key-value pairs. This is what several vector databases do. It brings compaction,
bloom filters, block caches, iterators and a decade of tuning.

**Memory-map the index files** and let the kernel handle persistence. Cheap
reads, no explicit write path, and no control over *when* a page reaches the
disk, which is exactly what durability is about. `msync` semantics for a
structure being mutated in place are also a poor fit for a graph.

**Write-ahead log plus periodic snapshot.** The log is the source of truth
between snapshots; the snapshot is a checkpoint; recovery is load-then-replay.

The decisive observation is that **the whole index has to be in memory to be
searched anyway**. A graph search hops to unpredictable slots hundreds of times
per query; an index that is not resident is a different design (DiskANN), not
this one with a cache in front. So the storage layer's job is not to serve reads
— it never does — it is only to make writes durable and to rebuild memory after a
restart. An LSM's entire value proposition is serving reads from disk
efficiently, and Proxima would pay for all of it and use none of it.

There is a second reason, which is the stronger one. A key-value store persists
the *vectors*. It does not persist the *graph*, and the graph is not a function
of the vectors: after any deletion, rebuilding from the stored vectors in the
same order produces a **different** graph, because the rebuild never creates the
edges the tombstone left behind. A design that recovers by replaying vectors into
a fresh index cannot promise that a restart answers queries identically. It would
answer them *about as well*, which is a much weaker and much harder-to-test
claim.

## Decision

An append-only write-ahead log, framed as `length | CRC32C | payload`, with a
torn tail detected and truncated on open. Plus a snapshot of the **whole index
including its graph**, with a versioned header, a byte-order mark and a checksum,
written to a temporary file and renamed atomically. Recovery is: read the
manifest, load the snapshot, replay the log tail.

The property this exists to provide, stated precisely and asserted:

> When `upsert` or `delete` returns `Ok`, the mutation is on stable storage. If
> the machine loses power in the next instant, reopening the collection produces
> an index that answers every query with **exactly** the same results, distances
> included.

`tests/recovery.rs` truncates a real log at every byte offset; `tests/snapshot.rs`
compares full result vectors across a save and load, and asserts that a rebuild
is a *different* graph so that the comparison is evidence of something.

## Consequences

**What it bought.**

* The strong reload property above, rather than "the same vectors come back".
* A format that is 400 lines of hand-written codec, readable end to end,
  versioned, and testable against a byte string. An on-disk format is a
  compatibility promise, and with derives the layout is a consequence of a struct's
  field order that somebody will one day reorder for readability.
* No dependency on a storage engine, consistent with ADR 0001.
* An honest, measurable `fsync` policy: 570-690 µs per record at `always`, 2.8 µs
  per record in a batch of a thousand.

**What it cost.**

* **Snapshot write amplification.** A snapshot rewrites the *entire* index. At
  the default threshold of 100 000 log records, a million-vector collection
  rewrites several hundred megabytes every hundred thousand writes. An LSM
  amortises this properly; this does not. The threshold is the dial and both ends
  of it are bad: lower means more rewriting, higher means longer recovery.
* **Recovery time is linear in the log tail** and each replayed record is a full
  graph insert, about 100 µs. A hundred thousand records is ten seconds of
  startup.
* **The whole corpus must fit in memory.** There is no paging, no block cache and
  no partial load. This is a consequence of the observation that motivated the
  decision, not an independent limitation, but it is a hard ceiling.
* **No transactions.** A batch is one `fsync`, not one atomic unit; a crash
  mid-batch leaves a prefix durable. There is no isolation level and no rollback.
  Prefix consistency is what is guaranteed and what is tested.
* **No point-in-time recovery.** The log is truncated at every snapshot, so there
  is one checkpoint and one tail. Restoring to an arbitrary earlier moment is not
  possible.
* **Compaction forces a snapshot**, because renumbering slots invalidates the
  existing snapshot-plus-log pair.

**What would reverse it.** A corpus that stops fitting in memory, which changes
the premise entirely, or a write rate at which snapshot amplification dominates —
at which point incremental or delta snapshots, not an LSM, are the first thing to
try.
