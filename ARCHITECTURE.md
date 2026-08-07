# Architecture

How Proxima works, where it departs from the papers it implements, and what each
departure cost. The decisions themselves — the ones with alternatives that were
considered and rejected — are in [`docs/adr/`](docs/adr/); this document is about
the mechanisms.

Every number here was measured. The commands that produce them are
`cargo run --release -p proxima-cli -- bench`, `cargo bench -p proxima-core`,
`cargo bench -p proxima-index` and
`cargo run --release -p proxima-store --example fsync_cost`.

---

## 1. The substrate

### Vectors live in one flat array

`VectorStore` holds `slot_count * dim` floats in a single `Vec<f32>`. Vector `s`
occupies `s*dim .. (s+1)*dim`, so its address is arithmetic rather than a load.

The alternative, `Vec<Vec<f32>>`, is the obvious representation and the single
worst decision available — and the cost is invisible in a microbenchmark that
iterates in order. A graph search does not iterate in order. It hops: examine a
node, read its neighbour list, evaluate the distance to each neighbour, each at
an unpredictable slot. With `Vec<Vec<f32>>` every one of those evaluations is
**two dependent cache misses** — load the 24-byte `Vec` header to learn the
address, then load the data there — and the second cannot even be issued until
the first returns, so the ~100 ns latencies serialise instead of overlapping.
With a flat array the pointer chase is gone and the prefetcher sees a contiguous
run the moment the first line is touched.

The price is that the store is dimension-homogeneous and that growing it copies
everything. Both are fine: a vector index has a fixed dimension by construction,
and the copy amortises to O(1) per insert.

### Two numbering schemes, kept apart

`VectorId` is the caller's `u64` name for a vector: never invented by Proxima,
never changed. `Slot` is Proxima's `u32` *address*: the row in the flat array and
the node number in the graph, dense and reused after deletion. They are separate
newtypes because they have the same shape and swapping them produces
plausible-looking wrong answers rather than a crash.

`SlotAllocator` recycles slots so that an index whose live set is constant has a
constant footprint no matter how many times it has been written to. The free list
is a **stack, not a queue**: the most recently released slot is the first reused,
so a delete-then-insert pair touches memory that is still in cache. A queue
sounds fairer and measurably is not.

### `detach` versus `release`, and why tombstones exist

A flat index can free a slot the instant a vector is deleted. A graph index
cannot: other nodes still hold edges pointing at that slot, and handing it back
to the allocator would let a later insert land there and **silently inherit those
edges**. So deletion is split three ways in `VectorStore`:

* `detach` unmaps the id, keeps the data, and does **not** free the slot. This is
  a tombstone.
* `release` puts a detached slot back on the free list. Only compaction calls it.
* `remove` is both, which is what a flat index wants.

This split is the reason `UnmappedSlots` exists in the snapshot loader: a
snapshot records which slots hold an id, not *why* the others do not, and
restoring a graph with `UnmappedSlots::Free` would recreate exactly the bug the
split prevents. It is a required argument, not a default, because choosing wrongly
is a silent corruption rather than an error.

### Cosine is enforced, not conventional

`VectorSpace::prepare` normalises when `Metric::requires_normalization` says to,
and is the only supported way to hand a raw vector to an index. `VectorSpace::distance`
then runs `cosine_normalized`, which is `1 - dot(a, b)` — no norms in the inner
loop.

The failure this removes is silent. An unnormalised query against a normalised
corpus still returns *plausible* neighbours, just the wrong ones, and no test
that does not measure recall will notice. Making it the API's job rather than the
caller's is 20 lines and it closes the trap structurally.

`prepare` returns a `Cow`, so the Euclidean, Manhattan and inner-product paths
borrow and do not allocate.

### Distances are ranking scores

`Metric::Euclidean` returns the **squared** value and `Metric::InnerProduct`
returns the **negated** one. Both are monotone in the real thing and the search
never looks at the magnitude, so the square root and the sign flip are pure
overhead in the hot loop. `Metric::to_true_distance` converts for display, and is
documented as never to be called inside a ranking loop.

### The kernels, and the one large optimisation

Each kernel has a scalar reference implementation and an unrolled variant that
keeps `LANES = 8` independent accumulators. Unrolling breaks the loop-carried
dependency on a single accumulator, which is what lets LLVM emit NEON. Measured
at dimension 768:

| kernel | scalar | unrolled | speedup |
|---|--:|--:|--:|
| squared euclidean | 349.9 ns | 115.9 ns | 3.0x |
| dot | 328.3 ns | 45.1 ns | 7.3x |
| manhattan | 323.3 ns | 47.4 ns | 6.8x |
| cosine | 449.8 ns | 76.7 ns | 5.9x |

That is the difference between an HNSW query at 53 µs and one at 150 µs or more,
and it is twenty lines of ordinary safe Rust. Squared Euclidean gains least
because its inner loop is a subtract, a multiply and an add against dot product's
multiply and add — it starts closer to issue-limited than dependency-limited.

Property tests assert that scalar and unrolled agree to within a documented
tolerance, and that the metric axioms hold where `Metric::is_true_metric` says
they do.

### NaN is rejected at the boundary, once

`VectorSpace::validate` runs on insert and on query — not per candidate — and
rejects non-finite components. That is what lets every kernel skip checking. A
single NaN in stored data makes every distance involving that vector NaN, which
sorts to the bottom of the result heap and so silently removes the vector from
every search. Recall degrades without an error: the worst kind of bug.

### Bounded top-k

`TopK` is a `BinaryHeap` capped at `k`, not a sort-then-truncate. On a
million-vector scan that is the difference between one 16-element heap and a
million-element vector. Ties in distance break by id, which is one of the three
things that make search results reproducible.

---

## 2. HNSW, as implemented

`HnswIndex` follows Malkov and Yashunin. What follows is the algorithm as it
actually runs, with the departures marked.

### Structure

* **Levels** are drawn from `floor(-ln(U) * m_L)` with `m_L = 1 / ln(M)`,
  giving a geometric distribution with ratio `1/M`, so each layer is about `M`
  times smaller than the one below and the descent is logarithmic. The generator
  is `ChaCha8Rng`, seeded from the config and advanced **exactly once per
  insert**.
* **The base layer** keeps `2M` neighbours and every upper layer keeps `M`,
  following the paper: layer 0 holds every vector and carries the whole burden of
  the final refinement, and its higher degree is what stops greedy descent from
  terminating in a local minimum.
* **Adjacency is split by layer.** Layer 0 is one flat `Vec<u32>` of
  `slot_count * 2M` entries padded with a `u32::MAX` sentinel — read once per
  visited node, so a pointer chase in front of it would be a second cache miss on
  the critical path. Upper layers are a `Vec<Vec<u32>>`, and here that *is* right:
  only about `1/M` of nodes have a level above zero, so the outer array is mostly
  empty `Vec`s costing no allocation, and a flat arena would either waste
  `max_level * M` entries per node or need an offset table.
* **The sentinel is `u32::MAX`, not `Option<u32>`.** Eight bytes per link instead
  of four would double the graph's footprint to save one comparison. `Slot` is
  documented as topping out one below `u32::MAX` for exactly this reason.

### Insert

1. Draw a level. **Departure:** the duplicate-id check happens *before* the draw,
   so a rejected insert does not advance the generator. Without that,
   "same seed, same insertion order, same graph" would be false in the presence
   of a failed insert.
2. Greedy descent with `ef = 1` from the entry point down to one layer above the
   new node's own. Pure navigation, nothing is linked.
3. On each layer the node belongs to: `search_layer` with `ef_construction`,
   `select_neighbours` to pick `M` of the result, link forward, then link back
   from each chosen neighbour.
4. If the new node's level exceeds the entry level, it becomes the entry point.

### The neighbour-selection heuristic, and what it buys

`select_neighbours` is algorithm 4 from the paper, and the rule is one line:
**keep candidate `c` only if no already-kept neighbour `r` is closer to `c` than
`c` is to the base node.**

What that buys is the "navigable" in navigable small world. Keeping the `M`
*nearest* candidates instead — the obvious thing — fills a node's neighbour list
with members of its own cluster, all of which are also each other's neighbours.
The graph becomes a set of dense, well-connected islands with no edges between
them, and greedy search gets stuck in whichever island it starts in. The
diversity rule keeps a candidate only if it opens a direction no kept neighbour
already covers, so a node in a dense cluster spends some of its budget on
long-range edges. Those long edges are the ones a query traverses to cross the
space.

The cost is `O(M * |candidates|)` distance evaluations per selection, which is
why both `M` and `ef_construction` appear in build time and why they multiply:
doubling `M` from 8 to 16 while doubling `ef_construction` costs 2.5x build time,
and doubling `M` again at fixed `ef_construction` costs a further 1.7x.

**`keep_pruned_connections` is on by default.** The heuristic can leave a node
with far fewer than `M` neighbours, so the pruned candidates top the list up in
distance order. Off, the graph is sparser, saves a little memory and costs
recall on clustered data; it is exposed as a flag so the effect can be measured
rather than argued about.

**`link_back` re-runs the heuristic rather than truncating.** When adding an edge
overflows a neighbour's list, the naive fix is to drop the furthest link.
Truncating to the nearest `degree_cap` would systematically delete exactly the
long-range edges the heuristic added, converting the graph back into disconnected
clusters over the course of a build. So the whole list plus the new node goes
back through `select_neighbours`.

### Search

`search_layer` is the paper's algorithm 2 with one addition. It keeps a min-heap
of unexplored candidates and a max-heap of the best `ef` found, and stops when
the closest unexplored candidate is further away than the worst kept result —
nothing reachable through it can improve the list. That single condition is the
entire cost control of HNSW, and it is why `ef` and not `n` bounds the work.

**Departure: the `observe` callback.** `search_layer` calls a closure for every
node whose distance is computed, including ones that never make the returned
list. Unfiltered search ignores it. Filtered search is built on it, and section 3
explains why.

Greedy descent uses strict `<` when moving to a better neighbour, so a plateau of
equidistant nodes cannot cycle.

### Deletion and compaction

`remove` tombstones: the id is unmapped, the data stays, the slot is not freed.
Tombstoned nodes are still **traversed through** — their coordinates are still
needed as waypoints — and are never returned, because `store.id_of(slot)`
returning `None` is exactly the liveness test.

The cost is real and is reported by `IndexStats::deleted`: past roughly 20 %
tombstones the graph spends a fifth of its distance evaluations on vectors that
can never be returned. `compact` is a **full rebuild**, `O(n log n)` distance
evaluations, because there is no cheaper correct answer: repairing the graph
around one removed node means re-running neighbour selection for everything that
pointed at it, and once a meaningful fraction are tombstoned that is most of the
graph. Ids, vectors and metadata are preserved exactly; the graph is not.

`upsert` is remove-then-insert, so a heavily-upserted index accumulates
tombstones. That is the honest cost of upsert on a graph, and the reason it is a
separate trait method rather than `insert`'s behaviour.

### Determinism

Three things make two builds from the same seed and the same insertion order
byte-identical graphs:

1. `ChaCha8Rng`, whose output stream is specified. `rand::rngs::StdRng` is
   explicitly documented as *not* reproducible across `rand` releases, which
   would make the guarantee a lie the moment the lockfile moved.
2. Nothing in construction or search iterates a `HashMap`. `VectorStore::iter`
   walks slots in order for this reason.
3. Ties in distance break by id, in `TopK` and in the graph's candidate ordering.

The build is single-threaded because of this; see
[ADR 0003](docs/adr/0003-determinism-over-parallel-build.md).

---

## 3. Filtering, and the recall trap it avoids

This is the part of the system most likely to be wrong in an implementation that
has not measured it.

### Post-filtering is not a slower correct answer, it is a wrong one

The obvious design is to search for `k`, then drop the results the filter
rejects. That returns fewer than `k` results whenever the filter rejects
anything, and there is **no `k` you can search for that fixes it in general** —
the fraction of the top `k` that passes depends on the query, and a filter that
matches one in a thousand needs a `k` of thousands for some queries and tens for
others.

So filters run **during** traversal. `Filter::accepts` is called from the
`observe` callback, once per node whose distance is computed, and an accepted
node is recorded the moment it is examined — even if a hundred nearer rejected
nodes later push it out of the `ef` list.

### The trap: filling `k` is not enough

Recording accepted nodes during traversal fixes the short-results problem and
introduces a subtler one. Under a 1-in-40 filter an `ef = 64` traversal examines
a few hundred nodes and about ten of them pass. It can therefore return ten
results — a full `k` — that are merely **the ten accepted nodes it happened to
walk past**, not the ten nearest.

Measured, on 10 000 vectors of dimension 64 in 40 clusters at `ef_search = 64`:
recall@10 under a 1-in-40 filter was **0.3880** with a "stop when `k` results are
filled" condition. Mean recall of 0.39 with a full `k` results every time is
exactly the failure mode that looks fine in production and is not.

### The fix, and its cliff

The search also requires the traversal to have **seen at least `ef` accepted
nodes** before it stops. That makes the filtered search as selective *among
accepted vectors* as an unfiltered search is among all of them. When it has not:

1. **`ef` escalates geometrically**, doubling to `filter_ef_limit` (1 024 by
   default). Geometric so that total work is at most twice the final attempt's; a
   linear ramp would be cheaper per step and need many more steps.
2. **Below that, an exhaustive scan.** Either the accepted vectors sit in a
   region the greedy walk never reaches, or the filter is so selective that no
   bounded traversal gathers enough of them. Below roughly `1/ef` selectivity a
   scan is the *right* algorithm and not merely a safety net: the graph's whole
   advantage is that it examines a small fraction of the corpus, and if the
   filter rejects almost all of that fraction the advantage is already gone.
   `O(n)` and correct beats sub-linear and quietly wrong.

With the fix, recall@10 at 1-in-40 is **1.0000** with zero short results. The
cost is the cliff, and it is a cliff rather than a slope: a query whose filter
crosses the threshold goes from 53 µs to a full scan. `HnswConfig::with_exhaustive_filter_fallback(false)`
chooses the other side of that trade explicitly, and then results can be short of
`k`.

### The filter language is deliberately weak

`proxima-server`'s filter DSL is a conjunction of field predicates — `eq`, `ne`,
`gt`, `gte`, `lt`, `lte`, `in`, `exists` — with no OR and no nesting. Two reasons.

A filter is evaluated tens of thousands of times per query, so its cost is
*multiplied* by the search's cost, and an expensive filter does not merely slow
the query down: it changes which `ef` the escalation settles on and therefore how
much of the graph is walked. A conjunction compiles to a fixed list of hash
lookups with no allocation and no branching on shape. An arbitrary boolean
expression tree would be an interpreter on that path.

And a disjunction is usually *less* selective than its parts, while the cliff is
about selectivity. A language that makes cheap, selective filters easy and
expensive ones hard is pointing users at the part of the envelope that works.

Types are not coerced: `{"year": {"gte": 2024}}` matches an `Int` field and not a
`Float` one, following `MetadataValue`'s own rule. A range filter that silently
starts matching different rows after an ingest change is a bug nobody notices for
months. The one exception is an integral bound against a `Float` field, because
JSON cannot express "this 1.0 is a float".

---

## 4. Quantization and IVF-PQ

### What the quantizers cost, measured

On 5 000 clustered 128-dimensional vectors, 200 queries
(`cargo test --release -p proxima-quantize`):

| quantizer | code | compression | recall@10 |
|---|--:|--:|--:|
| scalar, u8, per-dimension | 128 B | 4x | 0.972 |
| scalar, u8, global range | 128 B | 4x | 0.960 |
| product, `m = 32` | 32 B | 16x | 0.473 |
| product, `m = 8` | 8 B | 64x | 0.219 |
| product, `m = 8`, top-100 reranked exactly | 8 B | 64x | 0.833 |

There is no configuration under which scalar beats product on compression or
product beats scalar on accuracy.

### IVF-PQ

`IvfPqIndex` composes them into the index that makes the memory argument:

1. **Coarse quantization.** `nlist` centroids by k-means over a strided sample of
   the corpus. Every vector is assigned to its nearest, and the assignment *is*
   the inverted file.
2. **Residual product quantization.** What is encoded is `x - c(x)`, not `x`.
   Residuals are distributed around the origin the same way in every cell, so one
   codebook fits all of them; encoding raw vectors would ask 256 centroids per
   subspace to cover the whole corpus rather than the spread within one cell, and
   measurably costs recall.
3. **Probing.** The query is scored against all `nlist` centroids and the nearest
   `nprobe` lists are scanned.
4. **Asymmetric distance.** The query stays at full precision; only the stored
   vectors are quantized. One `256 * dim` table per probed list, then each
   candidate costs `m` table lookups and adds, *independent of the dimension*.
   For cosine the table depends only on the query, so it is built once per query
   rather than once per probe — at `nprobe = 16` that is fifteen fewer table
   builds.
5. **Optional exact reranking** of the top candidates against the stored
   full-precision vectors.

The algebra, which is the one part not obviously right:

* Euclidean: `||q - (c + r)||² = ||(q - c) - r||²`, so the table is built on the
  residual query `q - c` and `squared_l2(code)` is the whole distance.
* Cosine: prepared operands are unit vectors, so
  `1 - <q, c + r> = 1 - <q, c> - <q, r>` — the centroid term exactly, the
  residual term from the table.
* **Inner product and Manhattan are rejected at construction.** Maximum inner
  product search is not nearest-neighbour search under any metric, so the nearest
  centroid is not the most promising list; Manhattan has no decomposition through
  a distance table at all. Both would return plausible wrong answers, so neither
  is offered.

### What it costs, measured

50 000 vectors, dimension 128, `nlist = 256`, `m = 32`, `nprobe = 16`:

| | flat | hnsw | ivfpq | ivfpq + rerank | ivfpq, vectors dropped |
|---|--:|--:|--:|--:|--:|
| bytes per vector | 512 | 669 | 718 | 718 | **47** |
| mean query | 1 049 µs | 52.6 µs | 269 µs | 308 µs | 270 µs |
| recall@10 | 1.0000 | 0.9990 | 0.6400 | 0.9990 | 0.6400 |

Two findings worth the space:

**Bare recall saturates at 0.6400 and does not move.** Not at `nprobe = 64`, not
at `nprobe = 256`, which is an exhaustive scan of every code in the corpus. The
missing 36 % is inside the quantization error, not in a list the search failed to
visit. That is the diagnostic distinction between an approximate *search*, whose
dial converges on the exact answer, and a *lossy compression*, whose dial
converges on the codebook's answer.

**You cannot have both the memory and the recall.** Reranking recovers 0.6400 to
0.9990 for 38 µs — and needs the exact vectors, which is the memory being saved.
`IvfPqConfig::validate` rejects `rerank > 0` with `retain_vectors` off rather
than letting the combination look available.

Training is automatic at `train_threshold` live vectors, which makes exactly one
insert seconds rather than microseconds. That is documented rather than hidden,
and the threshold is explicit for the same reason.

`IvfPqIndex` does not use `VectorStore`, unlike every other index here, for
exactly one reason: it has to be able to *not* have the vectors, and a store
whose `vector(slot)` sometimes returns nothing would be a worse type for the two
indexes that always have them.

---

## 5. Storage

### The write-ahead log

```text
file    := header record*
header  := "PROXWAL\x01"
record  := u32 payload_len | u32 crc32c(payload) | payload[payload_len]
payload := u8 op | body
  op 1, upsert := u64 id | u32 dim | f32[dim] | metadata
  op 2, delete := u64 id
```

The length precedes the checksum so a reader can size one read; the checksum
covers the payload but not the length, because a torn write leaves a *shorter*
file than the length claims and that is detectable from the file size alone.

**Only two operations, and neither is "insert".** The log records the *effect* of
a request, not the request. Keeping insert distinct from upsert would mean replay
could fail with `DuplicateId` on a perfectly valid log — which is exactly what
happens if a snapshot is taken between two writes to the same id. For the same
reason, deleting something absent is logged and applied as a no-op: deletes must
be idempotent or a log with two deletes of one id breaks recovery.

**The torn tail.** On open, records are read until one is short or fails its
checksum; the file is then **truncated on disk** at that point, not merely
ignored. Leaving the bad bytes would mean the next append writes after garbage
and the record after it is unreachable to the next recovery.

Recovery stops at the **first** bad record and never scans past it. A good record
after a bad one would mean the corruption is not a torn tail but damage in the
middle, and skipping it would replay a suffix of history without its prefix —
producing an index that looks fine and is wrong. `Recovery::bytes_discarded` is
reported so an operator can see it happened. A crash inside the eight-byte header
itself is handled too: a file whose contents are a strict prefix of the magic is
a half-written header and is rewritten.

The residual risk is a torn record whose truncated payload happens to checksum to
the recorded value: a `2^-32` event per crash. CRC-32C rather than the zlib
polynomial because it has better minimum Hamming distance at these record sizes
and because it is the polynomial both x86-64 and aarch64 implement in hardware,
so an intrinsics version later would not change a byte on disk.

`crates/proxima-store/tests/recovery.rs` truncates a real log at **every byte
offset** and asserts that what comes back is a prefix of history, never an error,
and immediately writable again.

### fsync, and what it costs

| policy | per record | survives process crash | survives power loss |
|---|--:|---|---|
| `always` (default) | 570-690 µs | yes | everything acknowledged |
| `every_64` | 23-25 µs | yes | all but the last 63 |
| `never` | 1.8 µs | yes | nothing promised |
| batch of 1 000 under `always` | 2.8-3.0 µs | yes | everything acknowledged |

Every policy survives a *process* crash, `SIGKILL` included, because the bytes
are in the kernel's page cache by the time `append` returns. Only power loss and
kernel panics distinguish them.

The last row is the one that matters: **batching is the way out that costs
nothing.** A thousand-record batch is one `fsync`, acknowledged as a whole, so it
is exactly as durable as a thousand separate appends and 200x cheaper. That is
why the HTTP API has a batch upsert and no single-point one.

`sync_all` and not `sync_data`: the log's length is metadata, and a `sync_data`
that leaves the new length in the inode cache can lose the tail of the file even
though every data page reached the platter.

### Snapshots

```text
magic "PROXSNAP" | u32 version | u32 byte-order mark | u32 reserved
                 | u64 body_len | body | u32 crc32c(body)
```

The byte-order mark earns its four bytes: the codec is little-endian and a
big-endian reader would otherwise decode a `dim` of 4 096 as 65 536 and produce a
confidently wrong index rather than an error. The reserved word is checked too,
so that every byte of the file is covered by something — a corruption test that
flips every bit in turn found the gap.

Writing is a temporary file, `fsync`, `rename`, then `fsync` **of the
directory**. All four steps: the rename is atomic everywhere, but without the
first `fsync` the rename can be durable while the contents are not, and without
the last one the rename itself can be lost.

**The snapshot carries the graph, not just the vectors**, and that is the whole
design. "A reload answers every query identically" is stronger than "a reload
holds the same vectors", and three things make the difference:

* Slot numbers must be preserved. Ties break by id, but the order candidates are
  *examined* in is slot order, and on a graph that decides which local optimum a
  greedy walk finds.
* Tombstones must be preserved. A removed vector is still traversed through.
* The graph must be preserved. Rebuilding from the vectors in slot order
  reproduces it only if nothing was ever deleted; after a single deletion it does
  not, because the rebuild never creates the edges the tombstone left behind.
  `tests/snapshot.rs` asserts a rebuild is a *different* graph, so that the
  round-trip tests are evidence of something.

The level generator's draw count is carried too, and is fast-forwarded on
restore. That is `O(draws)`, about 20 ns each, so a million-vector index spends
20 ms replaying the level sequence. Skipping it would be free and would make the
determinism guarantee stop at the process boundary.

Encoding is **deterministic**: `Metadata` is a `HashMap` with a per-process random
seed, so the codec sorts keys. Two encodes of one index are byte-identical, which
is what lets a snapshot be diffed, deduplicated or content-addressed. "Identical"
is about answers, not footprint — a reload allocates exactly and a grown index
carries the allocator's slack, so `IndexStats::vector_bytes` can legitimately
shrink across a save and load.

### Recovery

```text
open() → read manifest → load snapshot if present → replay WAL → truncate torn tail
```

The manifest exists so that a collection created and never written to still knows
its dimension, metric and index configuration after a restart, which a
snapshot-only design would forget. It is checksummed, and a mismatch between it
and the shape a caller asks to open is a `Configuration` error rather than a
silent reconfiguration — reopening a 768-dimensional collection as a
384-dimensional one would make every replayed record fail and leave an empty index
where the data used to be.

The write path is **log, then apply**, never the reverse. Applying first and
crashing before the log write would leave a mutation recovery cannot reproduce.
Replay and the live write path share one `apply` function, which makes "recovery
reproduces the index" a structural property rather than two implementations that
have to agree.

`compact` snapshots unconditionally, because compaction renumbers slots and the
existing snapshot plus the existing log no longer describe the index in memory.

---

## 6. Concurrency

**One `RwLock` per collection, around the index and the log together.** Readers
share it, writers take it exclusively, and there is nothing finer-grained
anywhere. The argument:

* **Searches are concurrent.** Any number run at once on as many cores as
  available. This is what a vector database spends its life doing, and it scales.
* **Writes serialise, and the lock is not what makes them serial.** A write is an
  `fsync` — 570 to 690 µs — followed by a graph insert of about 100 µs. Even with
  a lock-free index the `fsync`s would queue on the device. Making the index
  concurrently mutable would buy the 100 µs and cost a rewrite of `HnswIndex`
  around atomics, plus the determinism guarantee: concurrent inserts finish in a
  nondeterministic order and "same seed, same insertion order, same graph" would
  become "…if you inserted single-threaded".
* **Writers block readers.** This is the honest cost. A ten-thousand-vector batch
  holds the write lock for about a second and every search arriving in that second
  waits. The mitigation available today is to size batches for the latency you can
  accept; the one that is not implemented is a copy-on-write index behind an
  atomic swap, which would double the memory.
* **Poisoning is recovered from, not propagated.** A panic while holding the write
  lock poisons it, and refusing every later call would turn one bug into a
  permanent outage. The guard is recovered from the poison and the fact is
  reported through `Collection::is_poisoned`, `GET /health` (which answers `503`)
  and `proxima_collection_lock_poisoned`.

Above that, the server holds an `RwLock<BTreeMap<String, Arc<Collection>>>`.
Handlers clone the `Arc` and release the map lock before doing work, so creating a
collection blocks every request only for the duration of a directory create, while
writing to one collection blocks only that collection's readers. Dropping a
collection uses `Arc::try_unwrap` to wait for in-flight requests: if it is still
shared, the collection goes back in the map and the caller gets `409`, which is far
better than deleting files out from under a running search.

The server itself is **one thread per connection**, capped. The work never awaits
anything — it takes a lock, burns CPU for tens of microseconds, returns — so
there is nothing for an async task to yield to, and an async runtime's advantage
is holding a hundred thousand *idle* connections cheaply. The cost is a hard
ceiling on concurrent connections; past it the listener answers `503` and closes,
because a queue in front of a saturated CPU-bound server converts a fast failure
into a slow one.

Shutdown is one `AtomicBool` set by `SIGTERM`, `SIGINT` or `POST /admin/shutdown`.
The accept loop polls it every 50 ms, stops accepting, drains, and **flushes every
collection's log** before returning. Under the default policy the flush is a
no-op; under the weaker ones it is the difference between losing the last few
hundred writes on a rolling restart and losing none.
