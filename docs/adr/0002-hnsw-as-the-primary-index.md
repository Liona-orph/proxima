# 0002. HNSW as the primary approximate index

* **Status:** Accepted
* **Date:** 2026-01
* **Visible in:** `crates/proxima-index/src/hnsw.rs`; the default
  `IndexSpec::Hnsw(HnswConfig::new())` in the server's create handler

## Context

Something has to be the index a user gets when they do not choose one. The
candidates:

* **IVF / IVF-PQ.** Cheap to build, small in memory, and the standard answer at
  billion scale. Its recall ceiling is set by the codebook, not by the search
  effort.
* **Annoy-style random projection forests.** Simple, memory-mappable, and
  needs many trees for competitive recall.
* **Scalar-quantized flat scan.** Exact ordering up to quantization error and
  `O(n)`. Genuinely correct below a few tens of thousands of vectors.
* **HNSW.** The best measured recall-per-query-cost on essentially every public
  benchmark for corpora that fit in memory.
* **DiskANN / Vamana.** Better when the index does not fit in RAM, at the cost
  of a much more complex build and an SSD-shaped design.

Proxima is single-node and holds everything in memory, so the DiskANN argument
does not apply and the flat-scan argument applies only at small scale, where
`FlatIndex` already exists and is documented as the right choice.

## Decision

`HnswIndex` is the default. `FlatIndex` is kept as an exact index *and* as the
ground truth every recall number is measured against. `IvfPqIndex` is offered for
the case HNSW is worst at — memory per vector — and is not the default.

Defaults follow the paper: `M = 16`, `ef_construction = 200`, `ef_search = 64`,
base layer degree `2M`.

## Consequences

**What it bought.** On the benchmark fixture — 50 000 vectors, dimension 128 —
HNSW answers in **52.6 µs** at recall@10 of **0.9990**, against a flat scan's
1 049 µs. The dial keeps paying: `ef_search = 128` reaches 1.0000 at 70.9 µs.
IVF-PQ on the same fixture saturates at 0.6400 without exact reranking.

**What it cost.**

* **Build time.** 6.1 s for 50 000 vectors against the flat index's 8.0 ms, a
  752x ratio. HNSW costs about 100 µs per vector to build and then answers
  queries 20x faster, so the graph needs roughly 6 100 queries at this size
  before it is ahead on total work — and about 58 000 at 1 000 vectors, which is
  why `FlatIndex` is documented as the right choice for small corpora even though
  it loses the latency table.
* **Memory.** `~8M` bytes of graph per vector. 669 bytes per vector against
  flat's 512 at dimension 128 — a 31 % overhead, which falls to about 5 % at
  dimension 768 and rises to 100 % at dimension 32. At low dimension `FlatIndex`
  wins outright.
* **Deletion is a tombstone.** The graph cannot free a slot whose edges other
  nodes still hold, so deletes accumulate dead nodes that are still traversed
  through, and `compact` is a full rebuild. An IVF index would have had none of
  this: no edges, so a removed vector just leaves its list.
* **The filtered-search cliff**, which is ADR 0004.
* **A build that is inherently sequential**, which is ADR 0003.

**What would reverse it.** A corpus that does not fit in memory, which is a
DiskANN-shaped problem, or a workload dominated by deletes, where the rebuild
cost of compaction would dominate.
