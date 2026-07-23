# proxima-index benchmarks

Every number below was produced by `cargo bench -p proxima-index` on the machine
described under *Environment*. They are **measured, not estimated**: the build
time and recall tables are printed by the benchmark harness itself
(`crates/proxima-index/benches/index.rs`), so the whole file can be reproduced
with one command.

```
cargo bench -p proxima-index -- --measurement-time 2 --warm-up-time 1 --sample-size 20
```

## Environment

| | |
|---|---|
| CPU | aarch64 (ARM64, Neoverse-class), 4 cores, no SIMD intrinsics used |
| OS | Linux 6.8, Ubuntu 22.04 userspace |
| Toolchain | rustc 1.83.0, `--release` (`opt-level=3`, `lto="fat"`, `codegen-units=1`) |
| Harness | criterion 0.7, 20 samples, 2 s measurement window, 1 s warm-up |
| Threads | single. Nothing in this crate is parallel; see *What these numbers do not show* |

## Workload

50 000 (and 10 000, and 1 000) vectors of **dimension 128**, drawn from 64
isotropic Gaussian clusters with unit within-cluster standard deviation and
cluster centres at scale 4. Queries are **perturbed corpus points** — a stored
vector plus Gaussian noise at 0.25 — 200 of them for recall, 64 rotating for
latency.

That query construction is the one methodological choice worth arguing about, so
it is stated up front. Sampling queries from a *fresh* set of cluster centres
puts every query far outside the corpus, where the 780 members of the nearest
cluster all sit within a few percent of the same distance and the "true" top ten
is decided by noise in the fifth decimal place. Recall measured that way comes
out around 0.7 for a *correct* implementation at `ef_search = 32` and is a
measurement of the fixture, not the index. A real query is an embedding of
something similar to something stored, which is what the fixture models.

Metric is squared Euclidean throughout. `k = 10`.

## Recall

50 000 vectors, 200 queries, recall@10 against exhaustive search.

| M | ef_construction | ef_search | mean recall | min recall | graph MiB |
|--:|--:|--:|--:|--:|--:|
| 8 | 100 | 16 | 0.6545 | 0.00 | 4.5 |
| 8 | 100 | 32 | 0.7950 | 0.00 | 4.5 |
| 8 | 100 | 64 | 0.8930 | 0.00 | 4.5 |
| 8 | 100 | 128 | 0.9660 | 0.00 | 4.5 |
| 8 | 100 | 256 | 0.9955 | 0.90 | 4.5 |
| 16 | 200 | 16 | 0.8705 | 0.00 | 7.5 |
| 16 | 200 | 32 | 0.9560 | 0.00 | 7.5 |
| 16 | 200 | 64 | **0.9880** | 0.00 | 7.5 |
| 16 | 200 | 128 | 0.9945 | 0.00 | 7.5 |
| 16 | 200 | 256 | **1.0000** | 1.00 | 7.5 |
| 32 | 200 | 16 | 0.9670 | 0.80 | 13.6 |
| 32 | 200 | 32 | 0.9960 | 0.90 | 13.6 |
| 32 | 200 | 64 | **1.0000** | 1.00 | 13.6 |
| 32 | 200 | 128 | 1.0000 | 1.00 | 13.6 |
| 32 | 200 | 256 | 1.0000 | 1.00 | 13.6 |

The vector array for the same corpus is **24.4 MiB** (512 bytes per vector), so
the graph costs 18 %, 31 % and 56 % on top at `M = 8`, `16` and `32`.

Three things in this table are worth more than the headline:

* **`min recall` is the column to read.** At `M = 16, ef_search = 128` the mean
  is 0.9945 and the minimum is **0.00** — one query in two hundred gets nothing
  right. A mean of 0.99 is compatible with a small population of queries that
  fail completely, and only `M = 32` or `ef_search = 256` eliminates them here.
  `RecallReport` reports the minimum and the below-threshold fraction for
  exactly this reason.
* **`M` buys more than `ef_search` does.** Going from `M = 8, ef = 256` (0.9955,
  62 µs) to `M = 32, ef = 64` (1.0000, 55 µs) is better on both axes and costs
  9 MiB. When memory is available, spend it on `M`.
* **`ef_construction` is nearly free at query time and is paid once.** It never
  appears in the latency table; it appears once, in the build table. Build with
  a generous value and tune `ef_search` afterwards.

## Query latency

Mean of criterion's estimate, `k = 10`, one query per iteration.

| corpus | flat | hnsw ef=32 | ef=64 | ef=128 | ef=256 |
|---|--:|--:|--:|--:|--:|
| 1 000 | 22.3 µs | 11.7 µs | 21.0 µs | 36.5 µs | 62.5 µs |
| 10 000 | 211.2 µs | 11.8 µs | 15.9 µs | 23.9 µs | 61.9 µs |
| 50 000 | 1 049 µs | 34.7 µs | 54.8 µs | 78.6 µs | 107.8 µs |

**Flat scales exactly linearly** — 22.3 µs, 211 µs, 1 049 µs for 1x, 10x, 50x
the data, which is 23.3 GiB/s of vector data and essentially memory bandwidth.
That is the honest baseline: an exhaustive scan is not slow, it is just `O(n)`.

**HNSW barely scales at all.** At `ef = 64`, going from 1 000 to 50 000 vectors
costs 21.0 µs to 54.8 µs — a 2.6x increase for 50x the data, and most of that
increase is cache, not comparisons.

**The crossover is around 1 000 vectors**, not 10 000: at `n = 1 000` flat takes
22.3 µs and HNSW at `ef = 64` takes 21.0 µs. But *query* crossover is the wrong
question, because HNSW's 75 ms build at `n = 1 000` has to be amortised. At that
size the graph needs roughly **58 000 queries** before it is ahead in total
work; at `n = 50 000` it needs about **6 100**. That is the real decision rule,
and it is why `FlatIndex` is documented as the right choice for small corpora
even though it loses the latency table.

Note also that at `n = 1 000`, `ef = 128` and `ef = 256` are *slower than flat*.
An `ef` larger than the corpus is pure overhead.

## Build and insert

| vectors | flat | hnsw (M=16, ef_c=200) | ratio |
|---|--:|--:|--:|
| 1 000 | 172.9 µs | 75.3 ms | 435x |
| 10 000 | 1.5 ms | 999.6 ms | 684x |
| 50 000 | 8.1 ms | 6.1 s | 752x |

Insert throughput, 5 000 vectors of dimension 128, whole build per iteration:

| index | total | per vector | vectors/s |
|---|--:|--:|--:|
| flat | 505 µs | 101 ns | 9 900 000 |
| hnsw M=8, ef_c=100 | 195.6 ms | 39.1 µs | 25 600 |
| hnsw M=16, ef_c=200 | 488.8 ms | 97.8 µs | 10 200 |
| hnsw M=32, ef_c=200 | 812.1 ms | 162 µs | 6 200 |

Doubling `M` from 8 to 16 while doubling `ef_construction` costs 2.5x build
time; doubling `M` again at fixed `ef_construction` costs a further 1.7x. The
`M` term is the `O(M^2)` neighbour-selection heuristic and the `ef_construction`
term is the candidate search, and they multiply.

The ratio to flat grows with `n` (435x to 752x) because HNSW's insert is
`O(log n)` in graph hops while flat's is a `memcpy`. Read it as: **the graph
costs about 100 µs per vector to build and then answers queries 20x faster.**

## Filtering

Measured in `tests/hnsw.rs` rather than here, because the interesting quantity
is recall and not latency. Different corpus from the tables above: 10 000
vectors of dimension 64 in 40 clusters, 100 queries, default `M = 16`,
`ef_search = 64`.

| filter selectivity | recall@10 | short results |
|---|--:|--:|
| none | 1.0000 | 0 |
| 1 in 2 | 1.0000 | 0 |
| 1 in 40 | 1.0000 | 0 |

The 1-in-40 row is the one that matters, and it was **0.3880** before the
stopping condition was fixed. Filling `k` results is not enough: an `ef = 64`
traversal under a 1-in-40 filter examines a few hundred nodes and about ten pass,
so it returns ten accepted nodes that are merely the ones it walked past. The
search therefore also requires the traversal to have *seen* at least `ef`
accepted nodes before it stops, escalating `ef` geometrically to
`filter_ef_limit` and falling back to an exhaustive scan below that selectivity.
See `HnswIndex::search` for the argument.

## Distance kernels

From `cargo bench -p proxima-core`, same machine. The unrolled kernels break the
loop-carried dependency on the accumulator, which is what lets LLVM emit NEON.

| kernel | dim 96 | dim 768 | dim 3072 | speedup at 768 |
|---|--:|--:|--:|--:|
| squared euclidean, scalar | 17.9 ns | 349.9 ns | 1 531 ns | |
| squared euclidean, unrolled | 14.8 ns | 115.9 ns | 451.7 ns | **3.0x** |
| dot, scalar | 17.2 ns | 328.3 ns | 1 512 ns | |
| dot, unrolled | 4.79 ns | 45.1 ns | 237.1 ns | **7.3x** |
| manhattan, scalar | 17.3 ns | 323.3 ns | 1 510 ns | |
| manhattan, unrolled | 4.74 ns | 47.4 ns | 236.5 ns | **6.8x** |
| cosine, scalar | 37.2 ns | 449.8 ns | 1 886 ns | |
| cosine, unrolled | 11.2 ns | 76.7 ns | 283.0 ns | **5.9x** |

Reading these:

* **The unrolling is worth 3x to 7x**, which is the difference between an HNSW
  query at 55 µs and one at at least 150 µs. This is the single largest
  optimisation in the repository and it is 20 lines of ordinary safe Rust.
* **Squared Euclidean gains least (3.0x)** because its inner loop is a subtract,
  a multiply and an add against dot product's multiply and add — it is closer to
  issue-limited than dependency-limited to begin with.
* **Cosine's 5.9x is on three accumulator arrays at once**, and it still beats
  running `dot` plus two norms separately because the operands are loaded once.
* At dim 96 the unrolled kernels run at 4.7 ns, i.e. about **20 f32 pairs per
  nanosecond**. That is roughly four 4-lane FMAs per cycle at 2.5 GHz, which is
  what "the compiler found the SIMD" looks like.

## What these numbers do not show

* **Single-threaded, warm cache, one process.** The 50 000-vector corpus is
  24 MiB and does not fit in this machine's L2; a 1M-vector corpus would be
  memory-bound in a way none of these measurements capture.
* **No parallel build.** Building 50 000 vectors takes 6.1 s on one core. A
  parallel build is the obvious next optimisation and is deliberately not
  implemented, because it changes the order in which neighbours are selected and
  therefore the graph, and determinism was judged worth more than build
  throughput. `HnswConfig::seed` would become a lie.
* **No SIMD intrinsics.** Every kernel is portable safe Rust that the
  autovectoriser happens to compile well. Hand-written NEON would plausibly gain
  another 1.5x-2x on the distance kernels, at the cost of `unsafe` and a
  per-architecture code path.
* **Dimension 128 only.** At dimension 768 the distance kernels are 3x more
  expensive per comparison and the graph's memory overhead falls from 31 % to
  5 %, so both tables shift in HNSW's favour.
* **No quantization.** `proxima-quantize` is not wired into the index; its own
  measured recall and compression numbers are in that crate's `tests/recall.rs`.

## IVF-PQ against flat and HNSW

Everything above comes from `cargo bench -p proxima-index`, which measures the
graph. The table below comes from a different command:

```
cargo run --release -p proxima-cli -- bench --vectors 50000 --dim 128
```

`proxima bench` builds all three indexes over the *same* generated fixture and
reports build time, latency, **measured** memory per vector and recall@10 in one
run, so it is the command to use to reproduce this section. The fixture is the
one described under *Workload*: 50 000 vectors of dimension 128 in 64 Gaussian
clusters, 200 perturbed-corpus-point queries, seed `0x9e3779b9`, `k = 10`.

Memory is read from `IndexStats`, which sums the actual `Vec` capacities the
index holds — the vectors, the graph or the codes, the inverted lists, the
codebooks and the id maps. It is **not** `dim * 4 + M * 8`. Memory per vector is
the entire reason IVF-PQ exists, so estimating it would be estimating the thing
under test.

| index | build | mean query | p95 query | B/vector | recall@10 | min |
|---|--:|--:|--:|--:|--:|--:|
| flat | 7.97 ms | 1 049 µs | 1.1 ms | 512 | 1.0000 | 1.00 |
| hnsw `M=16`, `ef=64` | 6.06 s | 52.6 µs | 58.9 µs | 669 | 0.9990 | 0.90 |
| ivfpq `m=32`, `nprobe=16` | 25.98 s | 269.2 µs | 282.7 µs | 718 | 0.6400 | 0.30 |
| ivfpq `m=32`, `nprobe=16`, rerank 100 | 26.07 s | 307.7 µs | 340.3 µs | 718 | **0.9990** | 0.90 |
| ivfpq `m=32`, `nprobe=16`, vectors dropped | 26.16 s | 270.2 µs | 286.9 µs | **47** | 0.6400 | 0.30 |

The raw corpus is 24.4 MiB, i.e. 512 bytes per vector. The last row is 2.2 MiB
total.

### The three things this table says

**IVF-PQ wins on memory by 11x and loses on everything else.** 47 bytes per
vector against flat's 512 and HNSW's 669 — a 10.9x and 14.2x reduction. That is
the whole case for this index and it is a large number. It also builds 4.3x
slower than the graph and answers queries 5x slower, and at this corpus size its
bare recall is 0.64. If a 50 000-vector corpus fits in memory, and 24 MiB does,
**HNSW is the better index and it is not close.** IVF-PQ starts making sense
where 512 bytes per vector does not fit — around 20 million vectors on a 16 GiB
machine, which is 40x larger than anything measured here.

**Reranking is not optional, and it is not free.** Bare PQ recall is 0.64;
rescoring the top 100 candidates against the exact vectors takes it to 0.9990 —
the same as HNSW — for 38 µs, a 14 % latency increase. But reranking needs the
exact vectors, so the row that reranks is the row that costs 718 bytes per
vector, which is *more* than flat. **You cannot have both the memory saving and
the recall.** That is the honest shape of this index and no configuration
escapes it: `IvfPqConfig::validate` rejects `rerank > 0` with `retain_vectors`
off rather than letting the combination look available.

The usual production answer is to keep the vectors on disk and rerank with a
random read per candidate, which trades the memory for 100 IOPS per query.
Proxima does not implement that; see the README's *What this is not*.

**`nprobe` is not the dial people expect it to be.** Scanning more lists helps
until it does not:

| nprobe | mean query | recall (bare) | recall (rerank 100) |
|--:|--:|--:|--:|
| 1 | 22.9 µs | 0.4240 | 0.4895 |
| 4 | 72.5 µs | 0.6230 | 0.9540 |
| 16 | 269.7 µs | 0.6400 | 0.9990 |
| 64 | 1.1 ms | 0.6400 | 0.9990 |
| 256 (= `nlist`, an exhaustive code scan) | 4.4 ms | 0.6400 | 0.9990 |

Bare recall saturates at **0.64 at `nprobe = 16`** and does not move for the
remaining 16x of work, including at `nprobe = nlist`, where every single code in
the corpus is scored. That is the diagnostic: the missing 36 % is not in a list
the search failed to visit, it is **inside the quantization error**. Scoring
every code cannot recover a neighbour whose 32-byte code is closer to the query
than the true nearest neighbour's is. Only more bytes per code, or an exact
rescoring pass, can.

The same sweep on the graph, for comparison:

| ef_search | mean query | recall@10 | min |
|--:|--:|--:|--:|
| 16 | 24.5 µs | 0.9630 | 0.00 |
| 32 | 37.2 µs | 0.9925 | 0.70 |
| 64 | 53.0 µs | 0.9990 | 0.90 |
| 128 | 70.9 µs | 1.0000 | 1.00 |
| 256 | 98.2 µs | 1.0000 | 1.00 |

HNSW's dial keeps paying all the way to 1.0000. IVF-PQ's stops at 0.64. That
difference — a dial that converges on the exact answer against one that
converges on the codebook's answer — is the real distinction between an
approximate *search* and a *lossy compression*.

### Code size against recall

`subspaces` is the compression ratio and the dominant term in recall loss.
Measured at `nprobe = 16` on the same fixture:

| subspaces | code | compression | B/vector (dropped) | recall bare | recall + rerank 100 |
|--:|--:|--:|--:|--:|--:|
| 16 | 16 B | 32x | 31 | 0.3970 | 0.8990 |
| 32 | 32 B | 16x | 47 | 0.6400 | 0.9990 |

Halving the code halves the recall and saves 16 bytes per vector. Note that
"B/vector" is larger than the code — 47 against 32 — because the inverted lists,
the `slot -> list` map, the metadata table and the 256-centroid codebooks are
all real memory too, and a paper calculation of `n * m` bytes would miss all of
them. At `m = 32` the codebooks alone are 128 KiB, which is 2.6 bytes per vector
at this corpus size and would be 0.13 at a million.

### Why the absolute recall is low, and what that does and does not mean

The fixture is **hostile to product quantization by construction**. Isotropic
Gaussian clusters in 128 dimensions have no low-dimensional structure for a
subspace codebook to exploit: every one of the 32 four-dimensional subspaces
sees an isotropic Gaussian, which is the worst case for a 256-entry codebook.
Real embeddings are not like this — they have strong correlations between
dimensions and an effective rank far below their nominal one, which is exactly
what PQ exists to exploit, and published PQ recall on SIFT or GloVe at 16x
compression is substantially higher than 0.64.

So: **the 0.64 is a property of this fixture and does not transfer.** What does
transfer is the *shape* — recall falls steeply as `m` shrinks, `nprobe`
saturates, and exact reranking recovers most of the loss for a small constant
cost. `proxima-quantize`'s own `tests/recall.rs` makes the same argument at
length about the same fixture.

This is also the reason there is no OPQ. A learned rotation before quantization
is the standard fix for exactly the correlation structure this fixture does not
have, so implementing it here would produce a measurement of nothing. See the
README.

### What this section does not show

* **A corpus large enough for IVF-PQ to be the right answer.** Everything here
  is 50 000 vectors, where the whole corpus fits in 24 MiB and the memory
  argument does not bite. The numbers are a demonstration of the mechanism, not
  a recommendation.
* **Reranking against on-disk vectors.** The only reranking measured is against
  vectors held in RAM, which is why the reranked row costs more memory than flat.
* **Parallel training.** The 26 s build is one core: `k`-means over 256
  centroids, then 32 more `k`-means runs over the residual subspaces. It is
  embarrassingly parallel and is not parallelised, for the same determinism
  reason the graph build is not.
