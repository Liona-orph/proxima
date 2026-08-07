# 0001. Build the engine from scratch rather than wrapping an ANN library

* **Status:** Accepted
* **Date:** 2026-01
* **Visible in:** the dependency policy comment in `Cargo.toml`; every crate
  under `crates/`

## Context

Approximate nearest neighbour search is a solved problem with excellent
implementations. `faiss` has a decade of engineering and hand-written SIMD for
every architecture. `hnswlib` is the reference HNSW implementation and is faster
than what is here. `usearch` is smaller and well maintained. In Rust,
`instant-distance` and `hora` both exist. Wrapping one of them would have
produced a working vector database in a fraction of the time, with better
absolute numbers.

The project's purpose is the opposite: to be a system whose behaviour is
inspectable and defensible at every layer. A wrapper's interesting properties —
which pruning heuristic, how deletion is handled, whether a filter runs during
or after traversal, what happens to recall when a filter is selective — are all
inside the dependency, and the wrapper's documentation can only repeat what the
upstream README says.

There is also a narrower, practical problem with the specific alternatives.
`faiss` is C++ and its Rust bindings are a build-system liability. `hnswlib`
gives no way to observe traversal, which is what filtered search needs. None of
them will tell you the recall of *your* corpus without a separate harness.

## Decision

Implement everything in-tree: the distance kernels, the slot-addressed store,
k-means, scalar and product quantization, the HNSW graph, IVF-PQ, the
write-ahead log, the snapshot format, the HTTP server and the JSON parser.

Four runtime dependencies, each with a paragraph in `Cargo.toml` saying why it
is there and what was considered instead: `thiserror` (a proc macro, no runtime
cost), `rand` and `rand_chacha` (a *specified* generator stream, which the
determinism guarantee needs), and `signal-hook` (installing a signal handler is
FFI and every crate is `#![forbid(unsafe_code)]`).

Explicitly not taken: any ANN or linear-algebra crate, `rayon`, `serde`,
`tokio`, `axum`, `clap`, `serde_json`, `rand_distr`.

## Consequences

**What it bought.** Every behaviour has a comment explaining why it is that way.
Filtered search could be built on a traversal callback that no external library
exposes. `recall_at_k` could be a library function rather than a benchmark
script. The whole system compiles from a lockfile with 17 transitive crates, and
a reviewer can read all of it.

**What it cost.**

* **Absolute performance.** `hnswlib` and `faiss` are faster. The kernels here
  are portable safe Rust that the autovectoriser compiles well; hand-written
  NEON or AVX-512 would plausibly gain another 1.5-2x. There is no GPU path and
  no `int8` SIMD.
* **Maturity.** The JSON parser and the HTTP reader are the two components most
  likely to have a bug that `serde_json` and `hyper` do not. This is stated
  plainly in `SECURITY.md`; the mitigation is caps on every input dimension plus
  `forbid(unsafe_code)`, which bounds the worst case at a crash rather than
  memory corruption.
* **Missing features that a library would have supplied for free**: OPQ,
  binary quantization, memory-mapped indexes, GPU builds, HTTP/2, TLS.
* **Time.** Roughly 20 000 lines including tests, most of which is not novel.

**What would reverse it.** A need for absolute throughput competitive with
`faiss`, or for a feature (GPU build, OPQ, on-disk graphs) whose implementation
cost here exceeds the value of having implemented everything else.
