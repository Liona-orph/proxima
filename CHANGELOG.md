# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Proxima is pre-1.0: while the version starts with a zero, a minor bump may break
the API or the on-disk format, and any such break is listed under `Changed` with
what to do about it.

## [0.1.0] - Unreleased

First release. Everything below is new, so this section is a description of what
the engine is rather than a diff.

### Added

#### `proxima-core`

- `VectorSpace`: dimension and metric, validated once at collection creation so
  that no later operation has to re-check them.
- Distance kernels for Euclidean, squared Euclidean, cosine and inner product,
  each with a straightforward scalar implementation and a four-way unrolled
  variant. The unrolled variants exist to give the compiler's auto-vectoriser
  independent accumulator chains without any `unsafe`; a property test asserts
  that the two agree to a documented tolerance.
- `SlotAllocator`: dense `u32` slot addressing with a free list, so deletes
  leave reusable holes instead of forcing a compaction.
- `FlatStore`: contiguous `f32` storage addressed by slot, with the tombstone
  bitmap kept alongside the vectors so a filtered scan needs one branch and not
  two.
- `BoundedTopK`: a bounded max-heap for k-nearest results, with a total,
  deterministic tie-break so that equal distances resolve the same way on every
  run.
- `Metadata` and `MetadataValue`: typed per-vector metadata (`Null`, `Bool`,
  `Int`, `Float`, `String`) with no implicit coercion between types.
- `ProximaError`: one error enum for the workspace, written so that a single log
  line names the operation, the input and the invariant that failed.

#### `proxima-quantize`

- `ScalarQuantizer`: per-dimension min/max `f32` to `u8` quantization, with the
  reconstruction error it costs measurable from the crate's tests.
- `ProductQuantizer`: subspace decomposition with per-subspace codebooks, code
  encoding and decoding, and asymmetric distance computation against a
  precomputed lookup table.
- `KMeans`: Lloyd's algorithm with k-means++ seeding, an explicit iteration and
  tolerance budget, and empty-cluster handling. Seeded from an explicit `u64`,
  so the same seed produces the same codebook.

#### `proxima-index`

- `VectorIndex`: the trait every index implements — insert, delete, search, and
  filtered search that applies the predicate *during* traversal rather than to
  the result list afterwards.
- `FlatIndex`: exhaustive search. Exact by construction, and the baseline every
  recall number is measured against.
- `HnswIndex`: hierarchical navigable small world graphs — multi-layer
  construction, the heuristic neighbour selection, bidirectional links with
  degree pruning, `ef_construction` and `ef_search` control, and an escalation
  bound for selective filters so a rare predicate cannot turn the search into a
  full scan. The build is single-threaded and seeded, so the same seed builds
  the same graph.
- `IvfPqIndex`: inverted file over k-means coarse centroids with product-
  quantized residuals, `nprobe` list selection, and optional exact reranking of
  the top candidates.
- `recall_at_k`: recall of an approximate index against exact ground truth, with
  both the mean and the worst-case query reported, because a mean recall hides
  the query that returned nothing useful.

#### `proxima-store`

- A write-ahead log with CRC32C-framed records: every record carries its own
  length and checksum, so a process that dies mid-write leaves a torn tail that
  recovery detects and truncates rather than a log that silently loads short.
- Torn-tail recovery on open, with the number of truncated bytes reported rather
  than swallowed.
- Versioned, checksummed snapshots: a format version in the header, a checksum
  over the payload, and a refusal to load a version it does not understand.
- Configurable fsync policy (`always`, `never`, `every_N`), with
  `examples/fsync_cost.rs` to measure what the choice costs on a given disk.
- `Collection`: an index plus its log plus its snapshots behind one type —
  insert, delete, search, snapshot, compaction of tombstoned slots, and
  automatic snapshotting after a configurable number of records.

#### `proxima-server`

- An HTTP/1.1 and JSON API built on the standard library only: a hand-written
  request parser with caps on the request line, header bytes, header count and
  body size, and a hand-written JSON parser with a nesting depth limit.
- Collection lifecycle, vector insert and delete, search with `k` and index
  parameters, and metadata-filtered search. Collection names are restricted to
  `[A-Za-z0-9_-]`.
- A metadata filter language that is a fixed conjunction of field predicates
  (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `exists`), chosen so that filter
  evaluation inside the graph traversal stays a fixed list of lookups.
- `GET /health` for liveness and readiness, and `GET /metrics` in Prometheus
  exposition format with request counts, failures and search latency.
- Graceful shutdown: SIGTERM and SIGINT stop the listener, drain in-flight
  requests and snapshot before exit. Signal handling is unix-only; elsewhere the
  server falls back to the admin endpoint.

#### `proxima-cli`

- `proxima serve` — run the API over a data directory, with `--data`,
  `--address`, `--sync`, `--snapshot-after` and `--max-connections`.
- `proxima bench` — build every index over a generated, seeded fixture and
  report recall, latency and measured memory per vector, plus `ef_search` and
  `nprobe` sweeps. This is the command that produces the numbers in the docs.
- `proxima query` — search a collection from the shell, with an explicit vector
  or a deterministic pseudo-random one, an optional metadata filter, and raw
  JSON output.

#### Project

- Apache-2.0 licence, `NOTICE`, contributing guide, code of conduct, security
  policy and threat model.
- `make ci` as the single gate: `cargo fmt --all --check`, clippy over all
  targets with `-D warnings`, the full test suite, docs with
  `RUSTDOCFLAGS="-D warnings"`, and a benchmark compile check. GitHub Actions
  runs the same steps with every action pinned to a commit SHA.
- Multi-stage `Dockerfile` producing a distroless, non-root image.

[0.1.0]: https://github.com/Liona-orph/proxima/releases/tag/v0.1.0
[unreleased]: https://github.com/Liona-orph/proxima/compare/v0.1.0...HEAD
