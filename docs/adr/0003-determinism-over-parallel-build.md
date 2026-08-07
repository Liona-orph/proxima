# 0003. Determinism over parallel index build

* **Status:** Accepted
* **Date:** 2026-01
* **Visible in:** the absence of `rayon` in `Cargo.toml`; `HnswConfig::with_seed`;
  the `draws` counter in `HnswIndex`; `VectorStore::iter` iterating in slot order

## Context

Building 50 000 vectors into an HNSW graph takes **6.1 s** on one core. The work
is embarrassingly parallel in the obvious sense — each insert is an independent
search followed by a local edge update — and four cores would take it to roughly
1.6 s. `rayon` would make it a one-line change to the insert loop plus a lock per
node's adjacency list.

The problem is what parallel insert does to the *result*. Concurrent inserts
finish in a nondeterministic order. Two vectors inserted "at the same time" see
different partial graphs depending on which finished first, so they select
different neighbours, so the graph differs run to run. Every downstream property
follows:

* A benchmark number is no longer reproducible from the config.
* `HnswConfig::seed` becomes decorative: same seed, different graph.
* The snapshot round-trip tests, which assert that a reload answers *identically*,
  can no longer distinguish "the reload is faithful" from "both runs happened to
  agree".
* A recall regression cannot be bisected, because two builds of the same commit
  give different recall.

## Decision

The index build is single-threaded, and the determinism guarantee is stated as a
guarantee rather than an observation: **the same seed, the same insertion order
and the same configuration produce a byte-identical graph, in every process and
on every run.**

Three things enforce it:

1. `ChaCha8Rng`, whose output stream is specified. `rand::rngs::StdRng` is
   documented as *not* reproducible across `rand` releases, which would make the
   guarantee a lie the moment the lockfile moved. Both `rand` and `rand_chacha`
   are pinned with `=`.
2. Nothing that affects a result iterates a `HashMap`. `VectorStore::iter` walks
   slots in order specifically so that a tie between two equidistant vectors
   resolves the same way every time.
3. The generator advances **exactly once per successful insert**. The duplicate
   check runs before the level draw, so a rejected insert does not advance it,
   and `HnswIndex::compact` resets it because a rebuild from the same seed and
   order must give the same graph.

The guarantee extends across a restart: the snapshot carries the number of draws
taken and `HnswIndex::from_parts` fast-forwards the generator, so inserts after a
reload land on the levels they would have without one.

## Consequences

**What it bought.** `crates/proxima-store/tests/snapshot.rs` can assert exact
equality of full `Vec<SearchResult>` values — ids, distances and order — across a
save and load, and a companion test asserts that a *rebuild* from the same
vectors is a different graph, so the round-trip tests are evidence of something.
`proxima bench` produces the same numbers twice. A recall regression is
bisectable.

**What it cost.**

* **6.1 s instead of about 1.6 s** for 50 000 vectors, and proportionally more at
  scale: a million vectors is roughly two minutes on one core.
* **IVF-PQ training too.** 26 s for the same fixture — one k-means over 256
  centroids and 32 more over the residual subspaces — all of it sequential for
  the same reason.
* **A large batch upsert holds the collection's write lock for its whole
  duration**, because there is no way to spread it across cores.
* The determinism is only as good as its weakest link, so it is a constraint on
  every future change: any use of `thread_rng`, of hash-map iteration order, or
  of a floating-point reduction whose order varies would silently break it.

**What would reverse it.** A deterministic parallel build — partition the corpus,
build sub-graphs, merge in a fixed order — is possible and is a substantially
larger piece of work than `rayon::par_iter`. If build throughput became the
binding constraint, that is the shape of the fix, not a lock around the current
insert loop.
