# Contributing to Proxima

Proxima is a vector search engine written from first principles: the graph, the
quantizers, the k-means and the distance kernels are all in-tree. That choice
sets most of the rules below. They are not style preferences — they are what
keeps the project's central claim ("same seed, same graph, and every number in
the docs is one you can reproduce") true.

Read this once before your first pull request. It is short, and every rule in it
is enforced either by `make ci` or by review.

## Getting set up

The toolchain is pinned in `rust-toolchain.toml`:

```toml
[toolchain]
channel = "1.83.0"
components = ["rustfmt", "clippy"]
profile = "minimal"
```

If you have `rustup`, you do not need to do anything: the first `cargo` command
in the repository will install 1.83.0 with `rustfmt` and `clippy` and use it.
Do not override it with `+stable` or `+nightly`. The pin exists because clippy's
lint set moves between releases, and a lint that fires for you but not for CI
(or the reverse) wastes everyone's afternoon.

```sh
git clone https://github.com/Liona-orph/proxima
cd proxima
make ci
```

The first `make ci` is the slow one. After that it is incremental.

## `make ci` is the gate

There is one command:

```sh
make ci
```

It runs, in this order:

| Step | Command |
| --- | --- |
| format | `cargo fmt --all --check` |
| lint | `cargo clippy --workspace --all-targets -- -D warnings` |
| test | `cargo test --workspace` |
| docs | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items` |
| benches compile | `cargo bench --workspace --no-run` |

`.github/workflows/ci.yml` runs exactly these, in the same order. If `make ci`
is green and CI is red, that is a bug in the repository and worth an issue on
its own.

`make help` lists the rest. `make fmt` *checks* formatting; `make fmt-fix`
rewrites the tree.

The bench compile step is there because benchmarks are not built by
`cargo test`, so a refactor can break `benches/distance.rs` or
`benches/index.rs` and nobody notices until someone tries to re-measure a
number in the docs — which is exactly the moment you least want a broken build.

### Running one test

The suite is not small. When you are iterating, run the one test you care
about:

```sh
cargo test -p proxima-index hnsw::tests::same_seed_same_graph
```

`-p <crate>` picks the crate; the rest is a substring filter on the test path.
`-- --nocapture` if you want your `println!`s. `cargo test --workspace` before
you push, always.

## House rules

These are the ones that actually get pull requests sent back.

### No `unsafe`

Every crate carries `#![forbid(unsafe_code)]`, and the workspace lint table
denies `unsafe_code` on top of that. This is deliberate belt-and-braces: if an
exception is ever justified, it has to be spelled out in two places and
argued for in review. A hand-written SIMD kernel is *not* an accepted reason —
the unrolled scalar kernels exist precisely so that the compiler's
auto-vectoriser can do that job safely.

### No `unwrap` outside tests

`clippy::unwrap_used` is `deny` at the workspace level. In library and binary
code, propagate the error with `?` or handle it. `expect` is `allow`ed but
rare, and every use names the invariant that makes it unreachable in a comment
on the same line:

```rust
// Bounds checked by `SlotAllocator::allocate` above; the slot cannot be free.
let vector = store.get(slot).expect("slot was just allocated");
```

Test modules opt out with a module-level attribute:

```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    // ...
}
```

That form is required rather than per-call `#[allow]`s, because it makes every
exception in the repository findable with one grep.

### Docs on every public item

`missing_docs` is `warn` at the workspace level and `-D warnings` in CI, so in
practice a `///` on every public item is mandatory — module, type, function,
field, variant, associated constant. `cargo doc` also runs with
`--document-private-items` and `-D warnings`, so a broken intra-doc link is a
build failure whether the item is public or not.

### Comments explain *why*

This is the part that is easiest to get wrong and hardest to fix later.

- A comment that restates the code is noise. Delete it.
- A comment that says why the code is *this* shape and not the obvious one is
  the reason the file is readable in a year. Write it.
- **State the trade-off and what it cost.** "We build the graph
  single-threaded" is half a sentence. "We build the graph single-threaded
  because parallel neighbour selection changes the insertion order and
  therefore the graph; determinism was worth more here than build throughput,
  and it costs us roughly a factor of N on build time" is the whole one.
- No marketing. Nothing is "blazingly fast", "state of the art" or
  "production-grade". If it is fast, say how fast, measured how.
- No emoji. Anywhere: source, docs, commit messages, PR descriptions.
- Prose in docs is allowed to be long and precise. `clippy::doc_markdown` is
  switched off for exactly that reason.

### Every number in the docs is a number you measured

This is the rule the project is most serious about. If a doc comment,
`README.md`, `ARCHITECTURE.md`, `BENCHMARKS.md` or a PR description contains a
latency, a throughput, a recall, a memory figure or a ratio, it must come from
a run you did, and the reader must be able to reproduce it.

Say what you measured on. Machine, and the exact command:

```sh
# End-to-end: build every index over a generated fixture, then report recall,
# latency and measured memory per vector. Deterministic given --seed.
cargo run --release -p proxima-cli -- bench --vectors 50000 --dim 128 --seed 0x9e3779b9

# Micro-benchmarks, criterion, statistically sound:
cargo bench -p proxima-core     # distance kernels
cargo bench -p proxima-index    # build and search

# What an fsync costs on this machine, which is the number that explains the
# WAL sync policy:
cargo run --release -p proxima-store --example fsync_cost
```

If you change a kernel, a graph parameter or the storage layout, re-run the
affected benchmark and update the numbers in the same pull request. A stale
number is worse than no number, because it is believed.

If you cannot measure it, do not claim it. "Should be faster" is fine in a PR
description as a hypothesis; it is not fine in a doc comment as a fact.

### Determinism

Same seed, same graph. This is a test, not an aspiration
(`proxima-index/src/hnsw.rs`, `hnsw::tests::same_seed_same_graph`).

- Randomness is `rand_chacha::ChaCha8Rng`, always constructed from an explicit
  `u64` seed. `ChaCha8Rng` is used rather than `StdRng` because `StdRng` is
  documented as *not* reproducible across `rand` releases, which would make the
  determinism claim a lie the next time the lockfile moves.
- **`thread_rng` is never used.** Not in fixtures, not in tests, not in
  examples.
- **No iteration over a `HashMap` or `HashSet` in anything that can affect a
  result.** Rust's default hasher is randomly seeded per process, so the order
  differs between runs. Collect and sort, use a `Vec`, or use a `BTreeMap`. This
  applies to tie-breaking in the search path as much as to build order — two
  candidates at the same distance must break the tie the same way every time.
- Anything parallel that changes visit order changes the graph. That is why
  there is no `rayon`.

### Dependencies

The workspace has three runtime dependencies plus one unix-only one:
`thiserror`, `rand` (`=0.8.5`), `rand_chacha` (`=0.3.1`), and `signal-hook`
(`=0.3.17`, used by `proxima-server` for SIGTERM/SIGINT). Dev-dependencies are
`proptest` and `criterion`. That is the whole list.

**Adding a dependency requires a paragraph in the workspace `Cargo.toml`**, in
the dependency-policy comment block, saying what it does, what writing it by
hand would cost, and why that trade lands on the side of taking it. A PR that
adds a dependency without that paragraph will be sent back without further
review. The same applies to a Dependabot PR that widens a pin: the
justification has to still hold.

There is deliberately **no ANN or linear-algebra dependency** — no hnswlib, no
faiss, no usearch, no instant-distance, no ndarray. The interesting parts are
in-tree on purpose; they are what the project is judged on. A PR that replaces
one of them with a crate is not a contribution, it is a fork.

Versions are pinned with `=` where the exact release matters for
reproducibility. Do not relax a pin without saying why in the same paragraph.

## Commits and pull requests

**Commits.** Imperative mood, one logical change each, subject under ~72
characters, no trailing period. The body wraps at 72 and explains *why* — the
diff already says what. A commit that changes a measured number says which
command produced the new one.

The message below is a **shape to copy, not a record of anything**: it describes
a change that was not made and its numbers are invented. It is the only place in
this repository where a number is not a measurement, and it is marked as such
here so that the rule above has no exception a reader has to guess at.

```
Bound the HNSW filtered-search escalation by seen nodes

A selective filter made the ef escalation loop run to the corpus size,
because it counted accepted results and a 1-in-1000 filter accepts almost
none. Escalation now also stops once the traversal has seen ef * 8 nodes.

Recall at k=10 on the 50k/128d fixture is unchanged at 0.982; p99 on the
same filter drops from 41ms to 3.2ms.
(cargo run --release -p proxima-cli -- bench --seed 0x9e3779b9)
```

Do not add `Co-authored-by` for tools, and do not put emoji in the subject.

**Pull requests.** Small and single-purpose. `.github/PULL_REQUEST_TEMPLATE.md`
is short and asks for four things: what changed, why, the trade-off it makes,
and how it was measured if it claims a performance or recall change. Fill it
in — a reviewer who has to reconstruct the trade-off from the diff will guess,
and they will guess wrong.

Rebase rather than merge `main` into your branch; keep the history linear.
Force-pushing your own branch during review is fine and preferred over
"fix review comments" commits, as long as you say so in a comment so the
reviewer knows the range moved.

Draft PRs are welcome for design discussion. Mark them draft, and say what you
want feedback on.

## Reporting things

- Bugs and features: the issue forms in `.github/ISSUE_TEMPLATE/`. The bug form
  asks for the seed and the index configuration, because without them a search
  result cannot be reproduced. For anything that looks like a recall or
  correctness problem, include a `recall_at_k` number or the output of
  `proxima bench` — "the results look wrong" without a recall figure is not
  actionable.
- Security: **do not open an issue.** See [SECURITY.md](SECURITY.md).
- Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md); use GitHub's repository
  reporting tools for a private report.

## Licence

Proxima is Apache-2.0. By contributing you agree that your contribution is
licensed under it, per section 5 of the licence. There is no separate CLA.
