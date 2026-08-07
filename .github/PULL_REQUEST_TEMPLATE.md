<!--
Keep it short. The diff says what the code does; this says why, and what it
cost. Delete any section that genuinely does not apply.
-->

## What changed

<!-- One or two sentences. -->

## Why

<!-- The problem this solves. Link the issue if there is one. -->

## Trade-off

<!--
What does this give up? Build time, query latency, memory per vector, recall,
determinism, API surface, on-disk compatibility. "None" is a valid answer, but
it is rarely the true one — say so deliberately rather than by omission.
-->

## How it was measured

<!--
Required if this claims a performance or recall change. Give the command, the
machine, and the before and after. A number without a command to reproduce it
is a claim, not a measurement.

  cargo run --release -p proxima-cli -- bench --vectors 50000 --dim 128 --seed 0x9e3779b9
  cargo bench -p proxima-core
  cargo bench -p proxima-index

| metric      | before  | after   |
| ----------- | ------- | ------- |
| recall@10   |         |         |
| mean query  |         |         |
| p95 query   |         |         |
| build time  |         |         |
| bytes/vec   |         |         |
-->

## Checklist

- [ ] `make ci` is green locally (fmt, clippy `-D warnings`, tests, docs, bench compile).
- [ ] Every new public item has a `///` doc comment.
- [ ] Comments explain *why*, name the trade-off, and contain no marketing and no emoji.
- [ ] Every number added or changed in the docs is one I measured, and the command that produces it is stated.
- [ ] No new runtime dependency — or, if there is one, the workspace `Cargo.toml` has a paragraph justifying it and this PR points at it.
- [ ] No `unwrap` outside tests; any new `expect` names the invariant on the same line.
- [ ] No `unsafe`, no `thread_rng`, and no hash-map iteration order affecting a result.
- [ ] `CHANGELOG.md` updated if this is user-visible.
