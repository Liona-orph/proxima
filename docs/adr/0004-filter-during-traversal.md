# 0004. Filter during traversal, with `ef` escalation and a scan fallback

* **Status:** Accepted
* **Date:** 2026-01
* **Visible in:** the `observe` callback in `HnswIndex::search_layer`;
  `HnswIndex::search_prepared`; `HnswConfig::filter_ef_limit` and
  `exhaustive_filter_fallback`; `crates/proxima-server/src/filter.rs`

## Context

"Find the 10 nearest vectors where `lang = "en"`" is the query every vector
database is actually asked. There are three ways to answer it.

**Post-filter.** Search for `k`, drop the rejects. Returns fewer than `k`
whenever the filter rejects anything, and there is **no `k` you can search for
that fixes it in general** — the pass rate of the top `k` depends on the query.

**Pre-filter.** Compute the accepted id set first, then scan it. Correct, and
`O(matching)` — which is `O(n)` for a filter that accepts half the corpus, i.e.
worse than the index for the common case.

**Filter during traversal.** Evaluate the predicate as the graph is walked. This
is what Proxima does, and it introduces a subtler failure than either of the
others.

That failure is worth stating precisely, because an implementation that stops as
soon as it has `k` accepted results *looks* correct: it returns a full `k` every
time. Under a 1-in-40 filter, an `ef = 64` traversal examines a few hundred nodes
and about ten pass, so those ten results are merely **the ten accepted nodes the
walk happened to pass**, not the ten nearest. Measured on 10 000 vectors of
dimension 64 in 40 clusters at `ef_search = 64`: recall@10 of **0.3880**, with a
full `k` results on every query.

## Decision

Filters run during traversal, via a callback that fires for **every** node whose
distance is computed — including nodes that never enter the `ef` list — so an
accepted node is recorded the moment it is examined.

The stopping condition is not "`k` results are filled". The traversal must also
have **seen at least `ef` accepted nodes**, which makes the filtered search as
selective among accepted vectors as an unfiltered search is among all of them.
When it has not:

1. `ef` **escalates geometrically**, doubling to `filter_ef_limit` (1 024).
   Geometric so total work is at most twice the final attempt's.
2. Below that, an **exhaustive filtered scan**.

The filter language exposed over HTTP is a conjunction of field predicates with
no OR and no nesting.

## Consequences

**What it bought.** Recall@10 under a 1-in-40 filter goes from 0.3880 to
**1.0000** with zero short results, and stays at 1.0000 at 1-in-2 and unfiltered.
`recall_at_k_filtered` exists so a user can check the same thing on their own
corpus and their own predicate.

**What it cost.**

* **A latency cliff, not a slope.** A query whose filter crosses the selectivity
  threshold stops being a 53 µs graph search and becomes an `O(n)` scan — 1 049 µs
  on the benchmark corpus, and proportionally worse at scale. Two queries with
  the same shape and different literals can differ by 20x. Nothing in the API
  predicts which side of the line a filter falls on.
* **Escalation wastes the attempts below the final one.** A search that ends up
  at `ef = 512` has already run at 64, 128 and 256. Geometric growth bounds that
  at 2x, which is the reason it is geometric.
* **The filter is on the hot path**, called tens of thousands of times per query,
  which is why the DSL is weak: a conjunction compiles to a fixed list of hash
  lookups with no allocation, and an arbitrary boolean expression tree would be
  an interpreter there. A user who needs a disjunction issues two searches and
  merges — which is what a query planner would do anyway, and it makes the cost
  visible.
* **The fallback can be turned off** (`with_exhaustive_filter_fallback(false)`),
  and then results can be short of `k`. That is occasionally the right trade for
  a latency-critical service and it should be a decision, so it is a flag rather
  than a hard-coded behaviour.

**What would reverse it.** A filtered-search design that avoids the cliff —
per-attribute sub-graphs, or an inverted index consulted to choose between graph
and scan up front. Both are substantially more machinery and both trade memory
for it.
