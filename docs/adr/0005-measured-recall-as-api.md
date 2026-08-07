# 0005. Measured recall as a first-class API

* **Status:** Accepted
* **Date:** 2026-01
* **Visible in:** `crates/proxima-index/src/recall.rs`; `RecallReport::min` and
  `below`; `proxima bench`; the `min` column in `BENCHMARKS.md`

## Context

An approximate index without a recall number is a claim with no content. The
usual practice is a benchmark: the vendor runs ANN-Benchmarks on SIFT-1M,
publishes a curve, and the user gets results with no idea what they did not get.

That is not transferable. Recall depends on the corpus's intrinsic dimension and
cluster structure, on the query distribution, on `M` and `ef_search`, and — most
sharply — on the filter. An index with 0.99 unfiltered recall on the vendor's
fixture can have 0.3 recall on the user's data under a selective filter, and
nothing in the API will say so. The failure is silent: results come back, they
are ranked, they look plausible.

There is a second problem, with the mean. An index with mean recall 0.95 might
miss half a neighbour on every query, or return perfect results for 90 % of
queries and **nothing at all** for the other 10 %. The second is a much worse
product and the mean cannot tell them apart. This is not hypothetical: at
`M = 16, ef_search = 128` on the benchmark fixture the mean is 0.9945 and the
minimum is **0.00** — one query in two hundred gets nothing right.

## Decision

`recall_at_k` is a **library function**, in `proxima-index`, not in a benchmark.
It takes any `VectorIndex` and a `FlatIndex` over the same data and returns a
`RecallReport`. A user measures their own corpus with their own parameters in
three lines, in their own test suite.

`RecallReport` reports the **distribution**, not just the mean: `min`,
`per_query` for a histogram, `below(threshold)` for the failing fraction, and
`short_results` counted separately from ranking error because the fixes are
different — a short result means the filter's escalation gave up, not that the
ranking was wrong.

`recall_at_k_filtered` applies a filter to both sides, because filtered recall is
the measurement people forget to take and the one that exposes the cliff in
ADR 0004.

`FlatIndex` exists partly to be the ground truth: it is exact by construction, so
"measure against exhaustive search" needs no separate reference implementation.

Every recall figure in this repository's documentation comes from one of these,
and `proxima bench` prints the whole set in one command.

## Consequences

**What it bought.**

* "We think recall is fine" becomes an assertion that fails when it stops being
  true. `proxima-quantize`'s own recall table is asserted by its test suite with
  a small margin, so those numbers cannot silently rot.
* The `min` column caught the filtered-search bug in ADR 0004. Mean recall looked
  acceptable; the minimum did not.
* `BENCHMARKS.md` reports `min` next to `mean` everywhere, and the `min` column
  is the more informative of the two.

**What it cost.**

* **It is expensive.** One exhaustive scan plus one approximate search per query:
  100 queries against a million vectors is 100 million distance evaluations. This
  is a test-time and build-time tool, not a monitoring one, and it is documented
  as such. There is no online recall estimate.
* **It needs a `FlatIndex` holding the same data**, which means a second copy of
  the corpus in memory during measurement. Nothing checks that the two indexes
  agree, because there is no cheap way to — a mismatch shows up as implausibly
  low recall.
* **It makes the honest numbers public.** Bare IVF-PQ recall of 0.6400 on the
  benchmark fixture is in the README. An engine that only published a curve would
  not have had to write that down.
* **Reporting the distribution invites the question "why is `min` zero"**, which
  is the correct question and is more work to answer than publishing a mean.
