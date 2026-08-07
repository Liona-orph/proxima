# Architecture decision records

One file per decision that is visible in the code and that had a real
alternative. Nygard format: **Context**, **Decision**, **Consequences**, with the
status at the top.

These are not design documents. Each one records a choice that was made, the
option that was rejected, and — the part that matters — **what the choice cost**.
A consequence section that lists only benefits means the decision was not a
decision.

| | decision | status |
|---|---|---|
| [0001](0001-build-from-scratch.md) | Build the engine from scratch rather than wrapping an ANN library | Accepted |
| [0002](0002-hnsw-as-the-primary-index.md) | HNSW as the primary approximate index | Accepted |
| [0003](0003-determinism-over-parallel-build.md) | Determinism over parallel index build | Accepted |
| [0004](0004-filter-during-traversal.md) | Filter during traversal, with `ef` escalation and a scan fallback | Accepted |
| [0005](0005-measured-recall-as-api.md) | Measured recall as a first-class API | Accepted |
| [0006](0006-wal-plus-snapshot-over-lsm.md) | Write-ahead log plus snapshot, not an LSM tree | Accepted |
