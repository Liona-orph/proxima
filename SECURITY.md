# Security Policy

## Supported versions

| Version | Supported | Notes |
| --- | --- | --- |
| 0.1.x | Yes | Current line. Fixes ship in the next patch release. |
| < 0.1 | No | Nothing was released before 0.1.0. |

Proxima is pre-1.0. There is **no LTS branch and no backport promise**: security
fixes land on `main` and go out in the next `0.1.x` patch release, and the
supported version is whatever the latest patch is. Breaking changes are allowed
between minor versions while the version starts with a zero, including in the
on-disk format. If you are running Proxima somewhere that matters, pin a version
and read `CHANGELOG.md` before upgrading.

## Reporting a vulnerability

Email **security@proxima-search.dev**. Do not open a public issue, a discussion,
or a pull request that describes the problem.

Include, as far as you have it:

- what you did, exactly enough for us to do it too — the request, the input file,
  the configuration;
- the version or commit, and `rustc -Vv`;
- what happened, and what you expected;
- your assessment of impact, and whether you have told anyone else.

**What to expect, and when.**

| | |
| --- | --- |
| Acknowledgement of your email | within 3 working days |
| Initial assessment: is it a vulnerability, and how bad | within 10 working days |
| Fix, or a plan with dates if the fix is not simple | within 30 days of the assessment |
| Public disclosure | at the fix, or 90 days after your report, whichever is first |

We work to **90-day coordinated disclosure**. If we have not shipped a fix by
day 90, you are free to publish; we would rather you did than that the issue sat
quiet. If a fix ships earlier, we publish then, and we will tell you before we
do. If you want credit, say so and how you want to be named; if you want to stay
anonymous, that is fine too. We do not run a bug bounty and cannot pay.

If we conclude that what you reported is *not* a vulnerability, we will say so
and why, in enough detail for you to disagree with us.

## Threat model

This is the part worth reading before you deploy anything. Proxima is a
single-node engine written from first principles, and being honest about what
that means is more useful than a security policy that implies guarantees the
code does not make.

### Proxima has no authentication and no TLS

`proxima serve` speaks plain HTTP and accepts every request it receives. There
are no accounts, no API keys, no tokens, no authorisation checks, and no
transport encryption. It binds `127.0.0.1:7700` by default, and that default is
the right one.

**Do not expose `proxima serve` to an untrusted network.** Anyone who can reach
the port can read every vector, read and write every collection, and delete
them. If you need it reachable from elsewhere, put it behind something that does
authentication and TLS — a reverse proxy, a service mesh, an SSH tunnel — and
keep Proxima itself bound to a loopback or private interface. If you run
`--address 0.0.0.0:7700` (as the Docker image's default command does, because a
container has nowhere else to bind), the container's network is the security
boundary and it is your job to draw it.

The same goes for the operational endpoints: `GET /health` and `GET /metrics`
are unauthenticated. `/metrics` exposes collection names, collection sizes,
request counts and latency distributions. That is not secret exactly, but it is
information, and it is public to anyone who can reach the port.

### The HTTP and JSON parsers are hand-written

Say it plainly: `proxima-server` parses HTTP/1.1 and JSON with code written for
this project, in `crates/proxima-server/src/http.rs` and
`crates/proxima-server/src/json.rs`. It does not use hyper, or serde_json, or
anything else that has been fuzzed by strangers for a decade. Parsers on an
untrusted network edge are where bugs live, and a hand-written one has had
fewer eyes on it than the alternatives. Treat that as the largest single risk
in the system.

What limits the damage:

- **`#![forbid(unsafe_code)]` in every crate**, and `unsafe_code = "deny"` at
  the workspace level. All indexing is bounds-checked by the compiler. The
  realistic worst case for a parser bug is therefore a **panic (a dropped
  connection, or a crashed process) or resource exhaustion — not memory
  corruption, not arbitrary code execution**. That is a real difference, and it
  is why the rule exists.
- **The request line is capped** at 8 KiB (`http::MAX_REQUEST_LINE`), so a
  client cannot make the server buffer an unbounded URL.
- **Headers are capped**, both in total bytes (32 KiB,
  `http::MAX_HEADER_BYTES`) and in count (100, `http::MAX_HEADERS`).
- **The body is capped** by `Content-Length` against a configured limit
  (`ApiConfig::max_body_bytes`), and an over-large declared body is refused
  before it is read.
- **JSON nesting is capped** at 32 (`json::MAX_DEPTH`). A recursive-descent
  parser without a depth limit is a stack overflow waiting for `[[[[[...`, and
  a stack overflow in Rust aborts rather than unwinding.
- **Concurrent connections are capped** (`--max-connections`, default 256).

None of that makes a parser bug harmless. A panic in a request handler still
costs you that request, and a bug that makes the server allocate or loop is
still a denial of service. If you find one, report it as above.

### Metadata filters

A search request can carry a metadata filter, and that filter is attacker-
supplied data evaluated inside the graph traversal. It is worth being precise
about what it can do.

The filter language is a **fixed conjunction of field predicates**: a JSON
object where every key names a metadata field and every value is a condition on
it, with operators `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in` and `exists`. All
of them must hold. There is no OR, no nesting, no expression tree, and **no
evaluation of user-supplied code** — no scripting, no regular expressions, no
SQL, no string interpolation into a query anywhere. A filter compiles to a
fixed list of hash lookups and comparisons. Types are not coerced, so a filter
cannot widen its own match set by accident.

The cost of a filter is bounded by the number of fields in it and the number of
nodes the traversal visits, and the traversal itself is bounded by `ef` and its
escalation cap. The realistic abuse is therefore a slow query — a highly
selective filter makes HNSW search degrade toward a scan, which is a documented
property of filtered ANN search and not a bug. If your deployment lets untrusted
callers choose filters, rate-limit them upstream.

### Collection names and the filesystem

A collection maps to a directory under the data directory, so a collection name
is the only user-controlled string that reaches a path. Names are restricted to
`[A-Za-z0-9_-]` at creation (`api::validate_name`); everything else is rejected.
That is an allowlist, not an attempt to filter `..` out of a string, and it is
what prevents path traversal. There is no other user-controlled path component.

### Snapshot and WAL files are trusted input

The write-ahead log and the snapshots in the data directory are treated as
**trusted**. They are CRC32C-framed and checksummed, and the loader detects a
torn tail and truncates it, and rejects an unknown format version — but those
checks are there to survive a crash or a bad disk, **not** to defend against a
file that someone crafted on purpose. A hostile snapshot can plausibly make the
loader allocate a great deal of memory or panic.

**Loading a snapshot or a WAL from an untrusted source is not a supported
operation.** Treat the data directory with the same care as the binary: if an
attacker can write to it, they own the process. Restoring a backup you produced
is fine. Importing a collection someone emailed you is not, and a report of the
form "I crafted a snapshot file and crashed the loader" will be closed as
working as documented — unless it demonstrates something beyond a crash, which
we do want to hear about.

### Out of scope

We will not treat these as vulnerabilities:

- Reaching `proxima serve` over an untrusted network and doing anything at all.
  There is no authentication; that is documented, not a bug.
- The absence of TLS, authentication, authorisation, multi-tenancy, per-
  collection access control, audit logging, or encryption at rest. Proxima does
  not have these features. Missing features are not vulnerabilities; a feature
  request is welcome as an issue.
- Information disclosure through `/metrics` or `/health` on a port you chose to
  expose.
- Crashes or memory growth from a deliberately corrupted snapshot or WAL, per
  the section above.
- Denial of service from a legitimately expensive query — a huge `k`, a huge
  `ef_search`, a very selective filter, a large batch insert. Sizing and rate
  limiting are the operator's job.
- Resource exhaustion when the caps above are configured away (a very large
  `max_body_bytes`, a very large `--max-connections`).
- Findings from an automated scanner with no demonstrated impact, and reports
  that consist of a tool's output.
- Vulnerabilities in a dependency with no path to exploiting them through
  Proxima. Tell us anyway — we will bump the pin — but it is not an incident.
- Anything requiring an attacker who already has local filesystem or process
  access on the host.

If you are not sure which side of the line something falls on, mail
security@proxima-search.dev and ask. We would rather read a report that turns
out to be out of scope than not read one that was not.
