//! The Prometheus exposition endpoint.
//!
//! # What is exported and what is not
//!
//! Counters and gauges only. There is **no histogram**, and that is the one
//! thing an operator will miss, so it is worth being direct about it: a
//! latency histogram needs bucket boundaries chosen in advance and a bucket
//! array updated on every request, and the useful version of it — one that can
//! answer "what is p99" rather than "what is the mean" — is a real amount of
//! machinery. What is exported instead is
//! `proxima_search_duration_microseconds_sum` and
//! `proxima_searches_total`, which give a rate and a mean and nothing about the
//! tail. The mean latency of a search engine is the least interesting number
//! about it, and this is a gap, not a design.
//!
//! Per-collection series are labelled with `collection="..."`, which makes the
//! cardinality proportional to the number of collections. That is fine for the
//! tens a single node holds and would not be for millions of tenants.
//!
//! # Format
//!
//! The text exposition format, version 0.0.4: a `# HELP` line, a `# TYPE` line,
//! then samples. Metric names carry their unit as a suffix and counters end in
//! `_total`, per the Prometheus naming conventions, because a dashboard built
//! on `rate()` breaks quietly if a counter is not recognisable as one.

use std::fmt::Write as _;

use crate::api::Api;

/// Render the whole exposition body.
pub fn render(api: &Api) -> String {
    let mut out = String::with_capacity(4096);
    let (requests, errors, searches, search_micros) = api.totals();

    metric(
        &mut out,
        "proxima_up",
        "gauge",
        "1 when the process is serving.",
        &[(String::new(), 1.0)],
    );
    metric(
        &mut out,
        "proxima_uptime_seconds",
        "gauge",
        "Seconds since the process started.",
        &[(String::new(), api.uptime().as_secs_f64())],
    );
    metric(
        &mut out,
        "proxima_requests_total",
        "counter",
        "HTTP requests handled, successful or not.",
        &[(String::new(), requests as f64)],
    );
    metric(
        &mut out,
        "proxima_request_errors_total",
        "counter",
        "Requests answered with 4xx or 5xx, excluding 404.",
        &[(String::new(), errors as f64)],
    );
    metric(
        &mut out,
        "proxima_searches_total",
        "counter",
        "Searches served across all collections.",
        &[(String::new(), searches as f64)],
    );
    metric(
        &mut out,
        "proxima_search_duration_microseconds_sum",
        "counter",
        "Total time spent inside search, in microseconds. Divide by proxima_searches_total for the mean; there is no histogram, so there is no percentile.",
        &[(String::new(), search_micros as f64)],
    );

    let collections = api.collections();
    metric(
        &mut out,
        "proxima_collections",
        "gauge",
        "Open collections.",
        &[(String::new(), collections.len() as f64)],
    );

    // One pass over the collections, building every per-collection series
    // together, because `describe` takes each collection's read lock and doing
    // it once per metric would take it six times.
    let mut vectors = Vec::new();
    let mut deleted = Vec::new();
    let mut bytes = Vec::new();
    let mut bytes_per_vector = Vec::new();
    let mut wal_records = Vec::new();
    let mut wal_bytes = Vec::new();
    let mut upserts = Vec::new();
    let mut deletes = Vec::new();
    let mut snapshots = Vec::new();
    let mut per_collection_searches = Vec::new();
    let mut poisoned = Vec::new();

    for (name, collection) in &collections {
        let description = collection.describe();
        let label = format!(
            "collection=\"{}\",index=\"{}\"",
            escape(name),
            description.index
        );
        vectors.push((label.clone(), description.stats.len as f64));
        deleted.push((label.clone(), description.stats.deleted as f64));
        bytes.push((label.clone(), description.stats.total_bytes() as f64));
        bytes_per_vector.push((
            label.clone(),
            f64::from(description.stats.bytes_per_vector()),
        ));
        wal_records.push((label.clone(), description.wal_records as f64));
        wal_bytes.push((label.clone(), description.wal_bytes as f64));
        upserts.push((label.clone(), description.counters.upserts as f64));
        deletes.push((label.clone(), description.counters.deletes as f64));
        snapshots.push((label.clone(), description.counters.snapshots as f64));
        per_collection_searches.push((label.clone(), description.counters.searches as f64));
        poisoned.push((label, f64::from(u8::from(collection.is_poisoned()))));
    }

    metric(
        &mut out,
        "proxima_collection_vectors",
        "gauge",
        "Live vectors.",
        &vectors,
    );
    metric(
        &mut out,
        "proxima_collection_deleted_vectors",
        "gauge",
        "Tombstoned vectors still occupying memory. Alert when this exceeds a fifth of the live count.",
        &deleted,
    );
    metric(
        &mut out,
        "proxima_collection_bytes",
        "gauge",
        "Heap bytes held by vectors and index structures.",
        &bytes,
    );
    metric(
        &mut out,
        "proxima_collection_bytes_per_vector",
        "gauge",
        "Heap bytes per live vector. Compare against dim * 4.",
        &bytes_per_vector,
    );
    metric(
        &mut out,
        "proxima_collection_wal_records",
        "gauge",
        "Records in the write-ahead log since the last snapshot; this bounds recovery time.",
        &wal_records,
    );
    metric(
        &mut out,
        "proxima_collection_wal_bytes",
        "gauge",
        "Bytes in the write-ahead log.",
        &wal_bytes,
    );
    metric(
        &mut out,
        "proxima_collection_upserts_total",
        "counter",
        "Vectors upserted.",
        &upserts,
    );
    metric(
        &mut out,
        "proxima_collection_deletes_total",
        "counter",
        "Vectors removed.",
        &deletes,
    );
    metric(
        &mut out,
        "proxima_collection_snapshots_total",
        "counter",
        "Snapshots written.",
        &snapshots,
    );
    metric(
        &mut out,
        "proxima_collection_searches_total",
        "counter",
        "Searches served.",
        &per_collection_searches,
    );
    metric(
        &mut out,
        "proxima_collection_lock_poisoned",
        "gauge",
        "1 if a panic was hit while holding this collection's write lock. Restart the process.",
        &poisoned,
    );

    out
}

/// Write one metric family: `# HELP`, `# TYPE`, then its samples.
///
/// A family with no samples is emitted with its `HELP` and `TYPE` and nothing
/// else, so that a dashboard querying it before any collection exists gets an
/// empty result rather than an unknown-metric error.
fn metric(out: &mut String, name: &str, kind: &str, help: &str, samples: &[(String, f64)]) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
    for (labels, value) in samples {
        if labels.is_empty() {
            let _ = writeln!(out, "{name} {}", format_value(*value));
        } else {
            let _ = writeln!(out, "{name}{{{labels}}} {}", format_value(*value));
        }
    }
}

/// Format a sample value the way Prometheus expects.
///
/// Integral values are written without a decimal point purely for readability
/// of `curl /metrics`; the parser accepts either.
fn format_value(value: f64) -> String {
    if !value.is_finite() {
        return "NaN".to_owned();
    }
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// Escape a label value.
///
/// Collection names are already restricted to `[A-Za-z0-9_-]` by the create
/// handler, so this can never fire today. It is here because the restriction is
/// enforced in a different file, and a metrics endpoint that can emit an
/// unparseable line the moment that changes is a trap.
fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn every_family_has_help_and_type_lines() {
        let mut out = String::new();
        metric(&mut out, "x_total", "counter", "Some help.", &[]);
        assert_eq!(out, "# HELP x_total Some help.\n# TYPE x_total counter\n");
    }

    #[test]
    fn labelled_and_unlabelled_samples_both_format() {
        let mut out = String::new();
        metric(
            &mut out,
            "x",
            "gauge",
            "h",
            &[(String::new(), 1.0), ("collection=\"a\"".to_owned(), 2.5)],
        );
        assert!(out.contains("\nx 1\n"));
        assert!(out.contains("\nx{collection=\"a\"} 2.5\n"));
    }

    #[test]
    fn non_finite_values_become_nan_rather_than_inf() {
        // `inf` is legal in the exposition format but means something specific;
        // a division by zero in a mean is not it.
        assert_eq!(format_value(f64::INFINITY), "NaN");
        assert_eq!(format_value(f64::NAN), "NaN");
        assert_eq!(format_value(3.0), "3");
        assert_eq!(format_value(-0.5), "-0.5");
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(escape(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(escape("a\nb"), "a\\nb");
    }
}
