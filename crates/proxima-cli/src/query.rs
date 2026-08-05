//! `proxima query`: search a collection from the shell.
//!
//! Opens the collection **in this process** rather than talking to a running
//! server. That is a deliberate choice with a real consequence: it takes the
//! same directory lock the server would, so it cannot be run against a live
//! server's data directory. In exchange it works with no server running, which
//! is the situation you are usually in when you want to know whether an index
//! is any good — and "is the data actually on disk" is the question a shell
//! query is for, which a request to a process holding everything in memory
//! cannot answer.

use std::path::PathBuf;

use proxima_index::AcceptAll;
use proxima_server::json::{parse, Json};
use proxima_server::MetadataFilter;
use proxima_store::{Collection, CollectionOptions, Manifest};

use crate::args::Args;

const FLAGS: &[&str] = &[
    "data",
    "collection",
    "vector",
    "random",
    "seed",
    "k",
    "filter",
    "json",
];

/// Run one query and print the results.
pub(crate) fn run(args: &Args) -> Result<(), String> {
    args.reject_unknown(FLAGS)?;

    let data = PathBuf::from(args.get("data").unwrap_or("./proxima-data"));
    let name = args
        .get("collection")
        .ok_or("--collection is required; try 'proxima help'")?;
    let directory = data.join(name);
    if !directory.join("manifest").exists() {
        return Err(format!("no collection '{name}' in {}", data.display()));
    }

    let manifest =
        Manifest::read(&directory).map_err(|err| format!("cannot read {name}: {err}"))?;
    let dim = manifest.space.dim();
    let (collection, report) = Collection::open(
        &directory,
        manifest.name.clone(),
        dim,
        manifest.space.metric(),
        manifest.spec,
        CollectionOptions::default(),
    )
    .map_err(|err| format!("cannot open {name}: {err}"))?;

    let query = build_query(args, dim)?;
    let k: usize = args.value("k", 10)?;
    let filter = match args.get("filter") {
        Some(text) => {
            let value = parse(text).map_err(|err| format!("--filter is not valid JSON: {err}"))?;
            Some(MetadataFilter::parse(&value).map_err(|err| format!("--filter: {err}"))?)
        }
        None => None,
    };

    let started = std::time::Instant::now();
    let hits = match &filter {
        Some(filter) => collection.search_with_metadata(&query, k, filter),
        None => collection.search_with_metadata(&query, k, &AcceptAll),
    }
    .map_err(|err| format!("search failed: {err}"))?;
    let elapsed = started.elapsed();

    let metric = collection.space().metric();
    if args.flag("json") {
        let results: Vec<Json> = hits
            .iter()
            .map(|(hit, metadata)| {
                Json::object([
                    ("id", Json::number(hit.id.get() as f64)),
                    ("score", Json::number(hit.distance)),
                    (
                        "distance",
                        Json::number(metric.to_true_distance(hit.distance)),
                    ),
                    ("metadata", metadata_json(metadata)),
                ])
            })
            .collect();
        println!(
            "{}",
            Json::object([
                ("results", Json::Array(results)),
                ("took_micros", Json::count(elapsed.as_micros() as usize)),
            ])
        );
        return Ok(());
    }

    println!(
        "collection '{name}': {} vectors, {dim}d/{metric}, replayed {} log records",
        collection.len(),
        report.replayed
    );
    if hits.is_empty() {
        println!("no results");
        return Ok(());
    }
    println!("{:<12} {:>14} {:>14}  metadata", "id", "score", "distance");
    for (hit, metadata) in &hits {
        let fields: Vec<String> = {
            let mut fields: Vec<String> = metadata
                .iter()
                .map(|(key, value)| format!("{key}={}", render(value)))
                .collect();
            // Sorted, because `Metadata` is a `HashMap` and an output that
            // reorders itself between runs cannot be diffed.
            fields.sort();
            fields
        };
        println!(
            "{:<12} {:>14.6} {:>14.6}  {}",
            hit.id.get(),
            hit.distance,
            metric.to_true_distance(hit.distance),
            fields.join(" ")
        );
    }
    println!("{} results in {elapsed:.1?}", hits.len());
    Ok(())
}

/// Build the query vector from `--vector` or `--random`.
fn build_query(args: &Args, dim: usize) -> Result<Vec<f32>, String> {
    if let Some(text) = args.get("vector") {
        let values: Result<Vec<f32>, String> = text
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<f32>()
                    .map_err(|_| format!("--vector: '{}' is not a number", part.trim()))
            })
            .collect();
        let values = values?;
        if values.len() != dim {
            return Err(format!(
                "--vector has {} components, the collection is {dim}-dimensional",
                values.len()
            ));
        }
        return Ok(values);
    }
    if args.flag("random") {
        let seed: u64 = args.u64_value("seed", 1)?;
        return Ok(pseudo_random(dim, seed));
    }
    Err("give either --vector a,b,c or --random".to_owned())
}

/// A deterministic pseudo-random vector.
///
/// A small xorshift rather than the `rand` dependency the rest of the workspace
/// uses: this is a convenience for eyeballing an index, not a fixture anything
/// is measured against, and it wants no more guarantee than "the same seed gives
/// the same vector".
fn pseudo_random(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to roughly [-1, 1) using the high bits, which are the
            // well-mixed ones in a xorshift.
            (state >> 40) as f32 / 8_388_608.0 - 1.0
        })
        .collect()
}

fn render(value: &proxima_core::MetadataValue) -> String {
    match value {
        proxima_core::MetadataValue::Text(text) => text.clone(),
        proxima_core::MetadataValue::Int(number) => number.to_string(),
        proxima_core::MetadataValue::Float(number) => number.to_string(),
        proxima_core::MetadataValue::Bool(flag) => flag.to_string(),
    }
}

fn metadata_json(metadata: &proxima_core::Metadata) -> Json {
    Json::Object(
        metadata
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    proxima_core::MetadataValue::Text(text) => Json::string(text),
                    proxima_core::MetadataValue::Int(number) => Json::number(*number as f64),
                    proxima_core::MetadataValue::Float(number) => Json::Number(*number),
                    proxima_core::MetadataValue::Bool(flag) => Json::Bool(*flag),
                };
                (key.to_owned(), value)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use proxima_core::{Metadata, Metric, VectorId};
    use proxima_store::IndexSpec;

    use super::*;

    fn args(line: &str) -> Args {
        Args::parse(line.split_whitespace().map(str::to_owned)).unwrap()
    }

    fn fixture(name: &str) -> PathBuf {
        let data = std::env::temp_dir().join(format!(
            "proxima-cli-query-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&data);
        let (collection, _) = Collection::open(
            data.join("docs"),
            "docs",
            4,
            Metric::Euclidean,
            IndexSpec::Flat,
            CollectionOptions::default(),
        )
        .unwrap();
        for id in 0..20u64 {
            let value = id as f32;
            collection
                .upsert(
                    VectorId::new(id),
                    &[value, value * 0.5, -value, 1.0],
                    Metadata::new()
                        .with("lang", if id % 2 == 0 { "en" } else { "fr" })
                        .with("n", id as i64),
                )
                .unwrap();
        }
        data
    }

    #[test]
    fn a_query_against_a_real_collection_prints_results() {
        let data = fixture("basic");
        let line = format!(
            "query --data {} --collection docs --vector 3,1.5,-3,1 --k 3",
            data.display()
        );
        run(&args(&line)).unwrap();
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn filters_json_output_and_random_queries_all_work() {
        let data = fixture("options");
        let base = format!("query --data {} --collection docs", data.display());
        run(&args(&format!("{base} --random --k 2"))).unwrap();
        run(&args(&format!("{base} --random --seed 9 --json"))).unwrap();
        run(&args(&format!(
            "{base} --vector 1,0.5,-1,1 --filter {{\"lang\":\"en\"}}"
        )))
        .unwrap();
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn bad_input_is_reported_rather_than_guessed_at() {
        let data = fixture("errors");
        let base = format!("query --data {} --collection docs", data.display());
        for (line, fragment) in [
            (base.clone(), "--vector"),
            (format!("{base} --vector 1,2"), "4-dimensional"),
            (format!("{base} --vector 1,x,3,4"), "not a number"),
            (
                format!("{base} --vector 1,2,3,4 --filter not-json"),
                "valid JSON",
            ),
            (
                format!("{base} --vector 1,2,3,4 --filter {{\"a\":{{\"zz\":1}}}}"),
                "unknown operator",
            ),
            (
                format!(
                    "query --data {} --collection missing --random",
                    data.display()
                ),
                "no collection",
            ),
        ] {
            let err = run(&args(&line)).unwrap_err();
            assert!(err.contains(fragment), "{line}: {err}");
        }
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn random_queries_are_reproducible_and_in_range() {
        let a = pseudo_random(64, 5);
        assert_eq!(a, pseudo_random(64, 5));
        assert_ne!(a, pseudo_random(64, 6));
        assert!(a.iter().all(|v| v.is_finite() && (-1.0..1.0).contains(v)));
    }
}
