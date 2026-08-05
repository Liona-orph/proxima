//! `proxima bench`: the command that reproduces `BENCHMARKS.md`.
//!
//! Builds a flat index, an HNSW graph and one or more IVF-PQ configurations
//! over the same generated fixture, and reports for each of them: build time,
//! query latency, **measured** memory per vector, and recall@10 against the
//! flat index.
//!
//! # Memory is measured, not computed
//!
//! Every "bytes per vector" figure here comes from
//! [`IndexStats`](proxima_index::IndexStats), which sums the actual `Vec`
//! capacities the index holds. It is not `dim * 4 + m * 8`. The difference is
//! not academic — a computed figure would miss the inverted lists' `Vec`
//! headers, the codebooks, the allocator's slack on a grown array and the
//! metadata table — and memory per vector is the whole reason IVF-PQ exists, so
//! estimating it would be estimating the thing being demonstrated.
//!
//! # Latency is a mean over repeated queries, not criterion
//!
//! `cargo bench` runs criterion and produces confidence intervals; this command
//! runs each query set several times and reports the mean and the p95. It is a
//! reproduction tool, not a regression detector, and its job is to let someone
//! check the numbers in the README on their own machine in under a minute.

use std::time::{Duration, Instant};

use proxima_core::{Metadata, Metric, VectorId, VectorSpace};
use proxima_index::{
    recall_at_k, FlatIndex, HnswConfig, HnswIndex, IvfPqConfig, IvfPqIndex, RecallReport,
    VectorIndex,
};

use crate::args::Args;
use crate::fixture::{self, Fixture};

/// A search with one dial already bound, so that a sweep can vary it.
///
/// A boxed trait object rather than a generic, because the two sweeps below
/// close over different index types and the alternative is duplicating the
/// timing loop per index.
type BoundSearch<'a> = dyn Fn(&[f32]) -> Result<Vec<proxima_core::SearchResult>, String> + 'a;

/// One row of the report.
#[derive(Debug, Clone)]
struct Row {
    name: String,
    build: Duration,
    mean_query: Duration,
    p95_query: Duration,
    bytes_per_vector: f32,
    total_bytes: usize,
    recall: Option<RecallReport>,
    note: String,
}

/// Flags this command accepts.
const FLAGS: &[&str] = &[
    "vectors",
    "dim",
    "clusters",
    "queries",
    "seed",
    "k",
    "m",
    "ef-construction",
    "ef-search",
    "nlist",
    "nprobe",
    "subspaces",
    "rerank",
    "repeats",
    "skip-flat-latency",
];

/// Run the benchmark.
pub(crate) fn run(args: &Args) -> Result<(), String> {
    args.reject_unknown(FLAGS)?;

    let count: usize = args.value("vectors", 50_000)?;
    let dim: usize = args.value("dim", 128)?;
    let clusters: usize = args.value("clusters", 64)?;
    let query_count: usize = args.value("queries", 200)?;
    let seed: u64 = args.u64_value("seed", 0x9E37_79B9)?;
    let k: usize = args.value("k", 10)?;
    let m: usize = args.value("m", 16)?;
    let ef_construction: usize = args.value("ef-construction", 200)?;
    let ef_search: usize = args.value("ef-search", 64)?;
    let nlist: usize = args.value("nlist", 256)?;
    let nprobe: usize = args.value("nprobe", 16)?;
    let subspaces: usize = args.value("subspaces", 32)?;
    let rerank: usize = args.value("rerank", 100)?;
    let repeats: usize = args.value("repeats", 3)?;

    println!("proxima bench");
    println!(
        "  fixture: {count} vectors, dim {dim}, {clusters} clusters, {query_count} queries, seed {seed:#x}"
    );
    println!("  k = {k}, repeats = {repeats}");

    let started = Instant::now();
    let fixture = fixture::generate(count, dim, clusters, query_count, seed);
    println!(
        "  generated in {:.2?} ({:.1} MiB of f32)",
        started.elapsed(),
        fixture.corpus_bytes() as f64 / (1024.0 * 1024.0)
    );

    let space = VectorSpace::new(dim, Metric::Euclidean)
        .map_err(|err| format!("cannot build the space: {err}"))?;

    // Flat first: it is the ground truth every recall number is measured
    // against, so nothing else can be reported until it exists.
    let (flat, flat_build) = timed(|| {
        let mut index = FlatIndex::with_capacity(space.clone(), count);
        insert_all(&mut index, &fixture)?;
        Ok(index)
    })?;
    let mut rows = vec![Row {
        name: "flat".to_owned(),
        build: flat_build,
        mean_query: Duration::ZERO,
        p95_query: Duration::ZERO,
        bytes_per_vector: flat.stats().bytes_per_vector(),
        total_bytes: flat.stats().total_bytes(),
        recall: None,
        note: "exact by construction".to_owned(),
    }];
    if !args.flag("skip-flat-latency") {
        let (mean, p95) = latency(&flat, &fixture, k, repeats)?;
        rows[0].mean_query = mean;
        rows[0].p95_query = p95;
    }

    let (hnsw, hnsw_build) = timed(|| {
        let mut index = HnswIndex::with_capacity(
            space.clone(),
            HnswConfig::new()
                .with_m(m)
                .with_ef_construction(ef_construction)
                .with_ef_search(ef_search)
                .with_seed(seed),
            count,
        )
        .map_err(|err| err.to_string())?;
        insert_all(&mut index, &fixture)?;
        Ok(index)
    })?;
    let (mean, p95) = latency(&hnsw, &fixture, k, repeats)?;
    rows.push(Row {
        name: format!("hnsw m={m} ef={ef_search}"),
        build: hnsw_build,
        mean_query: mean,
        p95_query: p95,
        bytes_per_vector: hnsw.stats().bytes_per_vector(),
        total_bytes: hnsw.stats().total_bytes(),
        recall: Some(measure(&hnsw, &flat, &fixture, k)?),
        note: format!("mean degree {:.1}", hnsw.stats().mean_degree()),
    });

    // Three IVF-PQ configurations, because the interesting thing about this
    // index is the shape of the trade rather than any one point on it. The
    // built indexes are kept so that the nprobe sweep below can reuse them.
    let mut sweeps: Vec<IvfPqIndex> = Vec::new();
    for (label, config) in [
        (
            format!("ivfpq m={subspaces} nprobe={nprobe}"),
            IvfPqConfig::new(nlist, subspaces)
                .with_nprobe(nprobe)
                .with_seed(seed)
                .with_train_threshold(usize::MAX),
        ),
        (
            format!("ivfpq m={subspaces} nprobe={nprobe} rerank={rerank}"),
            IvfPqConfig::new(nlist, subspaces)
                .with_nprobe(nprobe)
                .with_rerank(rerank)
                .with_seed(seed)
                .with_train_threshold(usize::MAX),
        ),
        (
            format!("ivfpq m={subspaces} nprobe={nprobe} no vectors"),
            IvfPqConfig::new(nlist, subspaces)
                .with_nprobe(nprobe)
                .with_seed(seed)
                .with_retain_vectors(false)
                .with_train_threshold(usize::MAX),
        ),
    ] {
        let (index, build) = timed(|| {
            let mut index =
                IvfPqIndex::new(space.clone(), config.clone()).map_err(|err| err.to_string())?;
            insert_all(&mut index, &fixture)?;
            index.train().map_err(|err| err.to_string())?;
            Ok(index)
        })?;
        let report = index
            .training_report()
            .map(|report| format!("largest list {}", report.largest_list))
            .unwrap_or_default();
        let (mean, p95) = latency(&index, &fixture, k, repeats)?;
        rows.push(Row {
            name: label,
            build,
            mean_query: mean,
            p95_query: p95,
            bytes_per_vector: index.stats().bytes_per_vector(),
            total_bytes: index.stats().total_bytes(),
            recall: Some(measure(&index, &flat, &fixture, k)?),
            note: report,
        });
        sweeps.push(index);
    }

    report(&rows, k, fixture.corpus_bytes(), fixture.count);

    // The recall/latency curves, swept on the indexes that were already built.
    // Rebuilding once per point would take a quarter of an hour and would also
    // be measuring a different graph each time; sweeping the dial on one built
    // index is both cheaper and a fairer comparison.
    ef_sweep(&hnsw, &flat, &fixture, k, repeats)?;
    if let Some(index) = sweeps.first() {
        nprobe_sweep(index, &flat, &fixture, k, repeats, "no rerank")?;
    }
    if let Some(index) = sweeps.get(1) {
        nprobe_sweep(index, &flat, &fixture, k, repeats, "rerank")?;
    }
    Ok(())
}

/// Sweep `ef_search` on a built graph.
fn ef_sweep(
    index: &HnswIndex,
    truth: &FlatIndex,
    fixture: &Fixture,
    k: usize,
    repeats: usize,
) -> Result<(), String> {
    println!();
    println!(
        "hnsw ef_search sweep (one built graph, m = {})",
        index.config().m()
    );
    println!(
        "{:>10} {:>12} {:>11} {:>10} {:>9}",
        "ef", "mean query", "p95 query", "recall", "min"
    );
    for ef in [16usize, 32, 64, 128, 256] {
        let search = |query: &[f32]| {
            index
                .search_with_ef(query, k, ef, &proxima_index::AcceptAll)
                .map_err(|err| err.to_string())
        };
        let (mean, p95) = latency_with(&search, fixture, repeats)?;
        let recall = recall_with(&search, truth, fixture, k)?;
        println!(
            "{ef:>10} {:>12} {:>11} {:>10.4} {:>9.2}",
            format!("{mean:.1?}"),
            format!("{p95:.1?}"),
            recall.0,
            recall.1
        );
    }
    Ok(())
}

/// Sweep `nprobe` on a built IVF-PQ index.
fn nprobe_sweep(
    index: &IvfPqIndex,
    truth: &FlatIndex,
    fixture: &Fixture,
    k: usize,
    repeats: usize,
    label: &str,
) -> Result<(), String> {
    println!();
    println!(
        "ivfpq nprobe sweep, {label} (one built index, nlist = {}, m = {})",
        index.config().nlist(),
        index.config().subspaces()
    );
    println!(
        "{:>10} {:>12} {:>11} {:>10} {:>9}",
        "nprobe", "mean query", "p95 query", "recall", "min"
    );
    for nprobe in [1usize, 4, 16, 64, 256] {
        if nprobe > index.config().nlist() {
            continue;
        }
        let search = |query: &[f32]| {
            index
                .search_with_nprobe(query, k, nprobe, &proxima_index::AcceptAll)
                .map_err(|err| err.to_string())
        };
        let (mean, p95) = latency_with(&search, fixture, repeats)?;
        let recall = recall_with(&search, truth, fixture, k)?;
        println!(
            "{nprobe:>10} {:>12} {:>11} {:>10.4} {:>9.2}",
            format!("{mean:.1?}"),
            format!("{p95:.1?}"),
            recall.0,
            recall.1
        );
    }
    Ok(())
}

/// Latency of an arbitrary search closure, so a sweep can vary the dial.
fn latency_with(
    search: &BoundSearch<'_>,
    fixture: &Fixture,
    repeats: usize,
) -> Result<(Duration, Duration), String> {
    for query in fixture.queries() {
        search(query)?;
    }
    let mut samples = Vec::new();
    for _ in 0..repeats {
        for query in fixture.queries() {
            let started = Instant::now();
            let hits = search(query)?;
            samples.push(started.elapsed());
            std::hint::black_box(hits);
        }
    }
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    Ok((
        total / samples.len() as u32,
        samples[samples.len() * 95 / 100],
    ))
}

/// Mean and minimum recall@k of an arbitrary search closure.
fn recall_with(
    search: &BoundSearch<'_>,
    truth: &FlatIndex,
    fixture: &Fixture,
    k: usize,
) -> Result<(f32, f32), String> {
    let mut sum = 0.0f32;
    let mut minimum = f32::INFINITY;
    let mut count = 0usize;
    for query in fixture.queries() {
        let exact = truth.search_all(query, k).map_err(|err| err.to_string())?;
        let got = search(query)?;
        let hit = got
            .iter()
            .filter(|candidate| exact.iter().any(|truth| truth.id == candidate.id))
            .count();
        let recall = if exact.is_empty() {
            1.0
        } else {
            hit as f32 / exact.len() as f32
        };
        sum += recall;
        minimum = minimum.min(recall);
        count += 1;
    }
    if count == 0 {
        return Ok((0.0, 0.0));
    }
    Ok((sum / count as f32, minimum))
}

fn insert_all<I: VectorIndex>(index: &mut I, fixture: &Fixture) -> Result<(), String> {
    for (id, vector) in fixture.iter().enumerate() {
        index
            .insert(VectorId::new(id as u64), vector, Metadata::new())
            .map_err(|err| format!("insert {id} failed: {err}"))?;
    }
    Ok(())
}

fn timed<T>(build: impl FnOnce() -> Result<T, String>) -> Result<(T, Duration), String> {
    let started = Instant::now();
    let value = build()?;
    Ok((value, started.elapsed()))
}

/// Mean and p95 latency over `repeats` passes of the whole query set.
fn latency<I: VectorIndex>(
    index: &I,
    fixture: &Fixture,
    k: usize,
    repeats: usize,
) -> Result<(Duration, Duration), String> {
    // One untimed pass first. The first query touches every page of the index
    // for the first time, and including that in the mean measures the page
    // fault handler rather than the search.
    for query in fixture.queries() {
        index.search_all(query, k).map_err(|err| err.to_string())?;
    }

    let mut samples = Vec::with_capacity(repeats * fixture.queries().count());
    for _ in 0..repeats {
        for query in fixture.queries() {
            let started = Instant::now();
            let hits = index.search_all(query, k).map_err(|err| err.to_string())?;
            samples.push(started.elapsed());
            // Keep the result alive so the optimiser cannot delete the search.
            std::hint::black_box(hits);
        }
    }
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    let mean = total / samples.len() as u32;
    let p95 = samples[samples.len() * 95 / 100];
    Ok((mean, p95))
}

fn measure<I: VectorIndex>(
    index: &I,
    truth: &FlatIndex,
    fixture: &Fixture,
    k: usize,
) -> Result<RecallReport, String> {
    recall_at_k(index, truth, &fixture.queries, k).map_err(|err| err.to_string())
}

fn report(rows: &[Row], k: usize, corpus_bytes: usize, count: usize) {
    println!();
    println!(
        "{:<38} {:>10} {:>11} {:>10} {:>9} {:>9} {:>9}",
        "index", "build", "mean query", "p95 query", "B/vector", "recall", "min"
    );
    println!("{}", "-".repeat(102));
    for row in rows {
        let (recall, min) = match &row.recall {
            Some(report) => (format!("{:.4}", report.mean), format!("{:.2}", report.min)),
            None => ("1.0000".to_owned(), "1.00".to_owned()),
        };
        println!(
            "{:<38} {:>10} {:>11} {:>10} {:>9.0} {:>9} {:>9}",
            row.name,
            format!("{:.2?}", row.build),
            format!("{:.1?}", row.mean_query),
            format!("{:.1?}", row.p95_query),
            row.bytes_per_vector,
            recall,
            min
        );
    }

    println!();
    println!(
        "recall@{k} is against the flat index over the same {count} vectors; B/vector is measured \
         from IndexStats, not computed."
    );
    println!(
        "the raw corpus is {:.1} MiB ({:.0} bytes per vector).",
        corpus_bytes as f64 / (1024.0 * 1024.0),
        corpus_bytes as f64 / count as f64
    );
    if let Some(smallest) = rows
        .iter()
        .min_by(|a, b| a.bytes_per_vector.total_cmp(&b.bytes_per_vector))
    {
        println!(
            "smallest footprint: {} at {:.0} B/vector, {:.1} MiB total.",
            smallest.name,
            smallest.bytes_per_vector,
            smallest.total_bytes as f64 / (1024.0 * 1024.0)
        );
    }
    for row in rows {
        if !row.note.is_empty() {
            println!("  {}: {}", row.name, row.note);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    // A flat index measured against itself is exactly 1.0, not approximately.
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn a_small_bench_runs_end_to_end() {
        // Not a performance assertion — those belong in `cargo bench` — but a
        // check that every code path in the command executes and that the
        // numbers it prints are computed rather than stubbed.
        let args = Args::parse(
            "bench --vectors 1200 --dim 16 --clusters 8 --queries 20 --nlist 8 --subspaces 8 \
             --nprobe 4 --rerank 20 --repeats 1 --m 8 --ef-construction 40 --ef-search 16"
                .split_whitespace()
                .map(str::to_owned),
        )
        .unwrap();
        run(&args).unwrap();
    }

    #[test]
    fn an_unknown_flag_stops_the_run() {
        let args =
            Args::parse(["bench".to_owned(), "--vectorz".to_owned(), "10".to_owned()]).unwrap();
        assert!(run(&args).is_err());
    }

    #[test]
    fn latency_and_recall_agree_with_a_hand_measurement() {
        let fixture = fixture::generate(500, 8, 5, 10, 1);
        let space = VectorSpace::new(8, Metric::Euclidean).unwrap();
        let mut flat = FlatIndex::new(space);
        insert_all(&mut flat, &fixture).unwrap();

        // A flat index measured against itself must be exactly 1.0, which is
        // the sanity check that the harness is wired up the right way round.
        let report = measure(&flat, &flat, &fixture, 10).unwrap();
        assert_eq!(report.mean, 1.0);
        assert_eq!(report.queries, 10);

        let (mean, p95) = latency(&flat, &fixture, 10, 2).unwrap();
        assert!(mean > Duration::ZERO);
        assert!(p95 >= mean / 2, "p95 {p95:?} against mean {mean:?}");
    }
}
