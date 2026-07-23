//! Flat versus HNSW: build time, query latency and recall.
//!
//! Produces every number in `BENCHMARKS.md`. The recall table is printed by the
//! harness itself rather than measured by criterion, because recall is not a
//! timing and pretending otherwise would put a confidence interval on a number
//! that has none.
//!
//! ```text
//! cargo bench -p proxima-index
//! ```

#![allow(clippy::unwrap_used)]
// `criterion_group!` expands to an undocumented `pub fn`.
#![allow(missing_docs)]

use std::hint::black_box;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use proxima_core::{Metadata, Metric, VectorId, VectorSpace};
use proxima_index::{recall_at_k, FlatIndex, HnswConfig, HnswIndex, VectorIndex};

const DIM: usize = 128;
const CLUSTERS: usize = 64;

/// splitmix64, so the fixture is byte-identical on every machine and run.
struct Prng(u64);

impl Prng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / f32::from(1u16 << 12) - 1.0
    }

    fn next_normal(&mut self) -> f32 {
        (0..6).map(|_| self.next_f32()).sum::<f32>() / 2.0
    }
}

fn corpus(count: usize, seed: u64) -> Vec<f32> {
    let mut rng = Prng(seed);
    let centres: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|_| (0..DIM).map(|_| rng.next_normal() * 4.0).collect())
        .collect();
    let mut out = Vec::with_capacity(count * DIM);
    for i in 0..count {
        for value in &centres[i % centres.len()] {
            out.push(value + rng.next_normal());
        }
    }
    out
}

/// Queries are perturbed corpus points, which is what a real query looks like
/// and what makes recall@10 a measurement of the index rather than of the
/// fixture. See `tests/hnsw.rs` for the longer version of that argument.
fn queries_near(data: &[f32], count: usize, seed: u64) -> Vec<f32> {
    let mut rng = Prng(seed);
    let total = data.len() / DIM;
    let stride = (total / count).max(1);
    let mut out = Vec::with_capacity(count * DIM);
    for q in 0..count {
        let source = (q * stride) % total;
        for value in &data[source * DIM..(source + 1) * DIM] {
            out.push(value + rng.next_normal() * 0.25);
        }
    }
    out
}

fn space() -> VectorSpace {
    VectorSpace::new(DIM, Metric::Euclidean).unwrap()
}

fn build_flat(data: &[f32]) -> FlatIndex {
    let mut index = FlatIndex::with_capacity(space(), data.len() / DIM);
    for (i, vector) in data.chunks_exact(DIM).enumerate() {
        index
            .insert(VectorId::new(i as u64), vector, Metadata::new())
            .unwrap();
    }
    index
}

fn build_hnsw(data: &[f32], config: HnswConfig) -> HnswIndex {
    let mut index = HnswIndex::with_capacity(space(), config, data.len() / DIM).unwrap();
    for (i, vector) in data.chunks_exact(DIM).enumerate() {
        index
            .insert(VectorId::new(i as u64), vector, Metadata::new())
            .unwrap();
    }
    index
}

/// Build time. Criterion is the wrong tool for a multi-second operation with a
/// warm-up, so this is measured directly and reported once.
fn build_times(_: &mut Criterion) {
    println!("\n## build time (dim {DIM})\n");
    println!("| vectors | flat | hnsw M=16 ef_c=200 | ratio |");
    println!("|---|--:|--:|--:|");
    for count in [1_000usize, 10_000, 50_000] {
        let data = corpus(count, 0xB1D5);
        let started = Instant::now();
        let flat = build_flat(&data);
        let flat_time = started.elapsed();

        let started = Instant::now();
        let hnsw = build_hnsw(&data, HnswConfig::new().with_seed(1));
        let hnsw_time = started.elapsed();

        println!(
            "| {count} | {:.1?} | {:.1?} | {:.0}x |",
            flat_time,
            hnsw_time,
            hnsw_time.as_secs_f64() / flat_time.as_secs_f64()
        );
        black_box((flat.len(), hnsw.len()));
    }
    println!();
}

/// Query latency across corpus sizes: the crossover measurement.
fn query_latency(c: &mut Criterion) {
    for count in [1_000usize, 10_000, 50_000] {
        let data = corpus(count, 0xB1D5);
        let queries = queries_near(&data, 64, 0x9E1D);
        let flat = build_flat(&data);
        let hnsw = build_hnsw(&data, HnswConfig::new().with_seed(1));

        let mut group = c.benchmark_group(format!("query_n{count}"));
        group.bench_with_input(BenchmarkId::new("flat", count), &count, |b, _| {
            let mut next = 0usize;
            b.iter(|| {
                let query = &queries[next * DIM..(next + 1) * DIM];
                next = (next + 1) % 64;
                black_box(flat.search_all(black_box(query), 10).unwrap())
            });
        });
        for ef in [32usize, 64, 128, 256] {
            group.bench_with_input(
                BenchmarkId::new(format!("hnsw_ef{ef}"), count),
                &ef,
                |b, _| {
                    let mut next = 0usize;
                    b.iter(|| {
                        let query = &queries[next * DIM..(next + 1) * DIM];
                        next = (next + 1) % 64;
                        black_box(
                            hnsw.search_with_ef(
                                black_box(query),
                                10,
                                ef,
                                &proxima_index::AcceptAll,
                            )
                            .unwrap(),
                        )
                    });
                },
            );
        }
        group.finish();
    }
}

/// Insert throughput, amortised over a full build.
fn insert_throughput(c: &mut Criterion) {
    let data = corpus(5_000, 0xB1D5);
    let mut group = c.benchmark_group("insert_5000");
    group.sample_size(10);
    group.bench_function("flat", |b| b.iter(|| black_box(build_flat(&data).len())));
    for (m, ef) in [(8usize, 100usize), (16, 200), (32, 200)] {
        group.bench_function(format!("hnsw_m{m}_ef{ef}"), |b| {
            b.iter(|| {
                black_box(
                    build_hnsw(
                        &data,
                        HnswConfig::new()
                            .with_m(m)
                            .with_ef_construction(ef)
                            .with_seed(1),
                    )
                    .len(),
                )
            });
        });
    }
    group.finish();
}

/// The recall/parameter table. Printed, not timed.
fn recall_table(_: &mut Criterion) {
    let data = corpus(50_000, 0xB1D5);
    let queries = queries_near(&data, 200, 0x9E1D);
    let flat = build_flat(&data);

    println!("\n## recall@10, 50 000 vectors of dimension {DIM}, 200 queries\n");
    println!("| M | ef_construction | ef_search | mean recall | min recall | graph MiB |");
    println!("|--:|--:|--:|--:|--:|--:|");

    for (m, ef_construction) in [(8usize, 100usize), (16, 200), (32, 200)] {
        let hnsw = build_hnsw(
            &data,
            HnswConfig::new()
                .with_m(m)
                .with_ef_construction(ef_construction)
                .with_seed(1),
        );
        let graph_mib = hnsw.stats().graph_bytes as f64 / (1024.0 * 1024.0);
        for ef_search in [16usize, 32, 64, 128, 256] {
            let adapter = FixedEf(&hnsw, ef_search);
            let report = recall_at_k(&adapter, &flat, &queries, 10).unwrap();
            println!(
                "| {m} | {ef_construction} | {ef_search} | {:.4} | {:.2} | {graph_mib:.1} |",
                report.mean, report.min
            );
        }
    }
    println!();

    let stats = flat.stats();
    println!(
        "Vector array: {:.1} MiB for {} vectors ({} bytes each).\n",
        stats.vector_bytes as f64 / (1024.0 * 1024.0),
        stats.len,
        DIM * 4
    );
}

/// Read-only adapter that pins `ef_search`, so the recall harness can sweep it
/// without rebuilding the index.
struct FixedEf<'a>(&'a HnswIndex, usize);

impl VectorIndex for FixedEf<'_> {
    fn insert(
        &mut self,
        _id: VectorId,
        _vector: &[f32],
        _metadata: Metadata,
    ) -> proxima_core::Result<()> {
        unreachable!("read-only adapter")
    }

    fn remove(&mut self, _id: VectorId) -> proxima_core::Result<()> {
        unreachable!("read-only adapter")
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: &dyn proxima_index::Filter,
    ) -> proxima_core::Result<Vec<proxima_core::SearchResult>> {
        self.0.search_with_ef(query, k, self.1, filter)
    }

    fn contains(&self, id: VectorId) -> bool {
        self.0.contains(id)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn stats(&self) -> proxima_index::IndexStats {
        self.0.stats()
    }
}

criterion_group!(
    benches,
    build_times,
    recall_table,
    query_latency,
    insert_throughput
);
criterion_main!(benches);
