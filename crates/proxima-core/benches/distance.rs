//! Scalar versus unrolled distance kernels.
//!
//! The unrolling in `proxima_core::distance` is justified by an argument about
//! loop-carried dependencies. This benchmark is where that argument is checked
//! rather than believed. Run with `cargo bench -p proxima-core`.

#![allow(clippy::unwrap_used)]
// `criterion_group!` expands to an undocumented `pub fn`.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use proxima_core::distance::{
    cosine_scalar, cosine_unrolled, dot_scalar, dot_unrolled, manhattan_scalar, manhattan_unrolled,
    squared_euclidean_scalar, squared_euclidean_unrolled,
};

/// Deterministic pseudo-random vectors: splitmix64 mapped into `[-1, 1)`.
///
/// Not `rand`, because `proxima-core` deliberately has no runtime dependency on
/// it and a benchmark is not a good enough reason to add a dev-dependency that
/// the crate's own tests do not need.
fn vectors(dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / f32::from(1u16 << 11) - 1.0
    };
    let a = (0..dim).map(|_| next()).collect();
    let b = (0..dim).map(|_| next()).collect();
    (a, b)
}

fn kernels(c: &mut Criterion) {
    // 96 and 768 bracket the range that matters: 96 is a small sentence
    // embedding where per-call overhead is visible, 768 is the size of the
    // models most people actually deploy. 3072 checks that nothing falls off a
    // cliff once the operands stop fitting in L1.
    for dim in [96usize, 768, 3072] {
        let (a, b) = vectors(dim, 0x5EED);
        let mut group = c.benchmark_group(format!("dim_{dim}"));
        // One "element" is one f32 pair, so throughput reads as dimensions per
        // second, which is the number that transfers between dimensions.
        group.throughput(Throughput::Elements(dim as u64));

        macro_rules! pair {
            ($name:literal, $scalar:expr, $unrolled:expr) => {
                group.bench_with_input(
                    BenchmarkId::new(concat!($name, "_scalar"), dim),
                    &dim,
                    |bencher, _| {
                        bencher
                            .iter(|| $scalar(std::hint::black_box(&a), std::hint::black_box(&b)));
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(concat!($name, "_unrolled"), dim),
                    &dim,
                    |bencher, _| {
                        bencher
                            .iter(|| $unrolled(std::hint::black_box(&a), std::hint::black_box(&b)));
                    },
                );
            };
        }

        pair!(
            "squared_euclidean",
            squared_euclidean_scalar,
            squared_euclidean_unrolled
        );
        pair!("dot", dot_scalar, dot_unrolled);
        pair!("manhattan", manhattan_scalar, manhattan_unrolled);
        pair!("cosine", cosine_scalar, cosine_unrolled);

        group.finish();
    }
}

criterion_group!(benches, kernels);
criterion_main!(benches);
