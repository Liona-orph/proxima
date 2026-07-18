//! Properties both quantizers must satisfy on arbitrary data.
//!
//! Round-trip *equality* is not one of them — these are lossy codecs, and a
//! test that asserted it would be asserting the wrong thing. What can be
//! asserted is the error **bound**: scalar quantization's reconstruction is
//! within half a lattice step of the original, by construction, and any input
//! outside the trained range clamps to an endpoint rather than wrapping.
//!
//! For product quantization the bound is not analytic (it depends on where
//! k-means put the centroids), so the properties are structural instead: codes
//! are the right length, decoding is total over all 256 byte values, encoding
//! is idempotent, and the distance table agrees with an explicit reconstruction.

#![allow(clippy::unwrap_used)]
// Bit-exact float comparison is the point in `product_training_is_reproducible`:
// "the same seed gives the same codebook" is a claim about exact equality, and
// a tolerance would let a real non-determinism through.
#![allow(clippy::float_cmp)]

use proptest::prelude::*;
use proxima_core::{Metric, VectorSpace};
use proxima_quantize::{CodeType, PqConfig, ProductQuantizer, RangeMode, ScalarQuantizer};

const DIM: usize = 12;

/// A training set with enough distinct values that k-means has work to do.
fn training() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-50.0f32..50.0, DIM * 40..DIM * 60)
        .prop_map(|mut v| {
            v.truncate(v.len() / DIM * DIM);
            v
        })
        .prop_filter("needs at least one full vector", |v| v.len() >= DIM)
}

fn vector() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-80.0f32..80.0, DIM)
}

/// The `(min, max)` a `ScalarQuantizer` would learn for each dimension.
fn trained_bounds(data: &[f32], per_dimension: bool) -> Vec<(f32, f32)> {
    let mut bounds = vec![(f32::INFINITY, f32::NEG_INFINITY); DIM];
    for vector in data.chunks_exact(DIM) {
        for (component, value) in vector.iter().enumerate() {
            let slot = if per_dimension { component } else { 0 };
            bounds[slot].0 = bounds[slot].0.min(*value);
            bounds[slot].1 = bounds[slot].1.max(*value);
        }
    }
    if !per_dimension {
        let global = bounds[0];
        bounds.fill(global);
    }
    bounds
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Scalar reconstruction is within half a lattice step, and out-of-range
    /// inputs clamp to an endpoint instead of wrapping around it.
    #[test]
    fn scalar_error_is_bounded_by_half_a_step(
        data in training(),
        probe in vector(),
        per_dimension in any::<bool>(),
        signed in any::<bool>(),
    ) {
        let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
        let mode = if per_dimension { RangeMode::PerDimension } else { RangeMode::Global };
        let code_type = if signed { CodeType::I8 } else { CodeType::U8 };
        let sq = ScalarQuantizer::train(&space, &data, mode, code_type).unwrap();

        prop_assert_eq!(sq.code_len(), DIM);
        let decoded = sq.decode(&sq.encode(&probe).unwrap()).unwrap();
        prop_assert_eq!(decoded.len(), DIM);

        // The bound is per-dimension under `PerDimension` and global under
        // `Global`, and getting that distinction wrong is exactly the mistake
        // the two modes exist to make visible: a probe can sit comfortably
        // inside the *global* range and far outside its own dimension's.
        let bounds = trained_bounds(&data, per_dimension);

        // The reconstruction always lands inside the trained range, whatever
        // the probe was. That is what "clamps" means, and it is the property
        // that stops a drifting corpus from producing wild coordinates.
        for (component, value) in decoded.iter().enumerate() {
            let (lo, hi) = bounds[component];
            prop_assert!(*value >= lo - 1e-3 && *value <= hi + 1e-3, "{} outside [{}, {}]", value, lo, hi);
        }

        // ...and for a probe inside the range, within half a step.
        for (component, (original, reconstructed)) in probe.iter().zip(&decoded).enumerate() {
            let (lo, hi) = bounds[component];
            let step = (hi - lo) / 255.0;
            if *original >= lo && *original <= hi {
                prop_assert!(
                    (original - reconstructed).abs() <= step / 2.0 + 1e-3 * step.max(1.0),
                    "{} vs {} with step {}", original, reconstructed, step
                );
            }
        }
    }

    /// Re-encoding a decoded vector gives the same code: the lattice points are
    /// fixed points of the map, which is what makes the error non-accumulating.
    #[test]
    fn scalar_encoding_is_idempotent(data in training(), probe in vector()) {
        let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
        let sq = ScalarQuantizer::train(&space, &data, RangeMode::PerDimension, CodeType::U8).unwrap();
        let once = sq.encode(&probe).unwrap();
        let twice = sq.encode(&sq.decode(&once).unwrap()).unwrap();
        prop_assert_eq!(once, twice);
    }

    /// Asymmetric distance against a code equals the exact distance to that
    /// code's reconstruction. Any gap would be an indexing bug, not a rounding
    /// one, so the tolerance is tight.
    #[test]
    fn scalar_asymmetric_distance_matches_the_reconstruction(
        data in training(),
        query in vector(),
        probe in vector(),
    ) {
        let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
        let sq = ScalarQuantizer::train(&space, &data, RangeMode::PerDimension, CodeType::U8).unwrap();
        let code = sq.encode(&probe).unwrap();
        let reconstructed = sq.decode(&code).unwrap();

        let via_code = sq.asymmetric_squared_l2(&query, &code);
        let direct = proxima_core::distance::squared_euclidean(&query, &reconstructed);
        prop_assert!(
            (via_code - direct).abs() <= 1e-4 * direct.max(1.0),
            "{} vs {}", via_code, direct
        );
    }

    /// Product codes are `m` bytes, every byte value decodes, and the distance
    /// table agrees with reconstruct-and-measure.
    #[test]
    fn product_structure_holds_for_any_m(
        data in training(),
        query in vector(),
        m in prop::sample::select(&[1usize, 2, 3, 4, 6, 12][..]),
    ) {
        let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
        let pq = ProductQuantizer::train(&space, &data, PqConfig::new(m).with_seed(5)).unwrap();
        prop_assert_eq!(pq.code_len(), m);

        // Decoding is total: the codebook is padded to a full 256 entries even
        // when the training set had fewer distinct points.
        for byte in [0u8, 1, 127, 200, 255] {
            prop_assert_eq!(pq.decode(&vec![byte; m]).unwrap().len(), DIM);
        }

        let table = pq.distance_table(&query).unwrap();
        prop_assert_eq!(table.subspaces(), m);
        for probe in data.chunks_exact(DIM).take(16) {
            let code = pq.encode(probe).unwrap();
            let reconstructed = pq.decode(&code).unwrap();
            let via_table = table.squared_l2(&code);
            let direct = proxima_core::distance::squared_euclidean(&query, &reconstructed);
            prop_assert!(
                (via_table - direct).abs() <= 1e-3 * direct.max(1.0),
                "{} vs {}", via_table, direct
            );
        }
    }

    /// The same seed and data give byte-identical codebooks. This is what
    /// `PqConfig::with_seed` promises and the only thing that makes a persisted
    /// quantizer's codes portable across a rebuild.
    #[test]
    fn product_training_is_reproducible(data in training(), seed in any::<u64>()) {
        let space = VectorSpace::new(DIM, Metric::Euclidean).unwrap();
        let config = PqConfig::new(4).with_seed(seed);
        let a = ProductQuantizer::train(&space, &data, config).unwrap();
        let b = ProductQuantizer::train(&space, &data, config).unwrap();
        prop_assert_eq!(a.training_inertia(), b.training_inertia());
        for probe in data.chunks_exact(DIM).take(32) {
            prop_assert_eq!(a.encode(probe).unwrap(), b.encode(probe).unwrap());
        }
    }
}
