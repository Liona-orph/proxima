//! The README's 60-second quickstart, as a program that compiles and runs.
//!
//! It is here, at the workspace root, rather than pasted into the README alone,
//! because a quickstart that does not compile is worse than no quickstart: it
//! costs the reader the time it takes to find out. `make ci` builds and runs it,
//! so the README's first code block cannot rot without turning CI red.
//!
//! ```text
//! cargo run --release --example quickstart
//! ```

#![forbid(unsafe_code)]

use proxima_core::{Metadata, Metric, VectorId};
use proxima_index::{recall_at_k, FlatIndex, HnswConfig, HnswIndex, VectorIndex};

fn main() -> Result<(), proxima_core::ProximaError> {
    // A graph index, and a flat one to measure it against.
    let space = proxima_core::VectorSpace::new(64, Metric::Cosine)?;
    let mut index = HnswIndex::new(space.clone(), HnswConfig::new().with_seed(42))?;
    let mut truth = FlatIndex::new(space);

    // 5 000 vectors with cluster structure, and every fiftieth one as a query.
    let mut queries = Vec::new();
    for id in 0..5_000u64 {
        let cluster = (id % 50) as f32;
        let vector: Vec<f32> = (0..64)
            .map(|d| (cluster * 0.7 + d as f32 * 0.11).sin() + id as f32 * 0.00002 * d as f32)
            .collect();
        let metadata = Metadata::new().with("lang", if id % 3 == 0 { "en" } else { "fr" });
        index.insert(VectorId::new(id), &vector, metadata.clone())?;
        truth.insert(VectorId::new(id), &vector, metadata)?;
        if id % 50 == 0 {
            queries.extend_from_slice(&vector);
        }
    }

    // Search, with a filter that runs *during* traversal, so `k` results really
    // are the `k` nearest accepted vectors and not the accepted ones it walked past.
    let english = |_id: VectorId, meta: &Metadata| meta.text("lang") == Some("en");
    let hits = index.search(&queries[..64], 10, &english)?;
    println!(
        "{} hits, nearest {} at {:.4}",
        hits.len(),
        hits[0].id,
        hits[0].distance
    );

    // And the number that makes the previous line mean something.
    let report = recall_at_k(&index, &truth, &queries, 10)?;
    println!("{report}");
    assert!(report.mean > 0.99, "{report}");
    Ok(())
}
