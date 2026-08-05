//! `proxima serve`: run the HTTP API.

use std::net::SocketAddr;
use std::sync::Arc;

use proxima_server::api::ApiConfig;
use proxima_server::server::ServerConfig;
use proxima_server::{Api, Server};
use proxima_store::SyncPolicy;

use crate::args::Args;

const FLAGS: &[&str] = &[
    "data",
    "address",
    "sync",
    "snapshot-after",
    "max-connections",
    "no-signals",
];

/// Start the server and block until it shuts down.
pub(crate) fn run(args: &Args) -> Result<(), String> {
    args.reject_unknown(FLAGS)?;

    let data = args.get("data").unwrap_or("./proxima-data").to_owned();
    let address: SocketAddr = args
        .get("address")
        .unwrap_or("127.0.0.1:7700")
        .parse()
        .map_err(|err| format!("--address is not a host:port pair: {err}"))?;
    let sync = match args.get("sync") {
        Some(name) => SyncPolicy::from_name(name)
            .ok_or_else(|| format!("--sync '{name}' is not always, never or every_N"))?,
        None => SyncPolicy::Always,
    };

    let api = Arc::new(
        Api::open(ApiConfig {
            data_directory: data.clone().into(),
            sync_policy: sync,
            snapshot_after_records: args.value("snapshot-after", 100_000u64)?,
            ..ApiConfig::default()
        })
        .map_err(|err| format!("cannot open the data directory: {err}"))?,
    );

    let server = Server::bind(
        ServerConfig {
            address,
            max_connections: args.value("max-connections", 256usize)?,
            ..ServerConfig::default()
        },
        Arc::clone(&api),
    )
    .map_err(|err| format!("cannot bind {address}: {err}"))?;

    if !args.flag("no-signals") {
        server
            .install_signal_handlers()
            .map_err(|err| format!("cannot install signal handlers: {err}"))?;
    }

    let bound = server
        .local_address()
        .map_err(|err| format!("cannot read the bound address: {err}"))?;
    let collections = api.collections();
    println!(
        "proxima {} listening on http://{bound}",
        env!("CARGO_PKG_VERSION")
    );
    println!("  data:  {data}");
    println!("  fsync: {sync}");
    if collections.is_empty() {
        println!("  collections: none yet");
    } else {
        for (name, collection) in &collections {
            let description = collection.describe();
            println!(
                "  collection '{name}': {} vectors, {}d/{}, {} index",
                description.stats.len, description.dim, description.metric, description.index
            );
        }
    }
    println!("  shutdown: SIGTERM, SIGINT, or POST /admin/shutdown");

    let stranded = server
        .serve()
        .map_err(|err| format!("the server stopped with an error: {err}"))?;
    if stranded > 0 {
        // Reported rather than swallowed: a non-zero count means the drain
        // timeout expired with requests still running, and their writes may not
        // be in the log.
        eprintln!(
            "proxima: {stranded} connection(s) were still in flight when the drain timed out"
        );
    }
    println!("proxima: stopped cleanly, write-ahead logs flushed");
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn args(line: &str) -> Args {
        Args::parse(line.split_whitespace().map(str::to_owned)).unwrap()
    }

    #[test]
    fn bad_flags_are_rejected_before_anything_is_bound() {
        assert!(run(&args("serve --addres 1.2.3.4:1")).is_err());
        assert!(run(&args("serve --address not-an-address")).is_err());
        assert!(run(&args("serve --sync sometimes")).is_err());
        assert!(run(&args("serve --snapshot-after lots")).is_err());
    }
}
