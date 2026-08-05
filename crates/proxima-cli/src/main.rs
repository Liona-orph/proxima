//! `proxima`: serve, bench and query.
//!
//! Three subcommands, and each of them exists for a specific reason.
//!
//! * `serve` runs the HTTP API over a data directory.
//! * `bench` builds every index over a generated fixture and reports build
//!   time, latency, measured memory per vector and recall. **This is the
//!   command that reproduces `BENCHMARKS.md` and the numbers in the README.** A
//!   benchmark nobody can rerun is a claim, not a measurement.
//! * `query` searches a collection from the shell, so that "is the index
//!   actually returning anything sensible" does not require writing a client.

#![forbid(unsafe_code)]

mod args;
mod bench;
mod fixture;
mod query;
mod serve;

use args::Args;

/// Usage text. Hand-written, because the argument parser is a hundred lines and
/// generating this from it would be more machinery than the text is long.
const USAGE: &str = "\
proxima — a vector search engine

USAGE:
    proxima <command> [options]

COMMANDS:
    serve     run the HTTP/JSON API
    bench     build every index over a generated dataset and report recall,
              latency and measured memory per vector
    query     search a collection from the command line
    help      this text

SERVE:
    --data <dir>              data directory              [./proxima-data]
    --address <host:port>     listen address              [127.0.0.1:7700]
    --sync <policy>           always | never | every_N    [always]
    --snapshot-after <n>      snapshot every n WAL records[100000]
    --max-connections <n>     concurrent connections      [256]
    --no-signals              do not install SIGTERM/SIGINT handlers

BENCH:
    --vectors <n>             corpus size                 [50000]
    --dim <n>                 dimension                   [128]
    --clusters <n>            Gaussian clusters           [64]
    --queries <n>             query count                 [200]
    --seed <n>                fixture and index seed      [0x9e3779b9]
    --k <n>                   neighbours per query        [10]
    --m <n>                   HNSW M                      [16]
    --ef-construction <n>     HNSW ef_construction        [200]
    --ef-search <n>           HNSW ef_search              [64]
    --nlist <n>               IVF-PQ coarse centroids     [256]
    --nprobe <n>              IVF-PQ lists probed         [16]
    --subspaces <n>           IVF-PQ bytes per code       [32]
    --rerank <n>              IVF-PQ exact rerank depth   [100]
    --repeats <n>             latency passes              [3]
    --skip-flat-latency       skip the O(n) baseline scan

QUERY:
    --data <dir>              data directory              [./proxima-data]
    --collection <name>       collection to search        (required)
    --vector <a,b,c>          query vector, comma separated
    --random                  use a deterministic pseudo-random query instead
    --seed <n>                seed for --random           [1]
    --k <n>                   neighbours to return        [10]
    --filter <json>           metadata filter, e.g. '{\"lang\":\"en\"}'
    --json                    print the raw JSON response

EXAMPLES:
    proxima bench --vectors 50000 --dim 128
    proxima serve --data /var/lib/proxima --address 0.0.0.0:7700
    proxima query --data ./proxima-data --collection docs --random --k 5
";

fn main() -> std::process::ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let args = match Args::parse(arguments) {
        Ok(args) => args,
        Err(message) => return fail(&message),
    };

    let result = match args.command.as_str() {
        "serve" => serve::run(&args),
        "bench" => bench::run(&args),
        "query" => query::run(&args),
        "help" | "" => {
            print!("{USAGE}");
            return std::process::ExitCode::SUCCESS;
        }
        other => Err(format!("unknown command '{other}'; try 'proxima help'")),
    };

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => fail(&message),
    }
}

/// Report a failure on stderr and exit non-zero.
///
/// Errors go to stderr and results go to stdout, without exception, so that
/// `proxima query --json ... | jq` works and `proxima bench > results.txt`
/// still shows the failure.
fn fail(message: &str) -> std::process::ExitCode {
    eprintln!("proxima: {message}");
    eprintln!("try 'proxima help'");
    std::process::ExitCode::FAILURE
}
