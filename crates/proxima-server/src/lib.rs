//! An HTTP/JSON API over [`proxima_store::Collection`], on the standard library.
//!
//! # Why this is not built on a web framework
//!
//! The obvious implementation is `axum` on `tokio` with `serde`. That would be
//! three or four hundred transitive dependencies to serve eleven endpoints of
//! request/response JSON over a lock, and it would put an async runtime in
//! front of work that never awaits anything — a Proxima request takes a lock,
//! burns CPU for tens to hundreds of microseconds, and returns.
//!
//! So: [`std::net::TcpListener`], a thread per connection, a hand-written
//! HTTP/1.1 reader and a hand-written JSON parser. One dependency,
//! `signal-hook`, and only because installing a signal handler is FFI and this
//! crate is `#![forbid(unsafe_code)]`.
//!
//! **What this costs, stated plainly.** `axum` and `serde_json` are faster,
//! better tested and better fuzzed than what is here. This server has no
//! HTTP/2, no TLS, no chunked request bodies and no compression, and its
//! connection model caps concurrency at a few hundred rather than a hundred
//! thousand. Every one of those is written down where it is decided:
//! [`http`] for the protocol, [`json`] for the parser, [`server`] for the
//! threading. If you are terminating public internet traffic, put something in
//! front of this.
//!
//! # Shape
//!
//! ```text
//!   TcpListener ──accept──> thread ──> http::read_request
//!                                          |
//!                                          v
//!                                     Api::handle  ──> Collection (RwLock)
//!                                          |
//!                                          v
//!                                   http::write_response
//! ```
//!
//! * [`api`] owns the collections and routes requests.
//! * [`filter`] compiles the JSON filter language into a
//!   [`Filter`](proxima_index::Filter) that runs inside graph traversal.
//! * [`metrics`] renders the Prometheus exposition format.
//! * [`server`] owns the listener, the connection threads and shutdown.
//!
//! ```no_run
//! use std::sync::Arc;
//! use proxima_server::{Api, ApiConfig, Server, ServerConfig};
//!
//! let api = Arc::new(Api::open(ApiConfig {
//!     data_directory: "/var/lib/proxima".into(),
//!     ..ApiConfig::default()
//! })?);
//! let server = Server::bind(ServerConfig::default(), api)?;
//! server.install_signal_handlers()?;
//! println!("listening on {}", server.local_address()?);
//! server.serve()?;
//! # Ok::<(), proxima_core::ProximaError>(())
//! ```
#![forbid(unsafe_code)]

pub mod api;
pub mod filter;
pub mod http;
pub mod json;
pub mod metrics;
pub mod server;

pub use api::{Api, ApiConfig, ApiError};
pub use filter::MetadataFilter;
pub use http::{Request, Response};
pub use json::{parse, Json};
pub use server::{Server, ServerConfig, ShutdownHandle};
