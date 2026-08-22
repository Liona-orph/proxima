//! The listener, the connection threads, and graceful shutdown.
//!
//! # Thread per connection
//!
//! One thread per open connection, capped by [`ServerConfig::max_connections`].
//! This is the design that async runtimes exist to replace, and the reason it is
//! the right one here is the shape of the work: a Proxima request holds a lock
//! and burns CPU for tens or hundreds of microseconds. It never waits on another
//! service, so there is nothing for a task to yield to. An async runtime's
//! advantage is holding a hundred thousand *idle* connections cheaply, and a
//! vector database's clients are a handful of application servers with
//! keep-alive pools, not a hundred thousand browsers.
//!
//! What it costs: 8 MiB of virtual address space per thread (a few pages of it
//! resident), and a hard ceiling on concurrent connections. Past the cap the
//! listener answers `503` and closes, rather than queueing — a queue in front of
//! a saturated CPU-bound server just converts a fast failure into a slow one.
//!
//! # Graceful shutdown
//!
//! Three things start it: `SIGTERM`, `SIGINT`, and `POST /admin/shutdown`. All
//! three set one flag. The accept loop polls it every
//! [`SHUTDOWN_POLL`], stops accepting, waits up to
//! [`ServerConfig::drain_timeout`] for in-flight connections to finish, and then
//! **flushes every collection's write-ahead log** before returning.
//!
//! The flush is the point. Under the default [`SyncPolicy::Always`] it is a
//! no-op, because every acknowledged write is already durable — but under
//! `EveryN` or `Never` it is the difference between losing the last few hundred
//! writes on a rolling restart and losing none. A server whose durability
//! depends on the sync policy should still lose nothing when it is asked to stop
//! politely.
//!
//! Polling rather than a self-pipe or a non-blocking accept loop with a
//! `Waker`: a 50 ms worst-case delay on shutdown is invisible next to the
//! drain, and the alternative is either a second socket or a dependency on an
//! event loop.
//!
//! # Signals need one dependency
//!
//! `signal-hook`. Installing a signal handler is `sigaction`, which is FFI,
//! which is `unsafe` — and the crate is `#![forbid(unsafe_code)]`. The safe API
//! `signal_hook::flag::register` sets an `AtomicBool` from the handler and is
//! exactly the four lines needed. The alternative was to have no signal
//! handling at all and document that `SIGTERM` is a hard kill, which is
//! defensible only because `SyncPolicy::Always` already makes it safe — but a
//! server that cannot be drained is not one you can do a rolling restart of.
//!
//! [`SyncPolicy::Always`]: proxima_store::SyncPolicy::Always

use std::io::BufReader;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use proxima_core::ProximaError;

use crate::api::Api;
use crate::http::{self, ReadError, Response};
use crate::json::Json;

/// How often the accept loop checks the shutdown flag.
pub const SHUTDOWN_POLL: Duration = Duration::from_millis(50);

/// Listener and connection settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    /// Address to bind.
    pub address: SocketAddr,
    /// Concurrent connections allowed before new ones are refused with `503`.
    pub max_connections: usize,
    /// Read and write timeout on a connection.
    ///
    /// Bounds how long a stalled client can hold a thread. Without it, one
    /// client that opens a socket and sends nothing occupies a thread until the
    /// process dies, and `max_connections` of them stop the server.
    pub connection_timeout: Duration,
    /// How long shutdown waits for in-flight connections.
    pub drain_timeout: Duration,
    /// Requests served on one keep-alive connection before it is closed.
    ///
    /// Bounds the damage from a client that never closes: the thread comes back
    /// to the pool eventually whatever the client does.
    pub max_keep_alive_requests: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            address: ([127, 0, 0, 1], 7700).into(),
            max_connections: 256,
            connection_timeout: Duration::from_secs(30),
            drain_timeout: Duration::from_secs(30),
            max_keep_alive_requests: 1024,
        }
    }
}

/// A handle that can ask a running [`Server`] to stop.
///
/// Cloneable and `Send`, so it can be handed to a signal handler, an admin
/// endpoint and a test at the same time.
#[derive(Debug, Clone)]
pub struct ShutdownHandle {
    flag: Arc<AtomicBool>,
}

impl ShutdownHandle {
    /// Ask the server to stop accepting and drain.
    ///
    /// Idempotent, and returns immediately: the shutdown happens on the accept
    /// thread, which is what makes it safe to call from inside a request
    /// handler.
    pub fn shutdown(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// `true` once shutdown has been requested.
    pub fn is_shutting_down(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// The HTTP server.
#[derive(Debug)]
pub struct Server {
    listener: TcpListener,
    config: ServerConfig,
    api: Arc<Api>,
    shutdown: Arc<AtomicBool>,
    connections: Arc<AtomicUsize>,
}

impl Server {
    /// Bind the listener.
    ///
    /// Separate from [`Server::serve`] so that a caller — a test, or a CLI that
    /// wants to print the port it actually got when asked for port 0 — can learn
    /// the address before the loop starts.
    pub fn bind(config: ServerConfig, api: Arc<Api>) -> Result<Self, ProximaError> {
        let listener = TcpListener::bind(config.address)
            .map_err(|err| ProximaError::io("bind listener", config.address, &err))?;
        listener
            .set_nonblocking(true)
            .map_err(|err| ProximaError::io("set listener non-blocking", config.address, &err))?;
        Ok(Server {
            listener,
            config,
            api,
            shutdown: Arc::new(AtomicBool::new(false)),
            connections: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// The address actually bound, which differs from the requested one when
    /// port 0 was asked for.
    pub fn local_address(&self) -> Result<SocketAddr, ProximaError> {
        self.listener
            .local_addr()
            .map_err(|err| ProximaError::io("read listener address", self.config.address, &err))
    }

    /// A handle that can stop this server.
    pub fn shutdown_handle(&self) -> ShutdownHandle {
        ShutdownHandle {
            flag: Arc::clone(&self.shutdown),
        }
    }

    /// Install `SIGTERM` and `SIGINT` handlers that trigger shutdown.
    ///
    /// Unix only. On other platforms this is a no-op and shutdown is available
    /// through [`Server::shutdown_handle`] and `POST /admin/shutdown`.
    pub fn install_signal_handlers(&self) -> Result<(), ProximaError> {
        #[cfg(unix)]
        {
            for signal in [
                signal_hook::consts::SIGTERM,
                signal_hook::consts::SIGINT,
                signal_hook::consts::SIGQUIT,
            ] {
                signal_hook::flag::register(signal, Arc::clone(&self.shutdown)).map_err(|err| {
                    ProximaError::io("register signal handler", format!("signal {signal}"), &err)
                })?;
            }
        }
        Ok(())
    }

    /// Accept connections until shutdown, then drain and flush.
    ///
    /// Returns the number of connections that were still in flight when the
    /// drain timed out, which is zero on a clean shutdown.
    pub fn serve(self) -> Result<usize, ProximaError> {
        while !self.shutdown.load(Ordering::SeqCst) {
            match self.listener.accept() {
                Ok((stream, _peer)) => self.spawn(stream),
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(SHUTDOWN_POLL);
                }
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {
                    // A signal arrived. The flag is already set by the handler,
                    // so the loop condition will catch it.
                }
                Err(err) => {
                    // A failed accept is not a reason to stop serving: it is
                    // usually a per-connection condition such as the peer
                    // resetting between the SYN and the accept.
                    eprintln!("proxima: accept failed: {err}");
                    std::thread::sleep(SHUTDOWN_POLL);
                }
            }
        }

        let deadline = Instant::now() + self.config.drain_timeout;
        while self.connections.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            std::thread::sleep(SHUTDOWN_POLL);
        }
        let stranded = self.connections.load(Ordering::SeqCst);

        // The reason graceful shutdown exists. Under `SyncPolicy::Always` this
        // is a no-op; under everything else it is the last chance to make the
        // acknowledged writes durable.
        self.api.flush_all()?;
        Ok(stranded)
    }

    fn spawn(&self, stream: TcpStream) {
        let in_flight = self.connections.fetch_add(1, Ordering::SeqCst) + 1;
        if in_flight > self.config.max_connections {
            self.connections.fetch_sub(1, Ordering::SeqCst);
            // Refuse rather than queue: the work behind this API is CPU-bound
            // and a queue in front of a saturated CPU turns a fast failure into
            // a slow one. `Retry-After: 1` tells a well-behaved client what to
            // do about it.
            let mut stream = stream;
            let response =
                Response::text(503, "too many connections\n").with_header("Retry-After", "1");
            let _ = http::write_response(&mut stream, &response, false);
            return;
        }

        let api = Arc::clone(&self.api);
        let shutdown = Arc::clone(&self.shutdown);
        let connections = Arc::clone(&self.connections);
        let config = self.config.clone();
        let spawned = std::thread::Builder::new()
            .name("proxima-conn".to_owned())
            .spawn(move || {
                handle_connection(stream, &api, &shutdown, &config);
                connections.fetch_sub(1, Ordering::SeqCst);
            });
        if spawned.is_err() {
            // Out of threads. Same answer as being over the cap.
            self.connections.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// Serve one connection until it closes, errors, or hits a limit.
fn handle_connection(
    stream: TcpStream,
    api: &Arc<Api>,
    shutdown: &Arc<AtomicBool>,
    config: &ServerConfig,
) {
    // BSD-derived platforms can propagate the listener's non-blocking mode to
    // accepted sockets. A keep-alive read would then see `WouldBlock` before
    // the client can send its next request and incorrectly answer with 408.
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(config.connection_timeout));
    let _ = stream.set_write_timeout(Some(config.connection_timeout));
    // Nagle's algorithm delays a small response waiting for more data that will
    // never come, which adds up to 40 ms to a request/response API's latency.
    let _ = stream.set_nodelay(true);

    let Ok(write_half) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(stream);
    let mut writer = write_half;

    for _ in 0..config.max_keep_alive_requests {
        let request = match http::read_request(&mut reader, api.config().max_body_bytes) {
            Ok(request) => request,
            Err(ReadError::Closed) => return,
            Err(ReadError::Io(_)) => return,
            Err(ReadError::Malformed(status, reason)) => {
                let response = Response::json(
                    status,
                    Json::object([(
                        "error",
                        Json::object([
                            ("kind", Json::string("bad_request")),
                            ("message", Json::string(reason)),
                        ]),
                    )])
                    .to_string(),
                );
                let _ = http::write_response(&mut writer, &response, false);
                return;
            }
        };

        // Intercepted here rather than in the router, because the router has no
        // way to reach the listener.
        let is_shutdown_request = request.method == "POST" && request.path == "/admin/shutdown";
        if is_shutdown_request {
            shutdown.store(true, Ordering::SeqCst);
        }

        let mut response = api.handle(&request);
        if is_shutdown_request {
            response = Response::json(
                200,
                Json::object([("status", Json::string("shutting_down"))]).to_string(),
            );
        }

        // Stop keeping the connection alive once shutdown has begun, so that
        // idle keep-alive clients do not hold the drain open for its full
        // timeout.
        let keep_alive = request.keep_alive && !shutdown.load(Ordering::SeqCst);
        if http::write_response(&mut writer, &response, keep_alive).is_err() || !keep_alive {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::io::{Read, Write};

    use super::*;
    use crate::api::ApiConfig;

    fn test_server(name: &str) -> (Server, SocketAddr, std::path::PathBuf) {
        let directory = std::env::temp_dir().join(format!(
            "proxima-server-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        let api = Arc::new(
            Api::open(ApiConfig {
                data_directory: directory.clone(),
                ..ApiConfig::default()
            })
            .unwrap(),
        );
        let server = Server::bind(
            ServerConfig {
                address: ([127, 0, 0, 1], 0).into(),
                ..ServerConfig::default()
            },
            api,
        )
        .unwrap();
        let address = server.local_address().unwrap();
        (server, address, directory)
    }

    /// Send one request and read the whole response.
    fn roundtrip(address: SocketAddr, request: &str) -> String {
        let mut stream = TcpStream::connect(address).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        stream.flush().unwrap();
        let mut response = String::new();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let _ = stream.read_to_string(&mut response);
        response
    }

    /// Read exactly one response off a keep-alive connection.
    ///
    /// A single `read` is not enough and assuming it is makes the test flaky:
    /// TCP is a byte stream, so the head and the body can arrive in separate
    /// segments and the next iteration would then read the *previous*
    /// response's tail. This reads until the header terminator, then until
    /// `Content-Length` bytes of body have arrived.
    fn read_one_response(stream: &mut TcpStream) -> String {
        let mut raw = Vec::new();
        let mut byte = [0u8; 1];
        let mut body_start = None;
        let mut length = 0usize;
        loop {
            if stream.read(&mut byte).unwrap_or(0) == 0 {
                break;
            }
            raw.push(byte[0]);
            if body_start.is_none() && raw.ends_with(b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&raw).to_ascii_lowercase();
                length = head
                    .split("content-length:")
                    .nth(1)
                    .and_then(|rest| rest.split("\r\n").next())
                    .and_then(|value| value.trim().parse().ok())
                    .unwrap_or(0);
                body_start = Some(raw.len());
            }
            if let Some(start) = body_start {
                if raw.len() - start >= length {
                    break;
                }
            }
        }
        String::from_utf8_lossy(&raw).into_owned()
    }

    fn get(path: &str) -> String {
        format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    }

    fn post(path: &str, body: &str) -> String {
        format!(
            "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn it_serves_requests_and_shuts_down_on_demand() {
        let (server, address, directory) = test_server("basic");
        let handle = server.shutdown_handle();
        let worker = std::thread::spawn(move || server.serve().unwrap());

        let health = roundtrip(address, &get("/health"));
        assert!(health.starts_with("HTTP/1.1 200 OK"), "{health}");
        assert!(health.contains("\"status\":\"ok\""));

        let created = roundtrip(
            address,
            &post(
                "/collections",
                r#"{"name":"t","dim":2,"metric":"euclidean","index":"flat"}"#,
            ),
        );
        assert!(created.starts_with("HTTP/1.1 201"), "{created}");

        let upserted = roundtrip(
            address,
            &post(
                "/collections/t/points",
                r#"{"points":[{"id":1,"vector":[1,0],"metadata":{"a":"b"}}]}"#,
            ),
        );
        assert!(upserted.contains("\"upserted\":1"), "{upserted}");

        let searched = roundtrip(
            address,
            &post("/collections/t/search", r#"{"vector":[1,0],"k":1}"#),
        );
        assert!(searched.contains("\"id\":1"), "{searched}");
        assert!(
            searched.contains("\"metadata\":{\"a\":\"b\"}"),
            "{searched}"
        );

        let metrics = roundtrip(address, &get("/metrics"));
        assert!(
            metrics.contains("proxima_collection_vectors{collection=\"t\""),
            "{metrics}"
        );

        assert!(!handle.is_shutting_down());
        handle.shutdown();
        assert!(handle.is_shutting_down());
        assert_eq!(worker.join().unwrap(), 0, "no connection was left stranded");
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn the_admin_endpoint_shuts_the_server_down() {
        let (server, address, directory) = test_server("admin");
        let worker = std::thread::spawn(move || server.serve().unwrap());

        let response = roundtrip(address, &post("/admin/shutdown", "{}"));
        assert!(response.contains("shutting_down"), "{response}");
        // The loop notices within one poll interval and returns.
        assert_eq!(worker.join().unwrap(), 0);
        // ...and the port is free again.
        assert!(TcpListener::bind(address).is_ok());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn keep_alive_serves_several_requests_on_one_connection() {
        let (server, address, directory) = test_server("keepalive");
        let handle = server.shutdown_handle();
        let worker = std::thread::spawn(move || server.serve().unwrap());

        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        for request in 0..3 {
            stream
                .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            let text = read_one_response(&mut stream);
            assert!(
                text.starts_with("HTTP/1.1 200"),
                "request {request}: {text}"
            );
            assert!(
                text.contains("Connection: keep-alive"),
                "request {request}: {text}"
            );
        }
        drop(stream);

        handle.shutdown();
        worker.join().unwrap();
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn a_malformed_request_gets_a_status_and_the_connection_closes() {
        let (server, address, directory) = test_server("malformed");
        let handle = server.shutdown_handle();
        let worker = std::thread::spawn(move || server.serve().unwrap());

        let chunked = roundtrip(
            address,
            "POST /collections HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n",
        );
        assert!(chunked.starts_with("HTTP/1.1 411"), "{chunked}");

        let huge = roundtrip(
            address,
            "POST /collections HTTP/1.1\r\nHost: x\r\nContent-Length: 999999999999\r\n\r\n",
        );
        assert!(huge.starts_with("HTTP/1.1 413"), "{huge}");

        let garbage = roundtrip(address, "NOT AN HTTP REQUEST\r\n\r\n");
        assert!(garbage.starts_with("HTTP/1.1 400"), "{garbage}");

        // ...and the server is still serving afterwards.
        assert!(roundtrip(address, &get("/health")).starts_with("HTTP/1.1 200"));
        handle.shutdown();
        worker.join().unwrap();
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn shutdown_flushes_every_collection() {
        // The property graceful shutdown exists for, under a policy where it
        // actually does something.
        let directory = std::env::temp_dir().join(format!(
            "proxima-server-flush-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        let api = Arc::new(
            Api::open(ApiConfig {
                data_directory: directory.clone(),
                sync_policy: proxima_store::SyncPolicy::Never,
                ..ApiConfig::default()
            })
            .unwrap(),
        );
        let server = Server::bind(
            ServerConfig {
                address: ([127, 0, 0, 1], 0).into(),
                ..ServerConfig::default()
            },
            Arc::clone(&api),
        )
        .unwrap();
        let address = server.local_address().unwrap();
        let handle = server.shutdown_handle();
        let worker = std::thread::spawn(move || server.serve().unwrap());

        roundtrip(
            address,
            &post(
                "/collections",
                r#"{"name":"f","dim":2,"metric":"euclidean","index":"flat","sync_policy":"never"}"#,
            ),
        );
        roundtrip(
            address,
            &post(
                "/collections/f/points",
                r#"{"points":[{"id":1,"vector":[1,0]}]}"#,
            ),
        );
        handle.shutdown();
        worker.join().unwrap();

        // The log was flushed on the way out, so a fresh open finds the write.
        let api = Api::open(ApiConfig {
            data_directory: directory.clone(),
            ..ApiConfig::default()
        })
        .unwrap();
        assert_eq!(api.collections()[0].1.len(), 1);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn signal_handlers_install_without_disturbing_anything() {
        let (server, _, directory) = test_server("signals");
        server.install_signal_handlers().unwrap();
        assert!(!server.shutdown_handle().is_shutting_down());
        let _ = std::fs::remove_dir_all(directory);
    }
}
