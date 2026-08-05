//! The HTTP/JSON API: routing, request shapes and the [`Api`] that owns the
//! collections.
//!
//! # The endpoints
//!
//! | method | path | does |
//! |---|---|---|
//! | `GET` | `/health` | liveness and readiness |
//! | `GET` | `/metrics` | Prometheus exposition |
//! | `GET` | `/collections` | list |
//! | `POST` | `/collections` | create |
//! | `GET` | `/collections/{name}` | describe |
//! | `DELETE` | `/collections/{name}` | drop, deleting the files |
//! | `POST` | `/collections/{name}/points` | **batch** upsert |
//! | `POST` | `/collections/{name}/points/delete` | batch delete |
//! | `POST` | `/collections/{name}/search` | search with a metadata filter |
//! | `POST` | `/collections/{name}/snapshot` | snapshot now |
//! | `POST` | `/collections/{name}/compact` | reclaim tombstones |
//! | `POST` | `/admin/shutdown` | begin graceful shutdown |
//!
//! # Two decisions worth defending
//!
//! **Upsert is a batch endpoint and there is no single-point variant.** One
//! `fsync` costs about 600 µs and a thousand-point batch costs 3.0 µs per point (see
//! `proxima_store::wal`), so a per-point endpoint would be a 200x slower API
//! that looks more convenient. A caller who has one point sends a batch of one.
//!
//! **Delete is `POST .../points/delete` rather than `DELETE .../points`.**
//! `DELETE` with a body is legal and widely mishandled — proxies drop it,
//! several HTTP clients refuse to send it. A `POST` to an action path is ugly
//! and works everywhere.
//!
//! # Error bodies
//!
//! Every failure returns `{"error": {"kind": ..., "message": ...}}` with the
//! message from [`ProximaError`]'s `Display`, which is designed to be
//! diagnosable from one line. The status is `4xx` when
//! [`ProximaError::is_caller_error`] says the fault is the caller's and `5xx`
//! otherwise, so the split between "your request" and "our bug" is made in one
//! place rather than at forty call sites.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use proxima_core::{Metadata, MetadataValue, Metric, ProximaError, VectorId};
use proxima_index::{AcceptAll, HnswConfig, IvfPqConfig};
use proxima_store::{Collection, CollectionOptions, IndexSpec, Manifest, SyncPolicy};

use crate::filter::MetadataFilter;
use crate::http::{Request, Response};
use crate::json::{parse, Json};
use crate::metrics;

/// Server-wide configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiConfig {
    /// Directory that holds one subdirectory per collection.
    pub data_directory: PathBuf,
    /// Default sync policy for new collections.
    pub sync_policy: SyncPolicy,
    /// Snapshot threshold for new collections.
    pub snapshot_after_records: u64,
    /// Largest request body accepted, in bytes.
    ///
    /// A batch upsert of 10 000 vectors at dimension 1 536 is about 150 MiB of
    /// JSON, so the default of 256 MiB admits a large batch and still refuses
    /// an unbounded one.
    pub max_body_bytes: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        ApiConfig {
            data_directory: PathBuf::from("./proxima-data"),
            sync_policy: SyncPolicy::Always,
            snapshot_after_records: 100_000,
            max_body_bytes: 256 << 20,
        }
    }
}

/// The collections, and everything the handlers need.
///
/// Shared across connection threads behind an `Arc`. The map itself is an
/// `RwLock` and each [`Collection`] has its own lock inside, so creating a
/// collection blocks every request for the duration of a directory create while
/// writing to one collection blocks only that collection's readers. That split
/// is the reason the outer lock is over `Arc<Collection>` and not over the
/// collections themselves: a handler clones the `Arc`, releases the map lock,
/// and then does its work.
#[derive(Debug)]
pub struct Api {
    config: ApiConfig,
    collections: RwLock<BTreeMap<String, Arc<Collection>>>,
    started: Instant,
    requests: AtomicU64,
    errors: AtomicU64,
    search_micros: AtomicU64,
    searches: AtomicU64,
}

impl Api {
    /// Open every collection already on disk under `config.data_directory`.
    ///
    /// A directory that is not a collection — no manifest — is skipped with a
    /// warning rather than failing the start, because the data directory is
    /// somewhere an operator will eventually put a `README` or a stray
    /// `lost+found`.
    pub fn open(config: ApiConfig) -> Result<Self, ProximaError> {
        std::fs::create_dir_all(&config.data_directory).map_err(|err| {
            ProximaError::io(
                "create data directory",
                config.data_directory.display(),
                &err,
            )
        })?;

        let mut collections = BTreeMap::new();
        let entries = std::fs::read_dir(&config.data_directory).map_err(|err| {
            ProximaError::io("read data directory", config.data_directory.display(), &err)
        })?;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() || !path.join("manifest").exists() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let manifest = Manifest::read(&path)?;
            let (collection, report) = Collection::open(
                &path,
                manifest.name,
                manifest.space.dim(),
                manifest.space.metric(),
                manifest.spec,
                CollectionOptions {
                    sync_policy: config.sync_policy,
                    snapshot_after_records: config.snapshot_after_records,
                },
            )?;
            if report.recovery.torn_tail {
                eprintln!(
                    "proxima: collection '{name}' had a torn log tail; discarded {} bytes",
                    report.recovery.bytes_discarded
                );
            }
            collections.insert(name.to_owned(), Arc::new(collection));
        }

        Ok(Api {
            config,
            collections: RwLock::new(collections),
            started: Instant::now(),
            requests: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            search_micros: AtomicU64::new(0),
            searches: AtomicU64::new(0),
        })
    }

    /// The configuration in force.
    pub fn config(&self) -> &ApiConfig {
        &self.config
    }

    /// How long the process has been up.
    pub fn uptime(&self) -> std::time::Duration {
        self.started.elapsed()
    }

    /// Requests handled, failures, and search latency, for `/metrics`.
    pub fn totals(&self) -> (u64, u64, u64, u64) {
        (
            self.requests.load(Ordering::Relaxed),
            self.errors.load(Ordering::Relaxed),
            self.searches.load(Ordering::Relaxed),
            self.search_micros.load(Ordering::Relaxed),
        )
    }

    /// Every open collection, by name.
    pub fn collections(&self) -> Vec<(String, Arc<Collection>)> {
        self.read_collections()
            .iter()
            .map(|(name, collection)| (name.clone(), Arc::clone(collection)))
            .collect()
    }

    fn read_collections(
        &self,
    ) -> std::sync::RwLockReadGuard<'_, BTreeMap<String, Arc<Collection>>> {
        self.collections
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write_collections(
        &self,
    ) -> std::sync::RwLockWriteGuard<'_, BTreeMap<String, Arc<Collection>>> {
        self.collections
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn collection(&self, name: &str) -> Result<Arc<Collection>, ApiError> {
        self.read_collections()
            .get(name)
            .map(Arc::clone)
            .ok_or_else(|| ApiError::new(404, "not_found", format!("no collection '{name}'")))
    }

    /// Flush every collection's log. Called by graceful shutdown.
    pub fn flush_all(&self) -> Result<(), ProximaError> {
        for (_, collection) in self.collections() {
            collection.flush()?;
        }
        Ok(())
    }

    /// Route and handle one request.
    ///
    /// Total: this is the only entry point, and every path through it returns a
    /// [`Response`] rather than propagating an error, so a handler bug becomes a
    /// `500` with a message instead of a dropped connection.
    pub fn handle(&self, request: &Request) -> Response {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let response = match self.route(request) {
            Ok(response) => response,
            Err(error) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                error.into_response()
            }
        };
        if response.status >= 400 && response.status != 404 {
            // 404s are counted as requests but not as errors: a probe for a
            // collection that does not exist is normal traffic, and alerting on
            // it would train an operator to ignore the error rate.
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        response
    }

    fn route(&self, request: &Request) -> Result<Response, ApiError> {
        let segments = request.segments();
        let method = request.method.as_str();
        match (method, segments.as_slice()) {
            ("GET", ["health"]) => Ok(self.health()),
            ("GET", ["metrics"]) => Ok(Response::prometheus(metrics::render(self))),
            ("GET", ["collections"]) => Ok(self.list_collections()),
            ("POST", ["collections"]) => self.create_collection(request),
            ("GET", ["collections", name]) => self.describe(name),
            ("DELETE", ["collections", name]) => self.drop_collection(name),
            ("POST", ["collections", name, "points"]) => self.upsert(name, request),
            ("POST", ["collections", name, "points", "delete"]) => self.delete(name, request),
            ("POST", ["collections", name, "search"]) => self.search(name, request),
            ("POST", ["collections", name, "snapshot"]) => self.snapshot(name),
            ("POST", ["collections", name, "compact"]) => self.compact(name),
            // `/admin/shutdown` is routed by the server itself, which owns the
            // listener; reaching it here means the server did not intercept it.
            ("POST", ["admin", "shutdown"]) => Ok(Response::json(
                200,
                Json::object([("status", Json::string("shutting_down"))]).to_string(),
            )),
            (_, ["health" | "metrics" | "collections" | "admin", ..]) => Err(ApiError::new(
                405,
                "method_not_allowed",
                format!("{method} is not allowed on {}", request.path),
            )),
            _ => Err(ApiError::new(
                404,
                "not_found",
                format!("no route for {method} {}", request.path),
            )),
        }
    }

    // -----------------------------------------------------------------
    // Handlers
    // -----------------------------------------------------------------

    fn health(&self) -> Response {
        let collections = self.collections();
        // Readiness, not just liveness: a poisoned collection lock means a
        // handler panicked while mutating an index, and the process should be
        // taken out of rotation even though it is still answering.
        let poisoned: Vec<&str> = collections
            .iter()
            .filter(|(_, collection)| collection.is_poisoned())
            .map(|(name, _)| name.as_str())
            .collect();
        let vectors: usize = collections
            .iter()
            .map(|(_, collection)| collection.len())
            .sum();

        let body = Json::object([
            (
                "status",
                Json::string(if poisoned.is_empty() {
                    "ok"
                } else {
                    "degraded"
                }),
            ),
            ("version", Json::string(env!("CARGO_PKG_VERSION"))),
            (
                "uptime_seconds",
                Json::count(self.uptime().as_secs() as usize),
            ),
            ("collections", Json::count(collections.len())),
            ("vectors", Json::count(vectors)),
            (
                "poisoned",
                Json::Array(poisoned.iter().map(|n| Json::string(*n)).collect()),
            ),
        ]);
        Response::json(
            if poisoned.is_empty() { 200 } else { 503 },
            body.to_string(),
        )
    }

    fn list_collections(&self) -> Response {
        let items: Vec<Json> = self
            .collections()
            .iter()
            .map(|(name, collection)| {
                let description = collection.describe();
                Json::object([
                    ("name", Json::string(name)),
                    ("dim", Json::count(description.dim)),
                    ("metric", Json::string(description.metric.to_string())),
                    ("index", Json::string(description.index)),
                    ("vectors", Json::count(description.stats.len)),
                ])
            })
            .collect();
        Response::json(
            200,
            Json::object([("collections", Json::Array(items))]).to_string(),
        )
    }

    fn create_collection(&self, request: &Request) -> Result<Response, ApiError> {
        let body = body_json(request)?;
        let name = required_str(&body, "name")?.to_owned();
        validate_name(&name)?;
        let dim = body
            .get("dim")
            .and_then(Json::as_usize)
            .ok_or_else(|| ApiError::new(400, "bad_request", "'dim' must be a positive integer"))?;
        let metric_name = body
            .get("metric")
            .and_then(Json::as_str)
            .unwrap_or("cosine");
        let metric = Metric::from_name(metric_name).ok_or_else(|| {
            ApiError::new(
                400,
                "bad_request",
                format!(
                    "unknown metric '{metric_name}'; the metrics are {}",
                    Metric::ALL
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
        })?;
        let spec = parse_index_spec(body.get("index"))?;

        let mut collections = self.write_collections();
        if collections.contains_key(&name) {
            return Err(ApiError::new(
                409,
                "conflict",
                format!("collection '{name}' already exists"),
            ));
        }
        let sync_policy = match body.get("sync_policy").and_then(Json::as_str) {
            Some(text) => SyncPolicy::from_name(text).ok_or_else(|| {
                ApiError::new(
                    400,
                    "bad_request",
                    format!("unknown sync_policy '{text}'; use always, never or every_N"),
                )
            })?,
            None => self.config.sync_policy,
        };

        let (collection, _) = Collection::open(
            self.config.data_directory.join(&name),
            name.clone(),
            dim,
            metric,
            spec,
            CollectionOptions {
                sync_policy,
                snapshot_after_records: self.config.snapshot_after_records,
            },
        )?;
        let description = collection.describe();
        collections.insert(name.clone(), Arc::new(collection));

        Ok(Response::json(
            201,
            Json::object([
                ("name", Json::string(name)),
                ("dim", Json::count(description.dim)),
                ("metric", Json::string(description.metric.to_string())),
                ("index", Json::string(description.index)),
                ("sync_policy", Json::string(sync_policy.to_string())),
            ])
            .to_string(),
        ))
    }

    fn describe(&self, name: &str) -> Result<Response, ApiError> {
        let collection = self.collection(name)?;
        let description = collection.describe();
        let stats = &description.stats;
        let body = Json::object([
            ("name", Json::string(&description.name)),
            ("dim", Json::count(description.dim)),
            ("metric", Json::string(description.metric.to_string())),
            ("index", Json::string(description.index)),
            ("vectors", Json::count(stats.len)),
            ("deleted", Json::count(stats.deleted)),
            ("layers", Json::count(stats.layers)),
            (
                "nodes_per_layer",
                Json::Array(
                    stats
                        .nodes_per_layer
                        .iter()
                        .map(|n| Json::count(*n))
                        .collect(),
                ),
            ),
            ("edges", Json::count(stats.edges)),
            ("vector_bytes", Json::count(stats.vector_bytes)),
            ("index_bytes", Json::count(stats.graph_bytes)),
            ("bytes_per_vector", Json::number(stats.bytes_per_vector())),
            ("mean_degree", Json::number(stats.mean_degree())),
            (
                "metadata_keys",
                Json::Array(description.metadata_keys.iter().map(Json::string).collect()),
            ),
            ("wal_records", Json::count(description.wal_records as usize)),
            ("wal_bytes", Json::count(description.wal_bytes as usize)),
            (
                "sync_policy",
                Json::string(description.sync_policy.to_string()),
            ),
            (
                "counters",
                Json::object([
                    (
                        "searches",
                        Json::count(description.counters.searches as usize),
                    ),
                    (
                        "upserts",
                        Json::count(description.counters.upserts as usize),
                    ),
                    (
                        "deletes",
                        Json::count(description.counters.deletes as usize),
                    ),
                    (
                        "snapshots",
                        Json::count(description.counters.snapshots as usize),
                    ),
                ]),
            ),
        ]);
        Ok(Response::json(200, body.to_string()))
    }

    fn drop_collection(&self, name: &str) -> Result<Response, ApiError> {
        let mut collections = self.write_collections();
        let Some(collection) = collections.remove(name) else {
            return Err(ApiError::new(
                404,
                "not_found",
                format!("no collection '{name}'"),
            ));
        };
        // The map no longer holds it, but an in-flight request might. Unwrapping
        // the `Arc` is how we wait for that to be false; if it is still shared
        // we put it back and tell the caller to retry, which is far better than
        // deleting the files out from under a running search.
        match Arc::try_unwrap(collection) {
            Ok(collection) => {
                collection.drop_collection()?;
                Ok(Response::json(
                    200,
                    Json::object([("dropped", Json::string(name))]).to_string(),
                ))
            }
            Err(collection) => {
                collections.insert(name.to_owned(), collection);
                Err(ApiError::new(
                    409,
                    "busy",
                    format!("collection '{name}' has requests in flight; retry"),
                ))
            }
        }
    }

    fn upsert(&self, name: &str, request: &Request) -> Result<Response, ApiError> {
        let collection = self.collection(name)?;
        let body = body_json(request)?;
        let points = body
            .get("points")
            .and_then(Json::as_array)
            .ok_or_else(|| ApiError::new(400, "bad_request", "'points' must be an array"))?;

        let mut batch = Vec::with_capacity(points.len());
        for (index, point) in points.iter().enumerate() {
            let at = |message: String| {
                ApiError::new(400, "bad_request", format!("points[{index}]: {message}"))
            };
            let id = point
                .get("id")
                .and_then(Json::as_u64)
                .ok_or_else(|| at("'id' must be a non-negative integer".into()))?;
            let vector = point
                .get("vector")
                .and_then(Json::as_f32_vec)
                .ok_or_else(|| at("'vector' must be an array of numbers".into()))?;
            let metadata = match point.get("metadata") {
                Some(value) => parse_metadata(value).map_err(at)?,
                None => Metadata::new(),
            };
            batch.push((VectorId::new(id), vector, metadata));
        }

        let count = batch.len();
        collection.upsert_batch(&batch)?;
        Ok(Response::json(
            200,
            Json::object([
                ("upserted", Json::count(count)),
                ("vectors", Json::count(collection.len())),
            ])
            .to_string(),
        ))
    }

    fn delete(&self, name: &str, request: &Request) -> Result<Response, ApiError> {
        let collection = self.collection(name)?;
        let body = body_json(request)?;
        let ids = body
            .get("ids")
            .and_then(Json::as_array)
            .ok_or_else(|| ApiError::new(400, "bad_request", "'ids' must be an array"))?;
        let mut parsed = Vec::with_capacity(ids.len());
        for (index, id) in ids.iter().enumerate() {
            parsed.push(VectorId::new(id.as_u64().ok_or_else(|| {
                ApiError::new(
                    400,
                    "bad_request",
                    format!("ids[{index}] is not a non-negative integer"),
                )
            })?));
        }
        let removed = collection.delete_batch(&parsed)?;
        Ok(Response::json(
            200,
            Json::object([
                ("requested", Json::count(parsed.len())),
                ("deleted", Json::count(removed)),
                ("vectors", Json::count(collection.len())),
            ])
            .to_string(),
        ))
    }

    fn search(&self, name: &str, request: &Request) -> Result<Response, ApiError> {
        let collection = self.collection(name)?;
        let body = body_json(request)?;
        let vector = body
            .get("vector")
            .and_then(Json::as_f32_vec)
            .ok_or_else(|| {
                ApiError::new(400, "bad_request", "'vector' must be an array of numbers")
            })?;
        let k = body.get("k").and_then(Json::as_usize).unwrap_or(10);
        let filter = match body.get("filter") {
            Some(value) => MetadataFilter::parse(value)
                .map_err(|message| ApiError::new(400, "bad_filter", message))?,
            None => MetadataFilter::default(),
        };
        let with_metadata = body
            .get("with_metadata")
            .and_then(Json::as_bool)
            .unwrap_or(true);

        let started = Instant::now();
        let hits = if filter.is_empty() {
            collection.search_with_metadata(&vector, k, &AcceptAll)?
        } else {
            collection.search_with_metadata(&vector, k, &filter)?
        };
        let elapsed = started.elapsed();
        self.searches.fetch_add(1, Ordering::Relaxed);
        self.search_micros
            .fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);

        let metric = collection.space().metric();
        let results: Vec<Json> = hits
            .into_iter()
            .map(|(hit, metadata)| {
                let mut fields = vec![
                    ("id", Json::number(hit.id.get() as f64)),
                    // Both are reported. `score` is what the index ranked by —
                    // squared for Euclidean, negated for inner product — and
                    // `distance` is the number a human expects. Returning only
                    // the first surprises people; returning only the second
                    // makes the ordering look wrong for inner product.
                    ("score", Json::number(hit.distance)),
                    (
                        "distance",
                        Json::number(metric.to_true_distance(hit.distance)),
                    ),
                ];
                if with_metadata {
                    fields.push(("metadata", metadata_json(&metadata)));
                }
                Json::object(fields)
            })
            .collect();

        Ok(Response::json(
            200,
            Json::object([
                ("results", Json::Array(results)),
                ("took_micros", Json::count(elapsed.as_micros() as usize)),
                ("filtered", Json::Bool(!filter.is_empty())),
            ])
            .to_string(),
        ))
    }

    fn snapshot(&self, name: &str) -> Result<Response, ApiError> {
        let collection = self.collection(name)?;
        let bytes = collection.snapshot()?;
        Ok(Response::json(
            200,
            Json::object([("snapshot_bytes", Json::count(bytes as usize))]).to_string(),
        ))
    }

    fn compact(&self, name: &str) -> Result<Response, ApiError> {
        let collection = self.collection(name)?;
        let reclaimed = collection.compact()?;
        Ok(Response::json(
            200,
            Json::object([
                ("reclaimed", Json::count(reclaimed)),
                ("vectors", Json::count(collection.len())),
            ])
            .to_string(),
        ))
    }
}

/// A failure with an HTTP status and a machine-readable kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    /// HTTP status.
    pub status: u16,
    /// Stable machine-readable tag, e.g. `"not_found"`.
    pub kind: &'static str,
    /// Human-readable message.
    pub message: String,
}

impl ApiError {
    /// Build one.
    pub fn new(status: u16, kind: &'static str, message: impl Into<String>) -> Self {
        ApiError {
            status,
            kind,
            message: message.into(),
        }
    }

    /// Render as a JSON error body.
    pub fn into_response(self) -> Response {
        Response::json(
            self.status,
            Json::object([(
                "error",
                Json::object([
                    ("kind", Json::string(self.kind)),
                    ("message", Json::string(self.message)),
                ]),
            )])
            .to_string(),
        )
    }
}

impl From<ProximaError> for ApiError {
    /// Map an engine error onto a status.
    ///
    /// The whole mapping is `is_caller_error`, which is why that method exists
    /// on `ProximaError` — a service layer wanting exactly this is the only
    /// justification for exposing the distinction.
    fn from(error: ProximaError) -> Self {
        let (status, kind) = match &error {
            ProximaError::DimensionMismatch { .. } => (400, "dimension_mismatch"),
            ProximaError::UnknownId { .. } => (404, "unknown_id"),
            ProximaError::DuplicateId { .. } => (409, "duplicate_id"),
            ProximaError::NonFinite { .. } => (400, "non_finite"),
            ProximaError::Configuration { .. } => (400, "invalid_configuration"),
            ProximaError::Capacity { .. } => (507, "capacity_exceeded"),
            ProximaError::Corruption { .. } => (500, "corruption"),
            ProximaError::Io { .. } => (500, "io_error"),
            _ => (500, "internal"),
        };
        ApiError::new(status, kind, error.to_string())
    }
}

fn body_json(request: &Request) -> Result<Json, ApiError> {
    let text = request
        .body_str()
        .ok_or_else(|| ApiError::new(400, "bad_request", "request body is not UTF-8"))?;
    if text.trim().is_empty() {
        return Ok(Json::Object(BTreeMap::new()));
    }
    parse(text).map_err(|err| ApiError::new(400, "bad_json", err.to_string()))
}

fn required_str<'a>(body: &'a Json, field: &str) -> Result<&'a str, ApiError> {
    body.get(field)
        .and_then(Json::as_str)
        .ok_or_else(|| ApiError::new(400, "bad_request", format!("'{field}' must be a string")))
}

/// Reject collection names that are not safe as a directory component.
///
/// This is the only place a request string becomes a path, so it is the only
/// place path traversal can happen. Allowing exactly `[A-Za-z0-9_-]` is far
/// stronger than filtering `..` out, because it has no cases to have missed.
fn validate_name(name: &str) -> Result<(), ApiError> {
    if name.is_empty() || name.len() > 64 {
        return Err(ApiError::new(
            400,
            "bad_request",
            "collection name must be 1 to 64 characters",
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(ApiError::new(
            400,
            "bad_request",
            "collection name may contain only letters, digits, '_' and '-'",
        ));
    }
    Ok(())
}

fn parse_index_spec(value: Option<&Json>) -> Result<IndexSpec, ApiError> {
    let Some(value) = value else {
        return Ok(IndexSpec::Hnsw(HnswConfig::new()));
    };
    let kind = value
        .get("type")
        .and_then(Json::as_str)
        .or_else(|| value.as_str())
        .unwrap_or("hnsw");
    let field = |name: &str| value.get(name).and_then(Json::as_usize);
    match kind {
        "flat" => Ok(IndexSpec::Flat),
        "hnsw" => {
            let mut config = HnswConfig::new();
            if let Some(m) = field("m") {
                config = config.with_m(m);
            }
            if let Some(ef) = field("ef_construction") {
                config = config.with_ef_construction(ef);
            }
            if let Some(ef) = field("ef_search") {
                config = config.with_ef_search(ef);
            }
            if let Some(seed) = value.get("seed").and_then(Json::as_u64) {
                config = config.with_seed(seed);
            }
            config.validate().map_err(ApiError::from)?;
            Ok(IndexSpec::Hnsw(config))
        }
        "ivfpq" => {
            let nlist = field("nlist").unwrap_or(256);
            let subspaces = field("subspaces").unwrap_or(16);
            let mut config = IvfPqConfig::new(nlist, subspaces);
            if let Some(nprobe) = field("nprobe") {
                config = config.with_nprobe(nprobe);
            }
            if let Some(rerank) = field("rerank") {
                config = config.with_rerank(rerank);
            }
            if let Some(seed) = value.get("seed").and_then(Json::as_u64) {
                config = config.with_seed(seed);
            }
            if let Some(threshold) = field("train_threshold") {
                config = config.with_train_threshold(threshold);
            }
            if let Some(retain) = value.get("retain_vectors").and_then(Json::as_bool) {
                config = config.with_retain_vectors(retain);
            }
            Ok(IndexSpec::IvfPq(config))
        }
        other => Err(ApiError::new(
            400,
            "bad_request",
            format!("unknown index type '{other}'; the types are flat, hnsw, ivfpq"),
        )),
    }
}

fn parse_metadata(value: &Json) -> Result<Metadata, String> {
    let Some(fields) = value.as_object() else {
        return Err(format!(
            "'metadata' must be an object, got {}",
            value.type_name()
        ));
    };
    let mut metadata = Metadata::new();
    for (key, value) in fields {
        let parsed = match value {
            Json::String(text) => MetadataValue::Text(text.clone()),
            Json::Bool(flag) => MetadataValue::Bool(*flag),
            // An integral JSON number becomes an `Int`, which is what a filter
            // on a year or a timestamp expects. There is no way to say "this
            // 1.0 is a float" in JSON, and inventing a tagged encoding for it
            // would make every client's life worse to serve a case nobody has.
            Json::Number(number) if number.fract() == 0.0 && number.abs() < 9.007e15 => {
                MetadataValue::Int(*number as i64)
            }
            Json::Number(number) => MetadataValue::Float(*number),
            other => {
                return Err(format!(
                    "metadata field '{key}' is a {}; only strings, numbers and booleans are stored",
                    other.type_name()
                ))
            }
        };
        metadata.insert(key.clone(), parsed);
    }
    Ok(metadata)
}

fn metadata_json(metadata: &Metadata) -> Json {
    Json::Object(
        metadata
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    MetadataValue::Text(text) => Json::string(text),
                    MetadataValue::Int(number) => Json::number(*number as f64),
                    MetadataValue::Float(number) => Json::Number(*number),
                    MetadataValue::Bool(flag) => Json::Bool(*flag),
                };
                (key.to_owned(), value)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn request(method: &str, path: &str, body: &str) -> Request {
        Request {
            method: method.to_owned(),
            path: path.to_owned(),
            query: String::new(),
            headers: BTreeMap::new(),
            body: body.as_bytes().to_vec(),
            keep_alive: true,
        }
    }

    fn api(name: &str) -> Api {
        let directory = std::env::temp_dir().join(format!(
            "proxima-api-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        Api::open(ApiConfig {
            data_directory: directory,
            ..ApiConfig::default()
        })
        .unwrap()
    }

    fn json_body(response: &Response) -> Json {
        parse(std::str::from_utf8(&response.body).unwrap()).unwrap()
    }

    fn create(api: &Api, body: &str) -> Response {
        api.handle(&request("POST", "/collections", body))
    }

    #[test]
    fn the_full_lifecycle_works_over_the_api() {
        let api = api("lifecycle");
        let created = create(
            &api,
            r#"{"name":"docs","dim":4,"metric":"euclidean","index":{"type":"flat"}}"#,
        );
        assert_eq!(created.status, 201);

        let upserted = api.handle(&request(
            "POST",
            "/collections/docs/points",
            r#"{"points":[
                {"id":1,"vector":[1,0,0,0],"metadata":{"lang":"en","year":2024}},
                {"id":2,"vector":[0,1,0,0],"metadata":{"lang":"fr","year":2020}},
                {"id":3,"vector":[0.9,0.1,0,0],"metadata":{"lang":"en","year":2025}}
            ]}"#,
        ));
        assert_eq!(upserted.status, 200);
        assert_eq!(
            json_body(&upserted)
                .get("upserted")
                .and_then(Json::as_usize),
            Some(3)
        );

        let searched = api.handle(&request(
            "POST",
            "/collections/docs/search",
            r#"{"vector":[1,0,0,0],"k":2}"#,
        ));
        assert_eq!(searched.status, 200);
        let results = json_body(&searched);
        let hits = results.get("results").and_then(Json::as_array).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].get("id").and_then(Json::as_u64), Some(1));
        // Distance and metadata both travel with the hit.
        assert!(hits[0].get("distance").is_some());
        assert_eq!(
            hits[0]
                .get("metadata")
                .and_then(|m| m.get("lang"))
                .and_then(Json::as_str),
            Some("en")
        );

        let filtered = api.handle(&request(
            "POST",
            "/collections/docs/search",
            r#"{"vector":[1,0,0,0],"k":3,"filter":{"lang":"en","year":{"gte":2025}}}"#,
        ));
        let hits = json_body(&filtered);
        let hits = hits.get("results").and_then(Json::as_array).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].get("id").and_then(Json::as_u64), Some(3));

        let deleted = api.handle(&request(
            "POST",
            "/collections/docs/points/delete",
            r#"{"ids":[1,999]}"#,
        ));
        assert_eq!(
            json_body(&deleted).get("deleted").and_then(Json::as_usize),
            Some(1)
        );

        let described = api.handle(&request("GET", "/collections/docs", ""));
        assert_eq!(described.status, 200);
        let description = json_body(&described);
        assert_eq!(description.get("vectors").and_then(Json::as_usize), Some(2));
        assert_eq!(
            description.get("index").and_then(Json::as_str),
            Some("flat")
        );
        assert_eq!(
            description
                .get("metadata_keys")
                .and_then(Json::as_array)
                .map(<[Json]>::len),
            Some(2)
        );

        assert_eq!(
            api.handle(&request("POST", "/collections/docs/snapshot", ""))
                .status,
            200
        );
        assert_eq!(
            api.handle(&request("POST", "/collections/docs/compact", ""))
                .status,
            200
        );
        assert_eq!(
            api.handle(&request("DELETE", "/collections/docs", ""))
                .status,
            200
        );
        assert_eq!(
            api.handle(&request("GET", "/collections/docs", "")).status,
            404
        );
        let _ = std::fs::remove_dir_all(api.config().data_directory.clone());
    }

    #[test]
    fn collections_survive_a_restart() {
        let directory = std::env::temp_dir().join(format!(
            "proxima-api-restart-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        let config = ApiConfig {
            data_directory: directory.clone(),
            ..ApiConfig::default()
        };

        {
            let api = Api::open(config.clone()).unwrap();
            create(&api, r#"{"name":"a","dim":3,"metric":"cosine"}"#);
            api.handle(&request(
                "POST",
                "/collections/a/points",
                r#"{"points":[{"id":7,"vector":[1,2,3]}]}"#,
            ));
        }

        let api = Api::open(config).unwrap();
        let described = api.handle(&request("GET", "/collections/a", ""));
        assert_eq!(described.status, 200);
        let body = json_body(&described);
        assert_eq!(body.get("vectors").and_then(Json::as_usize), Some(1));
        assert_eq!(body.get("metric").and_then(Json::as_str), Some("cosine"));
        assert_eq!(body.get("index").and_then(Json::as_str), Some("hnsw"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn health_and_metrics_answer_before_anything_exists() {
        let api = api("health");
        let health = api.handle(&request("GET", "/health", ""));
        assert_eq!(health.status, 200);
        assert_eq!(
            json_body(&health).get("status").and_then(Json::as_str),
            Some("ok")
        );

        let metrics = api.handle(&request("GET", "/metrics", ""));
        assert_eq!(metrics.status, 200);
        let text = String::from_utf8(metrics.body).unwrap();
        assert!(text.contains("proxima_up 1"));
        assert!(text.contains("# TYPE proxima_requests_total counter"));
        let _ = std::fs::remove_dir_all(api.config().data_directory.clone());
    }

    #[test]
    fn bad_requests_are_named_precisely() {
        let api = api("errors");
        create(&api, r#"{"name":"c","dim":3,"metric":"euclidean"}"#);

        let cases = [
            ("POST", "/collections", r#"{"dim":3}"#, 400, "'name'"),
            (
                "POST",
                "/collections",
                r#"{"name":"../escape","dim":3}"#,
                400,
                "only letters",
            ),
            (
                "POST",
                "/collections",
                r#"{"name":"x","dim":3,"metric":"hamming"}"#,
                400,
                "unknown metric",
            ),
            (
                "POST",
                "/collections",
                r#"{"name":"c","dim":3}"#,
                409,
                "already exists",
            ),
            ("POST", "/collections", "{not json", 400, "byte"),
            (
                "POST",
                "/collections/c/points",
                r#"{"points":[{"id":1,"vector":[1,2]}]}"#,
                400,
                "dimension mismatch",
            ),
            (
                "POST",
                "/collections/c/points",
                r#"{"points":[{"vector":[1,2,3]}]}"#,
                400,
                "points[0]",
            ),
            (
                "POST",
                "/collections/c/search",
                r#"{"vector":[1,2,3],"filter":{"a":{"nope":1}}}"#,
                400,
                "unknown operator",
            ),
            ("GET", "/collections/missing", "", 404, "no collection"),
            ("PUT", "/collections", "{}", 405, "not allowed"),
            ("GET", "/nowhere", "", 404, "no route"),
        ];
        for (method, path, body, status, fragment) in cases {
            let response = api.handle(&request(method, path, body));
            assert_eq!(response.status, status, "{method} {path} -> {response:?}");
            let text = String::from_utf8(response.body).unwrap();
            assert!(text.contains(fragment), "{method} {path}: {text}");
        }
        let _ = std::fs::remove_dir_all(api.config().data_directory.clone());
    }

    #[test]
    fn every_index_type_can_be_created() {
        let api = api("index-types");
        for (name, spec) in [
            ("flat", r#"{"type":"flat"}"#),
            (
                "hnsw",
                r#"{"type":"hnsw","m":8,"ef_construction":50,"ef_search":16,"seed":3}"#,
            ),
            (
                "ivfpq",
                r#"{"type":"ivfpq","nlist":8,"subspaces":4,"nprobe":2,"rerank":10,"train_threshold":100}"#,
            ),
        ] {
            let body =
                format!(r#"{{"name":"{name}","dim":8,"metric":"euclidean","index":{spec}}}"#);
            let response = create(&api, &body);
            assert_eq!(response.status, 201, "{name}: {response:?}");
        }
        let bad = create(&api, r#"{"name":"nope","dim":8,"index":{"type":"btree"}}"#);
        assert_eq!(bad.status, 400);
        let _ = std::fs::remove_dir_all(api.config().data_directory.clone());
    }

    #[test]
    fn metadata_types_round_trip_through_the_api() {
        let api = api("metadata");
        create(
            &api,
            r#"{"name":"m","dim":2,"metric":"euclidean","index":{"type":"flat"}}"#,
        );
        api.handle(&request(
            "POST",
            "/collections/m/points",
            r#"{"points":[{"id":1,"vector":[1,0],"metadata":{"s":"x","i":42,"f":1.5,"b":true}}]}"#,
        ));
        let response = api.handle(&request(
            "POST",
            "/collections/m/search",
            r#"{"vector":[1,0],"k":1}"#,
        ));
        let body = json_body(&response);
        let metadata = body.get("results").and_then(Json::as_array).unwrap()[0]
            .get("metadata")
            .unwrap()
            .clone();
        assert_eq!(metadata.get("s").and_then(Json::as_str), Some("x"));
        assert_eq!(metadata.get("i").and_then(Json::as_u64), Some(42));
        assert_eq!(metadata.get("f").and_then(Json::as_f64), Some(1.5));
        assert_eq!(metadata.get("b").and_then(Json::as_bool), Some(true));

        let rejected = api.handle(&request(
            "POST",
            "/collections/m/points",
            r#"{"points":[{"id":2,"vector":[0,1],"metadata":{"nested":{"a":1}}}]}"#,
        ));
        assert_eq!(rejected.status, 400);
        let _ = std::fs::remove_dir_all(api.config().data_directory.clone());
    }

    #[test]
    fn a_search_response_can_omit_metadata() {
        let api = api("no-metadata");
        create(
            &api,
            r#"{"name":"n","dim":2,"metric":"euclidean","index":{"type":"flat"}}"#,
        );
        api.handle(&request(
            "POST",
            "/collections/n/points",
            r#"{"points":[{"id":1,"vector":[1,0],"metadata":{"a":"b"}}]}"#,
        ));
        let response = api.handle(&request(
            "POST",
            "/collections/n/search",
            r#"{"vector":[1,0],"k":1,"with_metadata":false}"#,
        ));
        let body = json_body(&response);
        assert!(body.get("results").and_then(Json::as_array).unwrap()[0]
            .get("metadata")
            .is_none());
        let _ = std::fs::remove_dir_all(api.config().data_directory.clone());
    }
}
