//! The subset of HTTP/1.1 this server speaks, and the subset it refuses.
//!
//! # What is implemented
//!
//! Request line, headers, `Content-Length` bodies, `Connection: keep-alive`,
//! and status responses. That is enough for a JSON API and it is deliberately
//! all of it.
//!
//! # What is not, and why that is a decision rather than a gap
//!
//! * **No `Transfer-Encoding: chunked` on requests.** Chunked decoding is where
//!   HTTP parsers acquire their CVEs, and every client that talks to this API
//!   knows its body length. A chunked request is answered with `411 Length
//!   Required`.
//! * **No TLS.** Terminate it in front. A hand-rolled TLS stack would be an
//!   unambiguously worse idea than a hand-rolled HTTP parser.
//! * **No HTTP/2.** It needs HPACK and a multiplexer, and the win over
//!   keep-alive HTTP/1.1 for a request/response API on a local network is
//!   small.
//! * **No compression.** A `Content-Encoding: gzip` would need a deflate
//!   implementation, which is a larger project than this server.
//!
//! # The limits, which are the security posture
//!
//! Every one of these exists because a server that trusts its input is a server
//! that can be stopped by one client: [`MAX_REQUEST_LINE`],
//! [`MAX_HEADER_BYTES`], [`MAX_HEADERS`] and a caller-supplied body cap. A
//! request that exceeds any of them is answered and the connection closed,
//! rather than the reader growing a buffer until the process dies.

use std::collections::BTreeMap;
use std::io::{BufReader, Read, Write};
use std::net::TcpStream;

/// Longest `METHOD /path HTTP/1.1` line accepted.
pub const MAX_REQUEST_LINE: usize = 8 * 1024;
/// Total header bytes accepted.
pub const MAX_HEADER_BYTES: usize = 32 * 1024;
/// Number of header lines accepted.
pub const MAX_HEADERS: usize = 100;

/// A parsed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// `GET`, `POST`, `DELETE`, ... uppercased as received.
    pub method: String,
    /// Path with the query string removed and percent-escapes decoded.
    pub path: String,
    /// Raw query string, or empty.
    pub query: String,
    /// Headers, with **lowercased** names, because HTTP header names are
    /// case-insensitive and comparing them as received is a bug that only shows
    /// up against an unusual client.
    pub headers: BTreeMap<String, String>,
    /// Body bytes, exactly `Content-Length` of them.
    pub body: Vec<u8>,
    /// Whether the connection should be kept open afterwards.
    pub keep_alive: bool,
}

impl Request {
    /// The body as UTF-8, or `None` if it is not.
    pub fn body_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.body).ok()
    }

    /// The path split on `/`, with empty segments removed.
    ///
    /// `/collections/docs/points` becomes `["collections", "docs", "points"]`,
    /// which is what the router matches on.
    pub fn segments(&self) -> Vec<&str> {
        self.path.split('/').filter(|s| !s.is_empty()).collect()
    }

    /// The value of a query parameter, percent-decoded.
    pub fn query_param(&self, name: &str) -> Option<String> {
        for pair in self.query.split('&') {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            if percent_decode(key) == name {
                return Some(percent_decode(value));
            }
        }
        None
    }
}

/// A response, ready to write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// Status code.
    pub status: u16,
    /// `Content-Type` value.
    pub content_type: String,
    /// Body bytes.
    pub body: Vec<u8>,
    /// Extra headers.
    pub headers: Vec<(String, String)>,
}

impl Response {
    /// A JSON response with a status.
    pub fn json(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Response {
            status,
            content_type: "application/json".to_owned(),
            body: body.into(),
            headers: Vec::new(),
        }
    }

    /// A `text/plain` response, for `/metrics` and for `503`s.
    pub fn text(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Response {
            status,
            content_type: "text/plain; charset=utf-8".to_owned(),
            body: body.into(),
            headers: Vec::new(),
        }
    }

    /// A response for the Prometheus exposition format.
    pub fn prometheus(body: String) -> Self {
        Response {
            status: 200,
            content_type: "text/plain; version=0.0.4; charset=utf-8".to_owned(),
            body: body.into_bytes(),
            headers: Vec::new(),
        }
    }

    /// Add a header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// Why a request could not be read.
#[derive(Debug)]
pub enum ReadError {
    /// The peer closed the connection cleanly before sending anything.
    ///
    /// Not an error condition: it is what a keep-alive connection does when the
    /// client is finished, and it must not be logged as a fault.
    Closed,
    /// The request was malformed or exceeded a limit. The `u16` is the status to
    /// answer with and the `String` is the reason.
    Malformed(u16, String),
    /// The socket failed.
    Io(std::io::Error),
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Closed => f.write_str("connection closed by peer"),
            ReadError::Malformed(status, reason) => write!(f, "{status}: {reason}"),
            ReadError::Io(err) => write!(f, "io error: {err}"),
        }
    }
}

/// Read one request from `reader`.
///
/// `max_body` caps `Content-Length`; a larger declared body is refused with
/// `413` **before** anything is read, so an attacker cannot make the server
/// allocate by lying about a length it never intends to send.
pub fn read_request(
    reader: &mut BufReader<TcpStream>,
    max_body: usize,
) -> Result<Request, ReadError> {
    let mut line = String::new();
    let read = read_line(reader, &mut line, MAX_REQUEST_LINE)?;
    if read == 0 {
        return Err(ReadError::Closed);
    }

    // Owned copies, because `line` is reused for the header lines below and the
    // borrow checker is right to object to holding slices of it across that.
    let (method, target, version) = {
        let mut parts = line.trim_end().split(' ');
        let method = parts
            .next()
            .ok_or_else(|| ReadError::Malformed(400, "empty request line".into()))?
            .to_owned();
        let target = parts
            .next()
            .ok_or_else(|| ReadError::Malformed(400, "no request target".into()))?
            .to_owned();
        let version = parts.next().unwrap_or("HTTP/1.1").to_owned();
        (method, target, version)
    };
    if method.is_empty() || !version.starts_with("HTTP/1.") {
        return Err(ReadError::Malformed(400, "not an HTTP/1.x request".into()));
    }

    let (raw_path, query) = target.split_once('?').unwrap_or((target.as_str(), ""));
    let path = percent_decode(raw_path);
    let query = query.to_owned();

    let mut headers = BTreeMap::new();
    let mut header_bytes = 0usize;
    loop {
        line.clear();
        let read = read_line(reader, &mut line, MAX_REQUEST_LINE)?;
        if read == 0 {
            return Err(ReadError::Malformed(400, "headers not terminated".into()));
        }
        header_bytes += read;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(ReadError::Malformed(431, "headers too large".into()));
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if headers.len() >= MAX_HEADERS {
            return Err(ReadError::Malformed(431, "too many headers".into()));
        }
        let Some((name, value)) = trimmed.split_once(':') else {
            return Err(ReadError::Malformed(400, "malformed header".into()));
        };
        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
    }

    if headers
        .get("transfer-encoding")
        .is_some_and(|value| value.to_ascii_lowercase().contains("chunked"))
    {
        return Err(ReadError::Malformed(
            411,
            "chunked request bodies are not supported; send Content-Length".into(),
        ));
    }

    let length = match headers.get("content-length") {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| ReadError::Malformed(400, "invalid Content-Length".into()))?,
        None => 0,
    };
    if length > max_body {
        return Err(ReadError::Malformed(
            413,
            format!("body of {length} bytes exceeds the {max_body}-byte limit"),
        ));
    }

    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).map_err(ReadError::Io)?;

    // HTTP/1.1 defaults to keep-alive, HTTP/1.0 to close, and an explicit
    // `Connection` header overrides both.
    let keep_alive = match headers.get("connection").map(|v| v.to_ascii_lowercase()) {
        Some(value) if value.contains("close") => false,
        Some(value) if value.contains("keep-alive") => true,
        _ => version == "HTTP/1.1",
    };

    Ok(Request {
        method,
        path,
        query,
        headers,
        body,
        keep_alive,
    })
}

/// Read a line, bounded.
///
/// `BufRead::read_line` has no limit, so a client that opens a connection and
/// sends `A` forever makes the server allocate forever. This is the same
/// function with a ceiling.
fn read_line(
    reader: &mut BufReader<TcpStream>,
    out: &mut String,
    limit: usize,
) -> Result<usize, ReadError> {
    let mut total = 0usize;
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Ok(total),
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                return Err(ReadError::Malformed(408, "read timed out".into()))
            }
            Err(err) => return Err(ReadError::Io(err)),
        }
        total += 1;
        if total > limit {
            return Err(ReadError::Malformed(431, "request line too long".into()));
        }
        out.push(byte[0] as char);
        if byte[0] == b'\n' {
            return Ok(total);
        }
    }
}

/// Write a response.
pub fn write_response(
    stream: &mut TcpStream,
    response: &Response,
    keep_alive: bool,
) -> std::io::Result<()> {
    let mut head = String::with_capacity(256);
    head.push_str("HTTP/1.1 ");
    head.push_str(&response.status.to_string());
    head.push(' ');
    head.push_str(status_text(response.status));
    head.push_str("\r\n");
    head.push_str("Content-Type: ");
    head.push_str(&response.content_type);
    head.push_str("\r\nContent-Length: ");
    head.push_str(&response.body.len().to_string());
    head.push_str("\r\nConnection: ");
    head.push_str(if keep_alive { "keep-alive" } else { "close" });
    head.push_str("\r\n");
    for (name, value) in &response.headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");

    // One write for the head and one for the body, then an explicit flush:
    // `TcpStream` is unbuffered, so this is already two syscalls and combining
    // them would mean copying the body.
    stream.write_all(head.as_bytes())?;
    stream.write_all(&response.body)?;
    stream.flush()
}

/// The reason phrase for a status code.
///
/// Only the codes this server emits. An unknown code gets a generic phrase
/// rather than a panic, because the phrase is decorative — clients read the
/// number.
pub fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    }
}

/// Decode `%XX` escapes and `+` in a path or query value.
///
/// Invalid escapes are left alone rather than rejected: a path with a stray `%`
/// is a `404`, and turning it into a `400` tells a client less.
pub fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("");
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    out.push(byte);
                    index += 3;
                } else {
                    out.push(bytes[index]);
                    index += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn request(path: &str, query: &str) -> Request {
        Request {
            method: "GET".into(),
            path: path.into(),
            query: query.into(),
            headers: BTreeMap::new(),
            body: Vec::new(),
            keep_alive: true,
        }
    }

    #[test]
    fn segments_ignore_empty_components() {
        assert_eq!(
            request("/collections/docs/points", "").segments(),
            vec!["collections", "docs", "points"]
        );
        assert_eq!(request("/", "").segments(), Vec::<&str>::new());
        assert_eq!(request("//a//b//", "").segments(), vec!["a", "b"]);
    }

    #[test]
    fn query_parameters_are_decoded() {
        let request = request("/x", "k=10&name=my%20collection&flag");
        assert_eq!(request.query_param("k").as_deref(), Some("10"));
        assert_eq!(
            request.query_param("name").as_deref(),
            Some("my collection")
        );
        assert_eq!(request.query_param("flag").as_deref(), Some(""));
        assert_eq!(request.query_param("absent"), None);
    }

    #[test]
    fn percent_decoding_leaves_broken_escapes_alone() {
        assert_eq!(percent_decode("a%2Fb"), "a/b");
        assert_eq!(percent_decode("a+b"), "a b");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
        assert_eq!(percent_decode("caf%C3%A9"), "café");
    }

    #[test]
    fn status_texts_cover_everything_the_api_emits() {
        for status in [
            200, 201, 204, 400, 404, 405, 409, 411, 413, 415, 422, 429, 431, 500, 503,
        ] {
            assert_ne!(status_text(status), "Status", "{status}");
        }
        assert_eq!(status_text(418), "Status");
    }

    #[test]
    fn responses_carry_their_content_type() {
        assert_eq!(Response::json(200, "{}").content_type, "application/json");
        assert!(Response::prometheus(String::new())
            .content_type
            .contains("version=0.0.4"));
        let with_header = Response::text(503, "x").with_header("Retry-After", "1");
        assert_eq!(
            with_header.headers,
            vec![("Retry-After".into(), "1".into())]
        );
    }

    #[test]
    fn a_body_that_is_not_utf8_is_reported_rather_than_lossily_decoded() {
        let mut request = request("/x", "");
        request.body = vec![0xFF, 0xFE];
        assert_eq!(request.body_str(), None);
    }
}
