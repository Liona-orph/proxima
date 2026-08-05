//! A JSON value, a parser and a writer, in about four hundred lines.
//!
//! # Why not `serde_json`
//!
//! This is the one place in the repository where the "write it yourself"
//! instinct deserves the most scrutiny, because `serde_json` is excellent and
//! this is not better than it. The argument is narrow and it is about what the
//! server actually parses:
//!
//! * The request bodies are a closed set of six shapes, each a handful of
//!   fields. There is no user-extensible schema and no place where `#[derive]`
//!   would save meaningful work.
//! * The one large body — a batch upsert — is a list of `f32` arrays, and the
//!   fast path for that is a number parser, which is the same amount of code
//!   either way.
//! * `serde` plus `serde_json` plus their derive macro is the largest
//!   dependency the workspace would have taken, and the workspace's stated
//!   position is that the interesting parts are in-tree.
//!
//! What is given up, honestly: `serde_json` is faster on large bodies, it has
//! years of fuzzing behind it, and it handles number edge cases this does not
//! try to. See the limits below.
//!
//! # What this parser does and does not accept
//!
//! It accepts RFC 8259 JSON with these deliberate restrictions:
//!
//! * **Numbers are `f64`.** There is no integer type on the wire, so a metadata
//!   `Int` arrives as a float and is converted back if it is integral. That is
//!   the same lossiness JavaScript has and it is bounded at `2^53`.
//! * **Nesting is capped** at [`MAX_DEPTH`]. A JSON parser without a depth limit
//!   is a stack overflow waiting for `[[[[[...`, and a stack overflow in a
//!   server is an abort, not an error.
//! * **The whole body is buffered** before parsing. Streaming would matter for
//!   multi-gigabyte uploads; the server caps a request body instead.
//! * Duplicate object keys keep the last occurrence, which is what every other
//!   implementation does.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Maximum nesting depth. Deep enough for any body this API defines, shallow
/// enough that the recursive descent cannot exhaust an 8 MiB stack.
pub const MAX_DEPTH: usize = 32;

/// A JSON value.
///
/// Objects are a `BTreeMap`, not a `HashMap`, so that serialisation is
/// deterministic: an API whose response bytes differ between processes cannot
/// be tested against a fixture, cached by content, or diffed in a bug report.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// Any number, held as `f64`.
    Number(f64),
    /// A string, already unescaped.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object, with keys in sorted order.
    Object(BTreeMap<String, Json>),
}

impl Json {
    /// Build an object from pairs, for constructing responses inline.
    pub fn object<K: Into<String>>(fields: impl IntoIterator<Item = (K, Json)>) -> Self {
        Json::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
        )
    }

    /// A string value.
    pub fn string(value: impl Into<String>) -> Self {
        Json::String(value.into())
    }

    /// A number from anything that converts into `f64`.
    pub fn number(value: impl Into<f64>) -> Self {
        Json::Number(value.into())
    }

    /// A number from a `usize`, which has no `Into<f64>`.
    pub fn count(value: usize) -> Self {
        Json::Number(value as f64)
    }

    /// The field named `key`, if this is an object that has one.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(fields) => fields.get(key),
            _ => None,
        }
    }

    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(value) => Some(value),
            _ => None,
        }
    }

    /// The number, if this is one.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// The number as a `usize`, if it is a non-negative integer.
    ///
    /// Rejects `1.5` rather than truncating it: a `k` of `1.5` is a client bug
    /// and silently answering `1` hides it.
    pub fn as_usize(&self) -> Option<usize> {
        let value = self.as_f64()?;
        if value < 0.0 || value.fract() != 0.0 || value > 9_007_199_254_740_992.0 {
            return None;
        }
        Some(value as usize)
    }

    /// The number as a `u64`, under the same rules as [`Json::as_usize`].
    pub fn as_u64(&self) -> Option<u64> {
        self.as_usize().map(|value| value as u64)
    }

    /// The boolean, if this is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// The array, if this is one.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(values) => Some(values),
            _ => None,
        }
    }

    /// The object's fields, if this is one.
    pub fn as_object(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Object(fields) => Some(fields),
            _ => None,
        }
    }

    /// An array of numbers as a vector of `f32`.
    ///
    /// The hot path of the whole API: every upserted vector and every query
    /// goes through it.
    pub fn as_f32_vec(&self) -> Option<Vec<f32>> {
        let values = self.as_array()?;
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            out.push(value.as_f64()? as f32);
        }
        Some(out)
    }

    /// The name of this value's type, for an error message.
    pub fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "boolean",
            Json::Number(_) => "number",
            Json::String(_) => "string",
            Json::Array(_) => "array",
            Json::Object(_) => "object",
        }
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Number(value) => write_number(*value, out),
            Json::String(value) => write_string(value, out),
            Json::Array(values) => {
                out.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    value.write(out);
                }
                out.push(']');
            }
            Json::Object(fields) => {
                out.push('{');
                for (index, (key, value)) in fields.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

impl std::fmt::Display for Json {
    /// Serialise to compact JSON.
    ///
    /// `Display` rather than an inherent `to_string`, so that a `Json` can be
    /// interpolated into a format string and so that the blanket `ToString`
    /// impl is the one callers get.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::with_capacity(64);
        self.write(&mut out);
        f.write_str(&out)
    }
}

/// JSON has no `NaN` and no `Infinity`, so both become `null`.
///
/// The alternative — emitting a bare `NaN` — produces a body that no client can
/// parse, which turns a surprising distance into a broken response. Proxima
/// rejects non-finite values at every input boundary, so this is unreachable
/// for stored data and exists for computed statistics.
fn write_number(value: f64, out: &mut String) {
    if !value.is_finite() {
        out.push_str("null");
        return;
    }
    if value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0 {
        // Integral values are written without a decimal point, so that ids and
        // counts read as integers on the wire rather than as `42.0`.
        let _ = write!(out, "{}", value as i64);
    } else {
        let _ = write!(out, "{value}");
    }
}

fn write_string(value: &str, out: &mut String) {
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // Everything below 0x20 must be escaped; `\u00XX` is always legal
            // and avoids a table of the ones with short forms.
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Why a body could not be parsed.
///
/// Carries the byte offset, because "expected `,`" without a position is not
/// actionable on a body a client generated programmatically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// What went wrong.
    pub message: String,
    /// Byte offset into the input.
    pub offset: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.message, self.offset)
    }
}

impl std::error::Error for ParseError {}

/// Parse a complete JSON document.
///
/// Trailing content after the value is an error: a body of `{} garbage` is a
/// client that built its request wrongly, and accepting the prefix would hide
/// it.
pub fn parse(input: &str) -> Result<Json, ParseError> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        position: 0,
        depth: 0,
    };
    parser.skip_whitespace();
    let value = parser.value()?;
    parser.skip_whitespace();
    if parser.position != parser.bytes.len() {
        return Err(parser.error("trailing content after the JSON value"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
    depth: usize,
}

impl Parser<'_> {
    fn error(&self, message: &str) -> ParseError {
        ParseError {
            message: message.to_owned(),
            offset: self.position,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.position += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), ParseError> {
        if self.peek() == Some(byte) {
            self.position += 1;
            Ok(())
        } else {
            Err(self.error(&format!("expected '{}'", byte as char)))
        }
    }

    fn literal(&mut self, text: &str, value: Json) -> Result<Json, ParseError> {
        if self.bytes[self.position..].starts_with(text.as_bytes()) {
            self.position += text.len();
            Ok(value)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn value(&mut self) -> Result<Json, ParseError> {
        match self.peek() {
            None => Err(self.error("unexpected end of input")),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("unexpected character")),
        }
    }

    fn enter(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.error("nesting deeper than 32 levels"));
        }
        Ok(())
    }

    fn array(&mut self) -> Result<Json, ParseError> {
        self.enter()?;
        self.expect(b'[')?;
        let mut values = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.position += 1;
            self.depth -= 1;
            return Ok(Json::Array(values));
        }
        loop {
            self.skip_whitespace();
            values.push(self.value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b']') => {
                    self.position += 1;
                    self.depth -= 1;
                    return Ok(Json::Array(values));
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
    }

    fn object(&mut self) -> Result<Json, ParseError> {
        self.enter()?;
        self.expect(b'{')?;
        let mut fields = BTreeMap::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.position += 1;
            self.depth -= 1;
            return Ok(Json::Object(fields));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.value()?;
            fields.insert(key, value);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b'}') => {
                    self.position += 1;
                    self.depth -= 1;
                    return Ok(Json::Object(fields));
                }
                _ => return Err(self.error("expected ',' or '}'")),
            }
        }
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| self.error("unterminated string"))?;
            self.position += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escape = self
                        .peek()
                        .ok_or_else(|| self.error("unterminated escape"))?;
                    self.position += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        _ => return Err(self.error("unknown escape")),
                    }
                }
                // Bare control characters are illegal in JSON strings, and
                // accepting them is how a log line ends up with an embedded
                // newline that splits it in two.
                0x00..=0x1F => return Err(self.error("unescaped control character")),
                _ => {
                    // Multi-byte UTF-8: copy the whole sequence. The input is a
                    // `&str`, so it is valid UTF-8 by construction and the
                    // continuation bytes are exactly the ones with the high two
                    // bits `10`.
                    let start = self.position - 1;
                    while matches!(self.peek(), Some(b) if b & 0xC0 == 0x80) {
                        self.position += 1;
                    }
                    match std::str::from_utf8(&self.bytes[start..self.position]) {
                        Ok(text) => out.push_str(text),
                        Err(_) => return Err(self.error("invalid UTF-8")),
                    }
                }
            }
        }
    }

    fn unicode_escape(&mut self) -> Result<char, ParseError> {
        let high = self.hex4()?;
        // Surrogate pairs: a code point above the BMP arrives as two escapes,
        // and decoding only the first produces a lone surrogate, which is not a
        // `char`.
        if (0xD800..0xDC00).contains(&high) {
            if !self.bytes[self.position..].starts_with(b"\\u") {
                return Err(self.error("lone high surrogate"));
            }
            self.position += 2;
            let low = self.hex4()?;
            if !(0xDC00..0xE000).contains(&low) {
                return Err(self.error("high surrogate not followed by a low one"));
            }
            let combined = 0x1_0000 + ((high - 0xD800) << 10) + (low - 0xDC00);
            return char::from_u32(combined).ok_or_else(|| self.error("invalid code point"));
        }
        char::from_u32(high).ok_or_else(|| self.error("invalid code point"))
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        if self.bytes.len() - self.position < 4 {
            return Err(self.error("truncated \\u escape"));
        }
        let text = std::str::from_utf8(&self.bytes[self.position..self.position + 4])
            .map_err(|_| self.error("invalid \\u escape"))?;
        let value = u32::from_str_radix(text, 16).map_err(|_| self.error("invalid \\u escape"))?;
        self.position += 4;
        Ok(value)
    }

    fn number(&mut self) -> Result<Json, ParseError> {
        let start = self.position;
        if self.peek() == Some(b'-') {
            self.position += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.position += 1;
        }
        if self.peek() == Some(b'.') {
            self.position += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.position])
            .map_err(|_| self.error("invalid number"))?;
        text.parse::<f64>()
            .map(Json::Number)
            .map_err(|_| ParseError {
                message: "invalid number".to_owned(),
                offset: start,
            })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn scalars_round_trip() {
        for text in [
            "null",
            "true",
            "false",
            "0",
            "-1",
            "1.5",
            "1e3",
            "-2.5e-3",
            "\"\"",
            "\"hello\"",
        ] {
            let value = parse(text).unwrap();
            assert_eq!(parse(&value.to_string()).unwrap(), value, "{text}");
        }
    }

    #[test]
    fn integral_numbers_are_written_without_a_decimal_point() {
        assert_eq!(Json::Number(42.0).to_string(), "42");
        assert_eq!(Json::Number(-7.0).to_string(), "-7");
        assert_eq!(Json::Number(0.5).to_string(), "0.5");
    }

    #[test]
    fn non_finite_numbers_become_null_rather_than_invalid_json() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let text = Json::Number(value).to_string();
            assert_eq!(text, "null");
            assert!(parse(&text).is_ok());
        }
    }

    #[test]
    fn objects_serialise_in_key_order_every_time() {
        let text = r#"{"zebra":1,"apple":2,"mango":3}"#;
        let value = parse(text).unwrap();
        assert_eq!(value.to_string(), r#"{"apple":2,"mango":3,"zebra":1}"#);
        for _ in 0..8 {
            assert_eq!(parse(text).unwrap().to_string(), value.to_string());
        }
    }

    #[test]
    fn escapes_round_trip_including_control_characters() {
        let original = "quote \" backslash \\ newline \n tab \t bell \u{7}";
        let encoded = Json::string(original).to_string();
        assert!(encoded.contains("\\u0007"));
        assert_eq!(parse(&encoded).unwrap().as_str(), Some(original));
    }

    #[test]
    fn surrogate_pairs_decode_to_one_char() {
        assert_eq!(
            parse(r#""\ud83d\ude00""#).unwrap().as_str(),
            Some("\u{1F600}")
        );
        assert!(parse(r#""\ud83d""#).is_err());
        assert!(parse(r#""\ud83d\u0041""#).is_err());
    }

    #[test]
    fn multibyte_utf8_survives_unescaped() {
        let text = "\"naïve — 日本語\"";
        assert_eq!(parse(text).unwrap().as_str(), Some("naïve — 日本語"));
    }

    #[test]
    fn nesting_is_capped_rather_than_overflowing_the_stack() {
        let deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        let err = parse(&deep).unwrap_err();
        assert!(err.to_string().contains("32 levels"), "{err}");

        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(parse(&ok).is_ok());
    }

    #[test]
    fn malformed_input_reports_a_position() {
        for text in [
            "{",
            "[1,",
            "{\"a\"}",
            "{\"a\":}",
            "tru",
            "\"unterminated",
            "[] extra",
            "\"\\q\"",
            "\"\n\"",
        ] {
            let err = parse(text).unwrap_err();
            assert!(err.offset <= text.len(), "{text}: {err}");
        }
    }

    #[test]
    fn accessors_reject_the_wrong_shape() {
        let value = parse(r#"{"k":10,"f":1.5,"s":"x","b":true,"a":[1,2],"n":-1}"#).unwrap();
        assert_eq!(value.get("k").and_then(Json::as_usize), Some(10));
        assert_eq!(
            value.get("f").and_then(Json::as_usize),
            None,
            "1.5 is not a count"
        );
        assert_eq!(
            value.get("n").and_then(Json::as_usize),
            None,
            "-1 is not a count"
        );
        assert_eq!(value.get("s").and_then(Json::as_str), Some("x"));
        assert_eq!(value.get("b").and_then(Json::as_bool), Some(true));
        assert_eq!(
            value.get("a").and_then(Json::as_f32_vec),
            Some(vec![1.0, 2.0])
        );
        assert_eq!(value.get("s").and_then(Json::as_f32_vec), None);
        assert_eq!(value.get("missing"), None);
        assert_eq!(value.type_name(), "object");
        assert!(value.as_object().unwrap().contains_key("k"));
    }

    #[test]
    fn a_vector_body_parses_to_f32() {
        let body = parse(r#"{"vector":[0.1,-2,3e2]}"#).unwrap();
        assert_eq!(
            body.get("vector").and_then(Json::as_f32_vec),
            Some(vec![0.1f32, -2.0, 300.0])
        );
    }

    #[test]
    fn builders_produce_what_they_claim() {
        let value = Json::object([
            ("a", Json::count(3)),
            ("b", Json::string("x")),
            ("c", Json::number(1.5f32)),
            ("d", Json::Array(vec![Json::Null])),
        ]);
        assert_eq!(value.to_string(), r#"{"a":3,"b":"x","c":1.5,"d":[null]}"#);
    }
}
