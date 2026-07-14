//! The payload attached to a vector, and the values a filter can test.
//!
//! This is deliberately a *small*, closed value model rather than arbitrary
//! JSON. Filters run inside the graph traversal — once per visited node, tens
//! of thousands of times per query — so the cost of reading a field has to be a
//! hash lookup and a discriminant check, not a parse. Anything richer belongs
//! in the caller's own store, keyed by [`VectorId`](crate::VectorId).

use std::collections::HashMap;

/// A single metadata value.
///
/// Four cases, chosen because they are what filters actually test: string
/// equality and set membership (`Text`), range and equality on counts and
/// timestamps (`Int`), range on scores (`Float`), and flags (`Bool`).
///
/// There is no `Null`: an absent key is the null, and
/// [`Metadata::get`] returning `None` is how a filter sees it. Encoding absence
/// twice is how filter predicates end up with three-valued logic bugs.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// UTF-8 text.
    Text(String),
    /// Signed 64-bit integer.
    Int(i64),
    /// 64-bit float. `f64` rather than `f32` because metadata is not in the
    /// distance path and timestamps-as-seconds need the mantissa.
    Float(f64),
    /// Boolean flag.
    Bool(bool),
}

impl MetadataValue {
    /// The text, if this is [`MetadataValue::Text`].
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MetadataValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// The integer, if this is [`MetadataValue::Int`].
    pub fn as_int(&self) -> Option<i64> {
        match self {
            MetadataValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// The float, if this is [`MetadataValue::Float`].
    ///
    /// Does **not** widen an [`MetadataValue::Int`]. A filter that means
    /// "numeric" should say which, because silently coercing is how a range
    /// filter starts matching the wrong rows after an ingest change.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            MetadataValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// The flag, if this is [`MetadataValue::Bool`].
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Bytes this value occupies, heap included, for [`Metadata::heap_bytes`].
    fn heap_bytes(&self) -> usize {
        match self {
            MetadataValue::Text(s) => s.capacity(),
            _ => 0,
        }
    }
}

impl From<&str> for MetadataValue {
    fn from(v: &str) -> Self {
        MetadataValue::Text(v.to_owned())
    }
}

impl From<String> for MetadataValue {
    fn from(v: String) -> Self {
        MetadataValue::Text(v)
    }
}

impl From<i64> for MetadataValue {
    fn from(v: i64) -> Self {
        MetadataValue::Int(v)
    }
}

impl From<f64> for MetadataValue {
    fn from(v: f64) -> Self {
        MetadataValue::Float(v)
    }
}

impl From<bool> for MetadataValue {
    fn from(v: bool) -> Self {
        MetadataValue::Bool(v)
    }
}

/// Key/value payload stored alongside a vector.
///
/// Backed by a `HashMap`, and empty metadata allocates nothing — which matters,
/// because the store keeps one of these per slot whether or not the caller uses
/// them.
///
/// ```
/// use proxima_core::Metadata;
///
/// let meta = Metadata::new()
///     .with("lang", "en")
///     .with("year", 2024_i64)
///     .with("public", true);
///
/// assert_eq!(meta.text("lang"), Some("en"));
/// assert_eq!(meta.int("year"), Some(2024));
/// assert_eq!(meta.bool("public"), Some(true));
/// assert_eq!(meta.text("missing"), None);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metadata {
    fields: HashMap<String, MetadataValue>,
}

impl Metadata {
    /// Empty metadata. Does not allocate.
    pub fn new() -> Self {
        Metadata::default()
    }

    /// Builder-style insert, for constructing metadata in an expression.
    pub fn with(mut self, key: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Insert or replace a field, returning the previous value.
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Option<MetadataValue> {
        self.fields.insert(key.into(), value.into())
    }

    /// Remove a field, returning it.
    pub fn remove(&mut self, key: &str) -> Option<MetadataValue> {
        self.fields.remove(key)
    }

    /// Look a field up.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.fields.get(key)
    }

    /// Shorthand for `get(key).and_then(MetadataValue::as_text)`.
    pub fn text(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(MetadataValue::as_text)
    }

    /// Shorthand for `get(key).and_then(MetadataValue::as_int)`.
    pub fn int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(MetadataValue::as_int)
    }

    /// Shorthand for `get(key).and_then(MetadataValue::as_float)`.
    pub fn float(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(MetadataValue::as_float)
    }

    /// Shorthand for `get(key).and_then(MetadataValue::as_bool)`.
    pub fn bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(MetadataValue::as_bool)
    }

    /// Number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// `true` if there are no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Iterate over the fields.
    ///
    /// Order is `HashMap` order, which is neither stable nor meaningful. Nothing
    /// in Proxima's search path iterates metadata, precisely so that no result
    /// can depend on it.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &MetadataValue)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Approximate heap footprint in bytes.
    ///
    /// Counts key and string-value capacities plus the map's own table. Used by
    /// index statistics, where "approximate" is the honest word: the `HashMap`
    /// table size is estimated from its capacity.
    pub fn heap_bytes(&self) -> usize {
        let entries: usize = self
            .fields
            .iter()
            .map(|(k, v)| k.capacity() + v.heap_bytes())
            .sum();
        let table = self.fields.capacity()
            * (std::mem::size_of::<String>() + std::mem::size_of::<MetadataValue>() + 1);
        entries + table
    }
}

impl<K, V> FromIterator<(K, V)> for Metadata
where
    K: Into<String>,
    V: Into<MetadataValue>,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Metadata {
            fields: iter
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    // Exact float comparison is deliberate here: every constant below is
    // exactly representable and the point is that the kernel returns it
    // bit-for-bit.
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn accessors_do_not_coerce_between_numeric_types() {
        // Documented decision: `Int` is not readable as a float. A filter that
        // silently widened would change meaning when an ingest job starts
        // writing 3 instead of 3.0.
        let meta = Metadata::new().with("n", 3_i64);
        assert_eq!(meta.int("n"), Some(3));
        assert_eq!(meta.float("n"), None);
    }

    #[test]
    fn empty_metadata_does_not_allocate() {
        assert_eq!(Metadata::new().heap_bytes(), 0);
    }

    #[test]
    fn collects_from_pairs() {
        let meta: Metadata = [("a", 1_i64), ("b", 2_i64)].into_iter().collect();
        assert_eq!(meta.len(), 2);
        assert_eq!(meta.int("b"), Some(2));
    }

    #[test]
    fn remove_reports_the_old_value() {
        let mut meta = Metadata::new().with("k", "v");
        assert_eq!(meta.remove("k"), Some(MetadataValue::Text("v".into())));
        assert!(meta.is_empty());
    }
}
