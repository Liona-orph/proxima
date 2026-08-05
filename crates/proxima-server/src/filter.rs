//! The metadata filter language, and the reason it is small.
//!
//! # The language
//!
//! A filter is a JSON object. Every key names a metadata field, every value is
//! a condition on it, and **all of them must hold** — the top level is an AND.
//!
//! ```json
//! {
//!   "lang":   "en",
//!   "year":   {"gte": 2024},
//!   "score":  {"gt": 0.8, "lte": 1.0},
//!   "status": {"in": ["published", "reviewed"]},
//!   "draft":  {"exists": false}
//! }
//! ```
//!
//! A bare scalar is equality. The operators are `eq`, `ne`, `gt`, `gte`, `lt`,
//! `lte`, `in` and `exists`. That is the whole language.
//!
//! # Why there is no OR, and no nesting
//!
//! Because of where a filter runs. It is evaluated **inside the graph
//! traversal**, once per visited node — tens of thousands of times per query —
//! so its cost is multiplied by the search's cost, and a filter that is
//! expensive to evaluate does not merely slow the query down, it changes which
//! `ef` the escalation settles on and therefore how much of the graph is
//! walked. A conjunction of field predicates compiles to a fixed list of hash
//! lookups and comparisons with no allocation and no branching on shape. An
//! arbitrary boolean expression tree would be an interpreter on that path.
//!
//! The other half of the argument is selectivity. A disjunction is usually
//! *less* selective than its parts, and the filtered-search cliff described in
//! `HnswIndex::search` is about selectivity: past roughly 1-in-`ef` the search
//! degrades to a scan. A language that makes it easy to write a cheap, highly
//! selective filter and hard to write an expensive one is pointing users at the
//! part of the performance envelope that works.
//!
//! A caller who needs a disjunction can issue two searches and merge, which is
//! what the query planner of a real database would do anyway, and which makes
//! the cost visible.
//!
//! # Types are not coerced
//!
//! `{"year": {"gte": 2024}}` matches an `Int` field and not a `Float` one, and
//! `{"year": 2024}` does not match `"2024"`. This follows `MetadataValue`'s own
//! rule, and the reason is the same: a range filter that silently starts
//! matching a different set of rows after an ingest change is a bug nobody
//! notices for months.

use proxima_core::{Metadata, MetadataValue, VectorId};
use proxima_index::Filter;

use crate::json::Json;

/// One comparison against one field.
#[derive(Debug, Clone, PartialEq)]
enum Condition {
    Equals(MetadataValue),
    NotEquals(MetadataValue),
    Greater(Number),
    GreaterOrEqual(Number),
    Less(Number),
    LessOrEqual(Number),
    In(Vec<MetadataValue>),
    Exists(bool),
}

/// A numeric bound, kept typed so that an `Int` field is not compared against a
/// `Float` bound.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Number {
    Int(i64),
    Float(f64),
}

impl Number {
    /// Compare against a metadata value of the *same* variant, or fail.
    fn compare(self, value: &MetadataValue) -> Option<std::cmp::Ordering> {
        match (self, value) {
            (Number::Int(bound), MetadataValue::Int(field)) => Some(field.cmp(&bound)),
            (Number::Float(bound), MetadataValue::Float(field)) => Some(field.total_cmp(&bound)),
            // A JSON number with no fractional part parses as `Int`, so an
            // integral bound is also allowed against a `Float` field. The
            // reverse is not: `{"count": {"gt": 1.5}}` against an `Int` field
            // is a request that cannot be satisfied exactly, and answering it
            // approximately is how off-by-one filters happen.
            (Number::Int(bound), MetadataValue::Float(field)) => {
                Some(field.total_cmp(&(bound as f64)))
            }
            _ => None,
        }
    }
}

/// A compiled filter: a list of field conditions, all of which must hold.
///
/// Implements [`Filter`], so it is passed straight to
/// [`VectorIndex::search`](proxima_index::VectorIndex::search) and evaluated
/// during traversal.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetadataFilter {
    conditions: Vec<(String, Condition)>,
}

impl MetadataFilter {
    /// `true` if the filter has no conditions and therefore accepts everything.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Number of field conditions.
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    /// Compile a filter from its JSON form.
    ///
    /// Returns the reason as a `String` rather than a `ProximaError`: this is a
    /// client-input error whose only destination is a `400` body, and inventing
    /// an error variant per malformed-filter shape would bloat the workspace's
    /// error enum with cases nothing matches on.
    pub fn parse(value: &Json) -> Result<Self, String> {
        let Some(fields) = value.as_object() else {
            return Err(format!(
                "filter must be an object, got {}",
                value.type_name()
            ));
        };
        let mut conditions = Vec::new();
        for (field, condition) in fields {
            match condition {
                Json::Object(operators) => {
                    if operators.is_empty() {
                        return Err(format!("filter on '{field}' has no operators"));
                    }
                    for (operator, operand) in operators {
                        conditions
                            .push((field.clone(), compile_operator(field, operator, operand)?));
                    }
                }
                scalar => {
                    conditions.push((
                        field.clone(),
                        Condition::Equals(scalar_value(field, scalar)?),
                    ));
                }
            }
        }
        // Sorted by field name so that two filters that mean the same thing
        // evaluate in the same order, which keeps a benchmark reproducible.
        conditions.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(MetadataFilter { conditions })
    }
}

fn compile_operator(field: &str, operator: &str, operand: &Json) -> Result<Condition, String> {
    match operator {
        "eq" => Ok(Condition::Equals(scalar_value(field, operand)?)),
        "ne" => Ok(Condition::NotEquals(scalar_value(field, operand)?)),
        "gt" => Ok(Condition::Greater(number(field, operand)?)),
        "gte" => Ok(Condition::GreaterOrEqual(number(field, operand)?)),
        "lt" => Ok(Condition::Less(number(field, operand)?)),
        "lte" => Ok(Condition::LessOrEqual(number(field, operand)?)),
        "in" => {
            let Some(values) = operand.as_array() else {
                return Err(format!("'in' on '{field}' needs an array"));
            };
            if values.is_empty() {
                return Err(format!("'in' on '{field}' has an empty list, which can never match"));
            }
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                out.push(scalar_value(field, value)?);
            }
            Ok(Condition::In(out))
        }
        "exists" => operand
            .as_bool()
            .map(Condition::Exists)
            .ok_or_else(|| format!("'exists' on '{field}' needs true or false")),
        other => Err(format!(
            "unknown operator '{other}' on '{field}'; the operators are eq, ne, gt, gte, lt, lte, in, exists"
        )),
    }
}

fn scalar_value(field: &str, value: &Json) -> Result<MetadataValue, String> {
    match value {
        Json::String(text) => Ok(MetadataValue::Text(text.clone())),
        Json::Bool(flag) => Ok(MetadataValue::Bool(*flag)),
        Json::Number(value) if value.fract() == 0.0 && value.abs() < 9.007e15 => {
            Ok(MetadataValue::Int(*value as i64))
        }
        Json::Number(value) => Ok(MetadataValue::Float(*value)),
        other => Err(format!(
            "filter on '{field}' needs a string, number or boolean, got {}",
            other.type_name()
        )),
    }
}

fn number(field: &str, value: &Json) -> Result<Number, String> {
    match value {
        Json::Number(value) if value.fract() == 0.0 && value.abs() < 9.007e15 => {
            Ok(Number::Int(*value as i64))
        }
        Json::Number(value) => Ok(Number::Float(*value)),
        other => Err(format!(
            "range filter on '{field}' needs a number, got {}",
            other.type_name()
        )),
    }
}

impl Filter for MetadataFilter {
    fn accepts(&self, _id: VectorId, metadata: &Metadata) -> bool {
        for (field, condition) in &self.conditions {
            let value = metadata.get(field);
            let held = match condition {
                Condition::Exists(expected) => value.is_some() == *expected,
                // An absent field fails every condition except `exists: false`.
                // There is no three-valued logic here: `Metadata` has no null,
                // and a filter that treated "missing" as "matches" would return
                // documents the caller explicitly excluded.
                _ => match value {
                    None => false,
                    Some(value) => match condition {
                        Condition::Equals(expected) => value == expected,
                        Condition::NotEquals(expected) => value != expected,
                        Condition::Greater(bound) => {
                            bound.compare(value) == Some(std::cmp::Ordering::Greater)
                        }
                        Condition::GreaterOrEqual(bound) => matches!(
                            bound.compare(value),
                            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                        ),
                        Condition::Less(bound) => {
                            bound.compare(value) == Some(std::cmp::Ordering::Less)
                        }
                        Condition::LessOrEqual(bound) => matches!(
                            bound.compare(value),
                            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                        ),
                        Condition::In(options) => options.contains(value),
                        Condition::Exists(_) => unreachable!("handled above"),
                    },
                },
            };
            if !held {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::json::parse;

    fn compile(text: &str) -> MetadataFilter {
        MetadataFilter::parse(&parse(text).unwrap()).unwrap()
    }

    fn accepts(filter: &MetadataFilter, metadata: &Metadata) -> bool {
        filter.accepts(VectorId::new(1), metadata)
    }

    #[test]
    fn an_empty_filter_accepts_everything() {
        let filter = compile("{}");
        assert!(filter.is_empty());
        assert_eq!(filter.len(), 0);
        assert!(accepts(&filter, &Metadata::new()));
    }

    #[test]
    fn a_bare_scalar_is_equality() {
        let filter = compile(r#"{"lang":"en","year":2024,"public":true}"#);
        assert_eq!(filter.len(), 3);
        assert!(accepts(
            &filter,
            &Metadata::new()
                .with("lang", "en")
                .with("year", 2024_i64)
                .with("public", true)
        ));
        assert!(!accepts(
            &filter,
            &Metadata::new()
                .with("lang", "fr")
                .with("year", 2024_i64)
                .with("public", true)
        ));
    }

    #[test]
    fn every_condition_must_hold() {
        let filter = compile(r#"{"a":1,"b":2}"#);
        assert!(!accepts(&filter, &Metadata::new().with("a", 1_i64)));
        assert!(accepts(
            &filter,
            &Metadata::new().with("a", 1_i64).with("b", 2_i64)
        ));
    }

    #[test]
    fn ranges_work_on_both_sides() {
        let filter = compile(r#"{"year":{"gte":2020,"lt":2025}}"#);
        for (year, expected) in [(2019, false), (2020, true), (2024, true), (2025, false)] {
            assert_eq!(
                accepts(&filter, &Metadata::new().with("year", year as i64)),
                expected,
                "year {year}"
            );
        }
    }

    #[test]
    fn float_bounds_do_not_match_integer_fields() {
        // The type rule, asserted: `MetadataValue::as_float` does not widen an
        // `Int`, and neither does this.
        let filter = compile(r#"{"score":{"gt":0.5}}"#);
        assert!(accepts(&filter, &Metadata::new().with("score", 0.9_f64)));
        assert!(!accepts(&filter, &Metadata::new().with("score", 9_i64)));
    }

    #[test]
    fn integral_bounds_do_match_float_fields() {
        // The one coercion allowed, because JSON cannot express "1.0 is a
        // float": `{"score": {"gte": 1}}` has to mean something sensible.
        let filter = compile(r#"{"score":{"gte":1}}"#);
        assert!(accepts(&filter, &Metadata::new().with("score", 1.5_f64)));
        assert!(!accepts(&filter, &Metadata::new().with("score", 0.5_f64)));
    }

    #[test]
    fn membership_and_negation() {
        let filter = compile(r#"{"status":{"in":["a","b"]},"draft":{"ne":true}}"#);
        assert!(accepts(
            &filter,
            &Metadata::new().with("status", "a").with("draft", false)
        ));
        assert!(!accepts(
            &filter,
            &Metadata::new().with("status", "c").with("draft", false)
        ));
        assert!(!accepts(
            &filter,
            &Metadata::new().with("status", "a").with("draft", true)
        ));
    }

    #[test]
    fn a_missing_field_fails_everything_except_exists_false() {
        let empty = Metadata::new();
        for text in [
            r#"{"x":1}"#,
            r#"{"x":{"ne":1}}"#,
            r#"{"x":{"gt":0}}"#,
            r#"{"x":{"in":[1,2]}}"#,
            r#"{"x":{"exists":true}}"#,
        ] {
            assert!(!accepts(&compile(text), &empty), "{text}");
        }
        assert!(accepts(&compile(r#"{"x":{"exists":false}}"#), &empty));
    }

    #[test]
    fn compilation_errors_name_the_field_and_the_problem() {
        for (text, fragment) in [
            ("[]", "must be an object"),
            (r#"{"a":{}}"#, "no operators"),
            (r#"{"a":{"between":1}}"#, "unknown operator"),
            (r#"{"a":{"gt":"x"}}"#, "needs a number"),
            (r#"{"a":{"in":[]}}"#, "never match"),
            (r#"{"a":{"in":5}}"#, "needs an array"),
            (r#"{"a":{"exists":1}}"#, "true or false"),
            (r#"{"a":null}"#, "string, number or boolean"),
        ] {
            let err = MetadataFilter::parse(&parse(text).unwrap()).unwrap_err();
            assert!(err.contains(fragment), "{text}: {err}");
        }
    }

    #[test]
    fn conditions_are_ordered_so_two_equal_filters_evaluate_identically() {
        assert_eq!(
            compile(r#"{"b":1,"a":2}"#),
            compile(r#"{"a":2,"b":1}"#),
            "field order in the request must not change the compiled filter"
        );
    }
}
