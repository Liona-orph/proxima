//! A tiny long-option parser.
//!
//! # Why not `clap`
//!
//! `clap` is very good and this is not a replacement for it. The CLI has three
//! subcommands and about twenty options, all of them `--name value`, none of
//! them positional beyond the subcommand, no subcommand nesting, no shell
//! completion and no colour. That is a hundred lines, against a dependency tree
//! that is larger than the entire rest of the workspace.
//!
//! What is given up: no `-v` short forms, no `--flag=value` (though it is
//! accepted), no generated help beyond the hand-written text, and no suggestion
//! when a flag is misspelled. The last one is the real loss, so an unknown flag
//! is at least reported by name rather than ignored — silently ignoring an
//! unrecognised flag is how a benchmark ends up not measuring what its command
//! line said.

use std::collections::BTreeMap;

/// Parsed command-line arguments: a subcommand and a flag map.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Args {
    /// The subcommand, or empty if none was given.
    pub(crate) command: String,
    /// `--name value` pairs. A bare `--flag` maps to `"true"`.
    pub(crate) flags: BTreeMap<String, String>,
    /// Anything that was neither a flag nor its value.
    pub(crate) positional: Vec<String>,
}

impl Args {
    /// Parse `std::env::args`, minus the program name.
    pub(crate) fn parse<I: IntoIterator<Item = String>>(arguments: I) -> Result<Self, String> {
        let mut args = Args::default();
        let mut iterator = arguments.into_iter().peekable();

        if let Some(first) = iterator.peek() {
            if !first.starts_with('-') {
                args.command = iterator.next().unwrap_or_default();
            }
        }

        while let Some(argument) = iterator.next() {
            let Some(name) = argument.strip_prefix("--") else {
                if argument.starts_with('-') && argument.len() > 1 {
                    return Err(format!(
                        "unknown option '{argument}'; options are spelled --like-this"
                    ));
                }
                args.positional.push(argument);
                continue;
            };
            if let Some((name, value)) = name.split_once('=') {
                args.flags.insert(name.to_owned(), value.to_owned());
                continue;
            }
            // A flag whose next argument starts with `--` is a boolean flag; a
            // negative number is a value, which is why the check is for the
            // double dash and not for a leading `-`.
            let takes_value = iterator.peek().is_some_and(|next| !next.starts_with("--"));
            let value = if takes_value {
                iterator.next().unwrap_or_default()
            } else {
                "true".to_owned()
            };
            args.flags.insert(name.to_owned(), value);
        }
        Ok(args)
    }

    /// A flag's raw value.
    pub(crate) fn get(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    /// A flag parsed as `T`, or `default` if it was not given.
    ///
    /// A flag that *was* given and does not parse is an error, never the
    /// default: `--k twelve` silently meaning ten is the kind of thing that
    /// makes a benchmark result a lie.
    pub(crate) fn value<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.get(name) {
            None => Ok(default),
            Some(text) => text.parse().map_err(|_| {
                format!(
                    "--{name} '{text}' is not a valid {}",
                    std::any::type_name::<T>()
                )
            }),
        }
    }

    /// A `u64` flag, accepting `0x` hexadecimal and `_` separators.
    ///
    /// Seeds are the reason this exists. `HnswConfig`'s default seed prints as
    /// `0x50b01a15` and the benchmark's as `0x9e3779b9`, so the natural thing to
    /// paste back into a command line is the hexadecimal form — and
    /// `u64::from_str` rejects it. A flag whose documented form the parser
    /// refuses is worse than no flag.
    pub(crate) fn u64_value(&self, name: &str, default: u64) -> Result<u64, String> {
        let Some(text) = self.get(name) else {
            return Ok(default);
        };
        let cleaned = text.replace('_', "");
        let parsed = match cleaned
            .strip_prefix("0x")
            .or_else(|| cleaned.strip_prefix("0X"))
        {
            Some(hex) => u64::from_str_radix(hex, 16),
            None => cleaned.parse(),
        };
        parsed.map_err(|_| format!("--{name} '{text}' is not a number (decimal or 0x hexadecimal)"))
    }

    /// A boolean flag: present means true unless it was given an explicit
    /// `false`.
    pub(crate) fn flag(&self, name: &str) -> bool {
        matches!(self.get(name), Some(value) if value != "false")
    }

    /// Fail if any flag outside `known` was given.
    ///
    /// Called by every subcommand. A typo in a benchmark's flags must not
    /// silently produce a number for a different configuration.
    pub(crate) fn reject_unknown(&self, known: &[&str]) -> Result<(), String> {
        for name in self.flags.keys() {
            if !known.contains(&name.as_str()) {
                return Err(format!(
                    "unknown flag --{name}; this command takes: {}",
                    known
                        .iter()
                        .map(|k| format!("--{k}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    // The parsed constants are exactly representable, and the point is that the
    // parser returns them bit-for-bit.
    #![allow(clippy::float_cmp)]

    use super::*;

    fn parse(line: &str) -> Args {
        Args::parse(line.split_whitespace().map(str::to_owned)).unwrap()
    }

    #[test]
    fn a_subcommand_and_its_flags() {
        let args = parse("bench --vectors 50000 --dim 128 --quiet");
        assert_eq!(args.command, "bench");
        assert_eq!(args.get("vectors"), Some("50000"));
        assert_eq!(args.value("dim", 0usize).unwrap(), 128);
        assert!(args.flag("quiet"));
        assert!(!args.flag("verbose"));
    }

    #[test]
    fn equals_form_and_negative_numbers() {
        let args = parse("query --k=5 --threshold -1.5");
        assert_eq!(args.value("k", 0usize).unwrap(), 5);
        assert_eq!(args.value("threshold", 0.0f64).unwrap(), -1.5);
    }

    #[test]
    fn a_flag_that_does_not_parse_is_an_error_not_the_default() {
        let args = parse("bench --vectors many");
        let err = args.value("vectors", 100usize).unwrap_err();
        assert!(err.contains("many"), "{err}");
        // ...and an absent flag really is the default.
        assert_eq!(args.value("absent", 7usize).unwrap(), 7);
    }

    #[test]
    fn unknown_flags_are_refused_by_name() {
        let args = parse("bench --vectrs 10");
        let err = args.reject_unknown(&["vectors", "dim"]).unwrap_err();
        assert!(err.contains("--vectrs"), "{err}");
        assert!(err.contains("--vectors"), "{err}");
        assert!(parse("bench --vectors 10")
            .reject_unknown(&["vectors"])
            .is_ok());
    }

    #[test]
    fn short_options_are_refused_rather_than_treated_as_positional() {
        let err = Args::parse(["-v".to_owned()]).unwrap_err();
        assert!(err.contains("-v"), "{err}");
    }

    #[test]
    fn seeds_parse_in_decimal_and_hexadecimal() {
        // The form the CLI itself prints has to be the form it accepts.
        let args = parse("bench --seed 0x9e3779b9");
        assert_eq!(args.u64_value("seed", 0).unwrap(), 0x9e37_79b9);
        assert_eq!(
            parse("bench --seed 2654435769")
                .u64_value("seed", 0)
                .unwrap(),
            0x9e37_79b9
        );
        assert_eq!(
            parse("bench --seed 0x9E37_79B9")
                .u64_value("seed", 0)
                .unwrap(),
            0x9e37_79b9
        );
        assert_eq!(parse("bench").u64_value("seed", 7).unwrap(), 7);
        assert!(parse("bench --seed zzz").u64_value("seed", 0).is_err());
    }

    #[test]
    fn no_arguments_is_no_command() {
        let args = Args::parse(Vec::<String>::new()).unwrap();
        assert!(args.command.is_empty());
        assert!(args.flags.is_empty());
    }
}
