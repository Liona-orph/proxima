//! The workspace root is not a library anyone should depend on.
//!
//! It exists only so that `examples/quickstart.rs` — the README's first code
//! block — is a real Cargo target that `make ci` compiles and runs. Everything
//! that does anything lives in `crates/`.
#![forbid(unsafe_code)]
