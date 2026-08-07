# Proxima — developer entry points.
#
# `make ci` is the single gate. It runs exactly what .github/workflows/ci.yml
# runs, in the same order, so a green `make ci` locally means a green CI run.
# If the two ever drift, this file is the one that is wrong.
#
# Recipe lines are hard tabs; see .editorconfig.

.DEFAULT_GOAL := help

CARGO ?= cargo
# rustdoc has no `-D warnings` flag of its own; it is passed through the
# environment, which is why this is a variable and not part of the recipe.
RUSTDOCFLAGS ?= -D warnings

help: ## Show this help
	@echo "Proxima — make targets"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_%-]+:.*?## /{printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "  make ci  is the gate. Run it before you open a pull request."

build: ## Build the workspace in release mode
	$(CARGO) build --workspace --release

run: ## Run the server against ./proxima-data (127.0.0.1:7700)
	$(CARGO) run --release -p proxima-cli -- serve --data ./proxima-data --address 127.0.0.1:7700

test: ## Run the whole test suite
	$(CARGO) test --workspace

fmt: ## Check formatting (does not rewrite; use `make fmt-fix`)
	$(CARGO) fmt --all --check

fmt-fix: ## Rewrite the tree with rustfmt
	$(CARGO) fmt --all

lint: ## Clippy over every target, warnings are errors
	$(CARGO) clippy --workspace --all-targets -- -D warnings

bench: ## Run the criterion benchmarks (slow; this is where the numbers come from)
	$(CARGO) bench -p proxima-core
	$(CARGO) bench -p proxima-index

bench-check: ## Compile the benchmarks without running them
	$(CARGO) bench --workspace --no-run

doc: ## Build the docs; a broken intra-doc link is an error
	RUSTDOCFLAGS="$(RUSTDOCFLAGS)" $(CARGO) doc --workspace --no-deps --document-private-items

fsync-cost: ## Measure what an fsync costs on this machine
	$(CARGO) run --release -p proxima-store --example fsync_cost

ci: ## The gate: fmt, lint, test, doc, and a bench compile check
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) doc
	$(MAKE) bench-check

clean: ## Remove build output
	$(CARGO) clean

.PHONY: help build run test fmt fmt-fix lint bench bench-check doc fsync-cost ci clean
