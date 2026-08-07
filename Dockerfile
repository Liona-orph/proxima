# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
# Pinned to the same 1.83 the repository pins in rust-toolchain.toml. If the two
# disagree, cargo will try to download the pinned toolchain inside the image and
# the build gets slower and less reproducible, so keep them in step.
FROM rust:1.83-bookworm AS builder

WORKDIR /build

# Manifests first. Dependencies change far less often than sources, so copying
# the manifests and building a throwaway target caches the dependency
# compilation in its own layer. The dummy sources exist only so that
# `cargo build` has something to compile; they are overwritten below.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
# The workspace root is itself a package (it hosts the README quickstart
# example), so its lib target has to exist for cargo to parse the workspace at
# all — even in the dependency-only layer.
RUN mkdir -p src examples && : > src/root.rs && echo 'fn main() {}' > examples/quickstart.rs
COPY crates/proxima-core/Cargo.toml       crates/proxima-core/Cargo.toml
COPY crates/proxima-quantize/Cargo.toml   crates/proxima-quantize/Cargo.toml
COPY crates/proxima-index/Cargo.toml      crates/proxima-index/Cargo.toml
COPY crates/proxima-store/Cargo.toml      crates/proxima-store/Cargo.toml
COPY crates/proxima-server/Cargo.toml     crates/proxima-server/Cargo.toml
COPY crates/proxima-cli/Cargo.toml        crates/proxima-cli/Cargo.toml
RUN set -eux; \
    for crate in proxima-core proxima-quantize proxima-index proxima-store proxima-server; do \
        mkdir -p "crates/$crate/src"; \
        echo '' > "crates/$crate/src/lib.rs"; \
    done; \
    mkdir -p crates/proxima-cli/src; \
    echo 'fn main() {}' > crates/proxima-cli/src/main.rs; \
    cargo build --release -p proxima-cli; \
    rm -rf crates/*/src

# Now the real sources. Only this layer and the ones after it rebuild when code
# changes.
COPY crates crates
COPY src src
COPY examples examples

# Touch the roots so cargo does not reuse the dummy artefacts from the layer
# above; mtimes inside a COPY are not guaranteed to be newer.
RUN set -eux; \
    find crates src examples -name '*.rs' -exec touch {} +; \
    cargo build --release -p proxima-cli; \
    strip target/release/proxima

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
# distroless/cc, not scratch. The binary is a normal glibc dynamically linked
# executable — it needs libc, libgcc and the dynamic loader — so `scratch` would
# produce an image that cannot exec anything. `cc-debian12` is the smallest base
# that supplies those, and it also brings CA certificates and /etc/passwd, which
# the `nonroot` user needs. Switching to a fully static musl build would allow
# `scratch`, at the cost of a different allocator and different measured
# numbers; that trade has not been made.
FROM gcr.io/distroless/cc-debian12:nonroot

LABEL org.opencontainers.image.title="proxima" \
      org.opencontainers.image.description="Proxima: a from-scratch vector search engine (HNSW, IVF-PQ, WAL-backed storage)." \
      org.opencontainers.image.source="https://github.com/proxima-search/proxima" \
      org.opencontainers.image.url="https://github.com/proxima-search/proxima" \
      org.opencontainers.image.documentation="https://github.com/proxima-search/proxima#readme" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="The Proxima Authors" \
      org.opencontainers.image.version="0.1.0"

COPY --from=builder /build/target/release/proxima /usr/local/bin/proxima

USER nonroot:nonroot

# The data directory. Mount a volume here or the WAL and snapshots die with the
# container.
WORKDIR /data
VOLUME ["/data"]

EXPOSE 7700

# No HEALTHCHECK. A Docker HEALTHCHECK is a command run inside the container,
# and a distroless image has no shell, no curl and no wget to run — adding one
# would mean adding a binary whose only job is to defeat the point of the base
# image. Configure the probe at the orchestrator instead: `GET /health` on port
# 7700 returns 200 when the server is live and ready. In Kubernetes that is an
# httpGet readiness/liveness probe; in Compose, a healthcheck on the client
# side or an external prober.

# The container has nowhere to bind but 0.0.0.0, which is why the default
# command differs from the binary's own default of 127.0.0.1. Proxima has no
# authentication and no TLS: the container network is the security boundary.
# See SECURITY.md.
ENTRYPOINT ["/usr/local/bin/proxima"]
CMD ["serve", "--data", "/data", "--address", "0.0.0.0:7700"]
