# ── Build stage ───────────────────────────────────────────────────────────────
FROM rust:1.85-bookworm AS builder

WORKDIR /app

# protoc is required by tonic-build to compile the proto files at build time.
# pkg-config + libssl-dev are needed by catboost-rust's build.rs (uses ureq to
# download the CatBoost shared library).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        protobuf-compiler pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# ── Dependency layer cache ─────────────────────────────────────────────────────
# Copy manifests + proto definitions first so Docker can cache the expensive
# dependency-fetch step independently from source changes.
COPY Cargo.toml build.rs ./
COPY proto/ ./proto/

# Compile a stub binary to pull and cache all crate dependencies (including
# catboost-rust which downloads libcatboostmodel.so at compile time).
RUN mkdir -p src && \
    echo 'fn main() {}' > src/main.rs && \
    cargo build --release && \
    rm -rf src

# ── Source build ──────────────────────────────────────────────────────────────
COPY src/ ./src/
# Touch main.rs so cargo knows it changed and rebuilds the binary.
RUN touch src/main.rs && cargo build --release

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the server binary.
COPY --from=builder /app/target/release/kserve-catboost /app/kserve-catboost

# catboost-rust's build.rs downloads libcatboostmodel.so and copies it next to
# the compiled binary in the Cargo target directory.
COPY --from=builder /app/target/release/libcatboostmodel.so /app/libcatboostmodel.so

# Make the dynamic linker find libcatboostmodel.so at runtime.
ENV LD_LIBRARY_PATH=/app

# ── KServe standard defaults (all overridable via env or --flag) ──────────────
ENV MODEL_NAME=catboost
ENV MODEL_PATH=/mnt/models/model.cbm
ENV HTTP_PORT=8080
ENV GRPC_PORT=9000

EXPOSE 8080 9000

CMD ["/app/kserve-catboost"]
