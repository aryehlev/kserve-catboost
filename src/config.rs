use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "kserve-catboost", about = "KServe v2 backend for CatBoost models")]
pub struct Config {
    #[arg(long, env = "MODEL_NAME", default_value = "catboost")]
    pub model_name: String,

    #[arg(long, env = "MODEL_PATH", default_value = "/mnt/models/model.cbm")]
    pub model_path: String,

    #[arg(long, env = "MODEL_VERSION", default_value_t = 1)]
    pub model_version: u64,

    #[arg(long, env = "HTTP_PORT", default_value_t = 8080)]
    pub http_port: u16,

    #[arg(long, env = "GRPC_PORT", default_value_t = 9000)]
    pub grpc_port: u16,

    /// Maximum rows to accumulate before dispatching a batch.
    /// Set to 1 (default) for no batching — each request is dispatched immediately.
    #[arg(long, env = "MAX_BATCH_SIZE", default_value_t = 1)]
    pub max_batch_size: usize,

    /// How long (ms) to wait for more requests before flushing an incomplete batch.
    /// Only relevant when MAX_BATCH_SIZE > 1.
    #[arg(long, env = "MAX_BATCH_WAIT_MS", default_value_t = 5)]
    pub max_batch_wait_ms: u64,

    /// Comma-separated list of preferred batch sizes (e.g. "8,16,32").
    /// The batcher dispatches immediately when total_rows reaches any of these
    /// sizes, without waiting for the full MAX_BATCH_WAIT_MS deadline.
    /// Mirrors Triton's preferred_batch_size semantics.
    #[arg(long, env = "PREFERRED_BATCH_SIZES", value_delimiter = ',', num_args = 0..)]
    pub preferred_batch_sizes: Vec<usize>,

    /// Number of independent batch-loop workers. Each worker owns its own
    /// channel and can run a CatBoost call concurrently with the others.
    /// Defaults to the number of logical CPUs on the host.
    #[arg(
        long, env = "NUM_BATCH_WORKERS",
        default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    )]
    pub num_batch_workers: usize,
}
