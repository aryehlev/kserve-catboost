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
}
