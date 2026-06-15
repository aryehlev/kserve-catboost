use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "kserve-catboost", about = "KServe v2 backend for CatBoost models")]
pub struct Config {
    /// Name of the model (must match the name used in inference requests)
    #[arg(long, env = "MODEL_NAME", default_value = "catboost")]
    pub model_name: String,

    /// Path to the CatBoost model file (.cbm)
    #[arg(long, env = "MODEL_PATH", default_value = "/mnt/models/model.cbm")]
    pub model_path: String,

    /// Port for the HTTP/REST server
    #[arg(long, env = "HTTP_PORT", default_value_t = 8080)]
    pub http_port: u16,

    /// Port for the gRPC server
    #[arg(long, env = "GRPC_PORT", default_value_t = 9000)]
    pub grpc_port: u16,
}
