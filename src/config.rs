use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "kserve-catboost", about = "KServe v2 backend for CatBoost models")]
pub struct Config {
    #[arg(long, env = "MODEL_NAME", default_value = "catboost")]
    pub model_name: String,

    #[arg(long, env = "MODEL_PATH", default_value = "/mnt/models/model.cbm")]
    pub model_path: String,

    #[arg(long, env = "MODEL_VERSION", default_value = "1")]
    pub model_version: String,

    #[arg(long, env = "HTTP_PORT", default_value_t = 8080)]
    pub http_port: u16,

    #[arg(long, env = "GRPC_PORT", default_value_t = 9000)]
    pub grpc_port: u16,
}
