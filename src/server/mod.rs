pub mod grpc;
pub mod http;
pub mod proto;
pub mod types;

pub const PLATFORM: &str = "catboost";
pub const EXTENSIONS: &[&str] = &["model_repository_extension"];
