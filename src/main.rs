mod config;
mod inference;
mod model;
mod server;

use std::{path::PathBuf, sync::Arc};

use clap::Parser as _;
use tracing::info;

use model::ModelRegistry;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "kserve_catboost=info,tower_http=debug".parse().unwrap()),
        )
        .init();

    let cfg = config::Config::parse();

    info!(
        model_name  = %cfg.model_name,
        model_path  = %cfg.model_path,
        version     = %cfg.model_version,
        http_port   = cfg.http_port,
        grpc_port   = cfg.grpc_port,
        "starting KServe CatBoost backend",
    );

    let path = PathBuf::from(&cfg.model_path);
    let model = model::CatBoostModel::load_file(&path)?;
    let registry = Arc::new(ModelRegistry::new(
        model,
        path,
        cfg.model_name,
        cfg.model_version,
    ));

    let http_addr: std::net::SocketAddr = format!("0.0.0.0:{}", cfg.http_port).parse()?;
    let grpc_addr: std::net::SocketAddr = format!("0.0.0.0:{}", cfg.grpc_port).parse()?;

    info!(%http_addr, %grpc_addr, "servers starting");

    let (http_res, grpc_res) = tokio::join!(
        server::http::serve(http_addr, registry.clone()),
        server::grpc::serve(grpc_addr, registry.clone()),
    );

    http_res?;
    grpc_res?;
    Ok(())
}
