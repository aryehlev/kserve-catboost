mod config;
mod grpc;
mod http;
mod model;
mod proto;
mod registry;
mod tensor;
mod types;

use std::{path::PathBuf, sync::Arc};

use clap::Parser as _;
use tracing::info;

use registry::ModelRegistry;

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
        model_name = %cfg.model_name,
        model_path = %cfg.model_path,
        http_port  = cfg.http_port,
        grpc_port  = cfg.grpc_port,
        "starting KServe CatBoost backend",
    );

    let path = PathBuf::from(&cfg.model_path);
    let model = model::CatBoostModel::load_file(cfg.model_name.clone(), &path)?;
    let registry = Arc::new(ModelRegistry::new(model, path));

    // ── SIGUSR1 hot-reload (Unix only) ────────────────────────────────────────
    #[cfg(unix)]
    {
        let reg = registry.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sig = match signal(SignalKind::user_defined1()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("could not install SIGUSR1 handler: {}", e);
                    return;
                }
            };
            loop {
                sig.recv().await;
                info!("SIGUSR1 received — reloading model");
                let r = reg.clone();
                match tokio::task::spawn_blocking(move || r.reload()).await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => tracing::error!("model reload failed: {}", e),
                    Err(e) => tracing::error!("reload task panicked: {}", e),
                }
            }
        });
    }

    let http_addr: std::net::SocketAddr =
        format!("0.0.0.0:{}", cfg.http_port).parse()?;
    let grpc_addr: std::net::SocketAddr =
        format!("0.0.0.0:{}", cfg.grpc_port).parse()?;

    info!(%http_addr, %grpc_addr, "servers starting");

    let (http_res, grpc_res) = tokio::join!(
        http::serve(http_addr, registry.clone()),
        grpc::serve(grpc_addr, registry.clone()),
    );

    http_res?;
    grpc_res?;
    Ok(())
}
