/// KServe Open Inference Protocol v2 — HTTP/REST server.
///
/// Required endpoints (§ Required API):
///   GET  /v2/health/live
///   GET  /v2/health/ready
///   GET  /v2                                              ← server metadata
///   GET  /v2/models/{name}                               ← model metadata
///   GET  /v2/models/{name}/ready
///   POST /v2/models/{name}/infer
///   GET  /v2/models/{name}/versions/{ver}                ← versioned metadata
///   GET  /v2/models/{name}/versions/{ver}/ready
///   POST /v2/models/{name}/versions/{ver}/infer
///
/// Repository extension (hot-reload / model lifecycle):
///   GET  /v2/repository/index
///   POST /v2/repository/models/{name}/load
///   POST /v2/repository/models/{name}/unload
use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
};
use std::sync::Arc;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use uuid::Uuid;

use crate::{
    registry::ModelRegistry,
    tensor::{
        flatten_json_to_f32, flatten_json_to_strings, prediction_shape,
        reshape_strings, validate_and_get_num_features,
    },
    types::*,
};

pub async fn serve(addr: std::net::SocketAddr, registry: Arc<ModelRegistry>) -> anyhow::Result<()> {
    let app = build_router(registry);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "HTTP REST server listening");
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn build_router(registry: Arc<ModelRegistry>) -> Router {
    Router::new()
        // ── Server ────────────────────────────────────────────────────────
        .route("/v2/health/live",  get(health_live))
        .route("/v2/health/ready", get(health_ready))
        .route("/v2",              get(server_metadata_handler))

        // ── Model (unversioned) ───────────────────────────────────────────
        .route("/v2/models/{model_name}",       get(model_metadata_handler))
        .route("/v2/models/{model_name}/ready", get(model_ready_handler))
        .route("/v2/models/{model_name}/infer", post(model_infer_handler))

        // ── Model (versioned) ─────────────────────────────────────────────
        .route(
            "/v2/models/{model_name}/versions/{model_version}",
            get(versioned_metadata_handler),
        )
        .route(
            "/v2/models/{model_name}/versions/{model_version}/ready",
            get(versioned_ready_handler),
        )
        .route(
            "/v2/models/{model_name}/versions/{model_version}/infer",
            post(versioned_infer_handler),
        )

        // ── Repository extension (lifecycle / hot-reload) ─────────────────
        .route("/v2/repository/index",                              get(repository_index_handler))
        .route("/v2/repository/models/{model_name}/load",   post(repository_load_handler))
        .route("/v2/repository/models/{model_name}/unload", post(repository_unload_handler))

        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(registry)
}

// ── Server endpoints ──────────────────────────────────────────────────────────

async fn health_live() -> StatusCode {
    StatusCode::OK
}

async fn health_ready(State(registry): State<Arc<ModelRegistry>>) -> StatusCode {
    if registry.is_loaded() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn server_metadata_handler() -> impl IntoResponse {
    Json(ServerMetadataResponse {
        name: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        extensions: vec![
            "classification".to_string(),
            "regression".to_string(),
            "model-repository".to_string(),
        ],
    })
}

// ── Model metadata (unversioned) ──────────────────────────────────────────────

async fn model_metadata_handler(
    Path(model_name): Path<String>,
    State(registry): State<Arc<ModelRegistry>>,
) -> impl IntoResponse {
    if model_name != registry.name {
        return not_found(&model_name);
    }
    build_model_metadata(&registry, None).into_response()
}

async fn model_ready_handler(
    Path(model_name): Path<String>,
    State(registry): State<Arc<ModelRegistry>>,
) -> StatusCode {
    if model_name == registry.name && registry.is_loaded() {
        StatusCode::OK
    } else if model_name == registry.name {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::NOT_FOUND
    }
}

// ── Versioned model endpoints ─────────────────────────────────────────────────

async fn versioned_metadata_handler(
    Path((model_name, model_version)): Path<(String, String)>,
    State(registry): State<Arc<ModelRegistry>>,
) -> impl IntoResponse {
    if model_name != registry.name {
        return not_found(&model_name);
    }
    if model_version != "1" {
        return not_found_version(&model_name, &model_version);
    }
    build_model_metadata(&registry, Some(model_version)).into_response()
}

async fn versioned_ready_handler(
    Path((model_name, model_version)): Path<(String, String)>,
    State(registry): State<Arc<ModelRegistry>>,
) -> StatusCode {
    if model_name != registry.name || model_version != "1" {
        return StatusCode::NOT_FOUND;
    }
    if registry.is_loaded() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn versioned_infer_handler(
    Path((model_name, model_version)): Path<(String, String)>,
    State(registry): State<Arc<ModelRegistry>>,
    request: Json<InferRequest>,
) -> impl IntoResponse {
    if model_name != registry.name {
        return not_found(&model_name);
    }
    if model_version != "1" {
        return not_found_version(&model_name, &model_version);
    }
    run_infer(model_name, Some(model_version), registry, request.0).await
}

// ── Inference (unversioned) ───────────────────────────────────────────────────

async fn model_infer_handler(
    Path(model_name): Path<String>,
    State(registry): State<Arc<ModelRegistry>>,
    Json(request): Json<InferRequest>,
) -> impl IntoResponse {
    if model_name != registry.name {
        return not_found(&model_name);
    }
    run_infer(model_name, None, registry, request).await
}

// ── Repository extension ──────────────────────────────────────────────────────

async fn repository_index_handler(
    State(registry): State<Arc<ModelRegistry>>,
) -> impl IntoResponse {
    let state = if registry.is_loaded() {
        ModelState::Ready
    } else {
        ModelState::Unavailable
    };
    Json(vec![RepositoryIndexEntry {
        name: registry.name.clone(),
        version: Some("1".to_string()),
        state,
        reason: String::new(),
    }])
}

async fn repository_load_handler(
    Path(model_name): Path<String>,
    State(registry): State<Arc<ModelRegistry>>,
) -> impl IntoResponse {
    if model_name != registry.name {
        return not_found(&model_name);
    }
    match tokio::task::spawn_blocking(move || registry.repository_load()).await {
        Ok(Ok(())) => StatusCode::OK.into_response(),
        Ok(Err(e)) => internal_error(&e.to_string()),
        Err(e) => internal_error(&e.to_string()),
    }
}

async fn repository_unload_handler(
    Path(model_name): Path<String>,
    State(registry): State<Arc<ModelRegistry>>,
) -> impl IntoResponse {
    if model_name != registry.name {
        return not_found(&model_name);
    }
    registry.repository_unload();
    StatusCode::OK.into_response()
}

// ── Shared inference logic ────────────────────────────────────────────────────

async fn run_infer(
    model_name: String,
    model_version: Option<String>,
    registry: Arc<ModelRegistry>,
    request: InferRequest,
) -> axum::response::Response {
    if !registry.is_loaded() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: format!("model '{}' is not loaded", model_name),
            }),
        )
            .into_response();
    }

    let id = request.id.unwrap_or_else(|| Uuid::new_v4().to_string());
    let model = registry.load_model();
    let dims = model.dimensions_count;
    let default_float_feats = model.float_features_count;

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f64>> {
        let (flat, num_feats) = extract_float_flat(&request.inputs, default_float_feats)?;
        let rows: Vec<&[f32]> = flat.chunks(num_feats).collect();
        let cat = extract_cat_features(&request.inputs)?;
        model.predict(&rows, cat)
    })
    .await;

    match result {
        Ok(Ok(predictions)) => {
            let shape = prediction_shape(predictions.len(), dims);
            (
                StatusCode::OK,
                Json(InferResponse {
                    model_name,
                    model_version,
                    id,
                    outputs: vec![OutputTensor {
                        name: "predictions".to_string(),
                        shape,
                        datatype: "FP64".to_string(),
                        data: predictions,
                    }],
                }),
            )
                .into_response()
        }
        Ok(Err(e)) => bad_request(&e.to_string()),
        Err(e) => internal_error(&e.to_string()),
    }
}

// ── Input extraction ──────────────────────────────────────────────────────────

fn extract_float_flat(
    inputs: &[InferInput],
    default_num_features: usize,
) -> anyhow::Result<(Vec<f32>, usize)> {
    let input = inputs
        .iter()
        .find(|i| i.name == "float_features")
        .or_else(|| {
            inputs.iter().find(|i| {
                matches!(
                    i.datatype.as_str(),
                    "FP32" | "FP64" | "INT32" | "INT64" | "UINT32" | "UINT64"
                )
            })
        });

    match input {
        None => Ok((vec![], if default_num_features == 0 { 1 } else { default_num_features })),
        Some(inp) => {
            let flat = flatten_json_to_f32(&inp.data)?;
            let num_feats = validate_and_get_num_features(flat.len(), &inp.shape)?;
            Ok((flat, num_feats))
        }
    }
}

fn extract_cat_features(inputs: &[InferInput]) -> anyhow::Result<Vec<Vec<String>>> {
    let input = inputs
        .iter()
        .find(|i| i.name == "cat_features")
        .or_else(|| inputs.iter().find(|i| i.datatype == "BYTES"));

    match input {
        None => Ok(vec![]),
        Some(inp) => {
            let flat = flatten_json_to_strings(&inp.data)?;
            reshape_strings(flat, &inp.shape)
        }
    }
}

// ── Shared metadata builder ───────────────────────────────────────────────────

fn build_model_metadata(
    registry: &ModelRegistry,
    version: Option<String>,
) -> impl IntoResponse {
    let model = registry.load_model();

    let mut inputs = vec![TensorMetadata {
        name: "float_features".to_string(),
        datatype: "FP32".to_string(),
        shape: vec![-1, model.float_features_count as i64],
    }];
    if model.cat_features_count > 0 {
        inputs.push(TensorMetadata {
            name: "cat_features".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![-1, model.cat_features_count as i64],
        });
    }
    let out_shape = if model.dimensions_count > 1 {
        vec![-1_i64, model.dimensions_count as i64]
    } else {
        vec![-1_i64]
    };

    let versions = version
        .map(|v| vec![v])
        .unwrap_or_else(|| vec!["1".to_string()]);

    Json(ModelMetadataResponse {
        name: model.name.clone(),
        versions,
        platform: "catboost".to_string(),
        inputs,
        outputs: vec![TensorMetadata {
            name: "predictions".to_string(),
            datatype: "FP64".to_string(),
            shape: out_shape,
        }],
    })
}

// ── Response helpers ──────────────────────────────────────────────────────────

fn not_found(name: &str) -> axum::response::Response {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: format!("model '{}' not found", name),
        }),
    )
        .into_response()
}

fn not_found_version(name: &str, version: &str) -> axum::response::Response {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: format!("model '{}' version '{}' not found", name, version),
        }),
    )
        .into_response()
}

fn bad_request(msg: &str) -> axum::response::Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse { error: msg.to_string() }),
    )
        .into_response()
}

fn internal_error(msg: &str) -> axum::response::Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: msg.to_string() }),
    )
        .into_response()
}
