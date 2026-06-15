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
    let app = Router::new()
        .route("/v2/health/live", get(health_live))
        .route("/v2/health/ready", get(health_ready))
        .route("/v2/models/{model_name}", get(model_metadata_handler))
        .route("/v2/models/{model_name}/ready", get(model_ready_handler))
        .route("/v2/models/{model_name}/infer", post(model_infer_handler))
        // Hot-reload endpoint — triggers an atomic model swap from disk.
        .route("/v2/models/{model_name}/reload", post(reload_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(registry);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "HTTP REST server listening");
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Health ────────────────────────────────────────────────────────────────────

async fn health_live() -> StatusCode {
    StatusCode::OK
}

async fn health_ready() -> StatusCode {
    StatusCode::OK
}

// ── Metadata ─────────────────────────────────────────────────────────────────

async fn model_metadata_handler(
    Path(model_name): Path<String>,
    State(registry): State<Arc<ModelRegistry>>,
) -> impl IntoResponse {
    if model_name != registry.name {
        return not_found(&model_name);
    }

    let model = registry.load();

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

    (
        StatusCode::OK,
        Json(ModelMetadataResponse {
            name: model.name.clone(),
            versions: vec!["1".to_string()],
            platform: "catboost".to_string(),
            inputs,
            outputs: vec![TensorMetadata {
                name: "predictions".to_string(),
                datatype: "FP64".to_string(),
                shape: out_shape,
            }],
        }),
    )
        .into_response()
}

async fn model_ready_handler(
    Path(model_name): Path<String>,
    State(registry): State<Arc<ModelRegistry>>,
) -> StatusCode {
    if model_name == registry.name {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

// ── Inference ─────────────────────────────────────────────────────────────────

async fn model_infer_handler(
    Path(model_name): Path<String>,
    State(registry): State<Arc<ModelRegistry>>,
    Json(request): Json<InferRequest>,
) -> impl IntoResponse {
    if model_name != registry.name {
        return not_found(&model_name);
    }

    let id = request.id.unwrap_or_else(|| Uuid::new_v4().to_string());

    // Take a snapshot of the current model.  Capture metadata fields we need
    // for the response before moving the Arc into the blocking closure.
    let model = registry.load();
    let dims = model.dimensions_count;
    let resp_model_name = model.name.clone();
    let default_float_feats = model.float_features_count;

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f64>> {
        // ── Float features — zero-copy row slices ──────────────────────────
        let (flat, num_feats) = extract_float_flat(&request.inputs, default_float_feats)?;
        // Build row pointers into the flat buffer — no inner-Vec allocation.
        let rows: Vec<&[f32]> = flat.chunks(num_feats).collect();

        // ── Cat features ──────────────────────────────────────────────────
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
                    model_name: resp_model_name,
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

// ── Hot-reload ────────────────────────────────────────────────────────────────

async fn reload_handler(
    Path(model_name): Path<String>,
    State(registry): State<Arc<ModelRegistry>>,
) -> impl IntoResponse {
    if model_name != registry.name {
        return not_found(&model_name);
    }

    let result =
        tokio::task::spawn_blocking(move || registry.reload()).await;

    match result {
        Ok(Ok(())) => (StatusCode::OK, "model reloaded\n").into_response(),
        Ok(Err(e)) => internal_error(&e.to_string()),
        Err(e) => internal_error(&e.to_string()),
    }
}

// ── Input extraction helpers ──────────────────────────────────────────────────

/// Extract float features as a flat `Vec<f32>` and return the per-row feature
/// count so the caller can slice into rows without extra allocation.
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
        None => {
            // No float input — return a single empty row so CatBoost sees
            // a valid (empty) batch when the model has no float features.
            Ok((vec![], if default_num_features == 0 { 1 } else { default_num_features }))
        }
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

fn bad_request(msg: &str) -> axum::response::Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
        .into_response()
}

fn internal_error(msg: &str) -> axum::response::Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
        .into_response()
}
