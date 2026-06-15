use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::{
    inference::{self, RawTensor, TensorData},
    model::ModelRegistry,
};
use super::types::{
    ErrorResponse, InferRequest, InferResponse, InferOutputTensor,
    ModelMetadataResponse, RepositoryIndexEntry, ServerMetadataResponse,
};

// ── Error type ───────────────────────────────────────────────────────────────

struct AppError(StatusCode, String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, Json(ErrorResponse { error: self.1 })).into_response()
    }
}

fn bad_request(msg: impl ToString) -> AppError {
    AppError(StatusCode::BAD_REQUEST, msg.to_string())
}

fn not_found(msg: impl ToString) -> AppError {
    AppError(StatusCode::NOT_FOUND, msg.to_string())
}

fn unavailable(msg: impl ToString) -> AppError {
    AppError(StatusCode::SERVICE_UNAVAILABLE, msg.to_string())
}

fn internal(msg: impl ToString) -> AppError {
    AppError(StatusCode::INTERNAL_SERVER_ERROR, msg.to_string())
}

type ApiResult<T> = Result<Json<T>, AppError>;

// ── Health ───────────────────────────────────────────────────────────────────

async fn live() -> StatusCode {
    StatusCode::OK
}

async fn ready(State(reg): State<Arc<ModelRegistry>>) -> StatusCode {
    if reg.is_loaded() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

// ── Metadata ─────────────────────────────────────────────────────────────────

async fn server_metadata() -> Json<ServerMetadataResponse> {
    Json(ServerMetadataResponse {
        name: "kserve-catboost".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        extensions: vec!["model_repository_extension".to_string()],
    })
}

async fn model_metadata(
    State(reg): State<Arc<ModelRegistry>>,
    Path(model_name): Path<String>,
) -> ApiResult<ModelMetadataResponse> {
    if model_name != reg.name {
        return Err(not_found(format!("no model '{model_name}'")));
    }
    Ok(Json(ModelMetadataResponse {
        name: reg.name.clone(),
        versions: vec![reg.version.clone()],
        platform: "catboost".to_string(),
    }))
}

async fn model_ready(
    State(reg): State<Arc<ModelRegistry>>,
    Path(model_name): Path<String>,
) -> StatusCode {
    if model_name != reg.name || !reg.is_loaded() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    }
}

// ── Inference ─────────────────────────────────────────────────────────────────

async fn infer(
    State(reg): State<Arc<ModelRegistry>>,
    Path(model_name): Path<String>,
    Json(req): Json<InferRequest>,
) -> ApiResult<InferResponse> {
    if model_name != reg.name {
        return Err(not_found(format!("no model '{model_name}'")));
    }

    let model = reg.current_model().map_err(|e| unavailable(e))?;

    let tensors: Result<Vec<RawTensor>, AppError> = req
        .inputs
        .into_iter()
        .map(|t| {
            let data = inference::json_to_tensor_data(t.data, &t.datatype)
                .map_err(|e| bad_request(e))?;
            Ok(RawTensor {
                name: t.name,
                datatype: t.datatype,
                shape: t.shape,
                data,
            })
        })
        .collect();

    let inputs = inference::parse_inputs(tensors?).map_err(|e| bad_request(e))?;
    let out = inference::run_blocking(model, inputs)
        .await
        .map_err(|e| internal(e))?;

    let id = req.id.unwrap_or_else(|| Uuid::new_v4().to_string());

    Ok(Json(InferResponse {
        model_name: reg.name.clone(),
        model_version: reg.version.clone(),
        id,
        outputs: vec![InferOutputTensor {
            name: "output-0".to_string(),
            datatype: "FP64".to_string(),
            shape: out.shape,
            data: out.predictions,
        }],
    }))
}

// ── Repository ───────────────────────────────────────────────────────────────

async fn repository_index(State(reg): State<Arc<ModelRegistry>>) -> Json<Vec<RepositoryIndexEntry>> {
    let state = if reg.is_loaded() {
        "READY".to_string()
    } else {
        "UNAVAILABLE".to_string()
    };
    Json(vec![RepositoryIndexEntry {
        name: reg.name.clone(),
        version: reg.version.clone(),
        state,
        reason: String::new(),
    }])
}

async fn repository_load(
    State(reg): State<Arc<ModelRegistry>>,
    Path(model_name): Path<String>,
) -> Result<StatusCode, AppError> {
    if model_name != reg.name {
        return Err(not_found(format!("no model '{model_name}'")));
    }
    tokio::task::spawn_blocking(move || reg.repository_load())
        .await
        .map_err(|e| internal(e))?
        .map_err(|e| internal(e))?;
    Ok(StatusCode::OK)
}

async fn repository_unload(
    State(reg): State<Arc<ModelRegistry>>,
    Path(model_name): Path<String>,
) -> Result<StatusCode, AppError> {
    if model_name != reg.name {
        return Err(not_found(format!("no model '{model_name}'")));
    }
    reg.repository_unload();
    Ok(StatusCode::OK)
}

// ── Router ───────────────────────────────────────────────────────────────────

pub async fn serve(
    addr: std::net::SocketAddr,
    registry: Arc<ModelRegistry>,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/v2/health/live", get(live))
        .route("/v2/health/ready", get(ready))
        .route("/v2", get(server_metadata))
        .route("/v2/models/{model_name}", get(model_metadata))
        .route("/v2/models/{model_name}/ready", get(model_ready))
        .route("/v2/models/{model_name}/infer", post(infer))
        .route("/v2/repository/index", get(repository_index))
        .route("/v2/repository/models/{model_name}/load", post(repository_load))
        .route("/v2/repository/models/{model_name}/unload", post(repository_unload))
        .layer(TraceLayer::new_for_http())
        .with_state(registry);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
