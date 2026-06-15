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

use crate::inference::{self, DynamicBatcher, OverloadError, RawTensor, TensorData, OUTPUT_DTYPE, OUTPUT_TENSOR};
use super::{
    types::{
        ErrorResponse, InferOutputTensor, InferRequest, InferResponse, MetadataTensor,
        ModelMetadataResponse, RepositoryIndexEntry, ServerMetadataResponse,
    },
    EXTENSIONS, PLATFORM,
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

fn too_many_requests(msg: impl ToString) -> AppError {
    AppError(StatusCode::TOO_MANY_REQUESTS, msg.to_string())
}

fn internal(msg: impl ToString) -> AppError {
    AppError(StatusCode::INTERNAL_SERVER_ERROR, msg.to_string())
}

type ApiResult<T> = Result<Json<T>, AppError>;

// ── Health ───────────────────────────────────────────────────────────────────

async fn live() -> StatusCode {
    StatusCode::OK
}

async fn ready(State(svc): State<Arc<DynamicBatcher>>) -> StatusCode {
    if svc.registry.is_loaded() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

// ── Metadata ─────────────────────────────────────────────────────────────────

async fn server_metadata() -> Json<ServerMetadataResponse> {
    Json(ServerMetadataResponse {
        name: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        extensions: EXTENSIONS.iter().map(|s| s.to_string()).collect(),
    })
}

async fn model_metadata(
    State(svc): State<Arc<DynamicBatcher>>,
    Path(model_name): Path<String>,
) -> ApiResult<ModelMetadataResponse> {
    if model_name != svc.registry.name {
        return Err(not_found(format!("no model '{model_name}'")));
    }
    Ok(Json(ModelMetadataResponse {
        name: svc.registry.name.clone(),
        versions: vec![svc.registry.version().to_string()],
        platform: PLATFORM.to_string(),
        // CatBoost doesn't expose its feature schema, so we return empty
        // descriptor lists — consistent with the gRPC metadata response.
        inputs: vec![],
        outputs: vec![MetadataTensor {
            name: crate::inference::OUTPUT_TENSOR.to_string(),
            datatype: crate::inference::OUTPUT_DTYPE.to_string(),
            shape: vec![-1, 1],
        }],
    }))
}

async fn model_ready(
    State(svc): State<Arc<DynamicBatcher>>,
    Path(model_name): Path<String>,
) -> StatusCode {
    if model_name != svc.registry.name {
        StatusCode::NOT_FOUND
    } else if !svc.registry.is_loaded() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    }
}

// ── Inference ─────────────────────────────────────────────────────────────────

async fn infer(
    State(svc): State<Arc<DynamicBatcher>>,
    Path(model_name): Path<String>,
    Json(req): Json<InferRequest>,
) -> ApiResult<InferResponse> {
    if model_name != svc.registry.name {
        return Err(not_found(format!("no model '{model_name}'")));
    }
    if !svc.registry.is_loaded() {
        return Err(unavailable(format!("model '{}' is not loaded", svc.registry.name)));
    }

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
    let out = svc.infer(inputs).await.map_err(|e| {
        if e.is::<OverloadError>() {
            too_many_requests(e)
        } else {
            internal(e)
        }
    })?;

    let id = req.id.unwrap_or_else(|| Uuid::new_v4().to_string());
    let output_name = req
        .outputs
        .into_iter()
        .next()
        .map(|o| o.name)
        .unwrap_or_else(|| OUTPUT_TENSOR.to_string());

    Ok(Json(InferResponse {
        model_name: svc.registry.name.clone(),
        model_version: svc.registry.version().to_string(),
        id,
        outputs: vec![InferOutputTensor {
            name: output_name,
            datatype: OUTPUT_DTYPE.to_string(),
            shape: out.shape,
            data: out.predictions,
        }],
    }))
}

// ── Repository ───────────────────────────────────────────────────────────────

async fn repository_index(
    State(svc): State<Arc<DynamicBatcher>>,
) -> Json<Vec<RepositoryIndexEntry>> {
    let state = if svc.registry.is_loaded() { "READY" } else { "UNAVAILABLE" };
    Json(vec![RepositoryIndexEntry {
        name: svc.registry.name.clone(),
        version: svc.registry.version().to_string(),
        state: state.to_string(),
        reason: String::new(),
    }])
}

async fn repository_load(
    State(svc): State<Arc<DynamicBatcher>>,
    Path(model_name): Path<String>,
) -> Result<StatusCode, AppError> {
    if model_name != svc.registry.name {
        return Err(not_found(format!("no model '{model_name}'")));
    }
    let reg = svc.registry.clone();
    tokio::task::spawn_blocking(move || reg.repository_load())
        .await
        .map_err(|e| internal(e))?
        .map_err(|e| internal(e))?;
    Ok(StatusCode::OK)
}

async fn repository_unload(
    State(svc): State<Arc<DynamicBatcher>>,
    Path(model_name): Path<String>,
) -> Result<StatusCode, AppError> {
    if model_name != svc.registry.name {
        return Err(not_found(format!("no model '{model_name}'")));
    }
    svc.registry.repository_unload();
    Ok(StatusCode::OK)
}

// ── Router ───────────────────────────────────────────────────────────────────

pub async fn serve(
    addr: std::net::SocketAddr,
    batcher: Arc<DynamicBatcher>,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/v2/health/live", get(live))
        .route("/v2/health/ready", get(ready))
        .route("/v2", get(server_metadata))
        .route("/v2/models/{model_name}", get(model_metadata))
        .route("/v2/models/{model_name}/ready", get(model_ready))
        .route("/v2/models/{model_name}/infer", post(infer))
        .route("/v2/repository/index", get(repository_index))
        .route(
            "/v2/repository/models/{model_name}/load",
            post(repository_load),
        )
        .route(
            "/v2/repository/models/{model_name}/unload",
            post(repository_unload),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(batcher);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
