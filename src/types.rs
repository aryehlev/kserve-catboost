/// KServe Open Inference Protocol v2 — HTTP/REST request and response types.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ── Inference request ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct InferRequest {
    pub id: Option<String>,
    pub inputs: Vec<InferInput>,
    pub outputs: Option<Vec<RequestedOutput>>,
    #[serde(default)]
    pub parameters: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct InferInput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    /// JSON data — may be a flat array or a nested (row-major) array.
    pub data: Value,
    #[serde(default)]
    pub parameters: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct RequestedOutput {
    pub name: String,
    #[serde(default)]
    pub parameters: HashMap<String, Value>,
}

// ── Inference response ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct InferResponse {
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    pub id: String,
    pub outputs: Vec<OutputTensor>,
}

#[derive(Debug, Serialize)]
pub struct OutputTensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    pub data: Vec<f64>,
}

// ── Server metadata (GET /v2) ─────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ServerMetadataResponse {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

// ── Model metadata ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelMetadataResponse {
    pub name: String,
    pub versions: Vec<String>,
    pub platform: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

#[derive(Debug, Serialize)]
pub struct TensorMetadata {
    pub name: String,
    pub datatype: String,
    /// -1 indicates a dynamic (batch) dimension.
    pub shape: Vec<i64>,
}

// ── Repository extension ──────────────────────────────────────────────────────
//
// POST /v2/repository/models/{name}/load  — load / hot-reload a model
// POST /v2/repository/models/{name}/unload — unload a model
// GET  /v2/repository/index               — list models and their states

/// A single entry returned by GET /v2/repository/index.
#[derive(Debug, Serialize)]
pub struct RepositoryIndexEntry {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub state: ModelState,
    pub reason: String,
}

#[derive(Debug, Serialize, Clone, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelState {
    Unknown,
    Ready,
    Unavailable,
    Loading,
    Unloading,
}

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}
