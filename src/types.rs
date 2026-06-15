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

// ── Metadata ──────────────────────────────────────────────────────────────────

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

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}
