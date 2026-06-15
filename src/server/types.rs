use serde::{Deserialize, Serialize};

// ── Infer ────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct RequestedOutput {
    pub name: String,
}

#[derive(Deserialize)]
pub struct InferRequest {
    #[serde(default)]
    pub id: Option<String>,
    pub inputs: Vec<InferInputTensor>,
    /// Clients may name the output tensors they want back.
    /// When provided the first entry's name is used for the response tensor.
    #[serde(default)]
    pub outputs: Vec<RequestedOutput>,
}

#[derive(Deserialize)]
pub struct InferInputTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub data: serde_json::Value,
}

#[derive(Serialize)]
pub struct InferResponse {
    pub model_name: String,
    pub model_version: String,
    pub id: String,
    pub outputs: Vec<InferOutputTensor>,
}

#[derive(Serialize)]
pub struct InferOutputTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub data: Vec<f64>,
}

// ── Metadata ─────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ServerMetadataResponse {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

#[derive(Serialize)]
pub struct MetadataTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
}

#[derive(Serialize)]
pub struct ModelMetadataResponse {
    pub name: String,
    pub versions: Vec<String>,
    pub platform: String,
    pub inputs: Vec<MetadataTensor>,
    pub outputs: Vec<MetadataTensor>,
}

// ── Repository ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct RepositoryIndexEntry {
    pub name: String,
    pub version: String,
    pub state: String,
    pub reason: String,
}

// ── Error ────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}
