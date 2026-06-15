use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::model::CatBoostModel;

pub struct InferInputs {
    pub float_features: Vec<Vec<f32>>,
    pub cat_features: Vec<Vec<String>>,
}

pub struct InferOutput {
    pub predictions: Vec<f64>,
    pub shape: Vec<i64>,
}

pub struct RawTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub data: TensorData,
}

pub enum TensorData {
    Floats(Vec<f32>),
    Strings(Vec<String>),
    RawBytes(Vec<u8>),
}

/// Convert a JSON value (from an HTTP request's `data` field) into TensorData.
/// Recursively flattens nested arrays so both `[[1,2],[3,4]]` and `[1,2,3,4]` are accepted.
pub fn json_to_tensor_data(value: serde_json::Value, datatype: &str) -> Result<TensorData> {
    let arr = value
        .as_array()
        .ok_or_else(|| anyhow!("tensor 'data' must be a JSON array"))?;

    match datatype {
        "FP32" | "FP64" | "INT32" | "INT64" | "UINT32" | "UINT64" => {
            Ok(TensorData::Floats(flatten_floats(arr)?))
        }
        "BYTES" | "STRING" => Ok(TensorData::Strings(flatten_strings(arr)?)),
        dt => Err(anyhow!("unsupported datatype: {dt}")),
    }
}

fn flatten_floats(arr: &[serde_json::Value]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        match v {
            serde_json::Value::Array(inner) => out.extend(flatten_floats(inner)?),
            serde_json::Value::Number(n) => {
                out.push(n.as_f64().ok_or_else(|| anyhow!("invalid number: {n}"))? as f32)
            }
            other => return Err(anyhow!("expected number or array, got: {other}")),
        }
    }
    Ok(out)
}

fn flatten_strings(arr: &[serde_json::Value]) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        match v {
            serde_json::Value::Array(inner) => out.extend(flatten_strings(inner)?),
            serde_json::Value::String(s) => out.push(s.clone()),
            other => return Err(anyhow!("expected string or array, got: {other}")),
        }
    }
    Ok(out)
}

/// Decode raw little-endian bytes for numeric types into f32.
fn decode_raw_floats(bytes: &[u8], datatype: &str) -> Result<Vec<f32>> {
    match datatype {
        "FP32" => {
            if bytes.len() % 4 != 0 {
                return Err(anyhow!("FP32 bytes length {} is not a multiple of 4", bytes.len()));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        "FP64" => {
            if bytes.len() % 8 != 0 {
                return Err(anyhow!("FP64 bytes length {} is not a multiple of 8", bytes.len()));
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect())
        }
        "INT32" => {
            if bytes.len() % 4 != 0 {
                return Err(anyhow!("INT32 bytes length {} is not a multiple of 4", bytes.len()));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect())
        }
        "INT64" => {
            if bytes.len() % 8 != 0 {
                return Err(anyhow!("INT64 bytes length {} is not a multiple of 8", bytes.len()));
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect())
        }
        dt => Err(anyhow!("cannot decode {dt} from raw bytes")),
    }
}

/// Decode raw BYTES-type data as repeated 4-byte-length-prefixed UTF-8 strings.
fn decode_raw_strings(bytes: &[u8]) -> Result<Vec<String>> {
    let mut result = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + len > bytes.len() {
            return Err(anyhow!("malformed BYTES tensor: string overruns buffer"));
        }
        result.push(String::from_utf8(bytes[pos..pos + len].to_vec())?);
        pos += len;
    }
    Ok(result)
}

fn reshape<T: Clone>(flat: Vec<T>, shape: &[i64]) -> Vec<Vec<T>> {
    let cols = shape.get(1).copied().unwrap_or(1).max(1) as usize;
    flat.chunks(cols).map(|c| c.to_vec()).collect()
}

/// Convert a list of RawTensors into InferInputs.
/// The first numeric tensor becomes float_features; the first BYTES tensor becomes cat_features.
pub fn parse_inputs(tensors: Vec<RawTensor>) -> Result<InferInputs> {
    let mut float_features: Vec<Vec<f32>> = vec![];
    let mut cat_features: Vec<Vec<String>> = vec![];
    let mut got_floats = false;
    let mut got_cats = false;

    for t in tensors {
        match t.datatype.as_str() {
            dt @ ("FP32" | "FP64" | "INT32" | "INT64" | "UINT32" | "UINT64") if !got_floats => {
                let flat = match t.data {
                    TensorData::Floats(v) => v,
                    TensorData::RawBytes(b) => decode_raw_floats(&b, dt)?,
                    TensorData::Strings(_) => {
                        return Err(anyhow!("expected numeric data for tensor '{}'", t.name))
                    }
                };
                float_features = reshape(flat, &t.shape);
                got_floats = true;
            }
            "BYTES" | "STRING" if !got_cats => {
                let flat = match t.data {
                    TensorData::Strings(v) => v,
                    TensorData::RawBytes(b) => decode_raw_strings(&b)?,
                    TensorData::Floats(_) => {
                        return Err(anyhow!("expected string data for tensor '{}'", t.name))
                    }
                };
                cat_features = reshape(flat, &t.shape);
                got_cats = true;
            }
            _ => {} // skip duplicates / unsupported types
        }
    }

    Ok(InferInputs {
        float_features,
        cat_features,
    })
}

pub async fn run_blocking(model: Arc<CatBoostModel>, inputs: InferInputs) -> Result<InferOutput> {
    let num_rows = inputs
        .float_features
        .len()
        .max(inputs.cat_features.len()) as i64;

    let predictions = tokio::task::spawn_blocking(move || {
        model.predict(inputs.float_features, inputs.cat_features)
    })
    .await??;

    Ok(InferOutput {
        shape: vec![num_rows, 1],
        predictions,
    })
}
