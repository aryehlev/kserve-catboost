/// Tensor helpers shared by HTTP and gRPC handlers.
///
/// Float features are returned as a **flat** `Vec<f32>` so callers can build
/// row-slices with `.chunks(num_features)` — avoiding one `Vec<f32>` allocation
/// per row when feeding large batches into the CatBoost C API.
use anyhow::{anyhow, Result};

// ── JSON (HTTP) ───────────────────────────────────────────────────────────────

/// Recursively flatten a (possibly nested) JSON array to a flat `Vec<f32>`.
pub fn flatten_json_to_f32(val: &serde_json::Value) -> Result<Vec<f32>> {
    match val {
        serde_json::Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                match v {
                    serde_json::Value::Array(_) => out.extend(flatten_json_to_f32(v)?),
                    serde_json::Value::Number(n) => {
                        out.push(n.as_f64().ok_or_else(|| anyhow!("non-finite number"))? as f32);
                    }
                    _ => return Err(anyhow!("expected number in float tensor, got {}", v)),
                }
            }
            Ok(out)
        }
        _ => Err(anyhow!("expected JSON array for tensor data")),
    }
}

/// Recursively flatten a (possibly nested) JSON array to a flat `Vec<String>`.
pub fn flatten_json_to_strings(val: &serde_json::Value) -> Result<Vec<String>> {
    match val {
        serde_json::Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                match v {
                    serde_json::Value::Array(_) => out.extend(flatten_json_to_strings(v)?),
                    serde_json::Value::String(s) => out.push(s.clone()),
                    _ => return Err(anyhow!("expected string in cat tensor, got {}", v)),
                }
            }
            Ok(out)
        }
        _ => Err(anyhow!("expected JSON array for tensor data")),
    }
}

// ── Shape helpers ─────────────────────────────────────────────────────────────

/// Resolve a 2-D shape `[d0, d1]` where at most one dimension may be −1
/// (dynamic), inferred from `data_len`.
pub fn resolve_shape_2d(shape: &[i64], data_len: usize) -> Result<(usize, usize)> {
    match shape {
        [d0, d1] => match (*d0, *d1) {
            (a, b) if a > 0 && b > 0 => Ok((a as usize, b as usize)),
            (-1, b) if b > 0 => {
                let b = b as usize;
                Ok((data_len / b, b))
            }
            (a, -1) if a > 0 => {
                let a = a as usize;
                Ok((a, data_len / a))
            }
            _ => Err(anyhow!("invalid 2-D shape {:?}", shape)),
        },
        _ => Err(anyhow!(
            "expected 2-D shape for batched features, got {:?}",
            shape
        )),
    }
}

/// Validate that `flat.len() == batch * features` and return the number of
/// features per row.  Callers use the returned count to slice the flat buffer.
pub fn validate_and_get_num_features(flat_len: usize, shape: &[i64]) -> Result<usize> {
    match shape.len() {
        1 => Ok(flat_len),  // single row: all values are features
        2 => {
            let (_, num_features) = resolve_shape_2d(shape, flat_len)?;
            if flat_len % num_features != 0 {
                return Err(anyhow!(
                    "flat length {} is not divisible by {} features",
                    flat_len,
                    num_features
                ));
            }
            Ok(num_features)
        }
        _ => Err(anyhow!(
            "float features must be 1-D or 2-D, got {:?}",
            shape
        )),
    }
}

/// Reshape a flat string vec into `Vec<Vec<String>>` — cat features still need
/// owned values, so there is no zero-copy path here.
pub fn reshape_strings(flat: Vec<String>, shape: &[i64]) -> Result<Vec<Vec<String>>> {
    match shape.len() {
        1 => Ok(vec![flat]),
        2 => {
            let (batch, feats) = resolve_shape_2d(shape, flat.len())?;
            if flat.len() != batch * feats {
                return Err(anyhow!(
                    "cat data length {} ≠ shape {}×{}",
                    flat.len(),
                    batch,
                    feats
                ));
            }
            Ok(flat.chunks(feats).map(|c| c.to_vec()).collect())
        }
        _ => Err(anyhow!("cat features must be 1-D or 2-D")),
    }
}

// ── Raw binary helpers (gRPC raw_input_contents) ──────────────────────────────

/// Parse raw little-endian f32 bytes into a flat `Vec<f32>`.
pub fn raw_bytes_to_f32(raw: &[u8]) -> Result<Vec<f32>> {
    if raw.len() % 4 != 0 {
        return Err(anyhow!(
            "raw float bytes length {} is not a multiple of 4",
            raw.len()
        ));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Parse raw little-endian f64 bytes into floats (converts to f32).
pub fn raw_bytes_f64_to_f32(raw: &[u8]) -> Result<Vec<f32>> {
    if raw.len() % 8 != 0 {
        return Err(anyhow!(
            "raw double bytes length {} is not a multiple of 8",
            raw.len()
        ));
    }
    Ok(raw
        .chunks_exact(8)
        .map(|b| {
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        })
        .collect())
}

/// Parse KServe raw BYTES encoding: each element is a 4-byte LE `uint32`
/// length prefix followed by that many UTF-8 bytes.
pub fn raw_bytes_to_strings(raw: &[u8]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        if i + 4 > raw.len() {
            return Err(anyhow!("truncated length prefix at offset {}", i));
        }
        let len = u32::from_le_bytes([raw[i], raw[i + 1], raw[i + 2], raw[i + 3]]) as usize;
        i += 4;
        if i + len > raw.len() {
            return Err(anyhow!(
                "string data truncated: expected {} bytes at offset {}",
                len,
                i
            ));
        }
        let s = std::str::from_utf8(&raw[i..i + len])
            .map_err(|e| anyhow!("invalid UTF-8 in cat feature: {}", e))?
            .to_string();
        out.push(s);
        i += len;
    }
    Ok(out)
}

// ── Output shape ──────────────────────────────────────────────────────────────

/// Compute the response shape for a flat CatBoost prediction vector.
///
/// CatBoost returns `batch_size × dimensions_count` values in one flat `Vec`.
pub fn prediction_shape(predictions_len: usize, dimensions: usize) -> Vec<i64> {
    if dimensions <= 1 {
        vec![predictions_len as i64]
    } else {
        let batch = predictions_len / dimensions;
        vec![batch as i64, dimensions as i64]
    }
}
