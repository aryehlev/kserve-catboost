use anyhow::{anyhow, Result};

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

/// Decode little-endian raw bytes for a numeric datatype into f32.
pub fn decode_raw_floats(bytes: &[u8], datatype: &str) -> Result<Vec<f32>> {
    match datatype {
        "FP32" => chunked(bytes, 4, |c| f32::from_le_bytes(c.try_into().unwrap())),
        "FP64" => chunked(bytes, 8, |c| {
            f64::from_le_bytes(c.try_into().unwrap()) as f32
        }),
        "INT32" => chunked(bytes, 4, |c| {
            i32::from_le_bytes(c.try_into().unwrap()) as f32
        }),
        "INT64" => chunked(bytes, 8, |c| {
            i64::from_le_bytes(c.try_into().unwrap()) as f32
        }),
        "UINT32" => chunked(bytes, 4, |c| {
            u32::from_le_bytes(c.try_into().unwrap()) as f32
        }),
        "UINT64" => chunked(bytes, 8, |c| {
            u64::from_le_bytes(c.try_into().unwrap()) as f32
        }),
        dt => Err(anyhow!("cannot decode {dt} from raw bytes")),
    }
}

fn chunked<T>(bytes: &[u8], width: usize, f: impl Fn(&[u8]) -> T) -> Result<Vec<T>> {
    if bytes.len() % width != 0 {
        return Err(anyhow!(
            "raw bytes length {} is not a multiple of {width}",
            bytes.len()
        ));
    }
    Ok(bytes.chunks_exact(width).map(f).collect())
}

/// Decode raw BYTES data as repeated 4-byte-LE-length-prefixed UTF-8 strings.
pub fn decode_raw_strings(bytes: &[u8]) -> Result<Vec<String>> {
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
    if pos != bytes.len() {
        return Err(anyhow!("malformed BYTES tensor: trailing bytes after last string"));
    }
    Ok(result)
}

/// Reshape a flat vector into rows using shape[1] as the column count.
/// Returns an error if the data length is not evenly divisible by the column count.
pub fn reshape<T: Clone>(flat: Vec<T>, shape: &[i64]) -> Result<Vec<Vec<T>>> {
    let cols_i64 = shape.get(1).copied().unwrap_or(1);
    if cols_i64 <= 0 {
        return Err(anyhow!(
            "invalid tensor shape: non-positive column count {cols_i64}"
        ));
    }
    let cols = cols_i64 as usize;
    if flat.len() % cols != 0 {
        return Err(anyhow!(
            "tensor data length {} is not divisible by column count {cols}",
            flat.len()
        ));
    }
    Ok(flat.chunks_exact(cols).map(|c| c.to_vec()).collect())
}

/// Convert a list of RawTensors into InferInputs.
/// The first numeric tensor becomes float_features; the first BYTES tensor becomes cat_features.
pub fn parse_inputs(tensors: Vec<RawTensor>) -> Result<super::InferInputs> {
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
                float_features = reshape(flat, &t.shape)?;
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
                cat_features = reshape(flat, &t.shape)?;
                got_cats = true;
            }
            _ => {}
        }
    }

    Ok(super::InferInputs {
        float_features,
        cat_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── reshape ───────────────────────────────────────────────────────────────

    #[test]
    fn reshape_2d() {
        let r = reshape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(r, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    fn reshape_no_col_dim_defaults_to_1() {
        let r = reshape(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(r, vec![vec![1.0], vec![2.0], vec![3.0]]);
    }

    #[test]
    fn reshape_empty() {
        let r = reshape(Vec::<f32>::new(), &[0, 1]).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn reshape_not_divisible() {
        assert!(reshape(vec![1.0f32, 2.0, 3.0], &[1, 2]).is_err());
    }

    #[test]
    fn reshape_zero_cols() {
        assert!(reshape(vec![1.0f32], &[1, 0]).is_err());
    }

    #[test]
    fn reshape_negative_cols() {
        assert!(reshape(vec![1.0f32], &[1, -1]).is_err());
    }

    // ── decode_raw_floats ─────────────────────────────────────────────────────

    fn f32_bytes(v: f32) -> Vec<u8> { v.to_le_bytes().to_vec() }
    fn f64_bytes(v: f64) -> Vec<u8> { v.to_le_bytes().to_vec() }
    fn i32_bytes(v: i32) -> Vec<u8> { v.to_le_bytes().to_vec() }
    fn i64_bytes(v: i64) -> Vec<u8> { v.to_le_bytes().to_vec() }
    fn u32_bytes(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }
    fn u64_bytes(v: u64) -> Vec<u8> { v.to_le_bytes().to_vec() }

    #[test]
    fn raw_fp32() {
        let r = decode_raw_floats(&f32_bytes(1.5), "FP32").unwrap();
        assert_eq!(r, vec![1.5f32]);
    }

    #[test]
    fn raw_fp64() {
        let r = decode_raw_floats(&f64_bytes(1.5), "FP64").unwrap();
        assert!((r[0] - 1.5f32).abs() < 1e-6);
    }

    #[test]
    fn raw_int32_negative() {
        let r = decode_raw_floats(&i32_bytes(-7), "INT32").unwrap();
        assert_eq!(r, vec![-7.0f32]);
    }

    #[test]
    fn raw_int64() {
        let r = decode_raw_floats(&i64_bytes(1_000_000), "INT64").unwrap();
        assert_eq!(r, vec![1_000_000.0f32]);
    }

    #[test]
    fn raw_uint32() {
        let r = decode_raw_floats(&u32_bytes(300), "UINT32").unwrap();
        assert_eq!(r, vec![300.0f32]);
    }

    #[test]
    fn raw_uint64() {
        let r = decode_raw_floats(&u64_bytes(999), "UINT64").unwrap();
        assert_eq!(r, vec![999.0f32]);
    }

    #[test]
    fn raw_floats_multiple_values() {
        let mut bytes = f32_bytes(1.0);
        bytes.extend(f32_bytes(2.0));
        bytes.extend(f32_bytes(3.0));
        let r = decode_raw_floats(&bytes, "FP32").unwrap();
        assert_eq!(r, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn raw_floats_bad_length() {
        assert!(decode_raw_floats(&[1, 2, 3], "FP32").is_err());
        assert!(decode_raw_floats(&[1, 2, 3, 4, 5], "INT64").is_err());
    }

    #[test]
    fn raw_floats_unknown_dtype() {
        assert!(decode_raw_floats(&[], "BOOL").is_err());
    }

    // ── decode_raw_strings ────────────────────────────────────────────────────

    fn length_prefix(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u32).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }

    #[test]
    fn raw_strings_single() {
        let r = decode_raw_strings(&length_prefix("hello")).unwrap();
        assert_eq!(r, vec!["hello"]);
    }

    #[test]
    fn raw_strings_multiple() {
        let mut b = length_prefix("foo");
        b.extend(length_prefix("bar"));
        b.extend(length_prefix("baz"));
        let r = decode_raw_strings(&b).unwrap();
        assert_eq!(r, vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn raw_strings_empty_string() {
        let r = decode_raw_strings(&length_prefix("")).unwrap();
        assert_eq!(r, vec![""]);
    }

    #[test]
    fn raw_strings_empty_buffer() {
        let r = decode_raw_strings(&[]).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn raw_strings_overrun() {
        let mut b = 100u32.to_le_bytes().to_vec(); // length 100 but no payload
        assert!(decode_raw_strings(&b).is_err());

        // length header incomplete (3 bytes instead of 4)
        b = vec![0x05, 0x00, 0x00];
        assert!(decode_raw_strings(&b).is_err());
    }

    #[test]
    fn raw_strings_trailing_bytes() {
        let mut b = length_prefix("abc");
        b.push(0xff); // extra byte after last string
        assert!(decode_raw_strings(&b).is_err());
    }

    // ── json_to_tensor_data ───────────────────────────────────────────────────

    #[test]
    fn json_flat_floats() {
        let v = serde_json::json!([1.0, 2.0, 3.0]);
        match json_to_tensor_data(v, "FP32").unwrap() {
            TensorData::Floats(f) => assert_eq!(f, vec![1.0f32, 2.0, 3.0]),
            _ => panic!("expected floats"),
        }
    }

    #[test]
    fn json_nested_floats() {
        let v = serde_json::json!([[1.0, 2.0], [3.0, 4.0]]);
        match json_to_tensor_data(v, "FP32").unwrap() {
            TensorData::Floats(f) => assert_eq!(f, vec![1.0f32, 2.0, 3.0, 4.0]),
            _ => panic!("expected floats"),
        }
    }

    #[test]
    fn json_int_dtype_parsed_as_floats() {
        let v = serde_json::json!([10, 20]);
        match json_to_tensor_data(v, "INT32").unwrap() {
            TensorData::Floats(f) => assert_eq!(f, vec![10.0f32, 20.0]),
            _ => panic!("expected floats"),
        }
    }

    #[test]
    fn json_strings() {
        let v = serde_json::json!(["cat", "dog"]);
        match json_to_tensor_data(v, "BYTES").unwrap() {
            TensorData::Strings(s) => assert_eq!(s, vec!["cat", "dog"]),
            _ => panic!("expected strings"),
        }
    }

    #[test]
    fn json_string_dtype_alias() {
        let v = serde_json::json!(["x"]);
        assert!(json_to_tensor_data(v, "STRING").is_ok());
    }

    #[test]
    fn json_not_array() {
        assert!(json_to_tensor_data(serde_json::json!(42), "FP32").is_err());
        assert!(json_to_tensor_data(serde_json::json!({"k": 1}), "FP32").is_err());
    }

    #[test]
    fn json_unknown_dtype() {
        assert!(json_to_tensor_data(serde_json::json!([1]), "BOOL").is_err());
    }

    // ── parse_inputs ──────────────────────────────────────────────────────────

    #[test]
    fn parse_float_tensor() {
        let tensors = vec![RawTensor {
            name: "input".into(),
            datatype: "FP32".into(),
            shape: vec![2, 3],
            data: TensorData::Floats(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        }];
        let r = parse_inputs(tensors).unwrap();
        assert_eq!(r.float_features, vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert!(r.cat_features.is_empty());
    }

    #[test]
    fn parse_cat_tensor() {
        let tensors = vec![RawTensor {
            name: "cats".into(),
            datatype: "BYTES".into(),
            shape: vec![2, 2],
            data: TensorData::Strings(vec!["a".into(), "b".into(), "c".into(), "d".into()]),
        }];
        let r = parse_inputs(tensors).unwrap();
        assert!(r.float_features.is_empty());
        assert_eq!(r.cat_features, vec![vec!["a", "b"], vec!["c", "d"]]);
    }

    #[test]
    fn parse_float_and_cat() {
        let tensors = vec![
            RawTensor {
                name: "f".into(),
                datatype: "FP32".into(),
                shape: vec![1, 2],
                data: TensorData::Floats(vec![1.0, 2.0]),
            },
            RawTensor {
                name: "c".into(),
                datatype: "BYTES".into(),
                shape: vec![1, 1],
                data: TensorData::Strings(vec!["x".into()]),
            },
        ];
        let r = parse_inputs(tensors).unwrap();
        assert_eq!(r.float_features, vec![vec![1.0f32, 2.0]]);
        assert_eq!(r.cat_features, vec![vec!["x"]]);
    }

    #[test]
    fn parse_raw_bytes_float() {
        let bytes: Vec<u8> = [1.0f32, 2.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensors = vec![RawTensor {
            name: "f".into(),
            datatype: "FP32".into(),
            shape: vec![2, 1],
            data: TensorData::RawBytes(bytes),
        }];
        let r = parse_inputs(tensors).unwrap();
        assert_eq!(r.float_features, vec![vec![1.0f32], vec![2.0f32]]);
    }

    #[test]
    fn parse_raw_bytes_strings() {
        let mut bytes = length_prefix("hello");
        bytes.extend(length_prefix("world"));
        let tensors = vec![RawTensor {
            name: "c".into(),
            datatype: "BYTES".into(),
            shape: vec![2, 1],
            data: TensorData::RawBytes(bytes),
        }];
        let r = parse_inputs(tensors).unwrap();
        assert_eq!(r.cat_features, vec![vec!["hello"], vec!["world"]]);
    }

    #[test]
    fn parse_wrong_data_type_for_float_tensor() {
        let tensors = vec![RawTensor {
            name: "f".into(),
            datatype: "FP32".into(),
            shape: vec![1, 1],
            data: TensorData::Strings(vec!["x".into()]),
        }];
        assert!(parse_inputs(tensors).is_err());
    }

    #[test]
    fn parse_wrong_data_type_for_cat_tensor() {
        let tensors = vec![RawTensor {
            name: "c".into(),
            datatype: "BYTES".into(),
            shape: vec![1, 1],
            data: TensorData::Floats(vec![1.0]),
        }];
        assert!(parse_inputs(tensors).is_err());
    }

    #[test]
    fn parse_shape_mismatch_rejected() {
        // 6 elements but shape says 2 cols → 3 rows; should succeed
        let tensors = vec![RawTensor {
            name: "f".into(),
            datatype: "FP32".into(),
            shape: vec![3, 2],
            data: TensorData::Floats(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        }];
        assert!(parse_inputs(tensors).is_ok());

        // 5 elements not divisible by 2 cols → error
        let tensors = vec![RawTensor {
            name: "f".into(),
            datatype: "FP32".into(),
            shape: vec![2, 2],
            data: TensorData::Floats(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        }];
        assert!(parse_inputs(tensors).is_err());
    }
}
