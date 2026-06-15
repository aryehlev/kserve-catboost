mod batcher;
mod parse;
pub use batcher::DynamicBatcher;
pub use parse::{
    decode_raw_floats, decode_raw_strings, json_to_tensor_data, parse_inputs, reshape, RawTensor,
    TensorData,
};

use std::sync::Arc;

use anyhow::Result;

use crate::model::CatBoostModel;

// Output tensor descriptor — shared by both HTTP and gRPC response builders.
pub const OUTPUT_TENSOR: &str = "output-0";
pub const OUTPUT_DTYPE: &str = "FP64";

pub struct InferInputs {
    pub float_features: Vec<Vec<f32>>,
    pub cat_features: Vec<Vec<String>>,
}

pub struct InferOutput {
    pub predictions: Vec<f64>,
    pub shape: Vec<i64>,
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
