use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::{
    proto::inference::{
        grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer},
        InferInputTensor, InferOutputTensor, InferTensorContents, ModelInferRequest,
        ModelInferResponse, ModelMetadataInput, ModelMetadataOutput, ModelMetadataRequest,
        ModelMetadataResponse, ModelReadyRequest, ModelReadyResponse, ServerLiveRequest,
        ServerLiveResponse, ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest,
        ServerReadyResponse,
    },
    registry::ModelRegistry,
    tensor::{
        prediction_shape, raw_bytes_f64_to_f32, raw_bytes_to_f32, raw_bytes_to_strings,
        reshape_strings, validate_and_get_num_features,
    },
};

// ── Service ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct InferenceService {
    registry: Arc<ModelRegistry>,
}

impl InferenceService {
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        Self { registry }
    }

    pub fn into_server(self) -> GrpcInferenceServiceServer<Self> {
        GrpcInferenceServiceServer::new(self)
    }
}

pub async fn serve(
    addr: std::net::SocketAddr,
    registry: Arc<ModelRegistry>,
) -> anyhow::Result<()> {
    tracing::info!(%addr, "gRPC server listening");
    tonic::transport::Server::builder()
        .add_service(InferenceService::new(registry).into_server())
        .serve(addr)
        .await?;
    Ok(())
}

// ── Trait implementation ──────────────────────────────────────────────────────

#[tonic::async_trait]
impl GrpcInferenceService for InferenceService {
    // ── Health ────────────────────────────────────────────────────────────────

    async fn server_live(
        &self,
        _: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        Ok(Response::new(ServerReadyResponse { ready: true }))
    }

    async fn model_ready(
        &self,
        req: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let ready = req.into_inner().name == self.registry.name;
        Ok(Response::new(ModelReadyResponse { ready }))
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    async fn server_metadata(
        &self,
        _: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        Ok(Response::new(ServerMetadataResponse {
            name: "kserve-catboost".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            extensions: vec!["classification".to_string(), "regression".to_string()],
        }))
    }

    async fn model_metadata(
        &self,
        req: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let r = req.into_inner();
        if r.name != self.registry.name {
            return Err(Status::not_found(format!("model '{}' not found", r.name)));
        }

        let model = self.registry.load();
        let mut inputs = vec![ModelMetadataInput {
            name: "float_features".to_string(),
            datatype: "FP32".to_string(),
            shape: vec![-1, model.float_features_count as i64],
        }];
        if model.cat_features_count > 0 {
            inputs.push(ModelMetadataInput {
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

        Ok(Response::new(ModelMetadataResponse {
            name: model.name.clone(),
            versions: vec!["1".to_string()],
            platform: "catboost".to_string(),
            inputs,
            outputs: vec![ModelMetadataOutput {
                name: "predictions".to_string(),
                datatype: "FP64".to_string(),
                shape: out_shape,
            }],
        }))
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    async fn model_infer(
        &self,
        req: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let inner = req.into_inner();
        if inner.model_name != self.registry.name {
            return Err(Status::not_found(format!(
                "model '{}' not found",
                inner.model_name
            )));
        }

        let id = if inner.id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            inner.id.clone()
        };

        // Snapshot — survives reloads in parallel.
        let model = self.registry.load();
        let dimensions = model.dimensions_count;
        let model_name = model.name.clone();

        let predictions = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f64>> {
            // ── Float features: flat buffer → row slices ───────────────────
            let (flat, num_feats) =
                extract_float_flat_grpc(&inner.inputs, &inner.raw_input_contents)?;
            let rows: Vec<&[f32]> = flat.chunks(num_feats).collect();

            // ── Cat features ───────────────────────────────────────────────
            let cat = extract_cat_grpc(&inner.inputs, &inner.raw_input_contents)?;

            model.predict(&rows, cat)
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .map_err(|e| Status::internal(e.to_string()))?;

        let shape = prediction_shape(predictions.len(), dimensions);

        Ok(Response::new(ModelInferResponse {
            model_name,
            model_version: "1".to_string(),
            id,
            parameters: Default::default(),
            outputs: vec![InferOutputTensor {
                name: "predictions".to_string(),
                datatype: "FP64".to_string(),
                shape,
                parameters: Default::default(),
                contents: Some(InferTensorContents {
                    fp64_contents: predictions,
                    ..Default::default()
                }),
            }],
            raw_output_contents: vec![],
        }))
    }
}

// ── gRPC input extraction ─────────────────────────────────────────────────────

/// Returns a flat `Vec<f32>` and the per-row feature count.
fn extract_float_flat_grpc(
    inputs: &[InferInputTensor],
    raw: &[Vec<u8>],
) -> anyhow::Result<(Vec<f32>, usize)> {
    let (idx, input) = inputs
        .iter()
        .enumerate()
        .find(|(_, i)| i.name == "float_features")
        .or_else(|| {
            inputs
                .iter()
                .enumerate()
                .find(|(_, i)| matches!(i.datatype.as_str(), "FP32" | "FP64"))
        })
        .ok_or_else(|| anyhow::anyhow!("no float_features input found"))?;

    let flat: Vec<f32> = if !raw.is_empty() {
        let bytes = raw
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("raw_input_contents[{}] missing", idx))?;
        match input.datatype.as_str() {
            "FP64" => raw_bytes_f64_to_f32(bytes)?,
            _ => raw_bytes_to_f32(bytes)?,
        }
    } else if let Some(c) = &input.contents {
        if !c.fp32_contents.is_empty() {
            c.fp32_contents.clone()
        } else if !c.fp64_contents.is_empty() {
            c.fp64_contents.iter().map(|&v| v as f32).collect()
        } else if !c.int_contents.is_empty() {
            c.int_contents.iter().map(|&v| v as f32).collect()
        } else if !c.int64_contents.is_empty() {
            c.int64_contents.iter().map(|&v| v as f32).collect()
        } else {
            return Err(anyhow::anyhow!("float_features contents are empty"));
        }
    } else {
        return Err(anyhow::anyhow!(
            "float_features has neither contents nor raw bytes"
        ));
    };

    let num_feats = validate_and_get_num_features(flat.len(), &input.shape)?;
    Ok((flat, num_feats))
}

fn extract_cat_grpc(
    inputs: &[InferInputTensor],
    raw: &[Vec<u8>],
) -> anyhow::Result<Vec<Vec<String>>> {
    let found = inputs
        .iter()
        .enumerate()
        .find(|(_, i)| i.name == "cat_features")
        .or_else(|| {
            inputs
                .iter()
                .enumerate()
                .find(|(_, i)| i.datatype == "BYTES")
        });

    let (idx, input) = match found {
        None => return Ok(vec![]),
        Some(v) => v,
    };

    let strings: Vec<String> = if !raw.is_empty() {
        let bytes = raw
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("raw_input_contents[{}] missing", idx))?;
        raw_bytes_to_strings(bytes)?
    } else if let Some(c) = &input.contents {
        c.bytes_contents
            .iter()
            .map(|b| {
                String::from_utf8(b.clone())
                    .map_err(|e| anyhow::anyhow!("invalid UTF-8 in cat feature: {}", e))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    } else {
        return Err(anyhow::anyhow!(
            "cat_features has neither contents nor raw bytes"
        ));
    };

    reshape_strings(strings, &input.shape)
}
