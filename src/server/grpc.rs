use std::sync::Arc;

use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::inference::{DynamicBatcher, OverloadError, RawTensor, TensorData, OUTPUT_DTYPE, OUTPUT_TENSOR};
use super::{
    proto::inference::{
        grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer},
        InferOutputTensor, InferTensorContents, ModelInferRequest, ModelInferResponse,
        ModelMetadataRequest, ModelMetadataResponse, ModelReadyRequest, ModelReadyResponse,
        ServerLiveRequest, ServerLiveResponse, ServerMetadataRequest, ServerMetadataResponse,
        ServerReadyRequest, ServerReadyResponse,
    },
    EXTENSIONS, PLATFORM,
};

pub struct GrpcService {
    batcher: Arc<DynamicBatcher>,
}

impl GrpcService {
    pub fn new(batcher: Arc<DynamicBatcher>) -> Self {
        Self { batcher }
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for GrpcService {
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
        Ok(Response::new(ServerReadyResponse {
            ready: self.batcher.registry.is_loaded(),
        }))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let req = request.into_inner();
        let ready = req.name == self.batcher.registry.name && self.batcher.registry.is_loaded();
        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn server_metadata(
        &self,
        _: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        Ok(Response::new(ServerMetadataResponse {
            name: env!("CARGO_PKG_NAME").to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            extensions: EXTENSIONS.iter().map(|s| s.to_string()).collect(),
        }))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let req = request.into_inner();
        if req.name != self.batcher.registry.name {
            return Err(Status::not_found(format!("no model '{}'", req.name)));
        }
        Ok(Response::new(ModelMetadataResponse {
            name: self.batcher.registry.name.clone(),
            versions: vec![self.batcher.registry.version().to_string()],
            platform: PLATFORM.to_string(),
            inputs: vec![],
            outputs: vec![],
        }))
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let ModelInferRequest {
            model_name,
            id,
            inputs,
            outputs: requested_outputs,
            raw_input_contents: raw,
            ..
        } = request.into_inner();

        if model_name != self.batcher.registry.name {
            return Err(Status::not_found(format!("no model '{model_name}'")));
        }
        if !self.batcher.registry.is_loaded() {
            return Err(Status::unavailable(format!(
                "model '{}' is not loaded",
                self.batcher.registry.name
            )));
        }

        let tensors: Result<Vec<RawTensor>, Status> = inputs
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                let data = if let Some(c) = t.contents {
                    if !c.fp32_contents.is_empty() {
                        TensorData::Floats(c.fp32_contents)
                    } else if !c.bytes_contents.is_empty() {
                        let strings = c
                            .bytes_contents
                            .into_iter()
                            .map(|b| {
                                String::from_utf8(b)
                                    .map_err(|e| Status::invalid_argument(e.to_string()))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        TensorData::Strings(strings)
                    } else {
                        TensorData::Floats(vec![])
                    }
                } else if let Some(bytes) = raw.get(i) {
                    TensorData::RawBytes(bytes.clone())
                } else {
                    TensorData::Floats(vec![])
                };

                Ok(RawTensor {
                    name: t.name,
                    datatype: t.datatype,
                    shape: t.shape,
                    data,
                })
            })
            .collect();

        let inputs = crate::inference::parse_inputs(tensors?)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let out = self
            .batcher
            .infer(inputs)
            .await
            .map_err(|e| {
                if e.is::<OverloadError>() {
                    Status::resource_exhausted(e.to_string())
                } else {
                    Status::internal(e.to_string())
                }
            })?;

        let resp_id = if id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            id
        };
        let output_name = requested_outputs
            .into_iter()
            .next()
            .map(|o| o.name)
            .unwrap_or_else(|| OUTPUT_TENSOR.to_string());

        Ok(Response::new(ModelInferResponse {
            model_name: self.batcher.registry.name.clone(),
            model_version: self.batcher.registry.version().to_string(),
            id: resp_id,
            parameters: Default::default(),
            outputs: vec![InferOutputTensor {
                name: output_name,
                datatype: OUTPUT_DTYPE.to_string(),
                shape: out.shape,
                parameters: Default::default(),
                contents: Some(InferTensorContents {
                    fp64_contents: out.predictions,
                    ..Default::default()
                }),
            }],
            raw_output_contents: vec![],
        }))
    }
}

pub async fn serve(
    addr: std::net::SocketAddr,
    batcher: Arc<DynamicBatcher>,
) -> anyhow::Result<()> {
    let svc = GrpcInferenceServiceServer::new(GrpcService::new(batcher));
    tonic::transport::Server::builder()
        .add_service(svc)
        .serve(addr)
        .await?;
    Ok(())
}
