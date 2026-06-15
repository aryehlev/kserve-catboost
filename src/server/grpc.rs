use std::sync::Arc;

use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::{
    inference::{self, RawTensor, TensorData},
    model::ModelRegistry,
};
use super::proto::inference::{
    grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer},
    InferOutputTensor, InferTensorContents, ModelInferRequest, ModelInferResponse,
    ModelMetadataRequest, ModelMetadataResponse, ModelReadyRequest, ModelReadyResponse,
    ServerLiveRequest, ServerLiveResponse, ServerMetadataRequest, ServerMetadataResponse,
    ServerReadyRequest, ServerReadyResponse,
};

pub struct GrpcService {
    registry: Arc<ModelRegistry>,
}

impl GrpcService {
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        Self { registry }
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
            ready: self.registry.is_loaded(),
        }))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let req = request.into_inner();
        let ready = req.name == self.registry.name && self.registry.is_loaded();
        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn server_metadata(
        &self,
        _: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        Ok(Response::new(ServerMetadataResponse {
            name: "kserve-catboost".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            extensions: vec!["model_repository_extension".to_string()],
        }))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let req = request.into_inner();
        if req.name != self.registry.name {
            return Err(Status::not_found(format!("no model '{}'", req.name)));
        }
        Ok(Response::new(ModelMetadataResponse {
            name: self.registry.name.clone(),
            versions: vec![self.registry.version.clone()],
            platform: "catboost".to_string(),
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
            raw_input_contents: raw,
            ..
        } = request.into_inner();

        if model_name != self.registry.name {
            return Err(Status::not_found(format!("no model '{model_name}'")));
        }

        let model = self
            .registry
            .current_model()
            .map_err(|e| Status::unavailable(e.to_string()))?;

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

        let inputs = inference::parse_inputs(tensors?)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let out = inference::run_blocking(model, inputs)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let resp_id = if id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            id
        };

        Ok(Response::new(ModelInferResponse {
            model_name: self.registry.name.clone(),
            model_version: self.registry.version.clone(),
            id: resp_id,
            parameters: Default::default(),
            outputs: vec![InferOutputTensor {
                name: "output-0".to_string(),
                datatype: "FP64".to_string(),
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
    registry: Arc<ModelRegistry>,
) -> anyhow::Result<()> {
    let svc = GrpcInferenceServiceServer::new(GrpcService::new(registry));
    tonic::transport::Server::builder()
        .add_service(svc)
        .serve(addr)
        .await?;
    Ok(())
}
