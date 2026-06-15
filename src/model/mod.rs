mod registry;
pub use registry::ModelRegistry;

use anyhow::Result;
use catboost_rust::Model;
use std::path::Path;

pub struct CatBoostModel {
    inner: Model,
}

impl CatBoostModel {
    pub fn load_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        #[cfg(catboost_zero_copy)]
        let inner = Model::load_buffer_zero_copy(bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        #[cfg(not(catboost_zero_copy))]
        let inner = Model::load_buffer(&bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok(Self { inner })
    }

    pub fn predict(
        &self,
        float_features: Vec<Vec<f32>>,
        cat_features: Vec<Vec<String>>,
    ) -> Result<Vec<f64>> {
        self.inner
            .calc_model_prediction(float_features, cat_features)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}
