use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use anyhow::Result;
use arc_swap::ArcSwap;

use super::CatBoostModel;

pub struct ModelRegistry {
    pub name: String,
    pub version: String,
    model: ArcSwap<CatBoostModel>,
    loaded: AtomicBool,
    path: PathBuf,
}

impl ModelRegistry {
    pub fn new(model: CatBoostModel, path: PathBuf, name: String, version: String) -> Self {
        Self {
            name,
            version,
            model: ArcSwap::new(Arc::new(model)),
            loaded: AtomicBool::new(true),
            path,
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Acquire)
    }

    pub fn current_model(&self) -> Result<Arc<CatBoostModel>> {
        if !self.is_loaded() {
            anyhow::bail!("model '{}' is not loaded", self.name);
        }
        Ok(self.model.load_full())
    }

    pub fn repository_load(&self) -> Result<()> {
        let model = CatBoostModel::load_file(&self.path)?;
        self.model.store(Arc::new(model));
        self.loaded.store(true, Ordering::Release);
        tracing::info!(model = %self.name, "model loaded");
        Ok(())
    }

    pub fn repository_unload(&self) {
        self.loaded.store(false, Ordering::Release);
        tracing::info!(model = %self.name, "model unloaded");
    }
}
