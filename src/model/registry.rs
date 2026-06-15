use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};

use anyhow::Result;
use arc_swap::ArcSwap;

use super::CatBoostModel;

pub struct ModelRegistry {
    pub name: String,
    /// Monotonically increasing — starts at the configured initial version and
    /// increments by 1 on each successful hot-reload so callers can detect changes.
    version: AtomicU64,
    model: ArcSwap<CatBoostModel>,
    loaded: AtomicBool,
    path: PathBuf,
}

impl ModelRegistry {
    pub fn new(
        model: CatBoostModel,
        path: PathBuf,
        name: String,
        initial_version: u64,
    ) -> Self {
        Self {
            name,
            version: AtomicU64::new(initial_version),
            model: ArcSwap::new(Arc::new(model)),
            loaded: AtomicBool::new(true),
            path,
        }
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
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

    /// Reload the model from disk and bump the version counter.
    pub fn repository_load(&self) -> Result<()> {
        let model = CatBoostModel::load_file(&self.path)?;
        self.model.store(Arc::new(model));
        let new_ver = self.version.fetch_add(1, Ordering::AcqRel) + 1;
        self.loaded.store(true, Ordering::Release);
        tracing::info!(model = %self.name, version = new_ver, "model reloaded");
        Ok(())
    }

    pub fn repository_unload(&self) {
        self.loaded.store(false, Ordering::Release);
        tracing::info!(model = %self.name, "model unloaded");
    }
}
