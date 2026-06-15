/// Lock-free model registry with hot-reload support.
///
/// `ModelRegistry` wraps an [`arc_swap::ArcSwap`] so readers never block —
/// they take a snapshot of the current model with a single atomic operation.
/// A reload atomically replaces the pointer; in-flight inferences finish
/// against the old model while new requests immediately pick up the fresh one.
use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use arc_swap::ArcSwap;

use crate::model::CatBoostModel;

pub struct ModelRegistry {
    current: ArcSwap<CatBoostModel>,
    /// Name shared across reloads (the model name never changes).
    pub name: String,
    /// Path used when reloading.
    path: PathBuf,
}

// ArcSwap<T> is Send + Sync when T: Send + Sync, which CatBoostModel is.
// Arc<ModelRegistry> is therefore safe to share across threads.

impl ModelRegistry {
    pub fn new(model: CatBoostModel, path: PathBuf) -> Self {
        let name = model.name.clone();
        Self {
            current: ArcSwap::new(Arc::new(model)),
            name,
            path,
        }
    }

    /// Return a cheap snapshot of the current model.
    ///
    /// The returned `Arc` keeps the model alive independently of any swap,
    /// so it is safe to hold across `await` points and move into
    /// `spawn_blocking` closures.
    pub fn load(&self) -> Arc<CatBoostModel> {
        self.current.load_full()
    }

    /// Reload the model from disk, atomically replacing the live one.
    ///
    /// After this returns, new requests will use the fresh model.
    /// Requests that already obtained a snapshot via [`load`] will finish
    /// against the old model — both versions coexist safely.
    pub fn reload(&self) -> Result<()> {
        let model = CatBoostModel::load_file(self.name.clone(), &self.path)?;
        self.current.store(Arc::new(model));
        tracing::info!(model = %self.name, path = ?self.path, "model hot-reloaded");
        Ok(())
    }
}
