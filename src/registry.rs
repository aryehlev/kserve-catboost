/// Lock-free model registry with KServe repository-extension hot-reload support.
///
/// An `ArcSwap` holds the live model pointer so inference readers never block —
/// they take a snapshot with a single atomic load.  A `load` call atomically
/// replaces the pointer; in-flight inferences finish against the old model while
/// new requests immediately pick up the fresh one.
///
/// The registry also tracks `loaded` state so that `unload` can make the model
/// appear unavailable for new requests without actually freeing anything
/// (important for graceful degradation without a server restart).
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::Result;
use arc_swap::ArcSwap;

use crate::model::CatBoostModel;

pub struct ModelRegistry {
    current: ArcSwap<CatBoostModel>,
    loaded: AtomicBool,
    /// Model name — fixed across reloads.
    pub name: String,
    path: PathBuf,
}

impl ModelRegistry {
    pub fn new(model: CatBoostModel, path: PathBuf) -> Self {
        let name = model.name.clone();
        Self {
            current: ArcSwap::new(Arc::new(model)),
            loaded: AtomicBool::new(true),
            name,
            path,
        }
    }

    /// Return an independent `Arc` snapshot of the current model.
    ///
    /// The returned `Arc` keeps the model alive independently of any concurrent
    /// swap, so it is safe to hold across `await` points and move into
    /// `spawn_blocking` closures.
    pub fn load_model(&self) -> Arc<CatBoostModel> {
        self.current.load_full()
    }

    /// `true` if the model has been loaded and not subsequently unloaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Acquire)
    }

    /// Reload the model from disk (KServe repository `load` semantics).
    ///
    /// Reads the model file, constructs a new `CatBoostModel`, then atomically
    /// swaps it in.  After this returns the server serves the fresh weights.
    pub fn repository_load(&self) -> Result<()> {
        let model = CatBoostModel::load_file(self.name.clone(), &self.path)?;
        self.current.store(Arc::new(model));
        self.loaded.store(true, Ordering::Release);
        tracing::info!(model = %self.name, path = ?self.path, "model hot-reloaded");
        Ok(())
    }

    /// Mark the model as unloaded (KServe repository `unload` semantics).
    ///
    /// Does **not** free the underlying memory — in-flight requests will still
    /// complete.  New inference requests will receive 503 until `load` is
    /// called again.
    pub fn repository_unload(&self) {
        self.loaded.store(false, Ordering::Release);
        tracing::info!(model = %self.name, "model marked unloaded");
    }
}
