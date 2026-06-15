use anyhow::{Context, Result};
use catboost_rust::Model;
use std::path::Path;

// ── Model wrapper ─────────────────────────────────────────────────────────────

pub struct CatBoostModel {
    inner: Model,
    pub name: String,
    pub float_features_count: usize,
    pub cat_features_count: usize,
    pub dimensions_count: usize,
    pub tree_count: usize,
}

// SAFETY: CatBoost's C API is documented as thread-safe for concurrent
// read-only operations (prediction). The raw pointer held by `Model` is
// stable for the lifetime of the struct and is never written to after
// construction.
unsafe impl Send for CatBoostModel {}
unsafe impl Sync for CatBoostModel {}

impl CatBoostModel {
    /// Load a model from a file path.  Reads the file into memory first so
    /// that `load_bytes` can hand the buffer to the C library.
    pub fn load_file(name: String, path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read model file: {}", path.display()))?;
        Self::load_bytes(name, bytes)
            .with_context(|| format!("failed to parse model from {}", path.display()))
    }

    /// Load a model from an in-memory buffer.
    ///
    /// When compiled against CatBoost ≥ 1.2.9 the buffer is handed to the C
    /// library directly without an additional copy (zero-copy path).
    /// Older versions fall back to `load_buffer` which copies internally.
    pub fn load_bytes(name: String, bytes: Vec<u8>) -> Result<Self> {
        // `catboost_zero_copy` is a cfg flag set by catboost-rust's build.rs
        // when the downloaded library version supports zero-copy loading.
        #[cfg(catboost_zero_copy)]
        let inner = Model::load_buffer_zero_copy(bytes)
            .context("model load (zero-copy) failed")?;

        #[cfg(not(catboost_zero_copy))]
        let inner = Model::load_buffer(&bytes).context("model load failed")?;

        let float_features_count = inner.get_float_features_count();
        let cat_features_count = inner.get_cat_features_count();
        let dimensions_count = inner.get_dimensions_count();
        let tree_count = inner.get_tree_count();

        tracing::info!(
            model           = %name,
            float_features  = float_features_count,
            cat_features    = cat_features_count,
            dimensions      = dimensions_count,
            trees           = tree_count,
            zero_copy       = cfg!(catboost_zero_copy),
            "model loaded",
        );

        Ok(Self {
            inner,
            name,
            float_features_count,
            cat_features_count,
            dimensions_count,
            tree_count,
        })
    }

    /// Run inference on a batch expressed as a **slice of row-slices**.
    ///
    /// Callers should build `float_rows` from a flat `Vec<f32>` using
    /// `.chunks(num_features)` to avoid allocating an inner `Vec` per row.
    /// Example:
    /// ```ignore
    /// let flat: Vec<f32> = parse_float_data(&input)?;
    /// let rows: Vec<&[f32]> = flat.chunks(model.float_features_count).collect();
    /// let predictions = model.predict(&rows, cat_features)?;
    /// ```
    pub fn predict<'a, S>(
        &self,
        float_rows: &[S],
        cat_features: Vec<Vec<String>>,
    ) -> Result<Vec<f64>>
    where
        S: AsRef<[f32]>,
    {
        self.inner
            .calc_model_prediction(float_rows, cat_features)
            .context("CatBoost prediction failed")
    }
}
