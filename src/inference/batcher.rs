use std::{sync::Arc, time::Duration};

use anyhow::Result;
use tokio::sync::{mpsc, oneshot};

use crate::model::ModelRegistry;

use super::{run_blocking, InferInputs, InferOutput};

struct BatchItem {
    float_features: Vec<Vec<f32>>,
    cat_features: Vec<Vec<String>>,
    /// Number of rows this item contributes (pre-computed so we don't re-derive it later).
    row_count: usize,
    tx: oneshot::Sender<Result<Vec<f64>>>,
}

/// Collects concurrent inference requests into larger batches, running one
/// CatBoost call per batch instead of one per request.
///
/// With `max_batch_size = 1` (the default) the batcher is effectively a
/// passthrough — the first arriving request is dispatched immediately.
pub struct DynamicBatcher {
    pub registry: Arc<ModelRegistry>,
    tx: mpsc::Sender<BatchItem>,
}

impl DynamicBatcher {
    pub fn start(
        registry: Arc<ModelRegistry>,
        max_batch_size: usize,
        max_wait: Duration,
    ) -> Self {
        let (tx, rx) = mpsc::channel(1024);
        tokio::spawn(batch_loop(registry.clone(), rx, max_batch_size, max_wait));
        Self { registry, tx }
    }

    pub async fn infer(&self, inputs: InferInputs) -> Result<InferOutput> {
        let row_count = inputs.float_features.len().max(inputs.cat_features.len());
        let (result_tx, result_rx) = oneshot::channel();

        self.tx
            .send(BatchItem {
                float_features: inputs.float_features,
                cat_features: inputs.cat_features,
                row_count,
                tx: result_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("batcher shut down"))?;

        let predictions = result_rx
            .await
            .map_err(|_| anyhow::anyhow!("batcher dropped response"))??;

        Ok(InferOutput {
            shape: vec![row_count as i64, 1],
            predictions,
        })
    }
}

async fn batch_loop(
    registry: Arc<ModelRegistry>,
    mut rx: mpsc::Receiver<BatchItem>,
    max_batch_size: usize,
    max_wait: Duration,
) {
    loop {
        // Block until at least one request arrives.
        let first = match rx.recv().await {
            Some(item) => item,
            None => return, // all senders dropped; shut down
        };

        let mut items = vec![first];
        let mut total_rows = items[0].row_count;
        let deadline = tokio::time::Instant::now() + max_wait;

        // Fill the batch until it reaches max_batch_size or the window closes.
        while total_rows < max_batch_size {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(item)) => {
                    total_rows += item.row_count;
                    items.push(item);
                }
                _ => break,
            }
        }

        // Merge all items into a single feature matrix.
        let mut all_floats: Vec<Vec<f32>> = Vec::with_capacity(total_rows);
        let mut all_cats: Vec<Vec<String>> = Vec::with_capacity(total_rows);
        let mut row_counts: Vec<usize> = Vec::with_capacity(items.len());
        let mut senders: Vec<oneshot::Sender<Result<Vec<f64>>>> =
            Vec::with_capacity(items.len());

        // Detect whether any request carries categorical features.
        // If so, rows without cats are padded with empty vecs so dimensions align.
        let has_cats = items.iter().any(|i| !i.cat_features.is_empty());

        for mut item in items {
            row_counts.push(item.row_count);
            senders.push(item.tx);
            all_floats.append(&mut item.float_features);
            if has_cats {
                if item.cat_features.is_empty() {
                    all_cats.extend(std::iter::repeat_n(vec![], item.row_count));
                } else {
                    all_cats.append(&mut item.cat_features);
                }
            }
        }

        let model = match registry.current_model() {
            Ok(m) => m,
            Err(e) => {
                let msg = e.to_string();
                for tx in senders {
                    let _ = tx.send(Err(anyhow::anyhow!("{msg}")));
                }
                continue;
            }
        };

        let combined = InferInputs {
            float_features: all_floats,
            cat_features: all_cats,
        };

        match run_blocking(model, combined).await {
            Ok(out) => {
                let mut offset = 0;
                for (tx, count) in senders.into_iter().zip(row_counts) {
                    let _ = tx.send(Ok(out.predictions[offset..offset + count].to_vec()));
                    offset += count;
                }
            }
            Err(e) => {
                let msg = e.to_string();
                for tx in senders {
                    let _ = tx.send(Err(anyhow::anyhow!("{msg}")));
                }
            }
        }
    }
}
