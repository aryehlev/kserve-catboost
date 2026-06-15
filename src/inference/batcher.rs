use std::{sync::Arc, time::Duration};

use anyhow::Result;
use tokio::sync::{mpsc, oneshot};

use crate::model::ModelRegistry;

use super::{run_blocking, InferInputs, InferOutput};

/// Returned by [`DynamicBatcher::infer`] when the queue is full and the
/// request must be shed. Maps to HTTP 429 / gRPC RESOURCE_EXHAUSTED.
#[derive(Debug)]
pub struct OverloadError;

impl std::fmt::Display for OverloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "server overloaded, try again later")
    }
}

impl std::error::Error for OverloadError {}

struct BatchItem {
    float_features: Vec<Vec<f32>>,
    cat_features: Vec<Vec<String>>,
    row_count: usize,
    tx: oneshot::Sender<Result<Vec<f64>>>,
}

/// Collects concurrent inference requests into larger batches, running one
/// CatBoost call per batch instead of one per request.
///
/// Batch dispatch is triggered by the first of:
///   1. `total_rows >= max_batch_size`  (hard ceiling)
///   2. `total_rows >= any preferred_batch_size`  (Triton-style early dispatch)
///   3. `max_wait` has elapsed since the first request in the batch arrived
///
/// With `max_batch_size = 1` (the default) every request is dispatched
/// immediately — the batcher acts as a passthrough.
pub struct DynamicBatcher {
    pub registry: Arc<ModelRegistry>,
    tx: mpsc::Sender<BatchItem>,
}

impl DynamicBatcher {
    pub fn start(
        registry: Arc<ModelRegistry>,
        max_batch_size: usize,
        max_wait: Duration,
        preferred_batch_sizes: Vec<usize>,
    ) -> Self {
        // Buffer = 4× the batch ceiling so senders are rarely blocked under
        // normal load, but backpressure still engages under sustained overload.
        let capacity = (max_batch_size * 4).max(64);
        let (tx, rx) = mpsc::channel(capacity);
        tokio::spawn(batch_loop(
            registry.clone(),
            rx,
            max_batch_size,
            max_wait,
            preferred_batch_sizes,
        ));
        Self { registry, tx }
    }

    pub async fn infer(&self, inputs: InferInputs) -> Result<InferOutput> {
        let row_count = inputs.float_features.len().max(inputs.cat_features.len());
        let (result_tx, result_rx) = oneshot::channel();

        // Non-blocking send: shed load immediately rather than letting callers
        // queue up inside the server. Returns OverloadError when full.
        self.tx
            .try_send(BatchItem {
                float_features: inputs.float_features,
                cat_features: inputs.cat_features,
                row_count,
                tx: result_tx,
            })
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => anyhow::Error::new(OverloadError),
                mpsc::error::TrySendError::Closed(_) => anyhow::anyhow!("batcher shut down"),
            })?;

        let predictions = result_rx
            .await
            .map_err(|_| anyhow::anyhow!("batcher dropped response"))??;

        Ok(InferOutput {
            shape: vec![row_count as i64, 1],
            predictions,
        })
    }
}

/// True when `total_rows` has reached or passed any preferred dispatch size.
#[inline]
fn hits_preferred(preferred: &[usize], total_rows: usize) -> bool {
    preferred.iter().any(|&p| total_rows >= p)
}

async fn batch_loop(
    registry: Arc<ModelRegistry>,
    mut rx: mpsc::Receiver<BatchItem>,
    max_batch_size: usize,
    max_wait: Duration,
    preferred_batch_sizes: Vec<usize>,
) {
    let item_limit = max_batch_size.max(1);

    loop {
        let mut batch: Vec<BatchItem> = Vec::with_capacity(item_limit);

        // ── Phase 1: blocking wait ────────────────────────────────────────────
        // recv_many suspends until ≥1 request arrives, then atomically drains
        // as many as `item_limit` in one go — a single async yield may fill
        // the whole batch when many callers are already queued.
        let n = rx.recv_many(&mut batch, item_limit).await;
        if n == 0 {
            return; // all senders dropped; shut down
        }
        let mut total_rows: usize = batch.iter().map(|i| i.row_count).sum();

        // ── Phase 1b: synchronous drain ───────────────────────────────────────
        // Grab any items that are already in the channel without paying async
        // overhead. This closes the gap between Phase 1 and Phase 2 cheaply.
        while total_rows < max_batch_size {
            match rx.try_recv() {
                Ok(item) => {
                    total_rows += item.row_count;
                    batch.push(item);
                }
                Err(_) => break,
            }
        }

        // ── Phase 2: deadline loop ────────────────────────────────────────────
        // Collect stragglers until the batch hits a dispatch threshold or the
        // deadline fires. Using sleep_until (absolute) avoids drift when the
        // loop body takes non-zero time.
        let needs_wait = !max_wait.is_zero()
            && total_rows < max_batch_size
            && !hits_preferred(&preferred_batch_sizes, total_rows);

        if needs_wait {
            let deadline = tokio::time::Instant::now() + max_wait;

            'collect: loop {
                if total_rows >= max_batch_size
                    || hits_preferred(&preferred_batch_sizes, total_rows)
                {
                    break 'collect;
                }

                let remaining = (max_batch_size.saturating_sub(batch.len())).max(1);
                let mut extra: Vec<BatchItem> = Vec::new();

                tokio::select! {
                    // biased: prefer draining the channel over the timer so a
                    // burst of requests fills the batch before we give up.
                    biased;
                    got = rx.recv_many(&mut extra, remaining) => {
                        if got == 0 { break 'collect; } // channel closed
                        for item in extra {
                            total_rows += item.row_count;
                            batch.push(item);
                        }
                        // Drain any synchronously available items after the
                        // async receive — same pattern as Phase 1b.
                        while total_rows < max_batch_size {
                            match rx.try_recv() {
                                Ok(item) => { total_rows += item.row_count; batch.push(item); }
                                Err(_) => break,
                            }
                        }
                    }
                    _ = tokio::time::sleep_until(deadline) => break 'collect,
                }
            }
        }

        dispatch(&registry, batch, total_rows).await;
    }
}

async fn dispatch(registry: &Arc<ModelRegistry>, batch: Vec<BatchItem>, total_rows: usize) {
    let mut all_floats: Vec<Vec<f32>> = Vec::with_capacity(total_rows);
    let mut all_cats: Vec<Vec<String>> = Vec::with_capacity(total_rows);
    let mut row_counts: Vec<usize> = Vec::with_capacity(batch.len());
    let mut senders: Vec<oneshot::Sender<Result<Vec<f64>>>> = Vec::with_capacity(batch.len());

    let has_cats = batch.iter().any(|i| !i.cat_features.is_empty());

    for mut item in batch {
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
            return;
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
