use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

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
/// `num_workers` independent batch-loop tasks run in parallel, each with its
/// own channel. Incoming requests are distributed round-robin so multiple
/// CatBoost calls can overlap. Default is the number of logical CPUs.
///
/// With `max_batch_size = 1` (the default) every request is dispatched
/// immediately — the batcher acts as a passthrough.
pub struct DynamicBatcher {
    pub registry: Arc<ModelRegistry>,
    /// One sender per worker; round-robin dispatch across them.
    senders: Vec<mpsc::Sender<BatchItem>>,
    /// Monotonically increasing counter used to select the next sender.
    next: AtomicUsize,
}

impl DynamicBatcher {
    pub fn start(
        registry: Arc<ModelRegistry>,
        max_batch_size: usize,
        max_wait: Duration,
        preferred_batch_sizes: Vec<usize>,
        num_workers: usize,
    ) -> Self {
        let num_workers = num_workers.max(1);
        // Per-worker channel capacity: 4× batch ceiling so senders are rarely
        // blocked, but backpressure still engages under sustained overload.
        let capacity = (max_batch_size * 4).max(64);

        let mut senders = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let (tx, rx) = mpsc::channel(capacity);
            senders.push(tx);
            tokio::spawn(batch_loop(
                registry.clone(),
                rx,
                max_batch_size,
                max_wait,
                preferred_batch_sizes.clone(),
            ));
        }

        Self {
            registry,
            senders,
            next: AtomicUsize::new(0),
        }
    }

    pub async fn infer(&self, inputs: InferInputs) -> Result<InferOutput> {
        let row_count = inputs.float_features.len().max(inputs.cat_features.len());
        let (result_tx, result_rx) = oneshot::channel();

        let mut item = BatchItem {
            float_features: inputs.float_features,
            cat_features: inputs.cat_features,
            row_count,
            tx: result_tx,
        };

        // Round-robin starting point; Relaxed is fine — we only need rough
        // distribution, not strict ordering.
        let n = self.senders.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed) % n;

        // Try each worker in order; use the first one with queue space.
        // Only return OverloadError if every worker's queue is full.
        for i in 0..n {
            let idx = (start + i) % n;
            match self.senders[idx].try_send(item) {
                Ok(()) => {
                    let predictions = result_rx
                        .await
                        .map_err(|_| anyhow::anyhow!("batcher dropped response"))??;
                    return Ok(InferOutput {
                        shape: vec![row_count as i64, 1],
                        predictions,
                    });
                }
                Err(mpsc::error::TrySendError::Full(returned)) => {
                    item = returned; // try next worker
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Err(anyhow::anyhow!("batcher shut down"));
                }
            }
        }

        Err(anyhow::Error::new(OverloadError))
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
        // overhead. Closes the gap between Phase 1 and Phase 2 cheaply.
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
                        // Synchronous drain after each async receive.
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
