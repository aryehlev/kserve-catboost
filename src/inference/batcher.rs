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

/// Fan-out payload: shared prediction buffer + slice bounds for one request.
/// Using Arc avoids a per-request copy inside the blocking dispatch task;
/// each caller does its own slice-to-Vec in its async context instead.
type FanOut = Result<(Arc<[f64]>, usize, usize)>;

struct BatchItem {
    float_features: Vec<Vec<f32>>,
    cat_features: Vec<Vec<String>>,
    /// Number of rows this item contributes (pre-computed; equal to the
    /// number of feature-vectors in float_features or cat_features).
    row_count: usize,
    tx: oneshot::Sender<FanOut>,
}

/// Collects concurrent inference requests into larger batches, running one
/// CatBoost call per batch instead of one per request.
///
/// Batch dispatch is triggered by the first of:
///   1. `total_rows >= max_batch_size`  (soft row ceiling — see note below)
///   2. `total_rows >= any preferred_batch_size`  (Triton-style early dispatch)
///   3. `max_wait` has elapsed since the first request in the batch arrived
///
/// Note on the row ceiling: because we cannot inspect a request's row count
/// before receiving it, `max_batch_size` is a *soft* ceiling. A single
/// multi-row request received just as the ceiling is approached may push
/// `total_rows` slightly over it. In the typical KServe case every request
/// carries exactly one row, so the ceiling is exact.
///
/// `num_workers` independent batch-loop tasks run in parallel, each with its
/// own channel. Incoming requests are distributed round-robin so multiple
/// CatBoost calls can overlap. Default is the number of logical CPUs.
///
/// With `max_batch_size = 1` (the default) every request bypasses the channel
/// entirely and calls the model directly — zero batcher overhead.
pub struct DynamicBatcher {
    pub registry: Arc<ModelRegistry>,
    /// One sender per worker; round-robin dispatch across them.
    senders: Vec<mpsc::Sender<BatchItem>>,
    /// Monotonically increasing counter used to select the next sender.
    next: AtomicUsize,
    /// Cached from construction; enables the fast-path bypass in `infer`.
    max_batch_size: usize,
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
        // saturating_mul avoids overflow for very large max_batch_size values.
        let capacity = max_batch_size.saturating_mul(4).max(64);

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
            max_batch_size,
        }
    }

    pub async fn infer(&self, inputs: InferInputs) -> Result<InferOutput> {
        // Fast path: max_batch_size=1 means no batching is possible, so skip
        // the channel entirely. Callers get the same latency as a raw
        // spawn_blocking with no batcher overhead at all.
        if self.max_batch_size == 1 {
            let model = self.registry.current_model()?;
            return run_blocking(model, inputs).await;
        }

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
        // A closed worker is treated as a failed candidate so we keep
        // probing rather than aborting on the first dead channel.
        let mut saw_full = false;
        let mut saw_closed = false;
        for i in 0..n {
            let idx = (start + i) % n;
            match self.senders[idx].try_send(item) {
                Ok(()) => {
                    let (arc, start, end) = result_rx
                        .await
                        .map_err(|_| anyhow::anyhow!("batcher dropped response"))??;
                    return Ok(InferOutput {
                        shape: vec![row_count as i64, 1],
                        predictions: arc[start..end].to_vec(),
                    });
                }
                Err(mpsc::error::TrySendError::Full(returned)) => {
                    saw_full = true;
                    item = returned; // try next worker
                }
                Err(mpsc::error::TrySendError::Closed(returned)) => {
                    saw_closed = true;
                    item = returned; // try next worker
                }
            }
        }

        // Report the most actionable error: full queues before dead workers.
        if saw_full {
            Err(anyhow::Error::new(OverloadError))
        } else if saw_closed {
            Err(anyhow::anyhow!("batcher shut down"))
        } else {
            Err(anyhow::anyhow!("batcher has no workers"))
        }
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
    loop {
        let mut batch: Vec<BatchItem> = Vec::with_capacity(max_batch_size.max(1));

        // ── Phase 1: blocking wait ────────────────────────────────────────────
        // Block until at least one request arrives. Using a single recv()
        // here (rather than recv_many) keeps the row ceiling accurate: the
        // try_recv drain below then pulls additional items synchronously
        // while checking total_rows against max_batch_size after each one,
        // so we never overshoot by more than one item's row count.
        let first = match rx.recv().await {
            Some(item) => item,
            None => return, // all senders dropped; shut down
        };
        let mut total_rows = first.row_count;
        batch.push(first);

        // ── Phase 1b: synchronous drain ───────────────────────────────────────
        // Pull any already-queued items without async overhead.
        // Stops at the row ceiling so overshoot is bounded to ≤1 item.
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
        //
        // Deadline branch is listed first under `biased;` so it fires
        // immediately once the instant passes — continuous queue traffic
        // cannot starve it and violate the MAX_BATCH_WAIT_MS bound.
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

                // Receive one item at a time so we re-check the row ceiling
                // after each arrival and minimise overshoot.
                let mut extra: Vec<BatchItem> = Vec::new();
                tokio::select! {
                    biased;
                    _ = tokio::time::sleep_until(deadline) => break 'collect,
                    got = rx.recv_many(&mut extra, 1) => {
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
                }
            }
        }

        // Spawn dispatch independently so this loop immediately goes back to
        // collecting the next batch, overlapping collection with inference.
        tokio::spawn(dispatch(registry.clone(), batch, total_rows));
    }
}

async fn dispatch(registry: Arc<ModelRegistry>, batch: Vec<BatchItem>, total_rows: usize) {
    let mut all_floats: Vec<Vec<f32>> = Vec::with_capacity(total_rows);
    let mut all_cats: Vec<Vec<String>> = Vec::with_capacity(total_rows);
    let mut row_counts: Vec<usize> = Vec::with_capacity(batch.len());
    let mut senders: Vec<oneshot::Sender<FanOut>> = Vec::with_capacity(batch.len());

    let has_floats = batch.iter().any(|i| !i.float_features.is_empty());
    let has_cats = batch.iter().any(|i| !i.cat_features.is_empty());

    for mut item in batch {
        row_counts.push(item.row_count);
        senders.push(item.tx);
        // Pad missing float rows symmetrically with empty vecs when other
        // items in the batch carry float features.
        if has_floats {
            if item.float_features.is_empty() {
                all_floats.extend(std::iter::repeat_with(Vec::new).take(item.row_count));
            } else {
                all_floats.append(&mut item.float_features);
            }
        }
        // Pad missing cat rows symmetrically with empty vecs when other
        // items in the batch carry categorical features.
        if has_cats {
            if item.cat_features.is_empty() {
                all_cats.extend(std::iter::repeat_with(Vec::new).take(item.row_count));
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
            let expected: usize = row_counts.iter().sum();
            if out.predictions.len() != expected {
                let msg = format!(
                    "prediction length mismatch: expected {expected}, got {}",
                    out.predictions.len()
                );
                for tx in senders {
                    let _ = tx.send(Err(anyhow::anyhow!("{msg}")));
                }
                return;
            }
            // Wrap predictions in Arc once. Each caller receives a pointer +
            // bounds and does its own slice copy in its own async task,
            // parallelising the fan-out instead of doing N copies here.
            let arc: Arc<[f64]> = Arc::from(out.predictions);
            let mut offset = 0;
            for (tx, count) in senders.into_iter().zip(row_counts) {
                let end = offset + count;
                let _ = tx.send(Ok((arc.clone(), offset, end)));
                offset = end;
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
