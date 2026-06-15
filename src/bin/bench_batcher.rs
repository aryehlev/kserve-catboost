/// Benchmark: direct vs batched inference throughput.
///
/// `fake_predict` simulates CatBoost's vectorized execution:
///   - 10 ms fixed overhead per call (model setup, C FFI boundary)
///   - 0.1 ms per row  (actual scoring — sub-linear in batch size)
///
/// With direct dispatch every request pays the 10 ms overhead alone.
/// With batching N concurrent requests share one 10 ms call.
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::{
    sync::{mpsc, oneshot},
    task,
    time::timeout_at,
};

// ── fake model ────────────────────────────────────────────────────────────────

fn fake_predict(n_rows: usize) -> Vec<f64> {
    std::thread::sleep(Duration::from_micros(10_000 + 100 * n_rows as u64));
    vec![0.5; n_rows]
}

// ── direct: one spawn_blocking per request ────────────────────────────────────

async fn bench_direct(n_requests: usize) -> Duration {
    let start = Instant::now();
    let handles: Vec<_> = (0..n_requests)
        .map(|_| {
            task::spawn_blocking(|| fake_predict(1))
        })
        .collect();
    for h in handles {
        h.await.unwrap();
    }
    start.elapsed()
}

// ── batched: inline batcher using the same channel/loop pattern ───────────────

struct BenchItem {
    row_count: usize,
    tx: oneshot::Sender<Vec<f64>>,
}

async fn bench_batched(n_requests: usize, max_batch: usize, max_wait: Duration) -> Duration {
    let (tx, mut rx) = mpsc::channel::<BenchItem>(n_requests + 64);

    // batch loop
    let loop_handle = tokio::spawn(async move {
        loop {
            let first = match rx.recv().await {
                Some(item) => item,
                None => return,
            };
            let mut items = vec![first];
            let mut total = items[0].row_count;
            let deadline = tokio::time::Instant::now() + max_wait;

            while total < max_batch {
                match timeout_at(deadline, rx.recv()).await {
                    Ok(Some(item)) => {
                        total += item.row_count;
                        items.push(item);
                    }
                    _ => break,
                }
            }

            let row_counts: Vec<usize> = items.iter().map(|i| i.row_count).collect();
            let senders: Vec<_> = items.into_iter().map(|i| i.tx).collect();

            let predictions = task::spawn_blocking(move || fake_predict(total))
                .await
                .unwrap();

            // fan out results
            let mut offset = 0;
            for (count, sender) in row_counts.into_iter().zip(senders) {
                let _ = sender.send(predictions[offset..offset + count].to_vec());
                offset += count;
            }
        }
    });

    let start = Instant::now();
    let request_handles: Vec<_> = (0..n_requests)
        .map(|_| {
            let tx = tx.clone();
            tokio::spawn(async move {
                let (result_tx, result_rx) = oneshot::channel();
                tx.send(BenchItem { row_count: 1, tx: result_tx }).await.unwrap();
                result_rx.await.unwrap()
            })
        })
        .collect();
    drop(tx); // close sender so loop exits when all requests are done

    for h in request_handles {
        h.await.unwrap();
    }
    let elapsed = start.elapsed();

    loop_handle.abort();
    elapsed
}

// ── runner ────────────────────────────────────────────────────────────────────

fn print_result(label: &str, n: usize, elapsed: Duration) {
    let rps = n as f64 / elapsed.as_secs_f64();
    println!("  {label:<40} {:>7.1} ms   {:>8.0} req/s", elapsed.as_millis(), rps);
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        // restrict blocking threads so contention shows up clearly
        .max_blocking_threads(8)
        .build()
        .unwrap();

    const N: usize = 100;

    println!("\n=== KServe-CatBoost batcher benchmark ({N} concurrent single-row requests) ===\n");
    println!("  fake_predict cost: 10 ms fixed + 0.1 ms/row\n");
    println!("  {:<40} {:>10}   {:>12}", "config", "elapsed", "throughput");
    println!("  {}", "-".repeat(60));

    // direct
    let d = rt.block_on(bench_direct(N));
    print_result("direct (no batcher)", N, d);

    // batched at various window sizes
    for &(max_batch, wait_ms) in &[(16usize, 5u64), (32, 10), (64, 20), (N, 50)] {
        let label = format!("batched (max={max_batch}, wait={wait_ms}ms)");
        let d = rt.block_on(bench_batched(N, max_batch, Duration::from_millis(wait_ms)));
        print_result(&label, N, d);
    }

    println!();
}
