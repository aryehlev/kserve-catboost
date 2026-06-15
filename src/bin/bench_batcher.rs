/// Benchmark: dynamic batcher overhead and throughput.
///
/// Two measurement modes:
///
///   OVERHEAD  – predict is instant (0 µs).  Shows the raw coordination cost
///               of the batcher (channel send, batch-loop recv, oneshot fan-back)
///               vs direct spawn_blocking with no real work.
///
///   THROUGHPUT – predict costs 10 ms fixed + 0.1 ms/row, simulating CatBoost's
///                C-FFI boundary.  Shows how much batching reduces wall time.
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::{
    sync::{mpsc, oneshot},
    task,
};

// ── fake predict ──────────────────────────────────────────────────────────────

fn fake_predict(n_rows: usize, fixed_us: u64) -> Vec<f64> {
    if fixed_us > 0 {
        std::thread::sleep(Duration::from_micros(fixed_us + 100 * n_rows as u64));
    }
    vec![0.5; n_rows]
}

// ── direct: one spawn_blocking per request ────────────────────────────────────

async fn bench_direct(n_requests: usize, fixed_us: u64) -> (Duration, Vec<Duration>) {
    let start = Instant::now();
    let handles: Vec<_> = (0..n_requests)
        .map(|_| {
            tokio::spawn(async move {
                let t0 = Instant::now();
                task::spawn_blocking(move || fake_predict(1, fixed_us))
                    .await
                    .unwrap();
                t0.elapsed()
            })
        })
        .collect();

    let mut latencies = Vec::with_capacity(n_requests);
    for h in handles {
        latencies.push(h.await.unwrap());
    }
    (start.elapsed(), latencies)
}

// ── batched: inline batcher (same channel/loop pattern as DynamicBatcher) ─────

struct BenchItem {
    row_count: usize,
    tx: oneshot::Sender<Vec<f64>>,
}

async fn bench_batched(
    n_requests: usize,
    max_batch: usize,
    max_wait: Duration,
    fixed_us: u64,
) -> (Duration, Vec<Duration>) {
    let (tx, mut rx) = mpsc::channel::<BenchItem>(n_requests + 64);

    let loop_handle = tokio::spawn(async move {
        loop {
            // Phase 1: blocking wait
            let first = match rx.recv().await {
                Some(item) => item,
                None => return,
            };
            let mut items = vec![first];
            let mut total = items[0].row_count;

            // Phase 1b: synchronous drain
            while total < max_batch {
                match rx.try_recv() {
                    Ok(item) => {
                        total += item.row_count;
                        items.push(item);
                    }
                    Err(_) => break,
                }
            }

            // Phase 2: deadline loop
            if !max_wait.is_zero() && total < max_batch {
                let deadline = tokio::time::Instant::now() + max_wait;
                'collect: loop {
                    if total >= max_batch {
                        break;
                    }
                    let mut extra: Vec<BenchItem> = Vec::new();
                    tokio::select! {
                        biased;
                        _ = tokio::time::sleep_until(deadline) => break 'collect,
                        got = rx.recv_many(&mut extra, 1) => {
                            if got == 0 { break 'collect; }
                            for item in extra {
                                total += item.row_count;
                                items.push(item);
                            }
                            while total < max_batch {
                                match rx.try_recv() {
                                    Ok(item) => { total += item.row_count; items.push(item); }
                                    Err(_) => break,
                                }
                            }
                        }
                    }
                }
            }

            let row_counts: Vec<usize> = items.iter().map(|i| i.row_count).collect();
            let senders: Vec<_> = items.into_iter().map(|i| i.tx).collect();

            let predictions = task::spawn_blocking(move || fake_predict(total, fixed_us))
                .await
                .unwrap();

            let mut offset = 0;
            for (count, sender) in row_counts.into_iter().zip(senders) {
                let _ = sender.send(predictions[offset..offset + count].to_vec());
                offset += count;
            }
        }
    });

    let shared_tx = Arc::new(tx);
    let start = Instant::now();
    let handles: Vec<_> = (0..n_requests)
        .map(|_| {
            let tx = shared_tx.clone();
            tokio::spawn(async move {
                let t0 = Instant::now();
                let (result_tx, result_rx) = oneshot::channel();
                tx.send(BenchItem { row_count: 1, tx: result_tx }).await.unwrap();
                result_rx.await.unwrap();
                t0.elapsed()
            })
        })
        .collect();
    drop(shared_tx);

    let mut latencies = Vec::with_capacity(n_requests);
    for h in handles {
        latencies.push(h.await.unwrap());
    }
    let elapsed = start.elapsed();

    loop_handle.abort();
    (elapsed, latencies)
}

// ── stats ─────────────────────────────────────────────────────────────────────

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    let idx = ((sorted.len() * pct).saturating_sub(1)) / 100;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_latency(label: &str, n: usize, total: Duration, mut lats: Vec<Duration>) {
    lats.sort_unstable();
    let rps = n as f64 / total.as_secs_f64();
    let p50 = percentile(&lats, 50);
    let p99 = percentile(&lats, 99);
    println!(
        "  {label:<42} {:>7.1} ms   {:>8.0} req/s   p50={:.2}ms  p99={:.2}ms",
        total.as_secs_f64() * 1000.0,
        rps,
        p50.as_secs_f64() * 1000.0,
        p99.as_secs_f64() * 1000.0,
    );
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .max_blocking_threads(8)
        .build()
        .unwrap();

    // ── Section 1: Overhead (0 µs predict) ───────────────────────────────────
    const N_OVERHEAD: usize = 1_000;
    println!(
        "\n=== Batcher coordination overhead ({N_OVERHEAD} concurrent requests, predict = instant) ===\n"
    );
    println!(
        "  {:<42} {:>10}   {:>12}   {}",
        "config", "elapsed", "throughput", "per-request latency"
    );
    println!("  {}", "-".repeat(85));

    let (t, l) = rt.block_on(bench_direct(N_OVERHEAD, 0));
    print_latency("direct (spawn_blocking, instant)", N_OVERHEAD, t, l);

    // passthrough: max_batch=1, wait=0 → each request dispatched immediately
    let (t, l) = rt.block_on(bench_batched(N_OVERHEAD, 1, Duration::ZERO, 0));
    print_latency("batcher passthrough (max=1, wait=0ms)", N_OVERHEAD, t, l);

    // small batch window
    let (t, l) = rt.block_on(bench_batched(N_OVERHEAD, 64, Duration::from_millis(1), 0));
    print_latency("batcher (max=64, wait=1ms)", N_OVERHEAD, t, l);

    // large batch window — absorbs all requests into a few batches
    let (t, l) = rt.block_on(bench_batched(N_OVERHEAD, N_OVERHEAD, Duration::from_millis(5), 0));
    print_latency(&format!("batcher (max={N_OVERHEAD}, wait=5ms)"), N_OVERHEAD, t, l);

    // ── Section 2: Throughput (10 ms fixed predict overhead) ─────────────────
    const N_THROUGHPUT: usize = 100;
    println!(
        "\n=== Throughput ({N_THROUGHPUT} concurrent requests, predict = 10 ms fixed + 0.1 ms/row) ===\n"
    );
    println!(
        "  {:<42} {:>10}   {:>12}   {}",
        "config", "elapsed", "throughput", "per-request latency"
    );
    println!("  {}", "-".repeat(85));

    let (t, l) = rt.block_on(bench_direct(N_THROUGHPUT, 10_000));
    print_latency("direct (no batcher)", N_THROUGHPUT, t, l);

    for &(max_batch, wait_ms) in &[(16usize, 5u64), (32, 10), (64, 20), (N_THROUGHPUT, 50)] {
        let label = format!("batched (max={max_batch}, wait={wait_ms}ms)");
        let (t, l) = rt.block_on(bench_batched(
            N_THROUGHPUT,
            max_batch,
            Duration::from_millis(wait_ms),
            10_000,
        ));
        print_latency(&label, N_THROUGHPUT, t, l);
    }

    println!();
}
