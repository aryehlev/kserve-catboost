/// Benchmark: dynamic batcher overhead and throughput.
///
/// Two measurement modes:
///
///   OVERHEAD  – predict is instant (0 µs).  Shows the raw coordination cost
///               of the batcher vs direct spawn_blocking with no real work.
///
///   THROUGHPUT – predict costs 10 ms fixed + 0.1 ms/row, simulating CatBoost.
///
/// Two dispatch strategies are compared:
///
///   BLOCKING dispatch: batch loop awaits spawn_blocking before accepting the
///     next request. Serialises collection and dispatch — bad for passthrough.
///
///   PIPELINED dispatch: batch loop fires spawn_blocking as a spawned task
///     and immediately loops back to collect the next batch. Collection and
///     dispatch overlap: the loop never stalls waiting for predict to finish.
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

// ── shared types ──────────────────────────────────────────────────────────────

struct BenchItem {
    row_count: usize,
    tx: oneshot::Sender<Vec<f64>>,
}

// Collect a batch from the channel (shared logic for both dispatch strategies).
async fn collect_batch(
    rx: &mut mpsc::Receiver<BenchItem>,
    max_batch: usize,
    max_wait: Duration,
) -> Option<(Vec<BenchItem>, usize)> {
    // Phase 1: blocking wait for first item
    let first = rx.recv().await?;
    let mut items = vec![first];
    let mut total = items[0].row_count;

    // Phase 1b: synchronous drain
    while total < max_batch {
        match rx.try_recv() {
            Ok(item) => { total += item.row_count; items.push(item); }
            Err(_) => break,
        }
    }

    // Phase 2: deadline loop
    if !max_wait.is_zero() && total < max_batch {
        let deadline = tokio::time::Instant::now() + max_wait;
        'collect: loop {
            if total >= max_batch { break; }
            let mut extra: Vec<BenchItem> = Vec::new();
            tokio::select! {
                biased;
                _ = tokio::time::sleep_until(deadline) => break 'collect,
                got = rx.recv_many(&mut extra, 1) => {
                    if got == 0 { break 'collect; }
                    for item in extra { total += item.row_count; items.push(item); }
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

    Some((items, total))
}

fn dispatch_batch(items: Vec<BenchItem>, total: usize, fixed_us: u64) {
    let row_counts: Vec<usize> = items.iter().map(|i| i.row_count).collect();
    let senders: Vec<_> = items.into_iter().map(|i| i.tx).collect();
    let predictions = fake_predict(total, fixed_us);
    let mut offset = 0;
    for (count, sender) in row_counts.into_iter().zip(senders) {
        let _ = sender.send(predictions[offset..offset + count].to_vec());
        offset += count;
    }
}

// ── blocking dispatch: loop awaits dispatch before next recv ──────────────────

async fn bench_batched_blocking(
    n_requests: usize,
    max_batch: usize,
    max_wait: Duration,
    fixed_us: u64,
) -> (Duration, Vec<Duration>) {
    let (tx, mut rx) = mpsc::channel::<BenchItem>(n_requests + 64);

    let loop_handle = tokio::spawn(async move {
        while let Some((items, total)) = collect_batch(&mut rx, max_batch, max_wait).await {
            // Awaiting here means the loop stalls until dispatch finishes.
            task::spawn_blocking(move || dispatch_batch(items, total, fixed_us))
                .await
                .unwrap();
        }
    });

    let elapsed = drive_requests(n_requests, tx).await;
    loop_handle.abort();
    elapsed
}

// ── pipelined dispatch: loop fires dispatch as a task, loops back immediately ─

async fn bench_batched_pipelined(
    n_requests: usize,
    max_batch: usize,
    max_wait: Duration,
    fixed_us: u64,
) -> (Duration, Vec<Duration>) {
    let (tx, mut rx) = mpsc::channel::<BenchItem>(n_requests + 64);

    let loop_handle = tokio::spawn(async move {
        while let Some((items, total)) = collect_batch(&mut rx, max_batch, max_wait).await {
            // Spawn dispatch independently — loop returns to recv immediately.
            task::spawn_blocking(move || dispatch_batch(items, total, fixed_us));
        }
    });

    let elapsed = drive_requests(n_requests, tx).await;
    loop_handle.abort();
    elapsed
}

// ── shared request driver ─────────────────────────────────────────────────────

async fn drive_requests(
    n_requests: usize,
    tx: mpsc::Sender<BenchItem>,
) -> (Duration, Vec<Duration>) {
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
    (start.elapsed(), latencies)
}

// ── stats ─────────────────────────────────────────────────────────────────────

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    let idx = ((sorted.len() * pct).saturating_sub(1)) / 100;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_row(label: &str, n: usize, total: Duration, mut lats: Vec<Duration>) {
    lats.sort_unstable();
    let rps = n as f64 / total.as_secs_f64();
    let p50 = percentile(&lats, 50);
    let p99 = percentile(&lats, 99);
    println!(
        "  {label:<48} {:>7.1} ms   {:>8.0} req/s   p50={:.2}ms  p99={:.2}ms",
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
    const N: usize = 1_000;
    println!("\n=== Overhead ({N} concurrent requests, predict = instant) ===\n");
    println!(
        "  {:<48} {:>10}   {:>12}   {}",
        "config", "elapsed", "throughput", "per-request latency"
    );
    println!("  {}", "-".repeat(92));

    let (t, l) = rt.block_on(bench_direct(N, 0));
    print_row("direct (spawn_blocking, instant)", N, t, l);

    println!("  -- blocking dispatch --");
    let (t, l) = rt.block_on(bench_batched_blocking(N, 1, Duration::ZERO, 0));
    print_row("batcher blocking (max=1,  wait=0ms)", N, t, l);
    let (t, l) = rt.block_on(bench_batched_blocking(N, 64, Duration::from_millis(1), 0));
    print_row("batcher blocking (max=64, wait=1ms)", N, t, l);
    let (t, l) = rt.block_on(bench_batched_blocking(N, N, Duration::from_millis(5), 0));
    print_row(&format!("batcher blocking (max={N}, wait=5ms)"), N, t, l);

    println!("  -- pipelined dispatch --");
    let (t, l) = rt.block_on(bench_batched_pipelined(N, 1, Duration::ZERO, 0));
    print_row("batcher pipelined (max=1,  wait=0ms)", N, t, l);
    let (t, l) = rt.block_on(bench_batched_pipelined(N, 64, Duration::from_millis(1), 0));
    print_row("batcher pipelined (max=64, wait=1ms)", N, t, l);
    let (t, l) = rt.block_on(bench_batched_pipelined(N, N, Duration::from_millis(5), 0));
    print_row(&format!("batcher pipelined (max={N}, wait=5ms)"), N, t, l);

    // ── Section 2: Throughput (10 ms fixed predict) ───────────────────────────
    const NT: usize = 100;
    println!(
        "\n=== Throughput ({NT} concurrent requests, predict = 10 ms fixed + 0.1 ms/row) ===\n"
    );
    println!(
        "  {:<48} {:>10}   {:>12}   {}",
        "config", "elapsed", "throughput", "per-request latency"
    );
    println!("  {}", "-".repeat(92));

    let (t, l) = rt.block_on(bench_direct(NT, 10_000));
    print_row("direct (no batcher)", NT, t, l);

    println!("  -- blocking dispatch --");
    for &(mb, wms) in &[(16usize, 5u64), (32, 10), (64, 20), (NT, 50)] {
        let (t, l) = rt.block_on(bench_batched_blocking(NT, mb, Duration::from_millis(wms), 10_000));
        print_row(&format!("batcher blocking (max={mb}, wait={wms}ms)"), NT, t, l);
    }

    println!("  -- pipelined dispatch --");
    for &(mb, wms) in &[(16usize, 5u64), (32, 10), (64, 20), (NT, 50)] {
        let (t, l) = rt.block_on(bench_batched_pipelined(NT, mb, Duration::from_millis(wms), 10_000));
        print_row(&format!("batcher pipelined (max={mb}, wait={wms}ms)"), NT, t, l);
    }

    println!();
}
