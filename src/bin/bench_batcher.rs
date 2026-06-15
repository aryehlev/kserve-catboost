/// Benchmark: dynamic batcher overhead and throughput.
///
/// Dispatch strategies compared:
///
///   DIRECT       – one spawn_blocking per request, no batcher.
///
///   BLOCKING     – batch loop awaits dispatch; serialises collection/dispatch.
///
///   PIPELINED    – batch loop fires dispatch as a spawned task, immediately
///                  loops back. Collection and dispatch overlap.
///
///   FAST-PATH    – for max_batch=1 the batcher bypasses the channel entirely
///                  and calls spawn_blocking directly, matching direct dispatch.
///
///   ARC FAN-OUT  – pipelined + share predictions behind Arc<[f64]> instead of
///                  cloning a slice per request. Saves one allocation per item.
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
    let mut lats = Vec::with_capacity(n_requests);
    for h in handles { lats.push(h.await.unwrap()); }
    (start.elapsed(), lats)
}

// ── shared batch item (clone-based fan-out) ───────────────────────────────────

struct Item {
    row_count: usize,
    tx: oneshot::Sender<Vec<f64>>,
}

// ── shared batch item (Arc-based fan-out) ─────────────────────────────────────

struct ArcItem {
    row_count: usize,
    offset: usize, // filled in at dispatch time via a separate channel
    tx: oneshot::Sender<(Arc<[f64]>, usize, usize)>,
}

// ── collect batch (shared) ────────────────────────────────────────────────────

async fn collect<T: Send>(
    rx: &mut mpsc::Receiver<T>,
    max_batch: usize,
    get_rows: impl Fn(&T) -> usize,
    max_wait: Duration,
) -> Option<(Vec<T>, usize)> {
    let first = rx.recv().await?;
    let mut items = vec![first];
    let mut total = get_rows(&items[0]);

    while total < max_batch {
        match rx.try_recv() {
            Ok(item) => { total += get_rows(&item); items.push(item); }
            Err(_) => break,
        }
    }

    if !max_wait.is_zero() && total < max_batch {
        let deadline = tokio::time::Instant::now() + max_wait;
        'dl: loop {
            if total >= max_batch { break; }
            let mut extra: Vec<T> = Vec::new();
            tokio::select! {
                biased;
                _ = tokio::time::sleep_until(deadline) => break 'dl,
                n = rx.recv_many(&mut extra, 1) => {
                    if n == 0 { break 'dl; }
                    for item in extra { total += get_rows(&item); items.push(item); }
                    while total < max_batch {
                        match rx.try_recv() {
                            Ok(item) => { total += get_rows(&item); items.push(item); }
                            Err(_) => break,
                        }
                    }
                }
            }
        }
    }
    Some((items, total))
}

// ── blocking dispatch ─────────────────────────────────────────────────────────

async fn bench_blocking(
    n: usize, max_batch: usize, max_wait: Duration, fixed_us: u64,
) -> (Duration, Vec<Duration>) {
    let (tx, mut rx) = mpsc::channel::<Item>(n + 64);
    let lh = tokio::spawn(async move {
        while let Some((items, total)) = collect(&mut rx, max_batch, |i| i.row_count, max_wait).await {
            let preds = task::spawn_blocking(move || fake_predict(total, fixed_us)).await.unwrap();
            fan_out_clone(items, &preds);
        }
    });
    let r = drive(n, tx).await;
    lh.abort();
    r
}

// ── pipelined dispatch ────────────────────────────────────────────────────────

async fn bench_pipelined(
    n: usize, max_batch: usize, max_wait: Duration, fixed_us: u64,
) -> (Duration, Vec<Duration>) {
    let (tx, mut rx) = mpsc::channel::<Item>(n + 64);
    let lh = tokio::spawn(async move {
        while let Some((items, total)) = collect(&mut rx, max_batch, |i| i.row_count, max_wait).await {
            task::spawn_blocking(move || {
                let preds = fake_predict(total, fixed_us);
                fan_out_clone(items, &preds);
            });
        }
    });
    let r = drive(n, tx).await;
    lh.abort();
    r
}

// ── fast-path: bypass channel for max_batch=1 ────────────────────────────────

async fn bench_fastpath(
    n: usize, max_batch: usize, max_wait: Duration, fixed_us: u64,
) -> (Duration, Vec<Duration>) {
    if max_batch == 1 {
        // Bypass the channel entirely — same as direct dispatch.
        return bench_direct(n, fixed_us).await;
    }
    bench_pipelined(n, max_batch, max_wait, fixed_us).await
}

// ── pipelined + Arc fan-out ───────────────────────────────────────────────────

async fn bench_arc(
    n: usize, max_batch: usize, max_wait: Duration, fixed_us: u64,
) -> (Duration, Vec<Duration>) {
    let (tx, mut rx) = mpsc::channel::<ArcItem>(n + 64);
    let lh = tokio::spawn(async move {
        while let Some((items, total)) = collect(&mut rx, max_batch, |i| i.row_count, max_wait).await {
            task::spawn_blocking(move || {
                let preds: Arc<[f64]> = Arc::from(fake_predict(total, fixed_us));
                let mut offset = 0usize;
                for item in items {
                    let end = offset + item.row_count;
                    let _ = item.tx.send((preds.clone(), offset, end));
                    offset = end;
                }
            });
        }
    });

    let shared_tx = Arc::new(tx);
    let start = Instant::now();
    let handles: Vec<_> = (0..n)
        .map(|_| {
            let tx = shared_tx.clone();
            tokio::spawn(async move {
                let t0 = Instant::now();
                let (rtx, rrx) = oneshot::channel();
                tx.send(ArcItem { row_count: 1, offset: 0, tx: rtx }).await.unwrap();
                let (_arc, _start, _end) = rrx.await.unwrap();
                // Arc slice available: _arc[_start.._end] — no copy.
                t0.elapsed()
            })
        })
        .collect();
    drop(shared_tx);
    let mut lats = Vec::with_capacity(n);
    for h in handles { lats.push(h.await.unwrap()); }
    let elapsed = start.elapsed();
    lh.abort();
    (elapsed, lats)
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn fan_out_clone(items: Vec<Item>, preds: &[f64]) {
    let mut offset = 0;
    for item in items {
        let end = offset + item.row_count;
        let _ = item.tx.send(preds[offset..end].to_vec());
        offset = end;
    }
}

async fn drive(n: usize, tx: mpsc::Sender<Item>) -> (Duration, Vec<Duration>) {
    let shared = Arc::new(tx);
    let start = Instant::now();
    let handles: Vec<_> = (0..n)
        .map(|_| {
            let tx = shared.clone();
            tokio::spawn(async move {
                let t0 = Instant::now();
                let (rtx, rrx) = oneshot::channel();
                tx.send(Item { row_count: 1, tx: rtx }).await.unwrap();
                rrx.await.unwrap();
                t0.elapsed()
            })
        })
        .collect();
    drop(shared);
    let mut lats = Vec::with_capacity(n);
    for h in handles { lats.push(h.await.unwrap()); }
    (start.elapsed(), lats)
}

// ── stats ─────────────────────────────────────────────────────────────────────

fn pct(sorted: &[Duration], p: usize) -> Duration {
    sorted[((sorted.len() * p).saturating_sub(1)) / 100]
}

fn row(label: &str, n: usize, total: Duration, mut lats: Vec<Duration>) {
    lats.sort_unstable();
    println!(
        "  {label:<50} {:>7.1} ms  {:>8.0} req/s  p50={:.2}ms  p99={:.2}ms",
        total.as_secs_f64() * 1000.0,
        n as f64 / total.as_secs_f64(),
        pct(&lats, 50).as_secs_f64() * 1000.0,
        pct(&lats, 99).as_secs_f64() * 1000.0,
    );
}

fn header() {
    println!(
        "  {:<50} {:>10}  {:>12}  {}",
        "config", "elapsed", "throughput", "per-req latency"
    );
    println!("  {}", "-".repeat(94));
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .max_blocking_threads(8)
        .build()
        .unwrap();

    // ── Section 1: Overhead (0 µs predict, 1000 requests) ────────────────────
    const N: usize = 1_000;
    println!("\n=== Overhead ({N} concurrent requests, predict = instant) ===\n");
    header();

    let (t, l) = rt.block_on(bench_direct(N, 0));
    row("direct (spawn_blocking)", N, t, l);

    let (t, l) = rt.block_on(bench_blocking(N, 1, Duration::ZERO, 0));
    row("blocking  passthrough (max=1,  wait=0)", N, t, l);

    let (t, l) = rt.block_on(bench_pipelined(N, 1, Duration::ZERO, 0));
    row("pipelined passthrough (max=1,  wait=0)", N, t, l);

    let (t, l) = rt.block_on(bench_fastpath(N, 1, Duration::ZERO, 0));
    row("fast-path passthrough (max=1,  wait=0) [bypass]", N, t, l);

    println!();

    let (t, l) = rt.block_on(bench_pipelined(N, 64, Duration::from_millis(1), 0));
    row("pipelined (max=64, wait=1ms)", N, t, l);
    let (t, l) = rt.block_on(bench_arc(N, 64, Duration::from_millis(1), 0));
    row("arc       (max=64, wait=1ms)", N, t, l);

    println!();

    let (t, l) = rt.block_on(bench_pipelined(N, N, Duration::from_millis(5), 0));
    row(&format!("pipelined (max={N}, wait=5ms)"), N, t, l);
    let (t, l) = rt.block_on(bench_arc(N, N, Duration::from_millis(5), 0));
    row(&format!("arc       (max={N}, wait=5ms)"), N, t, l);

    // ── Section 2: Throughput (10 ms fixed + 0.1 ms/row, 100 requests) ───────
    const NT: usize = 100;
    println!(
        "\n=== Throughput ({NT} concurrent requests, predict = 10 ms fixed + 0.1 ms/row) ===\n"
    );
    header();

    let (t, l) = rt.block_on(bench_direct(NT, 10_000));
    row("direct (no batcher)", NT, t, l);

    println!();
    for &(mb, wms) in &[(16usize, 5u64), (32, 10), (64, 20), (NT, 50)] {
        let d = Duration::from_millis(wms);
        let (t, l) = rt.block_on(bench_pipelined(NT, mb, d, 10_000));
        row(&format!("pipelined (max={mb:3}, wait={wms:2}ms)"), NT, t, l);
        let (t, l) = rt.block_on(bench_arc(NT, mb, d, 10_000));
        row(&format!("arc       (max={mb:3}, wait={wms:2}ms)"), NT, t, l);
    }

    println!();
}
