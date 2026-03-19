
use bloom_lfs::flush_behaviour::FOUR_KB_PAGE;
use bloom_lfs::flush_buffer::*;


use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion,
    Throughput,
};

// use llama::flush_buffer::{BufferError, BufferMsg, FlushBufferRing};
// use llama::log_structured_store::FOUR_KB_PAGE;
use std::hint::black_box;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Instant;

// ─── constants ───────────────────────────────────────────────────────────────

const RING_SIZE: usize = 4;

const SMALL: usize = 64;
const MEDIUM: usize = 512;
const LARGE: usize = 2048;
const MAX: usize = 4096;

const MB: usize = 1024 * 1024;

// ─── retry helper ────────────────────────────────────────────────────────────

fn put_with_retry(ring: &FlushBufferRing, payload: &[u8]) -> Result<BufferMsg, BufferError> {
    loop {
        let current = unsafe {
            ring.current_buffer
                .load(std::sync::atomic::Ordering::Acquire)
                .as_ref()
                .ok_or(BufferError::InvalidState)?
        };

        let reserve_result = current.reserve_space(payload.len());

        match &reserve_result {
            Err(BufferError::FailedReservation) => continue,
            Err(BufferError::EncounteredSealedBuffer) => continue,
            _ => {}
        }

        match ring.put(current, reserve_result, payload) {
            Err(BufferError::ActiveUsers) => continue,
            Err(BufferError::EncounteredSealedBuffer) => {
                std::thread::yield_now();
                continue;
            }
            Err(BufferError::RingExhausted) => {
                std::thread::yield_now();
                continue;
            }
            other => return other,
        }
    }
}

// ─── multithreaded driver ────────────────────────────────────────────────────
//
// Criterion controls iteration count for single-threaded benches via its
// iter() loop. For multithreaded cases we can't let criterion drive the inner
// loop directly — instead we treat the entire N-thread × K-ops run as one
// timed sample and use iter_custom() to hand back the measured duration.

fn run_multithreaded(
    ring: Arc<FlushBufferRing>,
    num_threads: usize,
    ops_per_thread: usize,
    payload_size: usize,
) -> std::time::Duration {
    let payload = vec![0xAA_u8; payload_size];
    let barrier = Arc::new(Barrier::new(num_threads));
    let start_times: Arc<Mutex<Vec<Instant>>> = Arc::new(Mutex::new(Vec::new()));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let ring = Arc::clone(&ring);
            let payload = payload.clone();
            let barrier = Arc::clone(&barrier);
            let start_times = Arc::clone(&start_times);

            thread::spawn(move || {
                barrier.wait();
                start_times.lock().unwrap().push(Instant::now());

                for _ in 0..ops_per_thread {
                    black_box(put_with_retry(&ring, black_box(&payload)).unwrap());
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let join_time = Instant::now();
    let earliest = start_times.lock().unwrap().iter().copied().min().unwrap();

    join_time.duration_since(earliest)
}

// =============================================================================
// Benchmark groups
// =============================================================================

// ── 1. Single-threaded payload size scaling ───────────────────────────────────
//
// How does throughput degrade as payload grows?
// At small sizes the CAS + rotation overhead dominates.
// At large sizes memcpy dominates.
// A non-linear degradation means rotation overhead is significant.

fn bench_payload_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_thread/payload_size");

    for payload_size in [SMALL, MEDIUM, LARGE, MAX] {
        let payload = vec![0xAA_u8; payload_size];
        group.throughput(Throughput::Bytes(payload_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{payload_size}B")),
            &payload,
            |b, payload| {
                let ring = FlushBufferRing::with_buffer_amount(RING_SIZE, FOUR_KB_PAGE);
                b.iter(|| {
                    black_box(put_with_retry(&ring, black_box(payload)).unwrap());
                });
            },
        );
    }

    group.finish();
}

// ── 2. Single-threaded ring size scaling ──────────────────────────────────────
//
// Does a deeper ring improve single-threaded throughput?
// Larger ring = the rotation scan touches more cache lines but
// the seal/reset cycle happens less often relative to writes.
// Expect a small but measurable improvement up to ring_size ~8,
// then diminishing returns.

fn bench_ring_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_thread/ring_size");
    let payload = vec![0xAA_u8; MEDIUM];
    group.throughput(Throughput::Bytes(MEDIUM as u64));

    for ring_size in [1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::from_parameter(ring_size),
            &ring_size,
            |b, &ring_size| {
                let ring = FlushBufferRing::with_buffer_amount(ring_size, FOUR_KB_PAGE);
                b.iter(|| {
                    black_box(put_with_retry(&ring, black_box(&payload)).unwrap());
                });
            },
        );
    }

    group.finish();
}

// ── 3. Single-threaded buffer size scaling ────────────────────────────────────
//
// The paper used 4MB buffers. We default to 4KB.
// This bench answers: how much does buffer size matter single-threaded?
// Larger buffers = fewer seals per 100K ops = less rotation overhead.
// The improvement from 4KB→256KB reveals how much time is spent in
// the seal/rotate/reset path vs pure memcpy.
//
// Also includes 1MB, 2MB, 3MB, 4MB to bracket the paper's design point.

fn bench_buffer_size_scaling_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_thread/buffer_size");
    let payload = vec![0xAA_u8; MEDIUM];
    group.throughput(Throughput::Bytes(MEDIUM as u64));

    let sizes = [
        (FOUR_KB_PAGE, "4KB"),
        (FOUR_KB_PAGE * 4, "16KB"),
        (FOUR_KB_PAGE * 16, "64KB"),
        (FOUR_KB_PAGE * 64, "256KB"),
        (1 * MB, "1MB"),
        (2 * MB, "2MB"),
        (3 * MB, "3MB"),
        (4 * MB, "4MB"),
    ];

    for (buf_size, label) in sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &buf_size,
            |b, &buf_size| {
                let ring = FlushBufferRing::with_buffer_amount(RING_SIZE, buf_size);
                b.iter(|| {
                    black_box(put_with_retry(&ring, black_box(&payload)).unwrap());
                });
            },
        );
    }

    group.finish();
}

// ── 4. Multi-threaded thread count scaling ────────────────────────────────────
//
// The key scaling story. Near-linear 1→2→4 is good.
// Plateau or regression at 8 is expected with 4KB buffers —
// the CAS on the state word becomes the bottleneck as all threads
// fight over the same cache line.
// Each sample = all threads completing ops_per_thread ops.

fn bench_thread_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_thread/thread_count");
    const OPS: usize = 10_000;

    for num_threads in [2, 4, 8] {
        let total_bytes = (num_threads * OPS * MEDIUM) as u64;
        group.throughput(Throughput::Bytes(total_bytes));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_threads}T")),
            &num_threads,
            |b, &num_threads| {
                // Allocate once — 4 × 1MB = 4MB, done once per benchmark not per iteration
                let ring = Arc::new(FlushBufferRing::with_buffer_amount(RING_SIZE, 1 * MB));

                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        total += run_multithreaded(Arc::clone(&ring), num_threads, OPS, MEDIUM);
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

// =============================================================================

criterion_group!(
    benches,
    bench_payload_size_scaling,
    bench_ring_size_scaling,
    bench_buffer_size_scaling_single,
    bench_thread_count_scaling,
);
criterion_main!(benches);
