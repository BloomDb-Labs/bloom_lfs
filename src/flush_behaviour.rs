//! # Flush Behaviour — `io_uring`-backed Write Dispatchers
//!
//! This module defines the two write strategies LLAMA uses to flush sealed
//! [`FlushBuffer`]s to the log-structured backing file:
//!
//! | Strategy                       | Type                                 | Ordering |
//! |--------------------------------|--------------------------------------|----------|
//! | Tail-Localised Writes          | [`FlushBehavior::NoWaitAppender`]    | Parallel |
//! | Strictly Serialised Writes     | [`FlushBehavior::WaitAppender`]      | `IO_LINK`|
//!
//! Both strategies are backed by the same [`Appender`] struct; the difference
//! lies in the `io_uring` submission-queue flags applied at dispatch time.
//!
//! ## Why Two Strategies?
//!
//! ### Tail-Localised Writes
//!
//! Append-only write patterns deliver substantial throughput improvements on
//! both spinning-disk and SSD storage because they eliminate head seeks and
//! enable write coalescing by the device firmware.  LLAMA exploits this by
//! staging writes in a ring of 4 KB-aligned [`FlushBuffer`]s.  Each buffer
//! is assigned a unique, non-overlapping slot in the LSS address space at seal
//! time; once sealed, buffers are flushed independently with no synchronisation
//! between them.
//!
//! Because slots are claimed atomically (fetch-add) but flushed concurrently, a
//! buffer sealed *later* may land on disk *before* an earlier one.  This means
//! flushes are **tail-localised** rather than strictly sequential — the maximum
//! write distance from the logical tail is bounded by:
//!
//! ```text
//! max_distance = RING_SIZE × FOUR_KB_PAGE
//! ```
//!
//! ### Serialised Writes
//!
//! For workloads that require strict append order (e.g. WAL segments, recovery
//! logs), [`WriteMode::SerializedWrites`] applies `IO_LINK` to the SQE chain.
//! The kernel will not begin the *n+1*th write until the *n*th has completed,
//! enforcing submission-order on disk at the cost of reduced parallelism.
//!
//! ## Completion Handling
//!
//! LLAMA deliberately avoids a dedicated watchdog thread.  Instead, a calling
//! thread inspects the completion queue at a well-defined point on the write
//! path via [`LogStructuredStore::check_async_cque`].  Failed writes are
//! re-submitted from the stored SQE; successful writes advance the
//! `hi_stable` stability pointer.
//!
//! ## `O_DIRECT` Alignment Invariant
//!
//! All buffers submitted through this module **must** be aligned to
//! [`FOUR_KB_PAGE`] and their lengths must be a multiple of the device's
//! logical block size.  This invariant is upheld by `Buffer::new_aligned`
//! inside [`crate::flush_buffer::FlushBufferRing`].

use io_uring::{opcode, squeue, types, IoUring};

#[allow(unused_imports)]
use crate::log_structured_store::LogStructuredStore;

#[allow(unused_imports)]
use crate::flush_buffer::RING_SIZE;
use crate::{flush_buffer::FlushBuffer, log_structured_store::FOUR_KB_PAGE};
use std::{
    fs::File,
    io,
    os::fd::AsRawFd,
    sync::{atomic::Ordering, Arc},
};

/// Flush Buffers must adherer to Strict Serialized Ordered Writes
#[allow(unused)]
const SERIALIZED_ORDERING: u8 = 0;

/// Flag Buffers are permitted to write within a localized region
/// within [`RING_SIZE`] × [`FOUR_KB_PAGE`] of the tail
#[allow(unused)]
const LOCALIZED_WRITES: u8 = 1;

/// A shared, mutex-protected `io_uring` handle.
///
/// The `Mutex` is from [`parking_lot`] and is fair, making it suitable for use
/// across many short-lived critical sections on the write path.
pub type SharedAsyncFileWriter = Arc<parking_lot::Mutex<IoUring>>;

/// Controls the `io_uring` submission-queue flags used when dispatching writes.
///
/// Choose [`TailLocalizedWrites`](WriteMode::TailLocalizedWrites) for maximum
/// throughput and choose [`SerializedWrites`](WriteMode::SerializedWrites) when
/// strict append ordering is required (e.g. WAL segments).
///
/// # Examples
///
/// ```rust,no_run
/// use llama::flush_behaviour::WriteMode;
/// use llama::log_structured_store::LogStructuredStore;
///
/// // High-throughput ingestion path — writes may land out of order within
/// // RING_SIZE × FOUR_KB_PAGE of the tail.
/// let store = LogStructuredStore::open_with_behavior("data.lss", WriteMode::TailLocalizedWrites)?;
///
/// // Recovery-log path — each write completes before the next begins.
/// let wal = LogStructuredStore::open_with_behavior("wal.lss", WriteMode::SerializedWrites)?;
/// # Ok::<(), std::io::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub enum WriteMode {
    /// Parallel localized writes — lock-free
    TailLocalizedWrites,
    /// Serialized ordered writes — drain ordering enforced
    SerializedWrites,
}

/// Unified `io_uring` appender — handles both localised and serialised flush strategies.
///
/// [`Appender`] wraps a shared [`IoUring`] instance and an `O_DIRECT` [`File`]
/// handle.  The concrete write behaviour (parallel vs. ordered) is determined by
/// the [`WriteMode`] stored at construction time.
///
/// In normal operation callers should not use `Appender` directly; instead,
/// interact with the store through [`FlushBehavior`].
pub struct Appender {
    /// Shared `O_DIRECT` file handle — the LSS backing file.
    store: Arc<File>,
    /// Shared `io_uring` instance.  Protected by a [`parking_lot::Mutex`] so
    /// that multiple threads can submit SQEs without data races.
    flusher: SharedAsyncFileWriter,
    /// Determines SQE flags applied to every write submission.
    mode: WriteMode,
}

impl Appender {
    /// Create a new `Appender` from an existing `io_uring` instance and file handle.
    ///
    /// # Arguments
    ///
    /// * `io_uring`    — Shared, mutex-protected `io_uring` ring.
    /// * `file_handle` — `O_DIRECT` file handle to the LSS backing file.
    /// * `mode`        — Write ordering mode.
    pub fn new(io_uring: SharedAsyncFileWriter, file_handle: Arc<File>, mode: WriteMode) -> Self {
        Self {
            flusher: io_uring,
            store: file_handle,
            mode,
        }
    }

    /// Submit a **fire-and-forget** write for `buffer_data` at byte offset `at`.
    ///
    /// Returns immediately after the SQE is pushed to the submission ring; the
    /// kernel picks it up asynchronously.  Poll completions via
    /// [`Appender::cqueue`] or [`LogStructuredStore::check_async_cque`].
    ///
    /// The SQE is also stored inside `buffer.submit_queue_entry` so that a
    /// failed completion can re-submit the exact same write without
    /// re-constructing it.
    ///
    /// # Arguments
    ///
    /// * `buffer`      — The [`FlushBuffer`] whose contents are being flushed.
    ///                   Its `submit_queue_entry` cell is updated in-place.
    /// * `buffer_data` — Aligned slice covering exactly the bytes to write
    ///                   (`0..used_bytes`).
    /// * `at`          — Byte offset in the backing file (`slot × FOUR_KB_PAGE`).
    /// * `buffer_ptr`  — Raw pointer to `buffer` cast to `u64`, stored as the
    ///                   SQE's `user_data` so the completion handler can recover
    ///                   the buffer without an extra lookup.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the submission ring is full or if
    /// `io_uring::submit()` fails.
    ///
    /// # Safety
    ///
    /// The pointed-to memory must remain valid and unmodified until the
    /// corresponding CQE is observed.  This invariant is upheld structurally:
    /// buffers live in a `Pin<Box<[Arc<FlushBuffer>]>>` inside
    /// [`crate::flush_buffer::FlushBufferRing`], so they are never moved, and the flush-in-progress
    /// bit prevents the buffer from being reset while a write is in flight.
    /// No additional action is required from the caller.
    pub fn submit(
        &self,
        buffer: &FlushBuffer,
        buffer_data: &[u8],
        at: u64,
        buffer_ptr: u64,
    ) -> io::Result<()> {
        let flags = match self.mode {
            // Parallel writes — kernel may reorder freely.
            // Safe because each buffer owns a non-overlapping LSS address range.
            WriteMode::TailLocalizedWrites => squeue::Flags::empty(),

            // Serialized writes — each write is linked to the next.
            // Kernel will not start the next write until this one completes.
            // Ordering is enforced by submission order, not a drain barrier.
            WriteMode::SerializedWrites => squeue::Flags::IO_LINK,
        };

        let sqe = opcode::Write::new(
            types::Fd(self.store.as_raw_fd()),
            buffer_data.as_ptr(),
            buffer_data.len() as u32,
        )
        // Slots are flawed they assume buffers will be filled to capacity
        .offset(at)
        .build()
        .flags(flags)
        .user_data(buffer_ptr);

        let mut ring = self.flusher.lock();

        unsafe {
            ring.submission()
                .push(&sqe)
                .map_err(|_| io::Error::other("SQ full"))?;

            *buffer.submit_queue_entry.get() = Some(sqe);
        }

        // submit() returns immediately — kernel picks it up asynchronously.
        // We do NOT call submit_and_wait() here.
        ring.submit()?;

        Ok(())
    }

    /// Submit a write and **block** until the kernel reports completion.
    ///
    /// Semantically equivalent to [`submit`](Self::submit) followed by an
    /// immediate `submit_and_wait(1)`.  Suitable for tests, shutdown sequences,
    /// or any context where you need to guarantee durability before proceeding.
    ///
    /// For high-throughput production writes prefer the async
    /// [`submit`](Self::submit) path.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the SQE push, the wait, or the CQE result
    /// indicates failure.
    pub fn submit_blocking(
        &self,
        buffer: &FlushBuffer,
        buffer_data: &[u8],
        at: u64,
        buffer_ptr: u64,
    ) -> io::Result<()> {
        let flags = match self.mode {
            WriteMode::TailLocalizedWrites => squeue::Flags::empty(),
            WriteMode::SerializedWrites => squeue::Flags::IO_LINK,
        };

        let sqe = opcode::Write::new(
            types::Fd(self.store.as_raw_fd()),
            buffer_data.as_ptr(),
            buffer_data.len() as u32,
        )
        .offset(at)
        .build()
        .flags(flags)
        .user_data(buffer_ptr);

        let mut ring = self.flusher.lock();

        unsafe {
            ring.submission()
                .push(&sqe)
                .map_err(|_| io::Error::other("SQ full"))?;

            *buffer.submit_queue_entry.get() = Some(sqe);
        }

        // BLOCKING: Wait for at least 1 completion
        ring.submit_and_wait(1)?;

        // Immediately check the completion
        if let Some(cqe) = ring.completion().next() {
            if cqe.result() < 0 {
                return Err(io::Error::from_raw_os_error(-cqe.result()));
            }
            // Success!
            return Ok(());
        }

        Err(io::Error::other("No completion received"))
    }

    /// Acquire a mutex guard giving exclusive access to the underlying `io_uring`
    /// instance, including its completion queue.
    ///
    /// The guard is held for the duration of the caller's critical section; keep
    /// it as short-lived as possible to avoid starving the write path.

    pub fn cqueue(&self) -> parking_lot::lock_api::MutexGuard<'_, parking_lot::RawMutex, IoUring> {
        let flusher_ring = self.flusher.lock();
        flusher_ring
    }
}

/// The top-level flush dispatcher — selects between parallel and serialised write modes.
///
/// `FlushBehavior` is an enum over the two variants of [`Appender`] so that the
/// store can branch once at construction time and then call the same interface
/// everywhere on the hot path.
///
/// # Variants
///
/// * [`WaitAppender`](FlushBehavior::WaitAppender) — wraps an [`Appender`] in
///   [`WriteMode::SerializedWrites`].  Use for WAL segments or any workload that
///   requires each write to complete before the next begins.
/// * [`NoWaitAppender`](FlushBehavior::NoWaitAppender) — wraps an [`Appender`]
///   in [`WriteMode::TailLocalizedWrites`].  Use for high-throughput data
///   ingestion where write ordering within the ring is acceptable.
///
/// # Examples
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use llama::flush_behaviour::{FlushBehavior, WriteMode};
///
/// let file    = Arc::new(std::fs::File::open("/dev/null").unwrap());
/// let io_ring = Arc::new(parking_lot::Mutex::new(io_uring::IoUring::new(8).unwrap()));
///
/// let flusher = FlushBehavior::with_no_wait_appender(io_ring, file);
/// ```
pub enum FlushBehavior {
    /// Strictly serialised write appender (`IO_LINK` per SQE).
    WaitAppender(Appender),
    /// Parallel tail-localised write appender (no ordering flags).
    NoWaitAppender(Appender),
}

impl FlushBehavior {
    /// Construct a [`FlushBehavior::WaitAppender`] from an existing ring and file handle.
    ///
    /// Writes submitted through this variant will use [`WriteMode::SerializedWrites`].
    pub fn with_wait_appender(io_uring: SharedAsyncFileWriter, file: Arc<File>) -> Self {
        FlushBehavior::WaitAppender(Appender::new(io_uring, file, WriteMode::SerializedWrites))
    }

    /// Construct a [`FlushBehavior::NoWaitAppender`] from an existing ring and file handle.
    ///
    /// Writes submitted through this variant will use [`WriteMode::TailLocalizedWrites`].
    pub fn with_no_wait_appender(io_uring: SharedAsyncFileWriter, file: Arc<File>) -> Self {
        FlushBehavior::NoWaitAppender(Appender::new(
            io_uring,
            file,
            WriteMode::TailLocalizedWrites,
        ))
    }

    /// Submit an **asynchronous** flush of the given buffer to its assigned LSS slot.
    ///
    /// Reads the buffer's `local_lss_address_slot` to determine the byte offset
    /// (`slot × FOUR_KB_PAGE`) and dispatches a fire-and-forget write SQE.
    ///
    /// Returns immediately; the caller must poll [`check_async_cque`](crate::log_structured_store::LogStructuredStore::check_async_cque)
    /// to observe completion.
    ///
    /// # Safety
    ///
    /// `buffer` must remain at a stable memory address until the corresponding
    /// CQE is consumed, because its address is stored as `user_data` in the SQE.
    pub fn submit_buffer(&self, buffer: &FlushBuffer) {
        match self {
            FlushBehavior::WaitAppender(a) | FlushBehavior::NoWaitAppender(a) => {
                let ptr = unsafe { *buffer.buf.buffer.get() };

                // SAFETY: `reserve_space` guaranteed this range is exclusively ours
                // and the buffer is 4 KiB-aligned (upheld by Buffer::new_aligned).
                let slice: &[u8] = unsafe { &*std::ptr::slice_from_raw_parts(ptr, FOUR_KB_PAGE) };

                // Store the raw buffer pointer as `user_data` so the completion
                // handler can recover it without a separate lookup table.
                let ptr_to_buffer_position = buffer as *const FlushBuffer as u64;

                let _ = a.submit(
                    buffer,
                    slice,
                    buffer.local_lss_address_slot.load(Ordering::Acquire) as u64
                        * FOUR_KB_PAGE as u64,
                    ptr_to_buffer_position,
                );
            }
        }
    }

    /// Submit a flush of the given buffer and **block** until it completes.
    ///
    /// Suitable for tests, shutdown sequences, or any context where you need to
    /// guarantee that the buffer has been written to stable storage before
    /// continuing.  Avoid on the hot write path — use the async
    /// [`submit_buffer`](Self::submit_buffer) instead.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the write fails.
    pub fn submit_buffer_and_wait(&self, buffer: &FlushBuffer) -> io::Result<()> {
        match self {
            FlushBehavior::WaitAppender(a) | FlushBehavior::NoWaitAppender(a) => {
                let ptr = unsafe { *buffer.buf.buffer.get() };
                let slice: &[u8] = unsafe { &*std::ptr::slice_from_raw_parts(ptr, FOUR_KB_PAGE) };
                let ptr_to_buffer_position = buffer as *const FlushBuffer as u64;

                a.submit_blocking(
                    buffer,
                    slice,
                    buffer.local_lss_address_slot.load(Ordering::Acquire) as u64
                        * FOUR_KB_PAGE as u64,
                    ptr_to_buffer_position,
                )
            }
        }
    }

    /// Acquire exclusive access to the `io_uring` instance's completion queue.
    ///
    /// Used by [`LogStructuredStore::check_async_cque`] to drain CQEs without
    /// duplicating the lock acquisition pattern across call sites.
    pub fn get_cqueue(
        &self,
    ) -> parking_lot::lock_api::MutexGuard<'_, parking_lot::RawMutex, IoUring> {
        match self {
            FlushBehavior::WaitAppender(appender) | FlushBehavior::NoWaitAppender(appender) => {
                appender.cqueue()
            }
        }
    }
}
