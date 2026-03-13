//! # Log-Structured Store (`lss`)
//!
//! The [`LogStructuredStore`] is the single durable backing file for LLAMA's page store.
//! All writes flow through a [`FlushBufferRing`] and are dispatched to the underlying
//! file via [`FlushBehavior`] (either tail-localised or serialised `io_uring` writes).
//!
//! ## Design Goals
//!
//! | Goal                     | Mechanism                                                   |
//! |--------------------------|-------------------------------------------------------------|
//! | High write throughput    | Amortised batch writes via [`FlushBufferRing`]              |
//! | Low write amplification  | Sequential / tail-localised layout on disk                  |
//! | Kernel-cache bypass      | `O_DIRECT` — I/O lands directly on the device               |
//! | Async I/O                | `io_uring` dispatched through [`FlushBehavior`]             |
//! | Crash recoverability     | [`LogStructuredStore::hi_stable`] tracks the highest contiguous durable slot    |
//!
//! ## File Configuration
//!
//! The backing file is opened with:
//!
//! ```text
//! O_RDWR | O_CREAT | O_DIRECT
//! ```
//!
//! `O_DIRECT` bypasses the kernel page cache — the device DMA's directly into the
//! userspace buffer. This requires every buffer to be aligned to the device's logical
//! block size. This invariant is upheld by `Buffer::new_aligned` inside [`FlushBufferRing`],
//!  which aligns all allocations to [`FOUR_KB_PAGE`].
//!
//! ## Write Path
//!
//! ```text
//!  Caller
//!    │
//!    │  write_payload(&[u8], Reservation)
//!    ▼
//!  reserve_space(payload_size)       — atomically claims a byte range in the active buffer
//!    │
//!    ▼
//!  FlushBufferRing::put()            — copies payload into the aligned buffer
//!    │
//!    ├─ buffer not full ──────────────────────────────► Ok(SuccessfulWrite)
//!    │
//!    └─ buffer full (sealed)
//!         │
//!         ├─ rotate ring to the next available buffer
//!         │
//!         └─ FlushBufferRing::flush()
//!                │
//!                ├─ NoWaitAppender  (TailLocalizedWrites)
//!                │    └─ io_uring write_at(offset)           unordered writes
//!                │
//!                └─ WaitAppender   (SerializedWrites)
//!                     └─ io_uring write_at(offset, IO_LINK)  strict submission order
//! ```
//!
//! ## Address Space Layout
//!
//! Each [`FlushBuffer`] is assigned a unique `local_lss_address_slot` at seal time.
//! The on-disk byte offset is:
//!
//! ```text
//! byte_offset = local_lss_address_slot × FOUR_KB_PAGE
//! ```
//!
//! Slots are handed out by [`FlushBufferRing::next_address_range`] via a single
//! atomic fetch-add, guaranteeing that no two sealed buffers ever map to the same
//! region of the file — even when their flushes complete out of order.
//!
//! ## Tail-Localised vs. Serialised Writes
//!
//! Because slots are claimed at seal time but flushed concurrently, a slower buffer
//! may land on disk *after* a buffer with a higher slot index.  The maximum write
//! distance from the logical tail is bounded by:
//!
//! ```text
//! max_distance = RING_SIZE × FOUR_KB_PAGE
//! ```
//!
//! For workloads that require strict append order (e.g. WAL segments),
//! [`FlushBehavior::WaitAppender`] uses `IO_LINK` to serialise submissions.
//!
//! ## Stability Tracking
//!
//! [`LogStructuredStore::hi_stable`] records the highest *contiguous* LSS slot that
//! has been durably written to secondary storage.  Out-of-order completions are held
//! in `completed_islands` until their predecessors arrive, at which point
//! [`mark_slot_complete`](LogStructuredStore::mark_slot_complete) walks the island set
//! and advances `hi_stable` as far as possible.

use std::{
    collections::BTreeSet,
    fs::{File, OpenOptions},
    io,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use io_uring::{cqueue, IoUring};
use parking_lot::RwLock;

use crate::{
    flush_behaviour::*,
    flush_buffer::{BufferError, BufferMsg, FlushBuffer, FlushBufferRing, RING_SIZE},
};

/// Size of a single LSS page / buffer alignment unit (4 KiB).
///
/// Every on-disk slot occupies exactly one page.  All userspace buffers passed to
/// `O_DIRECT` reads and writes **must** be aligned to this value.
pub const FOUR_KB_PAGE: usize = 4096;

/// Sentinel return value indicating that an `io_uring` flush operation failed.
///
/// A CQE result less than [`FAILED_FLUSH`] (i.e. negative) signals an OS-level
/// error; the flusher will re-submit the SQE for that buffer automatically.
pub const FAILED_FLUSH: i32 = 0;

/// The Log-Structured Store — LLAMA's single durable backing file.
///
/// `LogStructuredStore` owns:
///
/// * An `O_DIRECT` [`File`] handle to the backing store.
/// * An in-memory [`FlushBufferRing`] — the staging area for in-flight writes.
/// * A [`FlushBehavior`] dispatcher — determines whether flushes are parallel
///   (tail-localised) or strictly ordered (serialised).
/// * Stability bookkeeping: [`hi_stable`](Self::hi_stable) and `completed_islands`.
///
/// The store is `Send + Sync` and is expected to be wrapped in an [`Arc`] when
/// shared across threads.
///
/// # Examples
///
/// ```rust,no_run
/// use llama::log_structured_store::{LogStructuredStore, WriteMode};
///
/// let store = LogStructuredStore::open_with_behavior(
///     "/var/lib/llama/data.lss",
///     WriteMode::TailLocalizedWrites,
/// )?;
///
/// // Reserve space, write a payload, then poll completions.
/// let reservation = store.reserve_space(64)?;
/// store.write_payload(b"hello, LLAMA", reservation)?;
/// store.check_async_cque();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
///
pub struct LogStructuredStore {
    /// Ring of aligned flush buffers.  Calling threads write here first.
    pub(crate) buffer: FlushBufferRing,

    /// Flush strategy:
    ///
    /// * Either parallel ([`FlushBehavior::NoWaitAppender`])
    ///
    /// * Or strictly serialised ([`FlushBehavior::WaitAppender`]).
    ///
    /// Also used to access the shared `io_uring` completion queue.
    pub(crate) flusher: Arc<FlushBehavior>,

    /// `O_DIRECT` file handle shared with [`flusher`](Self::flusher).
    store: Arc<File>,

    /// The highest LSS slot such that **every** slot in `0..=hi_stable` has been
    /// durably written to secondary storage.
    ///
    /// Advanced by [`mark_slot_complete`](Self::mark_slot_complete) and
    /// [`advance_high_stable`](Self::advance_high_stable) as CQEs arrive.
    pub hi_stable: AtomicU64,

    /// Out-of-order completions waiting for their predecessors.
    ///
    /// When slot *N* completes but slot *N - 1* has not yet completed, *N* is
    /// placed here.  Once *N - 1* completes and `hi_stable` reaches *N - 1*,
    /// the advancement loop drains the island set forward.
    completed_islands: RwLock<BTreeSet<u64>>,
}

/// A space reservation inside the currently active [`FlushBuffer`].
///
/// Created by [`LogStructuredStore::reserve_space`] and consumed by
/// [`LogStructuredStore::write_payload`].  The reservation captures both the
/// target buffer and the byte offset within it so that the write can be
/// committed atomically without re-acquiring any locks.
///
/// # Lifetimes
///
/// `'a` is bound to the lifetime of the [`FlushBufferRing`] from which the
/// reservation was made, preventing use-after-rotation bugs.
pub struct Reservation<'a> {
    /// Reference-counted handle to the buffer that owns the reserved range.
    pub buffer: Arc<&'a FlushBuffer>,
    /// Byte offset within `buffer` at which the payload should be written.
    pub(crate) offset: usize,
}

impl LogStructuredStore {
    /// Open (or create) an LSS backing file and initialise the store.
    ///
    /// # Arguments
    ///
    /// * `path`    — Filesystem path for the backing file.  Parent directories
    ///               are created automatically if they do not exist.
    /// * `ring`    — Pre-built [`FlushBufferRing`] (owns the aligned buffers).
    /// * `flusher` — Flush strategy; the `Arc<File>` inside `flusher` **must**
    ///               refer to the **same** file opened here.  Prefer
    ///               [`open_with_behavior`](Self::open_with_behavior) to have
    ///               both constructed together and avoid mismatches.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the file cannot be opened or created.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use std::sync::Arc;
    /// use llama::log_structured_store::LogStructuredStore;
    /// use llama::flush_buffer::FlushBufferRing;
    /// use llama::flush_behaviour::{FlushBehavior, WriteMode};
    ///
    /// let file = Arc::new(std::fs::File::open("/dev/null").unwrap());
    /// let flusher = Arc::new(FlushBehavior::with_no_wait_appender(/* ... */));
    /// let ring    = FlushBufferRing::with_flusher(8, 4096, flusher.clone());
    ///
    /// let store = LogStructuredStore::open("/var/lib/llama/data.lss", ring, flusher)?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn open(
        path: impl AsRef<Path>,
        ring: FlushBufferRing,
        flusher: Arc<FlushBehavior>,
    ) -> io::Result<Self> {
        let file = open_direct(path)?;

        Ok(Self {
            buffer: ring,
            flusher,
            store: Arc::new(file),
            hi_stable: AtomicU64::new(0),
            completed_islands: RwLock::new(BTreeSet::new()),
        })
    }

    /// Convenience constructor — opens the backing file **and** wires the
    /// flusher so that both share the same `Arc<File>`.
    ///
    /// This is the preferred entry point when creating a store from scratch
    /// because it eliminates the risk of passing a mismatched file handle.
    ///
    /// # Arguments
    ///
    /// * `path` — Filesystem path for the backing file.
    /// * `mode` — [`WriteMode::TailLocalizedWrites`] for maximum parallelism,
    ///            or [`WriteMode::SerializedWrites`] for strict append order.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the file cannot be opened/created or if
    /// the `io_uring` instance cannot be initialised (requires Linux ≥ 5.1).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use llama::log_structured_store::{LogStructuredStore, WriteMode};
    ///
    /// // Parallel tail-localised writes — best for high-throughput ingestion.
    /// let store = LogStructuredStore::open_with_behavior(
    ///     "/var/lib/llama/data.lss",
    ///     WriteMode::TailLocalizedWrites,
    /// )?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn open_with_behavior(path: impl AsRef<Path>, mode: WriteMode) -> io::Result<Self> {
        let file = Arc::new(open_direct(path)?);
        let io_uring = Arc::new(parking_lot::Mutex::new(IoUring::new(8)?));

        let flusher = Arc::new(match mode {
            WriteMode::TailLocalizedWrites => {
                FlushBehavior::NoWaitAppender(Appender::new(io_uring, Arc::clone(&file), mode))
            }
            WriteMode::SerializedWrites => {
                FlushBehavior::WaitAppender(Appender::new(io_uring, Arc::clone(&file), mode))
            }
        });

        let ring = FlushBufferRing::with_flusher(RING_SIZE, FOUR_KB_PAGE, flusher.clone());

        Ok(Self {
            buffer: ring,
            flusher,
            store: file,
            hi_stable: AtomicU64::new(0),
            completed_islands: RwLock::new(BTreeSet::new()),
        })
    }

    /// Write `payload` into the buffer at the position described by `reservation`.
    ///
    /// If the active buffer fills up during the write it is sealed automatically
    /// by the ring; the ring's flush path will drain it to disk via
    /// [`FlushBehavior`].
    ///
    /// # Arguments
    ///
    /// * `payload`     — Raw bytes to write.  Must fit within the reserved range.
    /// * `reservation` — A slot previously obtained from [`reserve_space`](Self::reserve_space).
    ///
    /// # Errors
    ///
    /// Propagates [`BufferError`] variants from the ring, most notably:
    ///
    /// * [`BufferError::RingExhausted`] — all buffers are sealed and awaiting flush.
    /// * [`BufferError::EncounteredSealedBuffer`] — the active buffer rotated
    ///   between reservation and write; callers should retry.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// let reservation = store.reserve_space(16)?;
    /// store.write_payload(b"my page state   ", reservation)?;
    /// # Ok::<(), llama::flush_buffer::BufferError>(())
    /// ```
    pub fn write_payload<'a>(
        &self,
        payload: &[u8],
        reservation: Reservation<'a>,
    ) -> Result<BufferMsg, BufferError> {
        let (current, offset) = { (reservation.buffer, reservation.offset) };
        self.buffer.put(*current, Ok(offset), payload)
    }

    /// Reserve a contiguous byte range in the currently active [`FlushBuffer`].
    ///
    /// The reservation atomically claims `payload_size` bytes and returns an
    /// opaque [`Reservation`] that must be passed to [`write_payload`](Self::write_payload).
    ///
    /// # Errors
    ///
    /// * [`BufferError::InvalidState`] — the current buffer pointer is null
    ///   (should not happen in normal operation).
    /// * [`BufferError::EncounteredSealedBuffer`] — the buffer was sealed
    ///   concurrently; the ring has already rotated and the caller should retry.
    ///
    /// # Safety
    ///
    /// Internally dereferences the ring's `current_buffer` atomic pointer.
    /// This is safe because the ring guarantees the pointer is valid and
    /// non-null for the lifetime of the store.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// let reservation = store.reserve_space(64)?;
    /// // ... prepare payload ...
    /// store.write_payload(&[0u8; 64], reservation)?;
    /// # Ok::<(), llama::flush_buffer::BufferError>(())
    /// ```
    pub fn reserve_space(&self, payload_size: usize) -> Result<Reservation, BufferError> {
        let current = unsafe {
            self.buffer
                .current_buffer
                .load(std::sync::atomic::Ordering::Acquire)
                .as_ref()
        }
        .ok_or(BufferError::InvalidState)?;

        match current.reserve_space(payload_size) {
            Ok(offset) => Ok(Reservation {
                buffer: Arc::new(current),
                offset,
            }),
            Err(e) => Err(e),
        }
    }

    /// Poll the `io_uring` completion queue and process all available CQEs.
    ///
    /// 1. **Detects write failures** (`result < 0`) and re-submits the original
    ///    SQE.  The slot is *not* marked complete until the retry succeeds,
    ///    which prevents `hi_stable` from advancing past a broken slot.
    /// 2. **Marks successful writes** via [`mark_slot_complete`](Self::mark_slot_complete)
    ///    and resets the buffer for reuse.
    ///
    /// This method is designed to be called on the hot write path — once per
    /// write or on a short polling interval — rather than on a dedicated thread.
    ///
    /// # Panics
    ///
    /// Panics if `user_data` is a non-null, non-sentinel value that does not
    /// point to a valid [`FlushBuffer`] (i.e. memory has been corrupted).
    pub fn check_async_cque(&self) {
        // Collect CQEs and drop the io_uring lock before processing them to
        // avoid holding it while we potentially re-submit SQEs.
        let cqes: Vec<cqueue::Entry> = {
            let mut ring = self.flusher.get_cqueue();
            ring.completion().sync();
            ring.completion().collect()
        };

        for cqe in cqes {
            let user_data = cqe.user_data();

            // Write completion 
            let ptr = user_data as *const FlushBuffer;
            let buffer: &FlushBuffer = unsafe { &*ptr };
            let lss_slot = buffer.local_lss_address_slot.load(Ordering::Acquire) as u64;

            if cqe.result() < FAILED_FLUSH {
                // Re-fire the original SQE; slot stays incomplete until retry succeeds.
                let sqe = unsafe {
                    (*buffer.submit_queue_entry.get())
                        .as_ref()
                        .expect("stored SQE must be present on retry")
                };

                let mut ring = self.flusher.get_cqueue();
                unsafe {
                    let _ = ring.submission().push(&sqe);
                };
                let _ = ring.submit();
            } else {
                // Success — advance stability tracking and recycle the buffer.
                self.mark_slot_complete(lss_slot);
                self.buffer.reset_buffer(buffer);
            }
        }
    }

    /// Record that `lss_slot` has been durably written and advance [`hi_stable`](Self::hi_stable)
    /// as far as contiguity allows.
    ///
    /// # Algorithm
    ///
    /// * If `lss_slot == hi_stable + 1` the slot is the *exact* next expected
    ///   completion.  A CAS advances `hi_stable` and then
    ///   [`advance_high_stable`](Self::advance_high_stable) drains any islands
    ///   that have since become contiguous.
    /// * If `lss_slot > hi_stable + 1` a gap exists — earlier slots have not
    ///   yet completed.  The slot is inserted into `completed_islands` and
    ///   `hi_stable` is left unchanged.
    /// * If `lss_slot <= hi_stable` the slot is already covered; it is removed
    ///   from `completed_islands` defensively and the method returns.
    ///
    /// The CAS loop retries on contention so that concurrent completions on
    /// different threads always converge.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// // Simulated out-of-order completions: slot 2 arrives before slot 1.
    /// store.mark_slot_complete(2);
    /// assert_eq!(store.hi_stable.load(std::sync::atomic::Ordering::Acquire), 0);
    ///
    /// store.mark_slot_complete(1);
    /// assert_eq!(store.hi_stable.load(std::sync::atomic::Ordering::Acquire), 2);
    /// ```
    pub fn mark_slot_complete(&self, lss_slot: u64) {
        loop {
            let current = self.hi_stable.load(Ordering::Acquire);

            if lss_slot <= current {
                self.completed_islands.write().remove(&lss_slot);
                return;
            }

            if lss_slot == current + 1 {
                match self.hi_stable.compare_exchange(
                    current,
                    lss_slot,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        self.completed_islands.write().remove(&lss_slot);
                        self.advance_high_stable();
                        return;
                    }
                    Err(_) => continue, // CAS lost — reload and retry.
                }
            } else {
                // Gap detected — park the slot until its predecessor arrives.
                self.completed_islands.write().insert(lss_slot);
                return;
            }
        }
    }

    /// Drain `completed_islands` and advance [`hi_stable`](Self::hi_stable) as far
    /// as contiguity allows.
    ///
    /// Called by [`mark_slot_complete`](Self::mark_slot_complete) after each
    /// successful CAS.  Loops until the next expected slot is absent from the
    /// island set or another thread wins the CAS race.
    pub fn advance_high_stable(&self) {
        loop {
            let current = self.hi_stable.load(Ordering::Acquire);
            let next_expected = current + 1;

            // Hold the read lock only long enough to check membership.
            let found = self.completed_islands.read().contains(&next_expected);
            if !found {
                return;
            }

            match self.hi_stable.compare_exchange(
                current,
                next_expected,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.completed_islands.write().remove(&next_expected);
                    // Continue: there may be further contiguous islands.
                }
                Err(_) => continue, // Another thread advanced; re-check.
            }
        }
    }

    /// Return the highest contiguous LSS **byte offset** that has been durably
    /// flushed to secondary storage.
    ///
    /// Callers that need a slot number should read [`hi_stable`](Self::hi_stable)
    /// directly.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// let durable_up_to = store.get_high_stable_offset();
    /// println!("data is durable up to byte offset {durable_up_to}");
    /// ```
    pub(crate) fn get_high_stable_offset(&self) -> u64 {
        self.hi_stable.load(Ordering::Acquire) * (FOUR_KB_PAGE as u64)
    }

    /// Return the number of out-of-order completed slots currently parked in the
    /// island set.
    ///
    /// Primarily useful for diagnostics and tests.  Under normal operation this
    /// value should stay close to zero; a growing island count may indicate that
    /// a slot is consistently failing to complete.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// if store.get_island_count() > 16 {
    ///     tracing::warn!("large island backlog — possible stuck flush");
    /// }
    /// ```
    pub(crate) fn get_island_count(&self) -> usize {
        self.completed_islands.read().len()
    }

    /// Return a cloned [`Arc`] handle to the backing file.
    ///
    /// Useful when constructing a [`FlushBehavior`] separately that must share
    /// the same file descriptor as the store.
    pub(crate) fn file_handle(&self) -> Arc<File> {
        Arc::clone(&self.store)
    }

    /// Return a shared reference to the currently active [`FlushBuffer`].
    ///
    /// # Safety
    ///
    /// Dereferences the ring's `current_buffer` atomic pointer.  The ring
    /// guarantees this pointer is valid and non-null for the lifetime of the
    /// store.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the pointer is null (indicates ring corruption).
    pub(crate) fn get_cur_buffer(&self) -> &FlushBuffer {
        unsafe {
            self.buffer
                .current_buffer
                .load(std::sync::atomic::Ordering::Acquire)
                .as_ref()
                .unwrap()
        }
    }

    /// Initiate an **asynchronous** flush of the currently active buffer.
    ///
    /// Returns immediately; use [`check_async_cque`](Self::check_async_cque) to
    /// observe completion.  For synchronous semantics use
    /// [`flush_cur_buffer_blocking`](Self::flush_cur_buffer_blocking).
    pub(crate) fn flush_cur_buffer(&self) {
        let cur = self.get_cur_buffer();
        self.buffer.flush(cur);
    }

    /// Flush the currently active buffer and **block** until the write completes.
    ///
    /// Suitable for tests and orderly shutdown.  Avoid on the hot write path —
    /// prefer the asynchronous [`flush_cur_buffer`](Self::flush_cur_buffer) instead.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the underlying `io_uring` submission or
    /// completion fails.
    pub(crate) fn flush_cur_buffer_blocking(&self) -> io::Result<()> {
        let cur = self.get_cur_buffer();
        cur.set_flush_in_progress();
        self.flusher.submit_buffer_and_wait(cur)
    }
}

// helpers

/// Open `path` with `O_RDWR | O_CREAT | O_DIRECT`, creating parent directories
/// as needed.
///
/// # Errors
///
/// Returns an [`io::Error`] on any filesystem or permission failure.
fn open_direct(path: impl AsRef<Path>) -> io::Result<File> {
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }

    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // O_DIRECT bypasses the kernel page cache.
        // INVARIANT: every buffer passed to read/write must be aligned to
        // FOUR_KB_PAGE — upheld by Buffer::new_aligned in flush_buffer.rs.
        .custom_flags(libc::O_DIRECT)
        .open(path.as_ref())
}
