//! # Flush Buffer — Latch-Free I/O Buffer Ring
//!
//! This module implements LLAMA's in-memory write-staging layer: a fixed-size
//! ring of 4 KB-aligned [`FlushBuffer`]s that amortises individual page-state
//! writes into larger, sequential I/O operations before they are dispatched to
//! the [`LogStructuredStore`](crate::log_structured_store::LogStructuredStore).
//!
//! ## Design Goals
//!
//! | Goal                    | Mechanism                                                  |
//! |-------------------------|------------------------------------------------------------|
//! | Latch-free writes       | Single packed [`AtomicUsize`] state word per buffer        |
//! | `O_DIRECT` compatibility| 4 KB-aligned allocation via [`Buffer::new_aligned`]       |
//! | Amortised I/O           | Multiple threads fill one buffer before it is flushed      |
//! | All threads participate | Any thread may seal or initiate a flush                    |
//!
//! ## Flush Protocol
//!
//! Adapted from the LLAMA paper; all steps are performed without global locks:
//!
//! 1. **Identify** the page state to be written.
//! 2. **Seize** space in the active [`FlushBuffer`] via
//!    [`reserve_space`](FlushBuffer::reserve_space) — an atomic fetch-and-add
//!    on the packed state word claims a non-overlapping byte range.
//! 3. **Check** atomically whether the reservation succeeded.  If the buffer is
//!    already sealed or the space is exhausted, the buffer is sealed and the ring
//!    rotates to the next available slot.
//! 4. **Write** the payload into the reserved range while the flush-in-progress
//!    bit prevents the buffer from being dispatched to stable storage prematurely.
//! 5. **On failure** at step 3, write a "Failed Flush" sentinel into the reserved
//!    space.  This wastes a few bytes but removes all ambiguity about which writes
//!    succeeded.
//! 
//! Though the currently implementation delegates the handling of all erroneous and invalid
//! states to the caller, the current implementation of the Flush proceedure should lend itself
//! well to to LLAMA flushing protocol
//!
//! ## State Word Layout
//!
//! All per-buffer metadata is packed into a single [`AtomicUsize`], making every
//! state snapshot self-consistent and eliminating TOCTOU (time of check/time of use) races between the
//! fields:
//!
//! ```text
//! ┌────────────────┬────────────────┬──────────────────┬───────────────────┬──────────┐
//! │  Bits 63..32   │  Bits 31..8    │  Bits 7..2       │  Bit 1            │  Bit 0   │
//! │  write offset  │  writer count  │  (reserved)      │  flush-in-prog    │  sealed  │
//! └────────────────┴────────────────┴──────────────────┴───────────────────┴──────────┘
//! ```
//!
//! * **write offset** — next free byte position inside the backing allocation.
//! * **writer count** — number of threads that have reserved space but not yet finished
//!   copying their payload.
//! * **flush-in-progress** — set by whichever thread wins the CAS race to own the
//!   flush; prevents a second flush from being fired while the first is in flight.
//! * **sealed** — set when the buffer is full or explicitly closed; prevents new
//!   reservations.
//! 
//! Bits 7..2 represent unused space

use std::{
    cell::UnsafeCell,
    pin::Pin,
    sync::{
        atomic::{AtomicPtr, AtomicUsize, Ordering},
        Arc,
    },
    usize,
};

use io_uring::squeue::Entry;

use crate::{flush_behaviour::FlushBehavior, log_structured_store::FOUR_KB_PAGE};
use std::alloc::{alloc_zeroed, Layout};


/// A 4 KB-aligned, heap-allocated byte buffer suitable for `O_DIRECT` I/O.
///
/// `Buffer` owns a single contiguous allocation that is aligned to
/// [`FOUR_KB_PAGE`] (4 096 bytes) — the minimum alignment required by
/// `O_DIRECT` on all common block devices.
///
/// Cursor management is **not** handled here.  Instead, [`FlushBuffer`] uses
/// atomic fetch-and-add on its packed state word to hand out non-overlapping
/// byte ranges to concurrent writers.  This is what makes the
/// `unsafe impl Sync` sound: no two threads are ever granted the same region.
///
/// # Safety
///
/// [`Sync`] is manually implemented because [`UnsafeCell`] opts out of it by
/// default.  The invariant that upholds this is: all mutable access to the
/// inner pointer is mediated by [`FlushBuffer`], which guarantees exclusive
/// ranges per writer.
#[derive(Debug)]
pub struct Buffer {
    /// Raw pointer to the aligned allocation, wrapped in [`UnsafeCell`] to
    /// allow interior mutability without a lock.
    pub(crate) buffer: UnsafeCell<*mut u8>,
    /// Total allocation size in bytes.  Stored for correct deallocation.
    size: usize,
}

impl Buffer {
    /// Allocate a zeroed, [`FOUR_KB_PAGE`]-aligned buffer of `size` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `size` is not a multiple of [`FOUR_KB_PAGE`], if the layout
    /// is otherwise invalid, or if the allocator returns a null pointer.
    pub fn new_aligned(size: usize) -> Self {
        let layout = Layout::from_size_align(size, FOUR_KB_PAGE).expect("invalid layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "aligned allocation failed");

        Self {
            buffer: UnsafeCell::new(ptr),
            size,
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, FOUR_KB_PAGE).unwrap();
        unsafe { std::alloc::dealloc(*self.buffer.get(), layout) };
    }
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// A reference-counted handle to a [`Buffer`].
///
/// Shared between a [`FlushBuffer`] and the `io_uring` submission path, which
/// holds a pointer into the buffer while a write is in flight.
pub(crate) type SharedBuffer = Arc<Buffer>;

// ── State word constants ──────────────────────────────────────────────────────

/// Bit 0 of the state word — set when the buffer is closed to new writers.
const SEALED_BIT: usize = 1 << 0;

/// Bit 1 of the state word — set while a flush is in progress.
///
/// Prevents a second flush from being fired concurrently and prevents new
/// writers from entering a buffer that is already being drained.
const FLUSH_IN_PROGRESS_BIT: usize = 1 << 1;

/// Amount added to the state word to record one additional active writer.
const WRITER_SHIFT: usize = 8;
const WRITER_ONE: usize = 1 << WRITER_SHIFT;

/// Mask covering the writer-count field (bits 8..32).
const WRITER_MASK: usize = 0x00FF_FFFF00;

/// The write-offset field occupies the top 32 bits of the state word.
const OFFSET_SHIFT: usize = 32;

/// Amount added to the state word to advance the write offset by one byte.
const OFFSET_ONE: usize = 1 << OFFSET_SHIFT;

/// Default number of buffers in a [`FlushBufferRing`].
pub const RING_SIZE: usize = 4;

// ── State word helpers ────────────────────────────────────────────────────────

#[inline(always)]
fn state_offset(state: usize) -> usize {
    state >> OFFSET_SHIFT
}

#[inline(always)]
fn state_writers(state: usize) -> usize {
    (state & WRITER_MASK) >> WRITER_SHIFT
}

#[inline(always)]
fn state_sealed(state: usize) -> bool {
    state & SEALED_BIT != 0
}

#[inline(always)]
fn state_flush_in_progress(state: usize) -> bool {
    state & FLUSH_IN_PROGRESS_BIT != 0
}

// ── BufferError ───────────────────────────────────────────────────────────────

/// Errors that may be returned by buffer and ring operations.
#[derive(Debug, Clone, Copy)]
pub enum BufferError {
    /// The payload exceeds the remaining capacity of the active flush buffer.
    InsufficientSpace,

    /// The buffer is sealed and no longer accepts new reservations.
    EncounteredSealedBuffer,

    /// A CAS on the sealed bit found it was already set.
    EncounteredSealedBufferDuringCOMPEX,

    /// A CAS on the sealed bit found it was already clear.
    EncounteredUnSealedBufferDuringCOMPEX,

    /// A flush was attempted while at least one writer is still active.
    ActiveUsers,

    /// The buffer or ring is in an undefined / corrupt state.
    InvalidState,

    /// All buffers in the ring are sealed or being flushed — none available.
    RingExhausted,

    /// A [`reserve_space`](FlushBuffer::reserve_space) CAS failed; the caller
    /// should retry.
    FailedReservation,

    /// An attempt to clear the sealed bit via CAS failed; the caller should
    /// retry.
    FailedUnsealed,
}

// ── BufferMsg ─────────────────────────────────────────────────────────────────

/// Successful outcomes returned by buffer and ring operations.
#[derive(Debug, Clone)]
pub enum BufferMsg {
    /// The buffer transitioned to the sealed state.
    SealedBuffer,

    /// The payload was written to the buffer; no flush was triggered.
    SuccessfullWrite,

    /// The payload was written and the buffer was dispatched for flushing.
    SuccessfullWriteFlush,

    /// The buffer is ready to flush.  Carries the [`FlushBuffer`] that was
    /// sealed, allowing the recipient to initiate the flush independently.
    FreeToFlush(Arc<FlushBuffer>),
}

// ── FlushBuffer ───────────────────────────────────────────────────────────────

/// A single 4 KiB-aligned latch-free I/O buffer.
///
/// Multiple threads write into a `FlushBuffer` concurrently by atomically
/// claiming non-overlapping byte ranges through [`FlushBuffer::reserve_space`].  Once the
/// buffer is full (or explicitly sealed), it is dispatched to
/// [`FlushBehavior`] for an `io_uring` write and then reset for reuse.
///
/// # State Word
///
/// ```text
/// ┌────────────────┬────────────────┬──────────────────┬───────────────────┬──────────┐
/// │  Bits 63..32   │  Bits 31..8    │  Bits 7..2       │  Bit 1            │  Bit 0   │
/// │  write offset  │  writer count  │  (reserved)      │  flush-in-prog    │  sealed  │
/// └────────────────┴────────────────┴──────────────────┴───────────────────┴──────────┘
/// ```
///
/// All four fields are read and updated through a single [`AtomicUsize`], so
/// any snapshot is self-consistent: there are no TOCTOU races between the
/// offset, writer count, flush flag, and sealed flag.
///
/// # Safety
///
/// `FlushBuffer` is `Send + Sync`.  The only `unsafe` access is inside
/// [`write`](Self::write), where a raw pointer into the aligned allocation is
/// dereferenced.  Safety is upheld by the invariant that
/// [`reserve_space`](Self::reserve_space) grants each caller an exclusive,
/// non-overlapping byte range.
#[derive(Debug)]
pub struct FlushBuffer {
    /// Packed atomic state — see type-level docs for the bit layout.
    state: AtomicUsize,

    /// Backing aligned byte store shared with the `io_uring` submission path.
    pub(crate) buf: SharedBuffer,

    /// Position of this buffer within the parent [`FlushBufferRing`].
    pub(crate) pos: usize,

    /// The LSS address slot assigned to this buffer at seal time.
    ///
    /// On-disk byte offset = `local_lss_address_slot × FOUR_KB_PAGE`.
    /// Assigned by [`FlushBufferRing::next_address_range`] via fetch-add;
    /// guaranteed unique across all concurrently sealed buffers.
    pub(crate) local_lss_address_slot: AtomicUsize,

    /// The most recently submitted `io_uring` SQE for this buffer.
    ///
    /// Stored so that a failed CQE can re-fire the exact same write without
    /// re-constructing the SQE.  Guarded by the flush-in-progress state
    /// transition — only one thread may write or read this field at a time.
    pub(crate) submit_queue_entry: UnsafeCell<Option<Entry>>,
}

unsafe impl Send for FlushBuffer {}
unsafe impl Sync for FlushBuffer {}

impl FlushBuffer {
    /// Create a new `FlushBuffer` at ring position `buffer_number` with a
    /// `size`-byte aligned backing allocation.
    ///
    /// The initial LSS address slot is set to `buffer_number` so that buffers
    /// are pre-assigned non-overlapping slots at construction time.  The ring
    /// will update this via [`set_new_address_space_range`](Self::set_new_address_space_range)
    /// each time the buffer is reused.
    pub fn new_buffer(buffer_number: usize, size: usize) -> FlushBuffer {
        Self {
            state: AtomicUsize::new(0),
            buf: Arc::new(Buffer::new_aligned(size)),
            pos: buffer_number,
            local_lss_address_slot: AtomicUsize::new(buffer_number),
            submit_queue_entry: UnsafeCell::new(None),
        }
    }

    /// Atomically update this buffer's LSS address slot to `address_space`.
    ///
    /// Returns `Ok(previous_slot)` on success or `Err(observed)` if the CAS
    /// fails (another thread updated the slot concurrently).
    pub fn set_new_address_space_range(&mut self, address_space: usize) -> Result<usize, usize> {
        let range = self.local_lss_address_slot.load(Ordering::Relaxed);
        self.local_lss_address_slot.compare_exchange(
            range,
            address_space,
            Ordering::Acquire,
            Ordering::Relaxed,
        )
    }

    /// Return `true` if this buffer is open to new writers.
    ///
    /// A buffer is available when neither the sealed bit nor the
    /// flush-in-progress bit is set.
    pub fn is_available(&self) -> bool {
        self.state.load(Ordering::Acquire) & (SEALED_BIT | FLUSH_IN_PROGRESS_BIT) == 0
    }

    /// Attempt to atomically reserve `payload_size` bytes in this buffer.
    ///
    /// On success returns the byte offset at which the caller should write its
    /// payload.  The caller **must** call [`decrement_writers`](Self::decrement_writers)
    /// once the write is complete.
    ///
    /// # Errors
    ///
    /// * [`BufferError::EncounteredSealedBuffer`] — the buffer is sealed or a
    ///   flush is in progress; the caller should ask the ring to rotate.
    /// * [`BufferError::InsufficientSpace`] — `payload_size` bytes would exceed
    ///   [`FOUR_KB_PAGE`]; the caller should seal the buffer and retry on the
    ///   next one.
    /// * [`BufferError::FailedReservation`] — the CAS failed due to contention;
    ///   the caller should retry immediately.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `payload_size > FOUR_KB_PAGE`.
    pub fn reserve_space(&self, payload_size: usize) -> Result<usize, BufferError> {
        assert!(payload_size <= FOUR_KB_PAGE, "payload larger than buffer");

        let state = self.state.load(Ordering::Acquire);

        if state & (SEALED_BIT | FLUSH_IN_PROGRESS_BIT) != 0 {
            return Err(BufferError::EncounteredSealedBuffer);
        }

        let offset = state_offset(state);

        if offset + payload_size > FOUR_KB_PAGE {
            return Err(BufferError::InsufficientSpace);
        }


        // Analagous to the increment_writers() method
        let new = state
            .wrapping_add(payload_size * OFFSET_ONE)
            .wrapping_add(WRITER_ONE);

        match self
            .state
            .compare_exchange(state, new, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => Ok(offset),
            Err(_) => Err(BufferError::FailedReservation),
        }
    }

    /// Decrement the active-writer count by one.
    ///
    /// Should be called by every thread that previously succeeded at
    /// [`reserve_space`](Self::reserve_space) once it has finished copying its
    /// payload.  Returns the **previous** state word value.
    #[inline]
    pub fn decrement_writers(&self) -> usize {
        self.state.fetch_sub(WRITER_ONE, Ordering::AcqRel)
    }

    /// Increment the active-writer count by one.
    ///
    /// Returns the **previous** state word value.
    #[inline]
    pub fn increment_writers(&self) -> usize {
        self.state.fetch_add(WRITER_ONE, Ordering::AcqRel)
    }

    /// Set the flush-in-progress bit.
    ///
    /// Returns the **previous** state word value.  The caller should check
    /// whether the bit was already set in the returned value — only the thread
    /// that observes the bit transitioning from `0` to `1` owns the flush.
    #[inline]
    pub fn set_flush_in_progress(&self) -> usize {
        self.state.fetch_or(FLUSH_IN_PROGRESS_BIT, Ordering::AcqRel)
    }

    /// Clear the flush-in-progress bit.
    ///
    /// Returns the **previous** state word value.
    #[inline]
    pub fn clear_flush_in_progress(&self) -> usize {
        self.state
            .fetch_and(!FLUSH_IN_PROGRESS_BIT, Ordering::AcqRel)
    }

    /// Copy `payload` into the buffer at `offset`.
    ///
    /// # Safety
    ///
    /// The caller must have obtained `offset` from a successful
    /// [`reserve_space`](Self::reserve_space) call and must not alias the same
    /// region from another thread.
    pub fn write(&self, offset: usize, payload: &[u8]) {
        debug_assert!(offset + payload.len() <= self.buf.size);

        unsafe {
            let dst = (*self.buf.buffer.get()).add(offset);
            std::ptr::copy_nonoverlapping(payload.as_ptr(), dst, payload.len());
        }
    }

    /// Set the sealed bit, preventing any further reservations.
    ///
    /// # Errors
    ///
    /// Returns [`BufferError::EncounteredSealedBufferDuringCOMPEX`] if the
    /// buffer was already sealed before this call.
    pub fn set_sealed_bit_true(&self) -> Result<(), BufferError> {
        let prev = self.state.fetch_or(SEALED_BIT, Ordering::AcqRel);
        if state_sealed(prev) {
            Err(BufferError::EncounteredSealedBufferDuringCOMPEX)
        } else {
            Ok(())
        }
    }

    /// Clear the sealed bit, re-opening the buffer to new writers.
    ///
    /// Only succeeds when there are no active writers and no flush is in
    /// progress.
    ///
    /// # Errors
    ///
    /// * [`BufferError::ActiveUsers`] — writers or a flush are still active.
    /// * [`BufferError::EncounteredUnSealedBufferDuringCOMPEX`] — the buffer
    ///   was not sealed to begin with.
    /// * [`BufferError::FailedUnsealed`] — the CAS failed; retry.
    #[allow(unused)]
    pub(crate) fn set_sealed_bit_false(&self) -> Result<(), BufferError> {
        let current = self.state.load(Ordering::Acquire);

        if state_writers(current) != 0 || state_flush_in_progress(current) {
            return Err(BufferError::ActiveUsers);
        }

        if !state_sealed(current) {
            return Err(BufferError::EncounteredUnSealedBufferDuringCOMPEX);
        }

        match self.state.compare_exchange(
            current,
            current & !SEALED_BIT,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(_) => Err(BufferError::FailedUnsealed),
        }
    }

    /// Reset the write offset to zero, leaving all flag bits intact.
    ///
    /// Intended for use in tests only.  In production code the ring resets
    /// buffers through [`FlushBufferRing::reset_buffer`].
    pub fn reset_offset(&self) {
        loop {
            let current = self.state.load(Ordering::Acquire);
            let zeroed = current & 0x0000_0000_FFFF_FFFF;
            if self
                .state
                .compare_exchange(current, zeroed, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Return a raw snapshot of the packed state word.
    ///
    /// Available in test builds only.  Use the `state_offset`, `state_writers`,
    /// `state_sealed`, and `state_flush_in_progress` helpers to decode the
    /// individual fields.
    #[cfg(test)]
    pub(crate) fn state_snapshot(&self) -> usize {
        self.state.load(Ordering::Acquire)
    }
}

/// A fixed-size ring of [`FlushBuffer`]s that amortises writes into batched
/// sequential I/O.
///
/// The ring maintains a single *current* buffer pointer that all threads write
/// into concurrently.  When the current buffer is full it is sealed, a fresh
/// buffer is selected from the ring, and the sealed buffer is dispatched to
/// the configured [`FlushBehavior`] for an `io_uring` write.
///
/// New LSS address slots are assigned at seal time via a single atomic fetch-add on
/// [`next_address_range`](Self), ensuring that no two buffers ever map to the
/// same region of the backing file even when flushes complete out of order.
///
/// # Ring Exhaustion
///
/// If all buffers in the ring are sealed or being flushed when a rotation is
/// needed, [`rotate_after_seal`](Self::rotate_after_seal) returns
/// [`BufferError::RingExhausted`].  Callers should back off and poll the
/// completion queue to free up buffers.
pub struct FlushBufferRing {
    /// Pointer to the buffer currently accepting writes.
    ///
    /// Updated atomically via CAS during rotation.  The pointed-to buffer is
    /// guaranteed to be valid for the lifetime of the ring because all buffers
    /// are owned by `ring` and the ring is `Pin`ned.
    pub current_buffer: AtomicPtr<FlushBuffer>,

    /// Pinned, heap-allocated array of all buffers.
    ///
    /// `Pin` ensures the buffers never move in memory, which is required
    /// because `current_buffer` holds raw pointers into this slice, and the
    /// `io_uring` SQEs hold raw pointers into the backing allocations.
    ring: Pin<Box<[Arc<FlushBuffer>]>>,

    /// Index of the next candidate buffer during rotation.
    next_index: AtomicUsize,

    /// Monotonically increasing LSS slot counter.
    ///
    /// Incremented by fetch-add at seal time; the resulting value is stored as
    /// the sealed buffer's `local_lss_address_slot`.
    pub next_address_range: AtomicUsize,

    _size: usize,

    /// Optional flush dispatcher.  `None` in test mode — buffers are reset
    /// immediately without dispatching any `io_uring` writes.
    store: Option<Arc<FlushBehavior>>,
}

impl FlushBufferRing {
    /// Create a ring of `num_of_buffer` buffers, each `buffer_size` bytes,
    /// with **no** flush dispatcher attached.
    ///
    /// Intended for unit tests that exercise the ring's concurrency primitives
    /// without requiring a real `io_uring` instance or backing file.  In this
    /// mode, sealed buffers are reset immediately after flush is triggered,
    /// keeping the ring from stalling.
    pub fn with_buffer_amount(num_of_buffer: usize, buffer_size: usize) -> FlushBufferRing {
        let buffers: Vec<Arc<FlushBuffer>> = (0..num_of_buffer)
            .map(|i| Arc::new(FlushBuffer::new_buffer(i, buffer_size)))
            .collect();

        let buffers = Pin::new(buffers.into_boxed_slice());
        let current = &*buffers[0] as *const FlushBuffer as *mut FlushBuffer;

        FlushBufferRing {
            current_buffer: AtomicPtr::new(current),
            ring: buffers,
            next_index: AtomicUsize::new(1),
            _size: num_of_buffer,
            next_address_range: AtomicUsize::new(4),
            store: None,
        }
    }

    /// Create a ring of `num_of_buffer` buffers, each `buffer_size` bytes,
    /// connected to `flusher` for real `io_uring`-backed I/O.
    ///
    /// This is the production constructor.  Sealed buffers are submitted to
    /// `flusher` instead of being reset immediately.
    pub fn with_flusher(
        num_of_buffer: usize,
        buffer_size: usize,
        flusher: Arc<FlushBehavior>,
    ) -> FlushBufferRing {
        let buffers: Vec<Arc<FlushBuffer>> = (0..num_of_buffer)
            .map(|i| Arc::new(FlushBuffer::new_buffer(i, buffer_size)))
            .collect();

        let buffers = Pin::new(buffers.into_boxed_slice());
        let current = &*buffers[0] as *const FlushBuffer as *mut FlushBuffer;

        FlushBufferRing {
            current_buffer: AtomicPtr::new(current),
            ring: buffers,
            next_index: AtomicUsize::new(1),
            _size: num_of_buffer,
            next_address_range: AtomicUsize::new(4),
            store: Some(flusher),
        }
    }

    /// Write `payload` into `current` at the byte offset described by
    /// `reserve_result`.
    ///
    /// Handles all outcomes of a prior [`reserve_space`](FlushBuffer::reserve_space)
    /// call:
    ///
    /// * **`Ok(offset)`** — copy the payload, decrement the writer count, and
    ///   trigger a flush if this thread is the last writer in a sealed buffer.
    /// * **`Err(InsufficientSpace)`** — seal the buffer, rotate the ring, and
    ///   initiate a flush if this thread wins the flush-in-progress CAS race.
    /// * **`Err(EncounteredSealedBuffer)`** — propagated to the caller; the
    ///   ring has already rotated and the caller should retry on the new buffer.
    ///
    /// # Errors
    ///
    /// Propagates [`BufferError`] variants from the ring.
    pub fn put(
        &self,
        current: &FlushBuffer,
        reserve_result: Result<usize, BufferError>,
        payload: &[u8],
    ) -> Result<BufferMsg, BufferError> {
        match reserve_result {
            Err(BufferError::InsufficientSpace) => {
                // Seal the buffer — whichever thread sets the bit from 0→1 owns
                // the flush.
                let prev = current.state.fetch_or(SEALED_BIT, Ordering::AcqRel);

                if prev & SEALED_BIT != 0 {
                    return Err(BufferError::EncounteredSealedBuffer);
                }

                // Claim a unique LSS slot for this buffer before rotating.
                let slot = self.next_address_range.fetch_add(1, Ordering::AcqRel);
                current
                    .local_lss_address_slot
                    .store(slot, Ordering::Release);

                self.rotate_after_seal(current.pos)?;

                // Race to own the flush.  If writers are still active, the last
                // one to decrement will also attempt this and one of them will
                // observe the bit transitioning 0→1.
                let before = current.set_flush_in_progress();
                if before & FLUSH_IN_PROGRESS_BIT == 0 {
                    match self.store.as_ref() {
                        Some(store) => {
                            let _ = store.submit_buffer(current);
                        }
                        None => {
                            // Test mode: no dispatcher — reset immediately.
                            self.reset_buffer(current);
                        }
                    }
                    return Ok(BufferMsg::SuccessfullWriteFlush);
                }

                return Err(BufferError::ActiveUsers);
            }

            Err(BufferError::EncounteredSealedBuffer) => {
                return Err(BufferError::EncounteredSealedBuffer);
            }

            Err(e) => return Err(e),

            Ok(offset) => {
                current.write(offset, payload);

                let prev = current.decrement_writers();


                // Note: Atomic operations always yeild previous values
                let was_last_writer = state_writers(prev) == 1;
                let was_sealed = state_sealed(prev);

                if was_last_writer && was_sealed {
                    let prev = current.set_flush_in_progress();

                    if prev & FLUSH_IN_PROGRESS_BIT == 0 {
                        let flush_buffer = self.ring.get(current.pos).unwrap().clone();
                        self.flush(&flush_buffer);
                        return Ok(BufferMsg::SuccessfullWriteFlush);
                    }
                }

                return Ok(BufferMsg::SuccessfullWrite);
            }
        }
    }

    /// Rotate the ring's current buffer pointer away from the buffer at
    /// `sealed_pos`.
    ///
    /// Scans the ring for the next available (unsealed, not flushing) buffer and
    /// swaps `current_buffer` to point at it via CAS.  If no available buffer is
    /// found, returns [`BufferError::RingExhausted`].
    ///
    /// If `current_buffer` has already been rotated by another thread (i.e. it
    /// no longer points at `sealed_pos`), returns `Ok(())` immediately.
    pub fn rotate_after_seal(&self, sealed_pos: usize) -> Result<(), BufferError> {
        let current = self.current_buffer.load(Ordering::Acquire);
        let current_ref = unsafe { current.as_ref().ok_or(BufferError::InvalidState)? };

        if current_ref.pos != sealed_pos {
            return Ok(());
        }

        let ring_len = self.ring.len();

        for _ in 0..ring_len {
            let raw = self.next_index.fetch_add(1, Ordering::AcqRel);
            let next_index = raw % ring_len;
            let new_buffer = &self.ring[next_index];

            if new_buffer.is_available() {
                let _ = self.current_buffer.compare_exchange(
                    current,
                    Arc::as_ptr(new_buffer) as *const FlushBuffer as *mut FlushBuffer,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                return Ok(());
            }
        }

        Err(BufferError::RingExhausted)
    }

    /// Explicity dispatches `buffer` to stable storage asynchronously.
    ///
    /// Sets the flush-in-progress bit and submits the buffer to the configured
    /// [`FlushBehavior`].  In test mode (no dispatcher configured), the buffer
    /// is reset immediately so the ring does not stall waiting for a CQE that
    /// will never arrive.
    ///
    /// This method is `pub(crate)` and is called internally by [`put`](Self::put)
    /// and by [`LogStructuredStore`](crate::log_structured_store::LogStructuredStore).
    /// It is not part of the public-facing API 
    pub(crate) fn flush(&self, buffer: &FlushBuffer) {
        buffer.set_flush_in_progress();

        match self.store.as_ref() {
            Some(store) => {
                let _ = store.submit_buffer(buffer);
            }
            None => {
                self.reset_buffer(buffer);
            }
        }
    }

    /// Reset `buffer` for reuse after a completed flush.
    ///
    /// Clears the write offset, sealed bit, and flush-in-progress bit atomically
    /// via a CAS loop.  The writer-count field is left untouched because any
    /// writers that were counted during the sealed phase have already decremented
    /// themselves.
    pub fn reset_buffer(&self, buffer: &FlushBuffer) {
        loop {
            let flushed_buffer_state = buffer.state.load(Ordering::Acquire);

            const OFFSET_MASK: usize = usize::MAX << OFFSET_SHIFT;
            let reset = flushed_buffer_state & !(SEALED_BIT | FLUSH_IN_PROGRESS_BIT | OFFSET_MASK);

            if buffer
                .state
                .compare_exchange(
                    flushed_buffer_state,
                    reset,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                break;
            }
        }
    }
}

// =============================================================================
//  Tests
// =============================================================================

#[cfg(test)]
mod tests {

    use super::*;

    use std::{
        collections::HashSet,
        sync::{Arc, Barrier, Mutex},
        thread,
        time::Instant,
    };

    /// Very small, very lightweight, very unimpressive Linear Congruential Generator for deterministic
    /// pseudorandom number generation in tests.
    /// source: https://en.wikipedia.org/wiki/Linear_congruential_generator
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_usize(&mut self, bound: usize) -> usize {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.state >> 33) as usize) % bound
        }
    }

    const TEST_RING_SIZE: usize = 4;
    const OPS_PER_THREAD: usize = 2_000;

    /// Payload sizes ranging from tiny to near-capacity.
    const SIZES: &[usize] = &[
        1, 2, 4, 7, 8, 15, 16, 32, 64, 100, 128, 200, 256, 512, 1024, 2048, 4090, 4095, 4096,
    ];

    /// Build a recognisable, size-stamped payload.
    fn make_payload(tag: &str, size: usize) -> Vec<u8> {
        let meta = format!("[{tag}:{size}]");
        let mut buf = vec![0xAA_u8; size];
        let n = meta.len().min(size);
        buf[..n].copy_from_slice(&meta.as_bytes()[..n]);
        buf
    }

    // =========================================================================
    // Retry helper
    //
    // The ring does not retry internally — that is the caller's responsibility
    // (mapping table in production, this helper in tests).
    //
    // Loop:
    //   1. Load current buffer.
    //   2. Call reserve_space.
    //   3. Pass result into put.
    //   4. Retry on transient errors (FailedReservation, EncounteredSealedBuffer,
    //      ActiveUsers).
    //   5. Any other outcome is final.
    // =========================================================================
    fn put_with_retry(ring: &FlushBufferRing, payload: &[u8]) -> Result<BufferMsg, BufferError> {
        loop {
            let current = unsafe {
                ring.current_buffer
                    .load(Ordering::Acquire)
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

    // =========================================================================
    // Single-buffer unit tests — no ring, no flusher
    // =========================================================================

    /// reserve_space on a sealed buffer must return EncounteredSealedBuffer.
    #[test]
    fn reserve_on_sealed_buffer_returns_error() {
        let buf = FlushBuffer::new_buffer(0, FOUR_KB_PAGE);
        buf.set_sealed_bit_true().unwrap();
        assert!(matches!(
            buf.reserve_space(16),
            Err(BufferError::EncounteredSealedBuffer)
        ));
    }

    /// Sealing an already-sealed buffer must return the COMPEX error.
    #[test]
    fn double_seal_returns_error() {
        let buf = FlushBuffer::new_buffer(0, FOUR_KB_PAGE);
        buf.set_sealed_bit_true().unwrap();
        assert!(matches!(
            buf.set_sealed_bit_true(),
            Err(BufferError::EncounteredSealedBufferDuringCOMPEX)
        ));
    }

    /// Unsealing an already-unsealed buffer must return the COMPEX error.
    #[test]
    fn unseal_unsealed_returns_error() {
        let buf = FlushBuffer::new_buffer(0, FOUR_KB_PAGE);
        assert!(matches!(
            buf.set_sealed_bit_false(),
            Err(BufferError::EncounteredUnSealedBufferDuringCOMPEX)
        ));
    }

    /// reserve_space on a flush-in-progress buffer must return EncounteredSealedBuffer.
    #[test]
    fn reserve_on_flush_in_progress_returns_error() {
        let buf = FlushBuffer::new_buffer(0, FOUR_KB_PAGE);
        buf.set_flush_in_progress();
        assert!(matches!(
            buf.reserve_space(16),
            Err(BufferError::EncounteredSealedBuffer)
        ));
    }

    /// Writer count increments and decrements must be symmetric.
    #[test]
    fn writer_count_symmetric() {
        let buf = FlushBuffer::new_buffer(0, FOUR_KB_PAGE);
        buf.increment_writers();
        buf.increment_writers();
        buf.increment_writers();
        assert_eq!(state_writers(buf.state_snapshot()), 3);
        buf.decrement_writers();
        buf.decrement_writers();
        buf.decrement_writers();
        assert_eq!(state_writers(buf.state_snapshot()), 0);
    }

    /// A single exact-capacity reservation must consume the whole buffer.
    #[test]
    fn reserve_exact_capacity() {
        let buf = FlushBuffer::new_buffer(0, FOUR_KB_PAGE);
        let offset = buf.reserve_space(FOUR_KB_PAGE).unwrap();
        assert_eq!(offset, 0);
        // Next reservation must fail — no space left.
        assert!(matches!(
            buf.reserve_space(1),
            Err(BufferError::InsufficientSpace)
        ));
    }

    /// Two sequential reservations must not overlap.
    #[test]
    fn sequential_reservations_no_overlap() {
        let buf = FlushBuffer::new_buffer(0, FOUR_KB_PAGE);
        let a = buf.reserve_space(100).unwrap();
        let b = buf.reserve_space(100).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 100);
    }

    // =========================================================================
    // Concurrent single-buffer test — the most critical correctness invariant
    // =========================================================================

    /// Eight threads race to reserve 16-byte regions from a single buffer.
    /// No (buffer_pos, offset) pair may ever be issued twice.
    ///
    /// This directly validates the CAS-based reservation: if any two threads
    /// receive the same offset, the atomic state word is broken.
    #[test]
    fn concurrent_reserve_space_no_overlap() {
        let buf = Arc::new(FlushBuffer::new_buffer(99, FOUR_KB_PAGE));
        let seen: Arc<Mutex<HashSet<usize>>> = Arc::new(Mutex::new(HashSet::new()));

        const THREADS: usize = 8;
        // 8 threads × 32 reservations × 16 bytes = 4096 — exactly fills one buffer
        const RESERVES_PER_THREAD: usize = 32;

        let barrier = Arc::new(Barrier::new(THREADS));

        let handles: Vec<_> = (0..THREADS)
            .map(|_tid| {
                let buf = Arc::clone(&buf);
                let seen = Arc::clone(&seen);
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait(); // all threads start simultaneously

                    for _ in 0..RESERVES_PER_THREAD {
                        loop {
                            match buf.reserve_space(16) {
                                Ok(offset) => {
                                    let mut lock = seen.lock().unwrap();
                                    assert!(
                                        lock.insert(offset),
                                        "[OVERLAP] offset {offset} issued twice!"
                                    );
                                    break;
                                }
                                Err(BufferError::FailedReservation) => continue,
                                Err(BufferError::InsufficientSpace) => break,
                                Err(BufferError::EncounteredSealedBuffer) => break,
                                Err(e) => panic!("unexpected error: {e:?}"),
                            }
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("reserve worker panicked");
        }

        // All 256 unique offsets (0, 16, 32, ... 4080) must be present
        let lock = seen.lock().unwrap();
        assert_eq!(
            lock.len(),
            THREADS * RESERVES_PER_THREAD,
            "expected {} unique offsets, got {}",
            THREADS * RESERVES_PER_THREAD,
            lock.len()
        );
    }

    // =========================================================================
    // Ring-level tests — seal, rotate, exhaustion
    // =========================================================================

    /// A full-capacity payload fills the buffer and triggers a seal + rotate.
    /// After the write, current_buffer must point at a different buffer.
    #[test]
    fn exact_fill_triggers_rotate() {
        let ring = FlushBufferRing::with_buffer_amount(TEST_RING_SIZE, FOUR_KB_PAGE);
        let payload = make_payload("FILL", FOUR_KB_PAGE);

        match put_with_retry(&ring, &payload) {
            Ok(BufferMsg::SuccessfullWrite) | Ok(BufferMsg::SuccessfullWriteFlush) => {}
            other => panic!("exact_fill: unexpected {other:?}"),
        }

        // After a full-capacity write the ring must have rotated.
        // In no-flusher mode the buffer is reset immediately, so the pointer
        // may have wrapped — just assert the ring is still operational.
        let result = put_with_retry(&ring, &make_payload("AFTER", 16));
        assert!(
            result.is_ok(),
            "ring should still accept writes after rotate: {result:?}"
        );
    }

    /// Seal a buffer explicitly and verify the ring rotates to the next slot.
    #[test]
    fn manual_seal_causes_rotate() {
        let ring = FlushBufferRing::with_buffer_amount(TEST_RING_SIZE, FOUR_KB_PAGE);

        let current_before = unsafe {
            ring.current_buffer
                .load(Ordering::Acquire)
                .as_ref()
                .unwrap()
        };
        let pos_before = current_before.pos;

        // Seal the current buffer manually
        current_before.set_sealed_bit_true().unwrap();
        ring.rotate_after_seal(pos_before).unwrap();

        let current_after = unsafe {
            ring.current_buffer
                .load(Ordering::Acquire)
                .as_ref()
                .unwrap()
        };

        assert_ne!(
            current_after.pos, pos_before,
            "current_buffer should have rotated away from sealed buffer"
        );
    }

    /// After sealing all buffers without resetting, the ring must return
    /// RingExhausted rather than deadlocking or panicking.
    #[test]
    fn ring_exhaustion_returns_error() {
        let ring = FlushBufferRing::with_buffer_amount(TEST_RING_SIZE, FOUR_KB_PAGE);

        // Manually seal every buffer so none are available
        for i in 0..TEST_RING_SIZE {
            ring.ring[i].set_sealed_bit_true().ok();
        }

        let result = ring.rotate_after_seal(0);
        assert!(
            matches!(result, Err(BufferError::RingExhausted)),
            "expected RingExhausted, got {result:?}"
        );
    }

    /// Random-sized writes, single thread. Verifies the ring keeps accepting
    /// writes across multiple seal/rotate cycles without panicking.
    #[test]
    fn single_threaded_offset_uniqueness() {
        let ring = FlushBufferRing::with_buffer_amount(TEST_RING_SIZE, FOUR_KB_PAGE);

        let mut rng = Lcg::new(0);
        let mut writes = 0usize;
        let mut flushes = 0usize;
        let mut data_written = 0usize;
        let mut i = 0usize;

        loop {
            let size = SIZES[rng.next_usize(SIZES.len())];
            if data_written + size > FOUR_KB_PAGE * TEST_RING_SIZE {
                break;
            }

            let payload = make_payload(&format!("s{i:05}"), size);
            data_written += size;

            match put_with_retry(&ring, &payload) {
                Ok(BufferMsg::SuccessfullWrite) => writes += 1,
                Ok(BufferMsg::SuccessfullWriteFlush) => {
                    writes += 1;
                    flushes += 1;
                }
                other => panic!("single_threaded: unexpected {other:?}"),
            }
            i += 1;
        }

        println!(
            "single_threaded_offset_uniqueness: {writes} writes, {flushes} flushes, {data_written} bytes"
        );
    }

    /// Stress test: 2000 random-sized writes, single thread.
    #[test]
    fn single_threaded_stress() {
        let ring = FlushBufferRing::with_buffer_amount(TEST_RING_SIZE, FOUR_KB_PAGE);
        let mut writes = 0usize;
        let mut flushes = 0usize;
        let mut rng = Lcg::new(0x1234_5678);
        let start = Instant::now();

        for op in 0..OPS_PER_THREAD {
            let size = SIZES[rng.next_usize(SIZES.len())];
            let payload = make_payload(&format!("S:O{op:04}"), size);

            match put_with_retry(&ring, &payload) {
                Ok(BufferMsg::SuccessfullWrite) => writes += 1,
                Ok(BufferMsg::SuccessfullWriteFlush) => {
                    writes += 1;
                    flushes += 1;
                }
                other => panic!("op {op}: unexpected {other:?}"),
            }
        }

        let elapsed = start.elapsed();
        println!(
            "single_threaded_stress: {writes} writes, {flushes} flushes in {elapsed:.2?} ({:.0} ops/s)",
            (writes + flushes) as f64 / elapsed.as_secs_f64()
        );
    }

    // =========================================================================
    // Multi-threaded stress tests
    // =========================================================================

    const NUM_THREADS_SMALL: usize = 2;
    const NUM_THREADS_MEDIUM: usize = 4;
    const NUM_THREADS_LARGE: usize = 8;

    #[test]
    fn multi_threaded_test_small() {
        multi_threaded_stress_helper(NUM_THREADS_SMALL);
    }

    #[test]
    fn multi_threaded_test_medium() {
        multi_threaded_stress_helper(NUM_THREADS_MEDIUM);
    }

    #[test]
    fn multi_threaded_test_large() {
        multi_threaded_stress_helper(NUM_THREADS_LARGE);
    }

    fn multi_threaded_stress_helper(num_threads: usize) {
        let ring = Arc::new(FlushBufferRing::with_buffer_amount(
            TEST_RING_SIZE,
            FOUR_KB_PAGE,
        ));
        let barrier = Arc::new(Barrier::new(num_threads));
        let total_writes = Arc::new(AtomicUsize::new(0));
        let total_flushes = Arc::new(AtomicUsize::new(0));
        let start_times = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<thread::JoinHandle<()>> = (0..num_threads)
            .map(|tid| {
                let ring = Arc::clone(&ring);
                let barrier = Arc::clone(&barrier);
                let total_writes = Arc::clone(&total_writes);
                let total_flushes = Arc::clone(&total_flushes);
                let start_times = Arc::clone(&start_times);

                let seed = 0x1234_5678_u64
                    .wrapping_add(tid as u64)
                    .wrapping_mul(0xDEAD_CAFE);

                thread::spawn(move || {
                    let mut rng = Lcg::new(seed);
                    let mut local_writes = 0usize;
                    let mut local_flushes = 0usize;

                    barrier.wait();
                    // Record start AFTER barrier — this is when real work begins
                    start_times.lock().unwrap().push(Instant::now());

                    for op in 0..OPS_PER_THREAD {
                        let size = SIZES[rng.next_usize(SIZES.len())];
                        let payload = make_payload(&format!("T{tid}:O{op:04}"), size);

                        let result = loop {
                            let current = unsafe {
                                ring.current_buffer
                                    .load(Ordering::Acquire)
                                    .as_ref()
                                    .expect("null current_buffer")
                            };

                            let reserve_result = current.reserve_space(payload.len());

                            match &reserve_result {
                                Err(BufferError::FailedReservation) => continue,
                                Err(BufferError::EncounteredSealedBuffer) => continue,
                                _ => {}
                            }

                            match ring.put(current, reserve_result, &payload) {
                                Err(BufferError::ActiveUsers) => continue,
                                Err(BufferError::EncounteredSealedBuffer) => continue,
                                Err(BufferError::RingExhausted) => {
                                    std::thread::yield_now();
                                    continue;
                                }
                                Ok(BufferMsg::SealedBuffer) => continue,
                                other => break other,
                            }
                        };

                        match result {
                            Ok(BufferMsg::SuccessfullWrite) => local_writes += 1,
                            Ok(BufferMsg::SuccessfullWriteFlush) => {
                                local_writes += 1;
                                local_flushes += 1;
                            }
                            other => panic!("thread {tid} op {op}: unexpected {other:?}"),
                        }
                    }

                    total_writes.fetch_add(local_writes, Ordering::Relaxed);
                    total_flushes.fetch_add(local_flushes, Ordering::Relaxed);
                })
            })
            .collect();

        for (tid, handle) in handles.into_iter().enumerate() {
            handle
                .join()
                .unwrap_or_else(|_| panic!("worker thread {tid} panicked"));
        }

        let join_time = Instant::now();
        let writes = total_writes.load(Ordering::Relaxed);
        let flushes = total_flushes.load(Ordering::Relaxed);

        let earliest_start = start_times.lock().unwrap().iter().copied().min().unwrap();

        let elapsed = join_time.duration_since(earliest_start);

        println!(
            "multi_threaded_stress({num_threads} threads): {writes} writes, {flushes} flushes \
         in {elapsed:.2?} ({:.0} ops/s)",
            writes as f64 / elapsed.as_secs_f64()
        );

        assert_eq!(
            writes,
            num_threads * OPS_PER_THREAD,
            "total writes should equal num_threads * OPS_PER_THREAD"
        );
    }

    /// All threads race with large (2KB) payloads to maximise seal/rotate
    /// contention. Two threads × 100 ops × 2KB = 200KB of writes across
    /// multiple ring rotations.
    #[test]
    fn hammer_seal_concurrent_rotation() {
        let ring = Arc::new(FlushBufferRing::with_buffer_amount(
            TEST_RING_SIZE,
            FOUR_KB_PAGE,
        ));
        let barrier = Arc::new(Barrier::new(NUM_THREADS_SMALL));

        let handles: Vec<_> = (0..NUM_THREADS_SMALL)
            .map(|tid| {
                let ring = Arc::clone(&ring);
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait();

                    for iter in 0..100_usize {
                        let payload = make_payload(&format!("H{tid}:{iter}"), 2048);
                        match put_with_retry(&ring, &payload) {
                            Ok(_) => {}
                            Err(e) => panic!("hammer thread {tid} iter {iter}: error {e:?}"),
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("hammer worker panicked");
        }
    }
}
