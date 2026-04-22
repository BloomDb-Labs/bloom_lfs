//! # Log-Structured Store (`lss`)
//!
//! The [`LogStructuredStore`] is the single durable backing file for LLAMA's page store.
//! All writes flow through a [`FlushBufferRing`] and are dispatched to the underlying
//! file via [`QuikIO`] (either tail-localised or serialised `io_uring` writes).
//!
//! ## Design Goals
//!
//! | Goal                     | Mechanism                                                                |
//! |--------------------------|--------------------------------------------------------------------------|
//! | High write throughput    | Amortised batch writes via [`FlushBufferRing`]                           |
//! | Low write amplification  | Sequential / tail-localised layout on disk                               |
//! | Kernel-cache bypass      | `O_DIRECT` — I/O lands directly on the device                            |
//! | Async I/O                | `io_uring` dispatched through [`QuikIO`]                                 |

//! ## Architecture Overview
//!
//! ```text
//!                        User API (Public)
//!                              │
//!                              │
//!                        writer functions
//!                              │
//!                 ┌────────────┴────────────┐
//!                 │                         │
//!         FlushBufferRing              QuikIO
//!         (staging area)               (io_uring dispatch)
//!                 │                         │
//!                 └────────────┬────────────┘
//!                              │
//!                      O_DIRECT File I/O
//!                              │
//!                      LSS Backing File
//! ```
//!
//!
//! ## Critical Insight: FlushBufferRing Responsibilities
//!
//! The [`FlushBufferRing::put`] method is **already a complete write coordinator**:
//!
//! *  Handles sealing when buffer is full
//! *  Assigns unique LSS slots atomically
//! *  Rotates the ring to the next buffer
//! *  Submits async flushes via `io_uring`
//! *  Coordinates the "last writer" flush trigger
//!
//!
//! ## Address Space Layout
//!
//! Each [`FlushBuffer`] is assigned a unique `local_address` at seal time.
//! The on-disk byte offset is:
//!
//! ```text
//! byte_offset = local_address
//! ```
//!
//! Slots are handed out by incrementing the next [`FlushBufferRing::next_address_range`] via a single
//! atomic fetch-add, guaranteeing that no two sealed buffers ever map to the same
//! region of the file — even when their flushes complete out of order.
//!
//! ## Tail-Localised vs. Serialised Writes
//!
//! Because slots are claimed at seal time but flushed concurrently, a slower buffer
//! may land on disk *after* a buffer with a higher slot index. The maximum write
//! distance from the logical tail is bounded by:
//!
//! ```text
//! max_distance = RING_SIZE × FlushBufferSize
//! ```
//!
//! For workloads that require strict append order (e.g. WAL segments),
//! [`QuikIO::Searalized`] uses `IO_LINK` to serialise submissions.
//!
//! ## Stability Tracking
//!
//! [`LogStructuredStore::hi_stable`] records the highest *contiguous* LSS slot that
//! has been durably written to secondary storage. Out-of-order completions are held
//! in `completed_islands` until their predecessors arrive, at which point
//! [`mark_slot_complete`](LogStructuredStore::mark_slot_complete) walks the island set
//! and advances `hi_stable` as far as possible.
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
//! which aligns all allocations to [`ONE_MEGABYTE_BLOCK`].
//!
//!
//!
//!

pub mod test;

use std::{
    fs::OpenOptions,
    io,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    u64,
};

// use buffer_ring::flush_behaviour::{FlushableBuffer, WriteMode};
use buffer_ring::{
    BufferError, BufferRing, BufferRingOptions, FOUR_KB_BLOCK, FlushBuffer, ONE_MEGABYTE_BLOCK,
    QuikIO, RING_SIZE, WriteMode,
};
use crossbeam_channel::{Receiver, Sender, unbounded};
use dashmap::DashSet;
use parking_lot::RwLock;

pub(crate) type Slot = u64;

pub(crate) type Offset = usize;

pub(crate) type FileOffset = u64;
pub(crate) type ByteCount = usize;

pub struct LogStructuredStore {
    /// Internal buffer ring
    ///
    pub(crate) buffer_ring: BufferRing,

    /// Internal I/O dispatcher
    pub(crate) dispatcher: Arc<QuikIO>,

    /// The highest LSS slot such that **every** slot in `0..=hi_stable` has been
    /// durably written to secondary storage.
    ///
    /// Advanced by [`mark_slot_complete`](Self::mark_slot_complete) and
    /// [`advance_high_stable`](Self::advance_high_stable) as CQEs arrive.
    pub hi_stable: AtomicU64,

    /// Out-of-order completions waiting for their predecessors.
    ///
    /// When slot *N* completes but slot *N - i* has not yet completed, *N* is
    /// placed here. Once *N - i* completes and `hi_stable` reaches *N - 1*,
    /// the advancement loop drains the island set forward.
    ///
    pub(crate) completed_islands: DashSet<u64>,

    pub(crate) persisted_slot: RwLock<Option<crossbeam_channel::Receiver<(Slot, Offset)>>>,
}

impl LogStructuredStore {
    /// Completely halts the completion queue untill all previous entries have succeeded
    /// sucessfully while using the BufferRing'ss provided interface
    ///
    ///
    /// Drains all available CQEs and:
    ///
    /// * Retries failed writes
    /// * Marks successful slots complete
    /// * Advances `hi_stable`
    /// * Resets buffers for reuse
    ///
    fn _halt_cq_with_receiver__(&self) -> Result<(), String> {
        if let Some(persisted_slot) = &**self.persisted_slot.try_read().as_mut().expect("msg") {
            match self.buffer_ring.check_cque() {
                Ok(_) => {
                    for (slot, _) in persisted_slot.try_iter() {
                        println!("Slot {:?}", slot);
                        self._mk_slt_cmplt__(slot);
                    }

                    return Ok(());
                }
                Err(err) => return Err(err),
            }
        }
        Err("Receiver not Pressent".to_owned())
    }

    /// Completely halts the completion queue untill all previous entries have succeeded
    /// sucessfully
    ///
    ///
    /// Drains all available CQEs and:
    ///
    /// * Retries failed writes
    /// * Marks successful slots complete
    /// * Advances `hi_stable`
    /// * Resets buffers for reuse
    ///
    fn _halt_cq__(&self) -> Result<(), ()> {
        loop {
            let cqes = self.dispatcher.cqe();
            if cqes.is_empty() {
                return Ok(());
            }

            for cqe in cqes {
                let user_data = cqe.user_data();

                if user_data == 0 {
                    // Skip fsync entries
                    continue;
                }

                let ptr = cqe.user_data() as *const FlushBuffer;
                let buffer: &FlushBuffer = unsafe { &*ptr };

                if cqe.result() < 0 {
                    // Retry failed write
                    let sqe = unsafe { buffer.sqe().get().as_mut() }
                        .expect("stored SQE must be present on retry");
                    let mut ring = self.dispatcher.ring();
                    unsafe {
                        let _ = ring.submission().push(sqe.as_ref().expect("Got refernce"));
                    }
                    let _ = ring.submit();
                } else {
                    let lss_slot = buffer.local_address(Ordering::Acquire) as u64;

                    // Success - mark complete and reset
                    self._mk_slt_cmplt__(lss_slot);
                    self.buffer_ring.reset_buffer(buffer);
                }
            }

            let mut ring = self.dispatcher.ring();
            if ring.submission().len() == 0 && ring.completion().len() == 0 {
                break;
            }
            std::thread::yield_now();
        }

        Ok(())
    }

    /// Record that `lss_slot` has been durably written and advance
    /// [`hi_stable`](Self::hi_stable) as far as contiguity allows.
    ///
    /// ## Algorithm
    ///
    /// * If `lss_slot == hi_stable + 1` → it's the next expected completion
    ///   * CAS to advance `hi_stable`
    ///   * Drain any islands that are now contiguous
    ///
    /// * If `lss_slot > hi_stable + 1` → there's a gap
    ///   * Insert into `completed_islands`
    ///   * Leave `hi_stable` unchanged
    ///
    /// * If `lss_slot <= hi_stable` → already covered
    ///   * Remove from `completed_islands` (defensive)
    ///   * Return
    ///
    ///
    ///
    fn _mk_slt_cmplt__(&self, lss_slot: u64) {
        loop {
            let current = self.hi_stable.load(Ordering::Acquire);

            if current == u64::MAX {
                // No state is currently durable
                if lss_slot == 0 {
                    // We found our first durable slot
                    match self.hi_stable.compare_exchange(
                        current,
                        lss_slot,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => self._adv_hi_stable__(),
                        Err(_) => continue,
                    }
                } else {
                    // We found an island, a non contiguous durable slot
                    self.completed_islands.insert(lss_slot);
                    return;
                }
            }

            if lss_slot <= current {
                // Not sure if this will actually happen
                self.completed_islands.remove(&lss_slot);
                return;
            }

            if lss_slot == current + 1 {
                // We found the N + 1 stable slot!
                match self.hi_stable.compare_exchange(
                    current,
                    lss_slot,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        self.completed_islands.remove(&lss_slot);
                        self._adv_hi_stable__();
                        return;
                    }
                    Err(_) => continue,
                }
            } else {
                // Another Island found
                self.completed_islands.insert(lss_slot);
                return;
            }
        }
    }

    /// Drain `completed_islands` and advance [`hi_stable`](Self::hi_stable) as far
    /// as contiguity allows.
    fn _adv_hi_stable__(&self) {
        loop {
            let current = self.hi_stable.load(Ordering::Acquire);
            let next_expected = current + 1;

            // Didn't de-fragment the stability list
            let found = self.completed_islands.contains(&next_expected);
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
                    // A hole was patched!
                    self.completed_islands.remove(&next_expected);
                }
                Err(_) => {
                    // Some other thread patched the hole!
                    continue;
                }
            }
        }
    }
}

pub struct Reservation<'a> {
    /// Snapshot of the buffer's offset at reserve time
    offset: usize,

    /// Pointer to the buffer
    reserved: &'a FlushBuffer,

    /// Payload to be writter
    payload: &'a [u8],
}
impl LogStructuredStore {
    /// ensures that ll previous file io operations are gurantees to be be durable on disk  
    pub fn sync(&self) {
        let _ = self._halt_cq_with_receiver__();

        // Ensures all previous flushed data is durabble
        let _ = self.dispatcher.sync_data();

        let _ = self.buffer_ring.check_cque();
    }

    /// Attempts to atomically reserve payload_size bytes within the currently active buffer.
    pub fn reserve<'a>(&self, payload: &'a [u8]) -> Result<Reservation<'a>, BufferError> {
        let cur = self.buffer_ring.current_buffer(Ordering::Acquire);

        let offset = cur.reserve_space(payload.len())?;

        Ok(Reservation {
            offset,
            reserved: cur,
            payload,
        })
    }

    ///
    pub fn try_write(&self, reservation: Reservation) -> Result<usize, BufferError> {
        loop {
            let current = reservation.reserved;
            let data = reservation.payload;
            let offset = reservation.offset;

            match self.buffer_ring.put(current, Ok(offset), data) {
                Ok(_) => {
                    let lss_location = current.local_address(Ordering::Acquire) + offset;
                    return Ok(lss_location);
                }

                Err(BufferError::EncounteredSealedBuffer) | Err(BufferError::RingExhausted) => {
                    let _ = self._halt_cq_with_receiver__();

                    std::thread::yield_now();
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }

    ///
    pub fn write(&self, reservation: Reservation) -> Result<usize, BufferError> {
        let current = reservation.reserved;
        let data = reservation.payload;
        let offset = reservation.offset;

        match self.buffer_ring.put(current, Ok(offset), data) {
            Ok(_) => {
                let lss_location = current.local_address(Ordering::Acquire) + offset;
                return Ok(lss_location);
            }

            Err(e) => return Err(e),
        }
    }

    /// Dispatches an asynchronous flush and rotates the current buffer pointer away from the current position
    ///
    /// Current buffer must contain some valid data
    ///
    pub fn roll(&self) -> Result<String, BufferError> {
        self.buffer_ring.flush_current()?;
        Ok("Flushed".to_string())
    }

    // Read data from a specifc LSS offset into a mutable buffer
    ///
    /// # Arguments
    /// * `dst` - dst must be valid for writes of len number of  bytes.
    /// * `byte_offset` - The LSS offset number to read from
    /// * `len` - Number of bytes to read
    ///
    /// # Returns
    /// The data read from the offset.
    ///
    /// # Errors
    /// Returns `io::Error` if the read fails.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    ///
    /// let buffer = vec![0u8; 4096];
    /// let record_offset = 0;
    /// let data = store.copy_from(buffer.as_mut_ptr(), record_offset , 4096)?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn read(&self, dst: *mut u8, len: usize, offset: u64) -> io::Result<()> {
        use std::alloc::{Layout, alloc_zeroed};
        let layout = Layout::from_size_align(4096, 4096).unwrap();
        let temp = unsafe { alloc_zeroed(layout) };

        // self.read(temp, 4096, 0).expect("Read"); // read full 4KB block

        self.dispatcher.read(temp, FOUR_KB_BLOCK, offset)?;

        unsafe {
            dst.copy_from(temp, len);
        };

        io::Result::Ok(())
    }

    /// Get the highest contiguous slot that is durable.
    /// All slots from 0 to this value (inclusive) are guaranteed to be
    /// on stable storage.
    ///
    /// A slot is a [`FlushBuffer`] sized address block within the LogStructured store
    ///
    /// Not all too dissimilar from the OS page cache, the LSS keeps keeps in-flight
    /// data in pinned memory reigions called buffers. Unlike, page caches these memory
    /// regions are specifically used collect records payloads into a contiguous buffer to allow
    /// for batched processing ie, all at once write, all at once durability.
    ///
    /// These batches map directly to a slot. In fact it can be interpreted that the FlushBufferRing
    /// direclty writes buffers to fill lss slots
    ///
    /// For this reason, The lss system alone cannot gurantee durability on an induvidual record level.
    ///
    /// ```
    pub fn stable_slot(&self) -> u64 {
        self.hi_stable.load(Ordering::Acquire)
    }

    /// Check if a specific offset is durable by comparing it to the starting offset
    /// of the most stable slot.
    ///
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// let record_offset = store.write_async(b"data")?;
    ///
    /// // Later...
    /// if store.is_durable(record_offset) {
    ///     println!("This record is on disk");
    /// }
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn is_durable(&self, record_offset: u64) -> bool {
        record_offset <= self.hi_stable.load(Ordering::Acquire)
    }
}

/// Builder for opening or creating a [`LogStructuredStore`].
///
/// Follows the same pattern as [`std::fs::OpenOptions`]: accumulate flags with
/// chained setters, open at the end.  
/// # Examples
///
/// ```rust
/// let store = LssOpenOptions::new()
///     .write(true)
///     .read(true)
///     .create(true)
///     .write_mode(WriteMode::TailLocalizedWrites)
///     .open("/var/lib/llama/data.lss")?;
/// ```
pub struct LssConfig {
    pub(crate) read: bool,
    pub(crate) write: bool,
    pub(crate) create: bool,
    pub(crate) write_mode: WriteMode,
    pub(crate) publisher: Option<Sender<(Slot, Offset)>>,
    pub(crate) receiver: bool,
}

impl LssConfig {
    pub fn new() -> Self {
        Self {
            read: false,
            write: false,
            create: false,
            write_mode: WriteMode::TailLocalizedWrites,
            publisher: None,
            receiver: false,
        }
    }
    pub fn read(&mut self, v: bool) -> &mut Self {
        self.read = v;
        self
    }
    pub fn write(&mut self, v: bool) -> &mut Self {
        self.write = v;
        self
    }
    pub fn create(&mut self, v: bool) -> &mut Self {
        self.create = v;
        self
    }

    pub fn write_mode(&mut self, m: WriteMode) -> &mut Self {
        self.write_mode = m;
        self
    }

    /// Sets up a single producer single sonsumer communication channel. The store uses
    /// this channel to publish durably commited slots to a receiver
    pub fn with_publusher(&mut self) -> Receiver<(u64, usize)> {
        let (sender, receriver) = unbounded();
        self.publisher = Some(sender);
        receriver
    }

    // Flag for setting it persistent write completion channels for low-level writers
    pub fn interior_receiver(&mut self) -> &mut Self {
        self.receiver = true;
        self
    }

    /// Open (or create) the store with the accumulated options.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the file cannot be opened or created, or if
    /// the `io_uring` instance cannot be initialised (requires Linux ≥ 5.1).
    pub fn open(&mut self, path: impl AsRef<Path>) -> io::Result<LogStructuredStore> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = Arc::new(
            OpenOptions::new()
                .read(self.read)
                .write(self.write)
                .create(self.create)
                // O_DIRECT bypasses the kernel page cache.
                // INVARIANT: every buffer passed to read/write must be aligned to
                // FOUR_KB_PAGE — upheld by Buffer::new_aligned in flush_buffer.rs.
                .custom_flags(libc::O_DIRECT)
                .open(path.as_ref())?,
        );

        let flusher = Arc::new(match self.write_mode {
            WriteMode::TailLocalizedWrites => QuikIO::new(file.clone()),
            WriteMode::SerializedWrites => QuikIO::link(file.clone()),
        });

        let mut options = BufferRingOptions::new();
        options
            .auto_flush(true)
            .auto_rotate(true)
            .buffer_size(ONE_MEGABYTE_BLOCK)
            .capacity(RING_SIZE)
            .io_instance(flusher.clone());

        let mut receiver = None;
        if self.receiver {
            // VERY BAD IDEA. FIX

            let std_rx = options.completion_receiver(); // std::sync::mpsc::Receiver
            let (tx, rx) = crossbeam_channel::unbounded();
            std::thread::spawn(move || {
                for msg in std_rx {
                    let _ = tx.send(msg);
                }
            });
            receiver = Some(rx);
        }

        let ring = BufferRing::with_options(&mut options);

        Ok(LogStructuredStore {
            buffer_ring: ring,
            dispatcher: flusher,
            hi_stable: AtomicU64::new(u64::MAX),
            completed_islands: DashSet::new(),
            // durable_slots: self.publisher.take(),
            persisted_slot: receiver.into(),
        })
    }

    pub fn with_buffer_options(
        &mut self,
        io: Arc<QuikIO>,
        mut ring_op: BufferRingOptions,
    ) -> io::Result<LogStructuredStore> {
        let ring = BufferRing::with_options(&mut ring_op);

        let mut receiver = None;
        if self.receiver {
            // VERY BAD IDEA. FIX
            let std_rx = ring_op.completion_receiver(); // std::sync::mpsc::Receiver
            let (tx, rx) = crossbeam_channel::unbounded();
            std::thread::spawn(move || {
                for msg in std_rx {
                    let _ = tx.send(msg);
                }
            });
            receiver = Some(rx);
        }

        Ok(LogStructuredStore {
            buffer_ring: ring,
            dispatcher: io,
            hi_stable: AtomicU64::new(u64::MAX),
            completed_islands: DashSet::new(),
            // durable_slots: self.publisher.take(),
            persisted_slot: receiver.into(),
        })
    }
}
