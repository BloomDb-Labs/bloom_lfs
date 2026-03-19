//! # Log-Structured Store - Clean Public API
//!
//! This refactored version provides a simple, high-level API while hiding
//! all the buffer management, ring rotation, and flush plumbing from users.

use std::{
    collections::BTreeSet,
    fs::File,
    io,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use io_uring::cqueue;
use parking_lot::RwLock;

use crate::{
    flush_behaviour::{FlushBehavior, FOUR_KB_PAGE},
    flush_buffer::{
        state_offset, BufferError, FlushBuffer, FlushBufferRing, FLUSH_IN_PROGRESS_BIT, MD,
    },
};

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API - This is what users interact with
// ═══════════════════════════════════════════════════════════════════════════

/// The Log-Structured Store — LLAMA's single durable backing file.
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
/// // Simple synchronous write
/// let slot = store.write(&b"hello, LLAMA")?;
///
/// // Async write (fire and forget)
/// let slot = store.write_async(&b"hello, LLAMA")?;
/// store.poll_completions(); // Drive completions forward
///
/// // Read from a specific slot
/// let data = store.(slot, 12)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct LogStructuredStore {
    /// Internal buffer ring (hidden from users)
    pub(crate) buffer: FlushBufferRing,
    /// Internal flush dispatcher (hidden from users)
    pub(crate) flusher: Arc<FlushBehavior>,
    /// Internal file handle (hidden from users)
    pub(crate) store: Arc<File>,

    // Public stability tracking
    pub hi_stable: AtomicU64,

    /// Internal completion tracking (hidden from users)
    pub(crate) completed_islands: RwLock<BTreeSet<u64>>,
}

impl LogStructuredStore {
    /// Write data asynchronously without waiting for completion.
    ///
    /// Returns immediately after submitting the write. Call `poll_completions()`
    /// periodically to drive I/O forward.
    ///
    /// # Returns
    /// The LSS slot number where the data will be written.
    ///
    /// # Errors
    /// Returns `io::Error` if the write submission fails.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// let slot = store.write_async(b"my data")?;
    /// // ... do other work ...
    /// store.poll_completions(); // Drive I/O
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn write_async(&self, data: &[u8]) -> io::Result<u64> {
        self.internal_write(data)
    }

    /// Read data from a specific LSS slot.
    ///
    /// # Arguments
    /// * `slot` - The LSS slot number to read from
    /// * `len` - Number of bytes to read
    ///
    /// # Returns
    /// The data read from the slot.
    ///
    /// # Errors
    /// Returns `io::Error` if the read fails.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// let data = store.read_at_slot(42, 1024)?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn read_at_slot(&self, byte_offset: u64, len: usize) -> io::Result<Vec<u8>> {
        self.internal_read(byte_offset, len)
    }

    /// Poll the completion queue to drive async I/O forward.
    ///
    /// Call this periodically when using `write_async()` to process
    /// completions and advance the stability tracking.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// // Submit many async writes
    /// for i in 0..1000 {
    ///     store.write_async(b"data")?;
    /// }
    ///
    /// // Drive completions
    /// store.poll_completions();
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn poll_completions(&self) {
        let _ = self.check_async_cque();
    }

    // ─────────────────────────────────────────────────────────────────────
    // STABILITY TRACKING API
    // ─────────────────────────────────────────────────────────────────────

    /// Get the highest contiguous slot that is durable.
    ///
    /// All slots from 0 to this value (inclusive) are guaranteed to be
    /// on stable storage.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// let stable_slot = store.stable_slot();
    /// println!("All data up to slot {} is durable", stable_slot);
    /// ```
    pub fn stable_slot(&self) -> u64 {
        self.hi_stable.load(Ordering::Acquire)
    }

    /// Get the highest contiguous byte offset that is durable.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// let stable_bytes = store.stable_offset();
    /// println!("All data up to byte {} is durable", stable_bytes);
    /// ```
    pub fn stable_offset(&self) -> u64 {
        let s = self.hi_stable.load(Ordering::Acquire);
        if s == u64::MAX {
            return 0;
        }
        s * FOUR_KB_PAGE as u64
    }

    /// Check if a specific slot is durable.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// let slot = store.write_async(b"data")?;
    ///
    /// // Later...
    /// if store.is_durable(slot) {
    ///     println!("Slot {} is on disk", slot);
    /// }
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn is_durable(&self, slot: u64) -> bool {
        slot <= self.hi_stable.load(Ordering::Acquire)
    }

    /// Get the number of out-of-order completions waiting for predecessors.
    ///
    /// Useful for diagnostics. A growing value may indicate stuck flushes.
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use llama::log_structured_store::LogStructuredStore;
    /// # let store: LogStructuredStore = todo!();
    /// if store.pending_completions() > 16 {
    ///     println!("Warning: large backlog of pending completions");
    /// }
    /// ```
    pub fn pending_completions(&self) -> usize {
        self.completed_islands.read().len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL IMPLEMENTATION - Hidden from users
// ═══════════════════════════════════════════════════════════════════════════

impl LogStructuredStore {
    pub fn flush_and_sync(&self) -> io::Result<()> {
        let current = unsafe {
            self.buffer
                .current_buffer
                .load(Ordering::Acquire)
                .as_ref()
                .ok_or_else(|| io::Error::other("invalid buffer state"))?
        };

        let state = current.state.load(Ordering::Acquire);
        let has_data = state_offset(state) > 0;

        if has_data {
            let slot = self
                .buffer
                .next_address_range
                .fetch_add(1, Ordering::AcqRel);
            current
                .local_lss_address_slot
                .store(slot, Ordering::Release);

            let _ = current.set_sealed_bit_true();
            let _ = self.buffer.rotate_after_seal(current.pos);

            let before = current.set_flush_in_progress();
            if before & FLUSH_IN_PROGRESS_BIT == 0 {
                self.flusher.submit_buffer(current);
            }
        }

        // Wait for any naturally-sealed buffers that are still in-flight
        // before submitting the fsync — otherwise we race against their SQEs
        let target = self.buffer.next_address_range.load(Ordering::Acquire) as u64;
        loop {
            self.check_async_cque().ok();
            let stable = self.hi_stable.load(Ordering::Acquire);
            if stable != u64::MAX && stable + 1 >= target {
                break;
            }
            std::hint::spin_loop();
        }

        self.flusher.sync_data()
    }
    pub fn sync(&self) -> io::Result<()> {
        self.flusher.sync_data()
    }

    /// Internal write implementation - handles retry logic and blocking
    fn internal_write(&self, data: &[u8]) -> io::Result<u64> {
        loop {
            let current = unsafe {
                self.buffer
                    .current_buffer
                    .load(Ordering::Acquire)
                    .as_ref()
                    .ok_or_else(|| io::Error::other("invalid buffer state"))?
            };

            let reserve_result = current.reserve_space(data.len());

            // Handle reservation outcomes
            match &reserve_result {
                Err(BufferError::FailedReservation) => continue,
                Err(BufferError::EncounteredSealedBuffer) => continue,
                _ => {}
            }

            // Try to write
            let offset = match &reserve_result {
                Ok(offset) => *offset,
                Err(_) => match self.buffer.put(current, reserve_result, data) {
                    Err(BufferError::ActiveUsers) => continue,
                    Err(BufferError::EncounteredSealedBuffer) => {
                        std::thread::yield_now();
                        continue;
                    }
                    Err(BufferError::RingExhausted) => {
                        self.check_async_cque().ok();
                        std::thread::yield_now();
                        continue;
                    }
                    Err(e) => return Err(io::Error::other(format!("{:?}", e))),
                    Ok(_) => continue,
                },
            };

            // Write succeeded - copy data
            let _ = self.buffer.put(current, Ok(offset), data);
            let slot = current.local_lss_address_slot.load(Ordering::Acquire) as u64;

            return Ok(slot * MD as u64 + offset as u64);
        }
    }

    /// Internal read implementation
    fn internal_read(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        use io_uring::{opcode, types};
        use std::os::fd::AsRawFd;

        // O_DIRECT requires file offset aligned to 4KB
        let aligned_offset = offset & !(FOUR_KB_PAGE as u64 - 1);
        let delta = (offset - aligned_offset) as usize;
        let aligned_len = (len + delta).next_multiple_of(FOUR_KB_PAGE);

        let layout = std::alloc::Layout::from_size_align(aligned_len, FOUR_KB_PAGE).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());

        let sqe = opcode::Read::new(types::Fd(self.store.as_raw_fd()), ptr, aligned_len as u32)
            .offset(aligned_offset)
            .build();

        let mut ring = self.flusher.get_reader().flusher.lock();

        unsafe {
            ring.submission()
                .push(&sqe)
                .map_err(|_| io::Error::other("submission queue full"))?;
        }

        ring.submit_and_wait(1)?;

        let cqe = ring
            .completion()
            .next()
            .ok_or_else(|| io::Error::other("no completion"))?;

        if cqe.result() < 0 {
            unsafe { std::alloc::dealloc(ptr, layout) };
            return Err(io::Error::from_raw_os_error(-cqe.result()));
        }

        // Slice out exactly the bytes requested, skipping the alignment delta
        let result = unsafe { std::slice::from_raw_parts(ptr.add(delta), len).to_vec() };
        unsafe { std::alloc::dealloc(ptr, layout) };

        Ok(result)
    }
    /// Internal completion queue processing
    fn check_async_cque(&self) -> Result<(), ()> {
        let cqes: Vec<cqueue::Entry> = {
            let mut ring = self.flusher.get_cqueue();
            ring.completion().sync();
            ring.completion().collect()
        };

        if cqes.is_empty() {
            return Ok(());
        }

        for cqe in cqes {
            let ptr = cqe.user_data() as *const FlushBuffer;
            let buffer: &FlushBuffer = unsafe { &*ptr };
            let lss_slot = buffer.local_lss_address_slot.load(Ordering::Acquire) as u64;

            if cqe.result() < 0 {
                // Retry failed write
                let sqe = unsafe {
                    (*buffer.submit_queue_entry.get())
                        .as_ref()
                        .expect("stored SQE must be present on retry")
                };
                let mut ring = self.flusher.get_cqueue();
                unsafe {
                    let _ = ring.submission().push(&sqe);
                }
                let _ = ring.submit();
            } else {
                // Success - mark complete and reset
                self.mark_slot_complete(lss_slot);
                self.buffer.reset_buffer(buffer);
            }
        }

        Ok(())
    }

    /// Internal stability tracking
    fn mark_slot_complete(&self, lss_slot: u64) {
        loop {
            let current = self.hi_stable.load(Ordering::Acquire);

            if current == u64::MAX {
                if lss_slot == 0 {
                    match self.hi_stable.compare_exchange(
                        current,
                        0,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            self.advance_high_stable();
                            return;
                        }
                        Err(_) => continue,
                    }
                } else {
                    self.completed_islands.write().insert(lss_slot);
                    return;
                }
            }

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
                    Err(_) => continue,
                }
            } else {
                self.completed_islands.write().insert(lss_slot);
                return;
            }
        }
    }

    /// Internal stability advancement
    fn advance_high_stable(&self) {
        loop {
            let current = self.hi_stable.load(Ordering::Acquire);
            let next_expected = current + 1;

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
                }
                Err(_) => continue,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{flush_behaviour::WriteMode, open_options::LssOpenOptions};
    use std::{sync::Arc, thread, time::Duration};

    fn open_test_store(path: &str) -> LogStructuredStore {
        let _ = std::fs::create_dir_all("test_store");
        let _ = std::fs::remove_file(path);

        LssOpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .write_mode(WriteMode::TailLocalizedWrites)
            .open(path)
            .expect("failed to open store")
    }

    #[test]
    fn simple_async_write() {
        let path = "test_store/simple_async.db";
        let store = open_test_store(path);

        let slot = store.write_async(b"hello world").unwrap();

        // Poll until durable
        for _ in 0..100 {
            store.poll_completions();
            if store.is_durable(slot) {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        assert!(store.is_durable(slot));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn multiple_writes_async() {
        let path = "test_store/multi_async.db";
        let store = open_test_store(path);

        let mut slots = Vec::new();
        for i in 0..100 {
            let data = format!("message {}", i);
            let slot = store.write_async(data.as_bytes()).unwrap();
            slots.push(slot);
        }

        // Poll until all are durable
        let max_slot = *slots.iter().max().unwrap();
        for _ in 0..1000 {
            store.poll_completions();
            if store.stable_slot() >= max_slot {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }

        // All should be durable now
        for slot in slots {
            assert!(store.is_durable(slot), "slot {} not durable", slot);
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn concurrent_writers() {
        let path = "test_store/concurrent.db";
        let store = Arc::new(open_test_store(path));

        let mut handles = vec![];
        for thread_id in 0..4 {
            let store = Arc::clone(&store);
            let handle = thread::spawn(move || {
                let mut slots = Vec::new();
                for i in 0..50 {
                    let data = format!("thread {} message {}", thread_id, i);
                    let slot = store.write_async(data.as_bytes()).unwrap();
                    slots.push(slot);
                }
                slots
            });
            handles.push(handle);
        }

        let mut all_slots = Vec::new();
        for handle in handles {
            all_slots.extend(handle.join().unwrap());
        }

        // Wait for all to be durable
        let max_slot = *all_slots.iter().max().unwrap();
        for _ in 0..1000 {
            store.poll_completions();
            if store.stable_slot() >= max_slot {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }

        // Verify all are durable
        for slot in all_slots {
            assert!(store.is_durable(slot), "slot {} not durable", slot);
        }

        let _ = std::fs::remove_file(path);
    }

    // =========================================================================
    // Mock Mapping Table - Shows integration pattern
    // =========================================================================

    /// Simple in-memory mapping table that tracks page_id -> LSS slot mappings
    struct MockMappingTable {
        mappings: Arc<parking_lot::RwLock<std::collections::HashMap<u64, u64>>>,
    }

    impl MockMappingTable {
        fn new() -> Self {
            Self {
                mappings: Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
            }
        }

        fn insert(&self, page_id: u64, lss_slot: u64) {
            self.mappings.write().insert(page_id, lss_slot);
        }

        fn get(&self, page_id: u64) -> Option<u64> {
            self.mappings.read().get(&page_id).copied()
        }

        fn len(&self) -> usize {
            self.mappings.read().len()
        }
    }

    #[test]
    fn with_mock_mapping_table_async() {
        let path = "test_store/mock_mapping_async.db";
        let store = Arc::new(open_test_store(path));
        let mapping_table = Arc::new(MockMappingTable::new());

        // Launch multiple writers
        let mut handles = vec![];
        for thread_id in 0..4 {
            let store = Arc::clone(&store);
            let mapping_table = Arc::clone(&mapping_table);

            let handle = thread::spawn(move || {
                for page_id in 0..25u64 {
                    let global_page_id = thread_id * 100 + page_id;
                    let page_data = format!("Thread {} Page {} data", thread_id, page_id);
                    let lss_slot = store.write_async(page_data.as_bytes()).unwrap();
                    mapping_table.insert(global_page_id, lss_slot);
                }
            });
            handles.push(handle);
        }

        // Wait for all writers
        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 4 threads × 25 pages = 100 mappings
        assert_eq!(mapping_table.len(), 100);

        // Poll until all are durable
        for _ in 0..1000 {
            store.poll_completions();

            // Check if all mapped slots are durable
            let all_durable = (0..4)
                .flat_map(|tid| (0..25u64).map(move |pid| tid * 100 + pid))
                .all(|page_id| {
                    mapping_table
                        .get(page_id)
                        .map(|slot| store.is_durable(slot))
                        .unwrap_or(false)
                });

            if all_durable {
                break;
            }

            thread::sleep(Duration::from_millis(1));
        }

        // Final verification - all pages should be durable
        for thread_id in 0..4 {
            for page_id in 0..25u64 {
                let global_page_id = thread_id * 100 + page_id;
                let slot = mapping_table
                    .get(global_page_id)
                    .expect("mapping should exist");
                assert!(
                    store.is_durable(slot),
                    "thread {} page {} (slot {}) not durable",
                    thread_id,
                    page_id,
                    slot
                );
            }
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn mapping_table_with_updates() {
        let path = "test_store/mock_mapping_updates.db";
        let store = open_test_store(path);
        let mapping_table = MockMappingTable::new();

        // Initial write
        let page_id = 42;
        let slot1 = store.write_async(b"version 1").unwrap();
        mapping_table.insert(page_id, slot1);

        assert_eq!(mapping_table.get(page_id), Some(slot1));

        // Update the same page (new LSS slot, same page_id)
        let slot2 = store.write_async(b"version 2").unwrap();
        mapping_table.insert(page_id, slot2);

        assert_eq!(mapping_table.get(page_id), Some(slot2));

        // Both slots should be durable
        assert!(store.is_durable(slot1));
        assert!(store.is_durable(slot2));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_and_read_single() {
        let path = "test_store/write_read_single.db";
        let store = open_test_store(path);

        let data = b"hello, LLAMA";
        let slot = store.write_async(data).unwrap();

        store.flush_and_sync().unwrap();

        let read_back = store.read_at_slot(slot, data.len()).unwrap();
        assert_eq!(&read_back[..data.len()], data);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_and_read_multiple_sequential() {
        let path = "test_store/write_read_sequential.db";
        let store = open_test_store(path);

        let writes: Vec<(&[u8; 15], u64)> =
            vec![b"first page data", b"second pagedata", b"third page data"]
                .into_iter()
                .map(|data| {
                    let slot = store.write_async(data).unwrap();
                    (data, slot)
                })
                .collect();

        store.flush_and_sync().unwrap();

        for (data, slot) in &writes {
            let read_back = store.read_at_slot(*slot, data.len()).unwrap();
            assert_eq!(&read_back[..data.len()], *data);
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_and_read_concurrent() {
        let path = "test_store/write_read_concurrent.db";
        let store = Arc::new(open_test_store(path));

        // Each thread writes its own pages and records slot -> expected data
        let mut handles = vec![];
        for thread_id in 0..4usize {
            let store = Arc::clone(&store);
            let handle = thread::spawn(move || {
                (0..25usize)
                    .map(|page_id| {
                        let data = format!(
                            "thread={} page={} payload={}",
                            thread_id,
                            page_id,
                            thread_id * 100 + page_id
                        );
                        let slot = store.write_async(data.as_bytes()).unwrap();
                        (slot, data)
                    })
                    .collect::<Vec<_>>()
            });
            handles.push(handle);
        }

        let all_writes: Vec<(u64, String)> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        store.flush_and_sync().unwrap();

        for (slot, expected) in &all_writes {
            let read_back = store.read_at_slot(*slot, expected.len()).unwrap();
            assert_eq!(
                &read_back[..expected.len()],
                expected.as_bytes(),
                "slot {} data mismatch",
                slot
            );
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_and_read_update() {
        let path = "test_store/write_read_update.db";
        let store = open_test_store(path);

        // Write v1
        let v1 = b"version 1 data";
        let slot1 = store.write_async(v1).unwrap();

        // Overwrite same logical page — new slot, new data
        let v2 = b"version 2 data";
        let slot2 = store.write_async(v2).unwrap();

        store.flush_and_sync().unwrap();

        // Both slots are independently readable — LSS is append-only
        let read1 = store.read_at_slot(slot1, v1.len()).unwrap();
        assert_eq!(&read1[..v1.len()], v1, "v1 slot corrupted");

        let read2 = store.read_at_slot(slot2, v2.len()).unwrap();
        assert_eq!(&read2[..v2.len()], v2, "v2 slot corrupted");

        assert_ne!(slot1, slot2, "slots must be distinct");

        let _ = std::fs::remove_file(path);
    }
    #[test]
    fn write_and_read_boundary_sizes() {
        let path = "test_store/write_read_boundary.db";
        let store = open_test_store(path);

        let tiny = vec![0xABu8; 1];
        let medium = vec![0xCDu8; 64];
        let full = vec![0xEFu8; 128];

        let off_tiny = store.write_async(&tiny).unwrap();
        let off_medium = store.write_async(&medium).unwrap();
        let off_full = store.write_async(&full).unwrap();

        store.flush_and_sync().unwrap();

        let r = store.read_at_slot(off_tiny, tiny.len()).unwrap();
        assert_eq!(&r[..tiny.len()], &tiny[..]);

        let r = store.read_at_slot(off_medium, medium.len()).unwrap();
        assert_eq!(&r[..medium.len()], &medium[..]);

        let r = store.read_at_slot(off_full, full.len()).unwrap();
        assert_eq!(&r[..full.len()], &full[..]);

        let _ = std::fs::remove_file(path);
    }
}
