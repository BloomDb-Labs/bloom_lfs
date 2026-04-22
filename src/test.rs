#[cfg(test)]
mod tests {
    use buffer_ring::{BufferError, BufferMsg, WriteMode};
    use crossbeam_channel::Receiver;
    use parking_lot::lock_api::RwLock;

    use crate::{ByteCount, FileOffset, LogStructuredStore, LssConfig};

    use std::{
        collections::HashMap,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
    };

    static PATH: &str = "test_store/simple_async.db";

    fn open_test_store(path: &str) -> (Receiver<(FileOffset, ByteCount)>, LogStructuredStore) {
        let _ = std::fs::create_dir_all("test_store");
        let _ = std::fs::remove_file(path);

        let mut options = LssConfig::new();
        let receiver = options.with_publusher();

        options
            .read(true)
            .write(true)
            .create(true)
            .write_mode(WriteMode::TailLocalizedWrites)
            .interior_receiver();

        (receiver, options.open(path).expect("failed to open store"))
    }

    // =========================================================================
    // Mock Mapping Table - Shows integration pattern
    // =========================================================================

    type PID = usize;

    #[derive(Clone)]
    pub struct Entry {
        pub(crate) payload: Box<[u8]>,
        pub(crate) offset: Option<u64>,
    }

    /// Simple in-memory mapping table that tracks page_id -> LSS slot mappings

    pub struct MockMappingTable {
        pub mappings: Arc<parking_lot::RwLock<std::collections::HashMap<PID, Entry>>>,
        pub lss: LogStructuredStore,

        pub index: AtomicUsize,
    }

    impl MockMappingTable {
        fn insert(&self, page_id: PID, payload: Box<[u8]>) -> Result<BufferMsg, BufferError> {
            loop {
                let ring = &self.lss.buffer_ring;
                let current = ring.current_buffer(Ordering::Acquire);
                let reserve_result = current.reserve_space(payload.len());

                match &reserve_result {
                    Err(BufferError::FailedReservation)
                    | Err(BufferError::EncounteredSealedBuffer) => {
                        continue;
                    }
                    _ => {}
                }

                match ring.put(current, reserve_result, &payload) {
                    Ok(msg) => {
                        if let Ok(offset) = reserve_result {
                            let entry = Entry {
                                payload: payload.into(),
                                offset: Some(offset as u64),
                            };
                            self.mappings.write().insert(page_id, entry);
                            return Ok(msg);
                        }
                    }

                    Err(BufferError::EncounteredSealedBuffer) | Err(BufferError::RingExhausted) => {
                        let _ = ring.check_cque();
                        std::thread::yield_now();
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        fn get(&self, page_id: PID) -> Option<Entry> {
            let lock = self.mappings.read();
            lock.get(&page_id).cloned()
        }

        fn len(&self) -> usize {
            self.mappings.read().len()
        }

        fn is_durable(&self, page_id: PID) -> bool {
            let entry = self.get(page_id).unwrap();
            self.lss.is_durable(entry.offset.unwrap())
        }

        fn flush(&self) {
            let _ = self.lss.roll();
            self.lss.sync();
            let _ = self.lss.buffer_ring.check_cque();
        }

        fn drain_cque(&self) {
            std::thread::sleep(std::time::Duration::from_millis(100)); // Temporary fix for now
            let _ = self.lss.buffer_ring.check_cque();
        }

        fn allocate_page(&self) -> usize {
            self.index.fetch_add(1, Ordering::Relaxed)
        }
    }

    #[test]
    fn with_mock_mapping_table_async() {
        let (_, lss) = open_test_store(PATH);

        let table = MockMappingTable {
            mappings: Arc::new(RwLock::new(HashMap::new())),
            lss: lss,
            index: AtomicUsize::new(0),
        };

        let page_id = table.allocate_page();

        let deadbeef: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let _ = table.insert(page_id, deadbeef.clone().into_boxed_slice());
        table.flush();

        let entry = table.get(page_id).unwrap();
        let mut ptr = vec![0u8; 4];

        table
            .lss
            .read(ptr.as_mut_ptr(), 4, entry.offset.unwrap())
            .expect("Read");

        assert_eq!(deadbeef, ptr);
        assert_eq!(ptr, *entry.payload);

        assert!(table.is_durable(page_id))
    }

    #[test]
    fn write_and_read_multiple_sequential() {
        let (_, lss) = open_test_store(PATH);
        let table = MockMappingTable {
            mappings: Arc::new(RwLock::new(HashMap::new())),
            lss,
            index: AtomicUsize::new(0),
        };

        // Write 10_000 distinct entries
        let pages: Vec<(usize, Vec<u8>)> = (0..1000)
            .map(|i| {
                let pid = table.allocate_page();
                let payload = format!("page-{:04}", i).into_bytes();
                let _ = table.insert(pid, payload.clone().into_boxed_slice());
                (pid, payload)
            })
            .collect();

        table.flush();
        table.drain_cque();

        // Every page must round-trip exactly
        for (pid, expected) in &pages {
            let entry = table.get(*pid).unwrap();
            let mut dst = vec![0u8; expected.len()];
            table
                .lss
                .read(dst.as_mut_ptr(), expected.len(), entry.offset.unwrap())
                .unwrap();
            // println!("Expectation {:?}", expected);
            // println!("Destination {:?}", dst);
            assert_eq!(&dst, expected, "mismatch on page {pid}");
        }

        // All pages must be durable
        for (pid, _) in &pages {
            assert!(table.is_durable(*pid), "page {pid} not durable");
        }
    }

    #[test]
    fn write_and_read_concurrent() {
        let (_, lss) = open_test_store("test_store/concurrent.db");
        let table = Arc::new(MockMappingTable {
            mappings: Arc::new(RwLock::new(HashMap::new())),
            lss,
            index: AtomicUsize::new(0),
        });

        const THREADS: usize = 4;
        const PER_THREAD: usize = 500;

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let tbl = Arc::clone(&table);
                thread::spawn(move || {
                    for i in 0..PER_THREAD {
                        let pid = tbl.allocate_page();
                        let payload = format!("t{t}-msg-{i:04}").into_bytes();
                        let _ = tbl.insert(pid, payload.into_boxed_slice());
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        table.flush();
        table.drain_cque();

        // All inserted pages must be readable and durable
        let total = table.len();
        assert_eq!(total, THREADS * PER_THREAD, "some inserts were lost");

        let mappings = table.mappings.read();
        for (pid, entry) in mappings.iter() {
            let payload = &entry.payload;
            let mut dst = vec![0u8; payload.len()];
            table
                .lss
                .read(dst.as_mut_ptr(), payload.len(), entry.offset.unwrap())
                .unwrap();
            assert_eq!(
                &dst[..payload.len()],
                payload.as_ref(),
                "bad read for page {pid}"
            );
        }
        drop(mappings);

        for pid in 0..(THREADS * PER_THREAD) {
            assert!(table.is_durable(pid as usize), "page {pid} not durable");
        }
    }
    #[test]
    fn write_and_read_update() {
        let (_, lss) = open_test_store(PATH);
        let table = MockMappingTable {
            mappings: Arc::new(RwLock::new(HashMap::new())),
            lss: lss,
            index: AtomicUsize::new(0),
        };
        unimplemented!()
    }
    #[test]
    fn write_and_read_boundary_sizes() {
        let (_, lss) = open_test_store(PATH);
        let table = MockMappingTable {
            mappings: Arc::new(RwLock::new(HashMap::new())),
            lss: lss,
            index: AtomicUsize::new(0),
        };
        unimplemented!()
    }
}
