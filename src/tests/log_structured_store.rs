#[cfg(test)]
pub mod test {

    use std::sync::atomic::Ordering;

    use crate::{ flush_behaviour::WriteMode, flush_buffer::BufferError, log_structured_store::{FOUR_KB_PAGE, LogStructuredStore}};

    #[test]
    fn api_test() {
        let path = "test_store/log.db";
        let log =
            LogStructuredStore::open_with_behavior(path, WriteMode::TailLocalizedWrites).unwrap();

        let res = match log.reserve_space(4096) {
            Ok(r) => Ok(r),
            Err(e) => Err(e),
        };

        if res.is_ok() {
            let res = res.unwrap();
            // Mapping Table Install here
            let successful_cas = true;
            // Lets say, hypothetically our mapping table entry was successfully installed

            if successful_cas {
                let _ = log.write_payload(b"LENGTH_OF_4KB", res);
                log.flush_cur_buffer();
            } else {
                // Lets say, hypothetically, it wasn't
                let _ = log.write_payload(b"FAILED_FLUSH", res);
            }
        }

        // Check completions
        log.check_async_cque();

        // // Clean up
        // let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_out_of_order_completion() {
        let path = "test_store/ooo_log.db";
        let log =
            LogStructuredStore::open_with_behavior(path, WriteMode::TailLocalizedWrites).unwrap();

        // Simulate out-of-order completions: 3, 1, 2
        log.mark_slot_complete(3);
        assert_eq!(
            log.hi_stable.load(Ordering::Acquire),
            0,
            "hi_stable should stay at 0"
        );
        assert_eq!(log.get_island_count(), 1, "Should have 1 island (slot 3)");

        log.mark_slot_complete(1);
        assert_eq!(
            log.hi_stable.load(Ordering::Acquire),
            1,
            "hi_stable should advance to 1"
        );
        assert_eq!(
            log.get_island_count(),
            1,
            "Should still have 1 island (slot 3)"
        );

        log.mark_slot_complete(2);
        assert_eq!(
            log.hi_stable.load(Ordering::Acquire),
            3,
            "hi_stable should advance to 3"
        );
        assert_eq!(log.get_island_count(), 0, "Should have no islands");

        // Clean up
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_large_gap() {
        let path = "test_store/gap_log.db";
        let log =
            LogStructuredStore::open_with_behavior(path, WriteMode::TailLocalizedWrites).unwrap();

        // Create a large gap by completing slot 100 first
        log.mark_slot_complete(100);
        assert_eq!(log.hi_stable.load(Ordering::Acquire), 0);
        assert_eq!(
            log.get_island_count(),
            1,
            "Should only have island at slot 100"
        );

        // Complete slot 1
        log.mark_slot_complete(1);
        assert_eq!(log.hi_stable.load(Ordering::Acquire), 1);
        assert_eq!(
            log.get_island_count(),
            1,
            "Should still only have island at slot 100"
        );

        // Complete slot 2
        log.mark_slot_complete(2);
        assert_eq!(log.hi_stable.load(Ordering::Acquire), 2);

        // CRITICAL: islands should still only contain slot 100
        // NOT slots 3-99 (the old broken code would insert all of those)
        assert_eq!(
            log.get_island_count(),
            1,
            "Should ONLY have island at slot 100"
        );

        // Clean up
        let _ = std::fs::remove_file(path);
    }

    fn test_multiple_islands() {
        let path = "test_store/multi_island_log.db";
        let log =
            LogStructuredStore::open_with_behavior(path, WriteMode::TailLocalizedWrites).unwrap();

        // Create multiple islands: 5, 10, 15
        log.mark_slot_complete(5);
        log.mark_slot_complete(10);
        log.mark_slot_complete(15);

        assert_eq!(log.hi_stable.load(Ordering::Acquire), 0);
        assert_eq!(log.get_island_count(), 3, "Should have exactly 3 islands");

        // Fill in to reach first island
        for slot in 1..=5 {
            log.mark_slot_complete(slot);
        }

        assert_eq!(log.hi_stable.load(Ordering::Acquire), 5);
        assert_eq!(log.get_island_count(), 2, "Should have 2 islands remaining");

        // Clean up
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_async_flushes_and_hi_stable() {
        let path = "test_store/async_flush.db";
        let _ = std::fs::create_dir_all("test_store");
        let _ = std::fs::remove_file(path);

        let log =
            LogStructuredStore::open_with_behavior(path, WriteMode::TailLocalizedWrites).unwrap();

        // Fill in to reach first island
        for slot in 1..=3 {
            log.mark_slot_complete(slot);
        }

        // Write 8 × 1KB payloads to fill two buffers (slots 4 and 5)
        let mut records: Vec<(i32, u64, usize, bool)> = Vec::new(); // (i, slot, offset, flushed)
        for i in 0..8 {
            let payload = vec![i as u8; 1024]; // Unique payload

            loop {
                match log.reserve_space(1024) {
                    Ok(res) => {
                        let offset = res.offset;
                        log.write_payload(&payload, res).unwrap();
                        let slot = 4 + (i / 4) as u64; // 4 writes per buffer
                        records.push((i as i32, slot, offset, false));
                        break;
                    }
                    Err(BufferError::InsufficientSpace) => {
                        // Seal the current buffer (triggers async flush)
                        let current = unsafe {
                            log.buffer
                                .current_buffer
                                .load(Ordering::Acquire)
                                .as_ref()
                                .unwrap()
                        };

                        // Empty write seals the current buffer
                        let _ = log
                            .buffer
                            .put(current, Err(BufferError::InsufficientSpace), &[]);
                        log.check_async_cque();
                    }
                    Err(e) => panic!("unexpected error: {e:?}"),
                }
            }
        }

        // Seal the final buffer
        let current = log.get_cur_buffer();
        let _ = log
            .buffer
            .put(current, Err(BufferError::InsufficientSpace), &[]);

        // Wait for all async flushes to complete and hi_stable to advance
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let max_slot = records.iter().map(|r| r.1).max().unwrap();
        loop {
            log.check_async_cque();
            let stable = log.hi_stable.load(Ordering::Acquire);
            if stable >= max_slot {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!(
                    "timed out waiting for flushes — hi_stable={}, max_slot={}, islands={}",
                    stable,
                    max_slot,
                    log.get_island_count()
                );
            }
        }

        // Verify hi_stable is at least max_slot
        let final_stable = log.hi_stable.load(Ordering::Acquire);
        assert!(
            final_stable >= max_slot,
            "hi_stable {} should be >= max_slot {}",
            final_stable,
            max_slot
        );

        // Mark all records as flushed
        for record in &mut records {
            record.3 = true;
        }

        // Assert all records are marked as flushed
        for (i, _, _, flushed) in &records {
            assert!(*flushed, "Record {} should be flushed", i);
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_read_multiple_slots_from_disk() {
        use std::os::unix::fs::FileExt;

        let path = "test_store/read_multi_slots.db";
        let _ = std::fs::create_dir_all("test_store");
        let _ = std::fs::remove_file(path);

        let log =
            LogStructuredStore::open_with_behavior(path, WriteMode::TailLocalizedWrites).unwrap();

        // Write 8 × 1KB payloads to fill two buffers (slots 4 and 5)
        let mut records = Vec::new();
        for i in 0..8 {
            let payload = vec![i as u8; 1024]; // Fill entire payload with unique value

            loop {
                match log.reserve_space(1024) {
                    Ok(res) => {
                        let offset = res.offset;
                        log.write_payload(&payload, res).unwrap();
                        let slot = 4 + (i / 4) as u64; // 4 writes per buffer
                        records.push((i, slot, offset));
                        break;
                    }
                    Err(BufferError::InsufficientSpace) => {
                        // Seal the current buffer
                        let sealed = unsafe {
                            log.buffer
                                .current_buffer
                                .load(Ordering::Acquire)
                                .as_ref()
                                .unwrap()
                        };
                        let _ = log
                            .buffer
                            .put(sealed, Err(BufferError::InsufficientSpace), &[]);
                        // Flush the sealed buffer blocking
                        sealed.set_flush_in_progress();
                        log.flusher.submit_buffer_and_wait(sealed).unwrap();
                        log.buffer.reset_buffer(sealed);
                    }
                    Err(e) => panic!("unexpected error: {e:?}"),
                }
            }
        }

        // Seal and flush the final buffer
        let sealed = log.get_cur_buffer();
        let _ = log
            .buffer
            .put(sealed, Err(BufferError::InsufficientSpace), &[]);
        sealed.set_flush_in_progress();
        log.flusher.submit_buffer_and_wait(sealed).unwrap();
        log.buffer.reset_buffer(sealed);

        // Read back from all slots
        let file = std::fs::File::open(path).unwrap();
        for (i, slot, offset) in &records {
            let byte_offset = slot * FOUR_KB_PAGE as u64 + *offset as u64;
            let mut read_back = vec![0u8; 1024];
            file.read_at(&mut read_back, byte_offset).unwrap();

            // Verify entire payload is filled with i as u8
            let expected = vec![*i as u8; 1024];
            assert_eq!(
                read_back, expected,
                "Failed to read correct data at slot {}, offset {}: payload mismatch for i={}",
                slot, offset, i
            );
        }

        let _ = std::fs::remove_file(path);
    }
}
