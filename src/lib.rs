//! # Bloom_lfs — LLAMA Log-Structured Storage
//!
//! A high-performance, latch-free log-structured storage layer built for modern flash storage and multi-core systems.
//!
//! Bloom_lfs implements the storage foundation of LLAMA (Latch-free, Log-structured, Access-Method Aware) — a concurrent caching and storage subsystem designed to exploit the performance characteristics of append-only writes on flash media while avoiding the costs of random I/O and excessive write amplification.
//!
//! The project currently implements the **log-structured secondary storage** component, which provides:
//!
//! - **High write throughput** through batched, sequential writes
//! - **Latch-free concurrency** via atomic operations on packed state words
//! - **Low write amplification** with tail-localized append patterns
//! - **Direct I/O (`O_DIRECT`)** to bypass kernel page cache overhead
//! - **Asynchronous I/O** via `io_uring` for efficient kernel interactions
//!
//! ## Core Components
//!
//! - [`flush_buffer::FlushBuffer`] & [`flush_buffer::FlushBufferRing`]: Latch-free I/O buffer ring
//! - [`log_structured_store::LogStructuredStore`]: Durable backing file with stability tracking
//! - [`flush_behaviour::FlushBehavior`]: `io_uring`-backed write dispatch (parallel/serial modes)
//!
//! ## Example
//!
//! ```rust,no_run
//! use bloom_lfs::log_structured_store::{LogStructuredStore, WriteMode};
//!
//! // Open store with tail-localized writes (high throughput)
//! let store = LogStructuredStore::open_with_behavior(
//!     "/var/lib/llama/data.lss",
//!     WriteMode::TailLocalizedWrites,
//! )?;
//!
//! // Reserve space, write payload, poll completions
//! let reservation = store.reserve_space(64)?;
//! store.write_payload(b"hello, LLAMA", reservation)?;
//! store.check_async_cque(); // Process completed writes
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! ## Design
//!
//! LLAMA is comprised of:
//!     Mapping Tables: In memory lock-free map to locate page deltas on both in memory in the caching layer as well on Disks whethre remote or local.
//!
//!     Caching Layer : The physically store hot deltas
//!
//!     Log-Structured Secondary Storage : To provides the usual advantages of avoiding random writes, reducing
//!                                        the number of writes via large multi-page buffers, and wear leveling  
//!                                        needed by flash memory
//!
//!     Recovery Protocal: To rebuild Mapping Table after crashes
//!
//!

pub mod flush_buffer;
pub mod log_structured_store;
pub mod flush_behaviour;
pub mod open_options; 



