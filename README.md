# Bloom_lfs (`lss`)

A **high-throughput append-oriented storage engine** built in Rust using **batched buffer-ring writes**, **direct I/O (`O_DIRECT`)**, and **async `io_uring` dispatch**.

## Overview

Bloom_lfs implements the storage foundation of LLAMA (Latch-free, Log-structured, Access-Method Aware) — a concurrent caching and storage subsystem designed to exploit the performance characteristics of append-only writes on flash media while avoiding the costs of random I/O and excessive write amplification.

The project currently implements the **log-structured secondary storage** component, which provides:

- **High write throughput** through batched, sequential writes
- **Latch-free concurrency** via atomic operations on packed state words
- **Low write amplification** with tail-localized append patterns
- **Direct I/O (`O_DIRECT`)** to bypass kernel page cache overhead
- **Asynchronous I/O** via `io_uring` for efficient kernel interactions

This project is designed as a durable write layer for Bloom Db. In general this system may be used for page stores, databases, WAL systems, or any workload that benefits from:

* Fast sequential persistence
* Reduced write amplification
* Predictable durability tracking
* Concurrent writers
* Kernel page-cache bypass

---

## Features

* Append-only log structured file layout
* Concurrent multi-threaded writers
* Buffer-ring batching for throughput
* `io_uring` async disk writes with `QuikIO`
* `O_DIRECT` page-cache bypass
* Durability tracking via stable offsets
* Sequential + tail-localized write modes
* Read by byte offset
* Flush / sync support

---

## Why This Exists

Traditional random writes are expensive.

This store converts many small writes into **large aligned sequential writes** by buffering data in memory first, then flushing to disk efficiently.

Ideal for:

* Databases
* Page stores
* LSM trees
* Write-ahead logs
* Persistent queues
* Storage engines

---

## High-Level Architecture

```text
                     Application Writes
                             │
                             ▼
                   LogStructuredStore API
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
           BufferRing               QuikIO Dispatcher
        (staging buffers)             (io_uring)
                 │                       │
                 └───────────┬───────────┘
                             ▼
                     O_DIRECT File I/O
                             ▼
                     Durable Backing File
```

---

## Core Concepts

### Buffer Ring

Incoming writes first reserve space inside an in-memory flush buffer.

Once full (or manually flushed), the buffer is sealed and asynchronously written to disk.

Benefits:

* Combines many writes into one disk operation
* Reduces syscall overhead
* Better SSD / NVMe throughput

### Stable Durability Tracking

The store tracks the highest contiguous durable offset:

```rust
store.stable_slot();
```

You can also verify whether a record is durable:

```rust
store.is_durable(offset);
```

Useful for:

* WAL commit acknowledgement
* Page checkpointing
* Replication progress
* Crash safety

### Write Modes

#### TailLocalizedWrites (Default)

Optimized for throughput.

Writes may complete out of order physically, but logical offsets remain correct.

#### SerializedWrites

Strict append ordering using linked submissions.

Useful for ordered logs or journaling.

---

## Installation

```toml
[dependencies]
lss = { path = "./lss" }
```

---

## Quick Start

```rust
use lss::{LssConfig, WriteMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = LssConfig::new();

    let store = config
        .read(true)
        .write(true)
        .create(true)
        .write_mode(WriteMode::TailLocalizedWrites)
        .open("data/store.db")?;

    let payload = b"hello world";

    let reservation = store.reserve(payload)?;
    let offset = store.write(reservation)?;

    store.roll()?;
    store.sync();

    store.buffer_ring.check_cque();
    println!("Written at offset {}", offset);

    Ok(())
}
```

---

## Reading Data

```rust
let mut dst = vec![0u8; 11];
store.read(dst.as_mut_ptr(), 11, offset as u64)?;
assert_eq!(&dst, b"hello world");
```

---

## Concurrency Example

```rust
use std::sync::Arc;
use std::thread;

let store = Arc::new(store);

for i in 0..4 {
    let db = store.clone();

    thread::spawn(move || {
        let payload = format!("thread-{i}");
        let r = db.reserve(payload.as_bytes()).unwrap();
        db.write(r).unwrap();
    });
}
```

---

## Included Tests

* Async mapping table example
* Sequential bulk writes
* Concurrent writes
* Durability validation

---

## Linux Requirements

Requires Linux with:

* `io_uring`
* `O_DIRECT`
* aligned block device writes

Recommended:

* Linux 5.10+
* NVMe SSD

---

## Roadmap

* Compaction layer
* Recovery scanner
* Snapshots
* Compression

---

## License

MIT


## Author Notes

This project demonstrates low-level systems programming topics:

lock-free coordination
atomic durability tracking
async kernel I/O
memory alignment
storage engine internals