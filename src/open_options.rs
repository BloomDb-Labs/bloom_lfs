use std::{
    collections::BTreeSet,
    fs::OpenOptions,
    io,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    sync::{
        atomic::AtomicU64,
        Arc,
    },
};

use io_uring::IoUring;
use parking_lot::RwLock;

use crate::{
    flush_behaviour::{Appender, FOUR_KB_PAGE, FlushBehavior, WriteMode},
    flush_buffer::{FlushBufferRing, RING_SIZE},
    log_structured_store::LogStructuredStore
};

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
pub struct LssOpenOptions {
    pub(crate) read: bool,
    pub(crate) write: bool,
    pub(crate) create: bool,
    pub(crate) write_mode: WriteMode,
}

impl LssOpenOptions {
    pub fn new() -> Self {
        Self {
            read: false,
            write: false,
            create: false,
            write_mode: WriteMode::TailLocalizedWrites,
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

        let io_uring = Arc::new(parking_lot::Mutex::new(IoUring::new(8)?));

        let flusher = Arc::new(match self.write_mode {
            WriteMode::TailLocalizedWrites => FlushBehavior::NoWaitAppender(Appender::new(
                io_uring,
                Arc::clone(&file),
                self.write_mode,
            )),
            WriteMode::SerializedWrites => FlushBehavior::WaitAppender(Appender::new(
                io_uring,
                Arc::clone(&file),
                self.write_mode,
            )),
        });

        let ring = FlushBufferRing::with_flusher(RING_SIZE, FOUR_KB_PAGE, flusher.clone());
        io::Result::Ok(LogStructuredStore {
            buffer: ring,
            flusher,
            store: file,
            hi_stable: AtomicU64::new(u64::MAX),
            completed_islands: RwLock::new(BTreeSet::new()),
        })
    }
}

