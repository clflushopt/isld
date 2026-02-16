//! Epoch-based memory reclamation.
//!
//! Provides safe deferred destruction for lock-free data structures. Threads
//! *pin* themselves to the current epoch before accessing shared pointers, and
//! *retire* pointers they remove. Retired pointers are only freed once every
//! thread has moved past the epoch in which the pointer was retired.
//!
//! # Usage
//!
//! ```ignore
//! let collector = Collector::new();
//!
//! // Each thread registers once.
//! let handle = collector.register();
//!
//! // Pin before accessing shared pointers.
//! let guard = handle.pin();
//! // ... read / CAS shared AtomicPtrs ...
//! guard.defer_destroy(retired_ptr);
//! // guard unpins on drop, may trigger GC.
//! ```

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

/// Type-erased record of a pointer waiting to be freed.
struct Garbage {
    epoch: usize,
    ptr: *mut u8,
    deleter: unsafe fn(*mut u8),
}

// SAFETY: The pointer is only accessed via the type-erased deleter which
// correctly reconstructs the original type.
unsafe impl Send for Garbage {}

/// Type-erased deleter that reconstructs and drops a `Box<T>`.
unsafe fn drop_box<T>(ptr: *mut u8) {
    unsafe {
        drop(Box::from_raw(ptr as *mut T));
    }
}

/// Owns all shared EBR state: the global epoch, the thread registry, and the
/// garbage list. Create one per logical "domain" of shared pointers.
pub struct Collector {
    epoch: AtomicUsize,
    threads: Mutex<Vec<Arc<AtomicUsize>>>,
    garbage: Mutex<Vec<Garbage>>,
}

impl Collector {
    /// Create a new collector. The returned `Arc` is cheap to clone and should
    /// be shared with every thread that will participate.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            epoch: AtomicUsize::new(0),
            threads: Mutex::new(Vec::new()),
            garbage: Mutex::new(Vec::new()),
        })
    }

    /// Register a thread and obtain a [`LocalHandle`] for pinning.
    pub fn register(self: &Arc<Self>) -> LocalHandle {
        let epoch = Arc::new(AtomicUsize::new(usize::MAX));
        self.threads.lock().unwrap().push(epoch.clone());
        LocalHandle {
            collector: Arc::clone(self),
            epoch,
        }
    }

    /// Try to advance the global epoch. Uses `try_lock` to avoid contention —
    /// if another thread is already checking, we simply skip this attempt.
    fn advance(&self) -> bool {
        let current = self.epoch.load(Ordering::Acquire);

        let min_epoch = {
            let threads = match self.threads.try_lock() {
                Ok(t) => t,
                Err(_) => return false,
            };
            threads
                .iter()
                .map(|t| t.load(Ordering::Acquire))
                .filter(|&e| e != usize::MAX)
                .min()
                .unwrap_or(current)
        };

        if min_epoch >= current.saturating_sub(1) {
            self.epoch.fetch_add(1, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Free garbage entries that are old enough to be safe. Drains reclaimable
    /// entries under the lock, then runs destructors *outside* the lock to
    /// avoid blocking concurrent `defer` calls.
    fn gc(&self) {
        let current = self.epoch.load(Ordering::Acquire);
        let safe_epoch = current.saturating_sub(3);

        // Take all entries out, release the lock quickly.
        let entries: Vec<Garbage> = {
            let mut list = match self.garbage.try_lock() {
                Ok(l) => l,
                Err(_) => return,
            };
            std::mem::take(&mut *list)
        };

        let mut remaining = Vec::new();
        for g in entries {
            if g.epoch <= safe_epoch {
                unsafe { (g.deleter)(g.ptr) };
            } else {
                remaining.push(g);
            }
        }

        // Put back entries that weren't old enough.
        if !remaining.is_empty() {
            let mut list = self.garbage.lock().unwrap();
            remaining.append(&mut *list);
            *list = remaining;
        }
    }

    /// Push a garbage entry.
    fn defer(&self, garbage: Garbage) {
        self.garbage.lock().unwrap().push(garbage);
    }

    /// Current epoch value.
    fn current_epoch(&self) -> usize {
        self.epoch.load(Ordering::Acquire)
    }
}

/// Per-thread handle to a [`Collector`]. Provides [`pin`](LocalHandle::pin)
/// for entering a critical section.
pub struct LocalHandle {
    collector: Arc<Collector>,
    epoch: Arc<AtomicUsize>,
}

impl LocalHandle {
    /// Pin the current thread to the global epoch, returning an RAII
    /// [`Guard`]. While the guard is alive, no pointer retired *after* this
    /// epoch can be freed.
    pub fn pin(&self) -> Guard<'_> {
        let epoch = self.collector.current_epoch();
        self.epoch.store(epoch, Ordering::Release);
        Guard { handle: self }
    }
}

impl Drop for LocalHandle {
    fn drop(&mut self) {
        // Mark as inactive.
        self.epoch.store(usize::MAX, Ordering::Release);
        // Remove from registry.
        let mut threads = self.collector.threads.lock().unwrap();
        threads.retain(|t| !Arc::ptr_eq(t, &self.epoch));
    }
}

/// RAII proof that the current thread is pinned. Provides
/// [`defer_destroy`](Guard::defer_destroy) to retire pointers.
pub struct Guard<'a> {
    handle: &'a LocalHandle,
}

impl Guard<'_> {
    /// Schedule `ptr` (which must have been allocated via `Box::into_raw`) to
    /// be freed once it is safe to do so.
    pub fn defer_destroy<T>(&self, ptr: *mut T) {
        let epoch = self.handle.collector.current_epoch();
        self.handle.collector.defer(Garbage {
            epoch,
            ptr: ptr as *mut u8,
            deleter: drop_box::<T>,
        });
    }
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        // Unpin.
        self.handle.epoch.store(usize::MAX, Ordering::Release);
        // Try to advance + collect.
        if self.handle.collector.advance() {
            self.handle.collector.gc();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    #[test]
    fn pin_unpin_advances_epoch() {
        let c = Collector::new();
        let h = c.register();

        let e0 = c.epoch.load(Ordering::Relaxed);
        {
            let _g = h.pin();
        }
        let e1 = c.epoch.load(Ordering::Relaxed);
        assert!(e1 > e0, "epoch should advance after unpin");
    }

    #[test]
    fn deferred_values_are_freed() {
        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct Tracked;
        impl Drop for Tracked {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        DROP_COUNT.store(0, Ordering::Relaxed);

        let c = Collector::new();
        let h = c.register();

        // Retire 100 values.
        for _ in 0..100 {
            let guard = h.pin();
            let ptr = Box::into_raw(Box::new(Tracked));
            guard.defer_destroy(ptr);
        }

        // Pump the collector to flush garbage.
        for _ in 0..10 {
            let _g = h.pin();
        }

        assert_eq!(DROP_COUNT.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn handle_drop_deregisters() {
        let c = Collector::new();

        let h1 = c.register();
        let h2 = c.register();
        assert_eq!(c.threads.lock().unwrap().len(), 2);

        drop(h1);
        assert_eq!(c.threads.lock().unwrap().len(), 1);

        drop(h2);
        assert_eq!(c.threads.lock().unwrap().len(), 0);
    }

    #[test]
    fn concurrent_register_and_pin() {
        let c = Collector::new();
        let barrier = Arc::new(std::sync::Barrier::new(8));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let c = Arc::clone(&c);
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let h = c.register();
                    barrier.wait();
                    for _ in 0..1_000 {
                        let guard = h.pin();
                        let ptr = Box::into_raw(Box::new(42u64));
                        guard.defer_destroy(ptr);
                    }
                    // Flush.
                    for _ in 0..10 {
                        let _g = h.pin();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All threads deregistered.
        assert_eq!(c.threads.lock().unwrap().len(), 0);
    }
}
