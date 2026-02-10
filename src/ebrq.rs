//! Lock-free unbounded queue with memory reclamation.

use std::{
    cell::RefCell,
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicPtr, AtomicUsize, Ordering},
    },
};

/// Global epoch manager keeps track of the current global epoch.
struct EpochManager {
    epoch: AtomicUsize,
}

impl EpochManager {
    const fn new() -> Self {
        Self {
            epoch: AtomicUsize::new(0),
        }
    }

    fn current(&self) -> usize {
        self.epoch.load(Ordering::Acquire)
    }

    fn advance(&self) -> bool {
        let current = self.current();

        // Find minimum epoch across all active threads.
        let min_epoch = {
            let threads = THREADS.lock().unwrap();

            threads
                .iter()
                .map(|t| t.load(Ordering::Acquire))
                .filter(|&e| e != usize::MAX)
                .min()
                .unwrap_or(current)
        };

        // We can only advance if all threads are at least at (current - 1).
        if min_epoch >= current.saturating_sub(1) {
            self.epoch.fetch_add(1, Ordering::Release);
            true
        } else {
            false
        }
    }
}

static EPOCH: EpochManager = EpochManager::new();

/// Values that are waiting to be freed are wrapped pointers with an associated
/// deletion function and an epoch designating when the value was retired.
struct Garbage {
    // When was this value retired.
    epoch: usize,
    // Pointer to free.
    ptr: *mut u8,
    // Deletion function.
    deleter: unsafe fn(*mut u8),
}

// SAFETY: The pointer is only accessed via the type-erased deleter which
// correctly reconstructs the original type. Safe to send across threads.
unsafe impl Send for Garbage {}

// Type-erased deleter that calls `drop`.
unsafe fn deleter<T>(ptr: *mut u8) {
    unsafe {
        drop(Box::from_raw(ptr as *mut T));
    }
}

/// Global garbage list so any thread's GC pass can collect all retired nodes.
static GARBAGE: Mutex<Vec<Garbage>> = Mutex::new(Vec::new());

fn gc() {
    let current_epoch = EPOCH.current();

    // Safe to free anything from `n` epochs ago.
    let safe_epoch = current_epoch.saturating_sub(3);

    let mut list = GARBAGE.lock().unwrap();

    list.retain(|g| {
        if g.epoch <= safe_epoch {
            unsafe {
                (g.deleter)(g.ptr);
            }
            false
        } else {
            true
        }
    });
}

/// Guard value to implement RAII style pin & drop which gives us ergonomics like the ones
/// used for say a `RwLock` or `Mutex` but instead will depend on the epoch metadata to decide
/// how the guarded value is retired.
struct Guard {
    _epoch: usize,
    thread_epoch: Arc<AtomicUsize>,
}

impl Guard {
    fn pin() -> Self {
        let epoch = EPOCH.current();
        let thread_epoch = LOCAL_STATE.with(|local| {
            let state = local.borrow();
            let tg = state
                .as_ref()
                .expect("thread not registered; call register_thread() first");
            tg.epoch.store(epoch, Ordering::Release);
            tg.epoch.clone()
        });

        Guard {
            _epoch: epoch,
            thread_epoch,
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        // Unpin this thread.
        self.thread_epoch.store(usize::MAX, Ordering::Release);

        // Advance the global epoch and run the GC pass.
        if EPOCH.advance() {
            gc();
        }
    }
}

/// Pins the current thread to the global epoch.
fn pin() -> Guard {
    Guard::pin()
}

/// Schedule a value to be freed later.
fn defer_destroy<T>(ptr: *mut T) {
    let epoch = EPOCH.current();

    let mut garbage = GARBAGE.lock().unwrap();
    garbage.push(Garbage {
        epoch,
        ptr: ptr as *mut u8,
        deleter: deleter::<T>,
    });
}

/// Thread-local guard that deregisters from the global registry on thread exit.
struct ThreadGuard {
    epoch: Arc<AtomicUsize>,
}

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        // Mark as exited.
        self.epoch.store(usize::MAX, Ordering::Release);
        // Remove from global registry.
        let mut threads = THREADS.lock().unwrap();
        threads.retain(|t| !Arc::ptr_eq(t, &self.epoch));
    }
}

/// Registry of all active threads.
static THREADS: Mutex<Vec<Arc<AtomicUsize>>> = Mutex::new(Vec::new());

thread_local! {
    static LOCAL_STATE: RefCell<Option<ThreadGuard>> = RefCell::new(None);
}

/// Register the current thread.
fn register_thread() {
    LOCAL_STATE.with(|local| {
        let mut state = local.borrow_mut();
        if state.is_none() {
            let epoch = Arc::new(AtomicUsize::new(usize::MAX));
            THREADS.lock().unwrap().push(epoch.clone());
            *state = Some(ThreadGuard { epoch });
        }
    })
}

struct Node<T> {
    value: Option<T>, // None for sentinel node
    next: AtomicPtr<Node<T>>,
}

struct Queue<T> {
    head: AtomicPtr<Node<T>>,
    tail: AtomicPtr<Node<T>>,
}

impl<T> Queue<T> {
    fn new() -> Self {
        // Create sentinel node
        let sentinel = Box::into_raw(Box::new(Node {
            value: None,
            next: AtomicPtr::new(ptr::null_mut()),
        }));

        Self {
            head: AtomicPtr::new(sentinel),
            tail: AtomicPtr::new(sentinel),
        }
    }

    fn enqueue(&self, value: T) {
        // Pin so that tail (and any node we dereference) can't be freed under us.
        let _guard = pin();

        let new_node = Box::into_raw(Box::new(Node {
            value: Some(value),
            next: AtomicPtr::new(ptr::null_mut()),
        }));

        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe { (*tail).next.load(Ordering::Acquire) };

            // Check tail is still consistent
            if tail != self.tail.load(Ordering::Acquire) {
                continue;
            }

            if next.is_null() {
                // Tail is indeed the last node, try to append
                unsafe {
                    if (*tail)
                        .next
                        .compare_exchange_weak(
                            ptr::null_mut(),
                            new_node,
                            Ordering::Release,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        // Success! Try to swing tail
                        let _ = self.tail.compare_exchange(
                            tail,
                            new_node,
                            Ordering::Release,
                            Ordering::Acquire,
                        );
                        return;
                    }
                }
            } else {
                // Tail is behind, help advance it
                let _ =
                    self.tail
                        .compare_exchange(tail, next, Ordering::Release, Ordering::Acquire);
            }
        }
    }

    fn dequeue(&self) -> Option<T> {
        // Pin to current epoch!
        let _guard = pin();

        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe { (*head).next.load(Ordering::Acquire) };

            // Check head is still consistent
            if head != self.head.load(Ordering::Acquire) {
                continue;
            }

            if head == tail {
                // Queue is empty or tail is behind
                if next.is_null() {
                    return None; // Empty
                }

                // Tail is behind, help advance it
                let _ =
                    self.tail
                        .compare_exchange(tail, next, Ordering::Release, Ordering::Acquire);
            } else {
                // Try to swing head
                if self
                    .head
                    .compare_exchange_weak(head, next, Ordering::Release, Ordering::Acquire)
                    .is_ok()
                {
                    // CAS succeeded — we have exclusive access to next's value
                    // since no other dequeuer can win the same CAS.
                    let value = unsafe { (*next).value.take() };

                    // Defer freeing the old head (now-retired sentinel).
                    defer_destroy(head);

                    return value;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // Tests share global EBR state (EPOCH, GARBAGE, THREADS) and must not
    // run in parallel — serialize them with a mutex.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_basic() {
        let _lock = TEST_LOCK.lock().unwrap();
        register_thread();
        let q = Queue::new();

        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);

        assert_eq!(q.dequeue(), Some(1));
        assert_eq!(q.dequeue(), Some(2));
        assert_eq!(q.dequeue(), Some(3));
        assert_eq!(q.dequeue(), None);
    }

    #[test]
    fn test_no_leaks() {
        let _lock = TEST_LOCK.lock().unwrap();
        register_thread();

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct DropCounter;
        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        let q = Queue::new();

        for _ in 0..1000 {
            q.enqueue(DropCounter);
        }

        for _ in 0..1000 {
            q.dequeue();
        }

        // Force garbage collection
        for _ in 0..10 {
            let _guard = pin();
        }

        // All 1000 should be dropped
        assert_eq!(DROP_COUNT.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_concurrent_with_done_signal() {
        let _lock = TEST_LOCK.lock().unwrap();
        register_thread();
        let q = Arc::new(Queue::new());
        const THREADS: usize = 8;
        const OPS_PER_THREAD: usize = 10_000;

        use std::sync::Barrier;

        // Synchronize start
        let start_barrier = Arc::new(Barrier::new(THREADS * 2));
        let mut handles = vec![];

        // Producers
        for t in 0..THREADS {
            let q = q.clone();
            let barrier = start_barrier.clone();
            handles.push(thread::spawn(move || {
                register_thread();
                barrier.wait(); // Wait for all threads to be ready

                for i in 0..OPS_PER_THREAD {
                    q.enqueue(t * OPS_PER_THREAD + i);
                }
            }));
        }

        // Consumers
        let consumed = Arc::new(AtomicUsize::new(0));
        let total_items = THREADS * OPS_PER_THREAD;

        for _ in 0..THREADS {
            let q = q.clone();
            let barrier = start_barrier.clone();
            let consumed = consumed.clone();

            handles.push(thread::spawn(move || {
                register_thread();
                barrier.wait(); // Wait for all threads to be ready

                loop {
                    if let Some(_) = q.dequeue() {
                        let prev = consumed.fetch_add(1, Ordering::Relaxed);

                        // Check if we're done
                        if prev + 1 >= total_items {
                            break;
                        }
                    } else {
                        // Queue empty — check if all items have been consumed.
                        if consumed.load(Ordering::Relaxed) >= total_items {
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(consumed.load(Ordering::Relaxed), total_items);
        assert_eq!(q.dequeue(), None);
    }
}
