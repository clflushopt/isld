//! Lock-free unbounded MPMC queue (Michael-Scott) backed by epoch-based
//! reclamation from [`crate::ebr`].

use std::{
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::ebr::LocalHandle;

struct Node<T> {
    value: Option<T>,
    next: AtomicPtr<Node<T>>,
}

/// A lock-free unbounded FIFO queue.
///
/// Operations require a [`LocalHandle`] obtained from an
/// [`ebr::Collector`](crate::ebr::Collector).
pub struct Queue<T> {
    head: AtomicPtr<Node<T>>,
    tail: AtomicPtr<Node<T>>,
}

impl<T> Queue<T> {
    /// Create an empty queue with a sentinel node.
    pub fn new() -> Self {
        let sentinel = Box::into_raw(Box::new(Node {
            value: None,
            next: AtomicPtr::new(ptr::null_mut()),
        }));

        Self {
            head: AtomicPtr::new(sentinel),
            tail: AtomicPtr::new(sentinel),
        }
    }

    /// Append `value` to the back of the queue.
    pub fn enqueue(&self, value: T, handle: &LocalHandle) {
        let _guard = handle.pin();

        let new_node = Box::into_raw(Box::new(Node {
            value: Some(value),
            next: AtomicPtr::new(ptr::null_mut()),
        }));

        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe { (*tail).next.load(Ordering::Acquire) };

            if tail != self.tail.load(Ordering::Acquire) {
                continue;
            }

            if next.is_null() {
                // Tail is the last node — try to append.
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
                // Tail is behind — help advance it.
                let _ =
                    self.tail
                        .compare_exchange(tail, next, Ordering::Release, Ordering::Acquire);
            }
        }
    }

    /// Remove and return the value at the front, or `None` if empty.
    pub fn dequeue(&self, handle: &LocalHandle) -> Option<T> {
        let guard = handle.pin();

        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe { (*head).next.load(Ordering::Acquire) };

            if head != self.head.load(Ordering::Acquire) {
                continue;
            }

            if head == tail {
                if next.is_null() {
                    return None;
                }
                // Tail is behind — help advance it.
                let _ =
                    self.tail
                        .compare_exchange(tail, next, Ordering::Release, Ordering::Acquire);
            } else {
                if self
                    .head
                    .compare_exchange_weak(head, next, Ordering::Release, Ordering::Acquire)
                    .is_ok()
                {
                    // CAS succeeded — exclusive access to next's value.
                    let value = unsafe { (*next).value.take() };
                    guard.defer_destroy(head);
                    return value;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ebr::Collector;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    #[test]
    fn basic_enqueue_dequeue() {
        let c = Collector::new();
        let h = c.register();
        let q = Queue::new();

        q.enqueue(1, &h);
        q.enqueue(2, &h);
        q.enqueue(3, &h);

        assert_eq!(q.dequeue(&h), Some(1));
        assert_eq!(q.dequeue(&h), Some(2));
        assert_eq!(q.dequeue(&h), Some(3));
        assert_eq!(q.dequeue(&h), None);
    }

    #[test]
    fn no_leaks() {
        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct DropCounter;
        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        DROP_COUNT.store(0, Ordering::Relaxed);

        let c = Collector::new();
        let h = c.register();
        let q = Queue::new();

        for _ in 0..1000 {
            q.enqueue(DropCounter, &h);
        }

        for _ in 0..1000 {
            q.dequeue(&h);
        }

        // Flush garbage.
        for _ in 0..10 {
            let _g = h.pin();
        }

        assert_eq!(DROP_COUNT.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn concurrent_mpmc() {
        let c = Collector::new();
        let q = Arc::new(Queue::new());

        const THREADS: usize = 8;
        const OPS: usize = 10_000;

        let barrier = Arc::new(std::sync::Barrier::new(THREADS * 2));
        let consumed = Arc::new(AtomicUsize::new(0));
        let total = THREADS * OPS;

        let mut handles = Vec::new();

        // Producers.
        for t in 0..THREADS {
            let c = Arc::clone(&c);
            let q = Arc::clone(&q);
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                let h = c.register();
                barrier.wait();
                for i in 0..OPS {
                    q.enqueue(t * OPS + i, &h);
                }
            }));
        }

        // Consumers.
        for _ in 0..THREADS {
            let c = Arc::clone(&c);
            let q = Arc::clone(&q);
            let barrier = barrier.clone();
            let consumed = consumed.clone();
            handles.push(thread::spawn(move || {
                let h = c.register();
                barrier.wait();
                loop {
                    if let Some(_) = q.dequeue(&h) {
                        let prev = consumed.fetch_add(1, Ordering::Relaxed);
                        if prev + 1 >= total {
                            break;
                        }
                    } else {
                        if consumed.load(Ordering::Relaxed) >= total {
                            break;
                        }
                        thread::yield_now();
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(consumed.load(Ordering::Relaxed), total);
        assert_eq!(q.dequeue(&c.register()), None);
    }
}
