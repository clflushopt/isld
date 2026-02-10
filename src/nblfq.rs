//! Lock-free bounded queue.
use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicU64, Ordering},
};

#[repr(transparent)]
pub struct Cell(AtomicU64);

impl Cell {
    /// All bits are set.
    const EMPTY: u32 = u32::MAX;
    /// Pack an index and counter into a single u64.
    fn pack(index: u32, counter: u32) -> u64 {
        // Counter is left-most 32 bits.
        (counter as u64) << 32 | index as u64
    }

    /// Unpack a packed cell into (index, counter).
    fn unpack(packed: u64) -> (u32, u32) {
        (packed as u32, (packed >> 32) as u32)
    }

    /// Creates a new empty cell.
    fn new() -> Self {
        Self(AtomicU64::new(Self::pack(Self::EMPTY, 0)))
    }

    /// Loads the value from the underlying atomic.
    fn load(&self, ordering: Ordering) -> u64 {
        self.0.load(ordering)
    }

    /// Write the value to the underlying atomic.
    fn store(&self, value: u64, ordering: Ordering) {
        self.0.store(value, ordering)
    }
}

pub struct Queue<T> {
    /// Ring buffer of packed cells.
    cells: Box<[Cell]>,

    /// Storage for the actual data.
    slots: Box<[UnsafeCell<MaybeUninit<T>>]>,

    /// Next position to enqueue.
    head: AtomicU64,

    /// Next position to dequeue.
    tail: AtomicU64,

    /// Capacity (must be power of 2 for fast modulo operations).
    capacity: usize,

    /// Bitmask for fast modulo: capacity - 1.
    mask: usize,
}

unsafe impl<T> Send for Queue<T> {}
unsafe impl<T> Sync for Queue<T> {}

/// Implementation of a lock-free queue.
///
///
/// The cells array holds the synchronization state i.e. who owns what and generation counters.
/// Each cell tracks which "lap" around the ring buffer it's on as such the positions are monotonically
/// increasing, the cell index wraps around and the "lap" is `position / capacity` which iteration around
/// the ring it is on.
///
/// A cell at index `i` is ready for :
///
/// - Enqueue when: cell's counter == current lap for that position.
/// - Dequeue when: cell contains a valid index and the counter matches.
///
/// The SM can be seen as :
///
/// 1. Initial state (cell `i`):
///   - index = EMPTY  | counter = 0 signaling that we are ready to enqueue at position `i` for `lap = 0`
///
/// 2. After enqueuing at position `p` where `p & mask == i`:
///   - index = i | counter = `p / capacity` signaling "Contains data ready for dequeue".
///
/// 3. After dequeuing at position `p`:
///   - index = EMPTY | counter = `p / capacity + 1` signaling "Empty, ready for enqueue at position `p`".
///
///
/// When enqueueing the process goes as follows :
///
/// 1. Claim a cell position via a CAS on `head`.
/// 2. Write data to the corresponding `slot` entry.
/// 3. Update the cell to indicate that the data is present.
///
/// When dequeueing we do the same thing but update to mark the slot as empty
/// and ready for the next lap.
///
/// 1. Claim a cell position via CAS on `tail`.
/// 2. Read data from the corresponding slot.
/// 3. Update the cell to indicate the slot is empty.
///
impl<T> Queue<T> {
    /// Creates a new empty queue with a bounded capacity. If `capacity` is not a power
    /// of two we `panic`.
    ///
    /// The maximum capacity is `u32::MAX`.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());
        assert!(capacity <= u32::MAX as usize);

        // Initialize all the cells, each cell starts ready for position `i` and `lap = 0`.
        let cells: Box<[Cell]> = (0..capacity)
            .map(|_| {
                // Empty, counter = 0 means "ready for lap = 0".
                Cell::new()
            })
            .collect();

        // Initialize slots as uninit.
        let slots: Box<[UnsafeCell<MaybeUninit<T>>]> = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect();

        Self {
            cells,
            slots,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            capacity,
            mask: capacity - 1,
        }
    }

    /// Enqueues a value into the queue, returns `Ok(())` on success
    /// and the original value `Err` wrapped.
    pub fn enqueue(&self, value: T) -> Result<(), T> {
        loop {
            // Reading the current head position has no synchronization requirements.
            let pos = self.head.load(Ordering::Relaxed);

            let cell_index = pos as usize & self.mask;
            let cell = &self.cells[cell_index];

            // Load the value in the cell, this requires `Acquire` semantics since
            // we must see writes from previous dequeue ops.
            let packed = cell.load(Ordering::Acquire);

            // Unpack the index, counter pair.
            let (index, counter) = Cell::unpack(packed);
            let lap = (pos as u32) / (self.capacity as u32);

            // This cell is ready.
            if counter == lap && index == Cell::EMPTY {
                // Position claim is not a synchronization the write to the cell synchronizes.
                match self
                    .head
                    .compare_exchange(pos, pos + 1, Ordering::Relaxed, Ordering::Relaxed)
                {
                    Ok(_) => {
                        // We own this cell and can write data to the slot.
                        unsafe {
                            self.slots[cell_index].replace(MaybeUninit::new(value));
                        }
                        cell.store(Cell::pack(cell_index as u32, counter), Ordering::Release);
                        return Ok(());
                    }
                    // Mark cell as containing data. Release ensures slot write is visible.
                    Err(_) => continue, // failure
                }
            } else if counter < lap {
                // Cell is behind - queue is full and the tail hasn't caught up to free
                // this cell yet.
                return Err(value);
            }
        }
    }

    /// Dequeue a value from the queue.
    pub fn dequeue(&self) -> Option<T> {
        loop {
            // Reading position, no data depends on this yet.
            let pos = self.tail.load(Ordering::Relaxed);
            let cell_index = pos as usize & self.mask;
            let cell = &self.cells[cell_index];

            // The load on the cell must see writes from previous operations it must synchronize
            // with enqueue's `Release` store to see slot data.
            let packed = cell.load(Ordering::Acquire);
            let (index, counter) = Cell::unpack(packed);
            let lap = pos as u32 / self.capacity as u32;

            if counter == lap && index != Cell::EMPTY {
                // Position is claimed, loading with `Acquire` makes it visible.
                match self
                    .tail
                    .compare_exchange(pos, pos + 1, Ordering::Relaxed, Ordering::Relaxed)
                {
                    Ok(_) => {
                        let value = unsafe { self.slots[cell_index].get().read().assume_init() };
                        // Publishes `counter + 1` so next-lap enqueue sees the cell is free via its `Acquire` load.
                        cell.store(Cell::pack(Cell::EMPTY, counter + 1), Ordering::Release);

                        return Some(value);
                    }
                    Err(_) => continue,
                }
            } else if counter < lap || counter == lap && index == Cell::EMPTY {
                // Queue is empty.
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use shuttle::thread;

    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let cases = [
            (0, 0),
            (1, 0),
            (0, 1),
            (Cell::EMPTY, 0),
            (123456, 67890),
            (u32::MAX - 1, u32::MAX),
        ];

        for (idx, ctr) in cases {
            let packed = Cell::pack(idx, ctr);
            assert_eq!((idx, ctr), Cell::unpack(packed));
        }
    }

    #[test]
    fn test_packed_layout() {
        // Counter in high bits and index in low bits.
        let packed = Cell::pack(0x1234_5678, 0xABCD_EF00);
        assert_eq!(packed, 0xABCD_EF00_1234_5678);
    }

    #[test]
    fn shuttle_test_mpmc() {
        shuttle::check_random(
            || {
                let queue = Arc::new(Queue::new(128));
                let mut handles = vec![];

                for i in 0..8 {
                    let q = queue.clone();
                    handles.push(thread::spawn(move || {
                        for j in 0..4 {
                            let _ = q.enqueue(i * 10 + j);
                        }
                    }))
                }

                let results = Arc::new(shuttle::sync::Mutex::new(vec![]));
                for _ in 0..4 {
                    let q = queue.clone();
                    let r = results.clone();
                    handles.push(thread::spawn(move || {
                        for _ in 0..4 {
                            loop {
                                if let Some(v) = q.dequeue() {
                                    r.lock().unwrap().push(v);
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

                let mut results = results.lock().unwrap();
                results.sort();
                assert_eq!(results.len(), 16);
            },
            100,
        );
    }
}
