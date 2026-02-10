#![feature(core_intrinsics)]
#![feature(unsafe_cell_access)]
pub mod ebrq;
pub mod nblfq;
pub mod select;

pub struct EytzingerTree<T> {
    data: Vec<T>,
}

impl<T: Ord + Copy + Default> EytzingerTree<T> {
    pub fn new(sorted: &[T]) -> Self {
        let mut data = vec![T::default(); sorted.len()];
        let mut index = 0;
        Self::build(&mut data, 0, sorted, &mut index);
        Self { data }
    }

    /// Converts sorted slice into 0-indexed Eytzinger layout
    fn build(data: &mut [T], k: usize, sorted: &[T], index: &mut usize) {
        if k >= data.len() {
            return;
        }

        // Left subtree
        Self::build(data, 2 * k + 1, sorted, index);

        // Current node
        data[k] = sorted[*index];
        *index += 1;

        // Right subtree
        Self::build(data, 2 * k + 2, sorted, index);
    }

    pub fn search(&self, target: T) -> Option<usize> {
        let n = self.data.len();
        let mut i = 0;

        // Navigate to leaf level
        while i < n {
            if target < self.data[i] {
                i = 2 * i + 1; // go left
            } else if target > self.data[i] {
                i = 2 * i + 2; // go right
            } else {
                return Some(i); // found it
            }
        }

        None
    }

    pub fn search_optimized(&self, target: T) -> Option<usize> {
        let n = self.data.len();
        let mut i = 0;
        let mut candidate = n; // Invalid index means "not found"

        while i < n {
            // Branchless: update candidate if we found target
            // If data[i] == target, use i, otherwise keep candidate
            let is_match = (self.data[i] == target) as usize;
            candidate = is_match * i + (1 - is_match) * candidate;

            // Branch-free navigation
            i = 2 * i + 1 + (self.data[i] < target) as usize;
        }

        if candidate < n { Some(candidate) } else { None }
    }
    pub fn as_ref(&self) -> &[T] {
        &self.data
    }

    pub fn search_prefetch(&self, target: T) -> Option<usize> {
        let n = self.data.len();
        let mut i = 0;
        let mut candidate = n;

        while i < n {
            // Prefetch both children
            let left = 2 * i + 1;
            let right = 2 * i + 2;

            if left < n {
                unsafe {
                    std::intrinsics::prefetch_read_data::<T, 3>(self.data.as_ptr().add(left));
                }
            }
            if right < n {
                unsafe {
                    std::intrinsics::prefetch_read_data::<T, 3>(self.data.as_ptr().add(right));
                }
            }

            // Branchless search
            let is_match = (self.data[i] == target) as usize;
            candidate = is_match * i + (1 - is_match) * candidate;

            let go_right = (self.data[i] < target) as usize;
            i = 2 * i + 1 + go_right;
        }

        if candidate < n { Some(candidate) } else { None }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eytzinger_layout() {
        {
            let sorted = vec![1, 2, 3, 4, 5, 6, 7];
            // Expected layout:
            //
            //       4
            //      / \
            //     2   6
            //    / \ / \
            //   1  3 5  7
            //
            // Position: [0,  1,  2,  3,  4,  5,  6,  7]
            // Value:    [4,  2,  6,  1,  3,  5,  7]
            let eytz = EytzingerTree::new(&sorted);
            assert_eq!(eytz.as_ref(), vec![4, 2, 6, 1, 3, 5, 7]);
        }

        {
            let sorted = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            let expected = vec![6, 3, 8, 1, 5, 7, 9, 0, 2, 4];
            assert_eq!(EytzingerTree::new(&sorted).as_ref(), expected);
        }
    }

    #[test]
    fn test_search() {
        let sorted = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let tree = EytzingerTree::new(&sorted);

        // Should find all elements
        for i in 0..10 {
            assert!(tree.search(i).is_some());
            assert!(tree.search_optimized(i).is_some());
            assert!(tree.search_prefetch(i).is_some());
        }
        println!("{:?}", tree.data);

        // Should not find these
        assert!(tree.search(10).is_none());
        assert!(tree.search(-1).is_none());
        assert!(tree.search_optimized(10).is_none());
        assert!(tree.search_optimized(-1).is_none());
        assert!(tree.search_prefetch(10).is_none());
        assert!(tree.search_prefetch(-1).is_none());
    }
}
