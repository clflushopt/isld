/// Quickselect is a selection algorithm to find the kth smallest element in an unordered list.
///
/// Quickselect has a good average-case performance and is very cache friendly and in cases where
/// you need a Top-K when K > √N it gives better algorithmic performance due to O(N) being somewhat
/// less than O(N log K).
struct QuickSelect {}

impl QuickSelect {
    /// Lomuto partition scheme as used in quicksort.
    fn partition<T: Ord>(data: &mut [T], left: usize, right: usize, pivot: usize) -> usize {
        if data.is_empty() || left >= right {
            return left;
        }
        // Move pivot to the end.
        data.swap(pivot, right);
        let mut store = left;

        for i in left..right {
            if data[i] < data[right] {
                data.swap(store, i);
                store += 1;
            }
        }

        // Move pivot to its final place.
        data.swap(right, store);

        store
    }

    /// Returns the k-th smallest element of the list within left..right inclusive.
    fn select<T: Ord + Copy>(data: &mut [T], left: usize, right: usize, k: usize) -> T {
        let mut left = left;
        let mut right = right;
        loop {
            if left == right {
                return data[left];
            }

            // Pivot is the middle of the current range.
            let mut pivot = left + (right - left) / 2;

            pivot = Self::partition(data, left, right, pivot);

            if k == pivot {
                return data[k];
            } else if k < pivot {
                right = pivot - 1;
            } else {
                left = pivot + 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QuickSelect;

    #[test]
    fn test_partition_empty_slice() {
        let mut data: Vec<i32> = vec![];
        // Can't partition empty slice meaningfully - just verify no panic
        // Don't call partition with invalid indices
        let _ = QuickSelect::partition(data.as_mut_slice(), 0, 0, 0);
    }

    #[test]
    fn test_partition_one_element() {
        let mut data = vec![42];
        let pivot_pos = QuickSelect::partition(&mut data, 0, 0, 0);
        assert_eq!(pivot_pos, 0);
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn test_partition_two_elements_sorted() {
        let mut data = vec![1, 2];
        let pivot_pos = QuickSelect::partition(&mut data, 0, 1, 1); // pivot=2
        assert_eq!(pivot_pos, 1);
        assert!(data[0] <= data[1]);
    }

    #[test]
    fn test_partition_two_elements_unsorted() {
        let mut data = vec![2, 1];
        let pivot_pos = QuickSelect::partition(&mut data, 0, 1, 1); // pivot=1
        assert_eq!(pivot_pos, 0);
        assert_eq!(data, vec![1, 2]);
    }

    #[test]
    fn test_partition_already_sorted() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let pivot_pos = QuickSelect::partition(&mut data, 0, 9, 4); // pivot value=5

        // Everything before pivot_pos should be <= 5
        for i in 0..pivot_pos {
            assert!(data[i] <= data[pivot_pos]);
        }
        // Everything after pivot_pos should be > 5
        for i in pivot_pos + 1..data.len() {
            assert!(data[i] > data[pivot_pos]);
        }
    }

    #[test]
    fn test_partition_reverse_sorted() {
        let mut data = vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        let pivot_pos = QuickSelect::partition(&mut data, 0, 9, 4); // pivot value=6

        for i in 0..pivot_pos {
            assert!(data[i] <= data[pivot_pos]);
        }
        for i in pivot_pos + 1..data.len() {
            assert!(data[i] > data[pivot_pos]);
        }
    }

    #[test]
    fn test_partition_all_equal() {
        let mut data = vec![5, 5, 5, 5, 5];
        let pivot_pos = QuickSelect::partition(&mut data, 0, 4, 2);
        // Should not infinite loop, pivot goes somewhere valid
        assert!(pivot_pos <= 4);
    }

    #[test]
    fn test_partition_subarray() {
        let mut data = vec![100, 5, 3, 8, 2, 7, 100];
        // Only partition indices 1..5, leave 0 and 6 untouched
        let pivot_pos = QuickSelect::partition(&mut data, 1, 5, 3); // pivot value=8

        assert_eq!(data[0], 100); // Untouched
        assert_eq!(data[6], 100); // Untouched

        for i in 1..pivot_pos {
            assert!(data[i] <= data[pivot_pos]);
        }
        for i in pivot_pos + 1..6 {
            assert!(data[i] > data[pivot_pos]);
        }
    }

    #[test]
    fn test_select_1() {
        let mut data = vec![7, 2, 5, 1, 8, 3];
        let right = data.len() - 1;
        // left = 0
        // right = 5
        // 3-rd smallest is "index = 2"
        let elem = QuickSelect::select(data.as_mut_slice(), 0, 5, 2);
        assert_eq!(elem, 3);
    }
}
