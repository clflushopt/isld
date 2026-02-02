use criterion::{Criterion, criterion_group, criterion_main};
use isld::EytzingerTree;
use std::hint::black_box;
fn benchmark_eytzinger(c: &mut Criterion) {
    let sizes = vec![100usize, 1_000, 10_000, 100_000, 1_000_000];

    for size in sizes {
        let data: Vec<usize> = (0..size).collect();
        let eytz = EytzingerTree::new(&data);

        let mut group = c.benchmark_group(format!("search_size_{}", size));

        group.bench_function("std_binary_search", |b| {
            b.iter(|| {
                for i in (0..size).step_by(size / 100) {
                    black_box(data.binary_search(&i));
                }
            });
        });

        group.bench_function("eytzinger_simple", |b| {
            b.iter(|| {
                for i in (0..size).step_by(size / 100) {
                    black_box(eytz.search(i));
                }
            });
        });

        group.bench_function("eytzinger_branchless", |b| {
            b.iter(|| {
                for i in (0..size).step_by(size / 100) {
                    black_box(eytz.search_optimized(i));
                }
            });
        });

        group.bench_function("eytzinger_prefetch", |b| {
            b.iter(|| {
                for i in (0..size).step_by(size / 100) {
                    black_box(eytz.search_prefetch(i));
                }
            });
        });

        group.finish();
    }
}

criterion_group!(benches, benchmark_eytzinger);
criterion_main!(benches);
