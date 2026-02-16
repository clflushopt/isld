//! JOIN benchmark: BTreeMap vs HashMap vs UnchainedHashTable
//!
//! Simulates a database equi-join workload:
//!   SELECT * FROM probe_side JOIN build_side ON probe_side.key = build_side.key
//!
//! Measures:
//!   - Build throughput (tuples/sec to construct the index)
//!   - Probe throughput (lookups/sec across varying selectivity & multiplicity)
//!
//! Workload parameters:
//!   - Build size: number of tuples on the build side
//!   - Probe size: number of lookups to perform
//!   - Selectivity: fraction of probe keys that have a match (0.0 = no matches, 1.0 = all match)
//!   - Multiplicity: number of build-side duplicates per key (1 = unique, N = 1:N join)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, HashMap};
use std::hint::black_box;
use std::time::Duration;

use isld::sch::{BuildConfig, LocalCollector, UnchainedHashTable, build};

// How long to record measurements for.
const MEASURE_DURATION_SECS: u64 = 60;

struct JoinWorkload {
    /// (key, payload) pairs for the build side
    build_tuples: Vec<(u32, u64)>,
    /// Keys to probe (mix of matching and non-matching)
    probe_keys: Vec<u32>,
    /// Human-readable label
    label: String,
}

impl JoinWorkload {
    /// Generate a join workload.
    ///
    /// - `build_keys`: number of distinct keys on the build side
    /// - `multiplicity`: duplicates per key (total build tuples = build_keys * multiplicity)
    /// - `probe_count`: number of probe operations
    /// - `selectivity`: fraction of probe keys that exist in the build side
    fn generate(
        build_keys: usize,
        multiplicity: usize,
        probe_count: usize,
        selectivity: f64,
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Build side: keys 0..build_keys, each repeated `multiplicity` times
        let mut build_tuples = Vec::with_capacity(build_keys * multiplicity);
        for key in 0..build_keys as u32 {
            for dup in 0..multiplicity {
                let payload = (key as u64) * 1000 + dup as u64;
                build_tuples.push((key, payload));
            }
        }
        // Shuffle to simulate unordered input
        build_tuples.shuffle(&mut rng);

        // Probe side: selectivity% of keys hit, rest miss
        let matching_probes = (probe_count as f64 * selectivity) as usize;
        let missing_probes = probe_count - matching_probes;

        let mut probe_keys = Vec::with_capacity(probe_count);

        // Keys that exist in build side
        for _ in 0..matching_probes {
            probe_keys.push(rng.random_range(0..build_keys as u32));
        }
        // Keys that don't exist (offset beyond build key range)
        let miss_base = build_keys as u32;
        for _ in 0..missing_probes {
            probe_keys.push(miss_base + rng.random_range(0..build_keys as u32));
        }
        // Shuffle probe order
        probe_keys.shuffle(&mut rng);

        let total_build = build_keys * multiplicity;
        let label =
            format!("build={total_build}/probe={probe_count}/sel={selectivity}/mul={multiplicity}");

        Self {
            build_tuples,
            probe_keys,
            label,
        }
    }
}

trait JoinIndex {
    fn build_from(tuples: &[(u32, u64)]) -> Self;
    fn probe(&self, key: u32) -> u64; // returns sum of matched payloads (to prevent elision)
}

struct HashMapIndex {
    map: HashMap<u32, Vec<u64>>,
}

impl JoinIndex for HashMapIndex {
    fn build_from(tuples: &[(u32, u64)]) -> Self {
        let mut map: HashMap<u32, Vec<u64>> = HashMap::with_capacity(tuples.len());
        for &(key, payload) in tuples {
            map.entry(key).or_default().push(payload);
        }
        Self { map }
    }

    #[inline]
    fn probe(&self, key: u32) -> u64 {
        match self.map.get(&key) {
            Some(payloads) => {
                let mut sum = 0u64;
                for &p in payloads {
                    sum = sum.wrapping_add(p);
                }
                sum
            }
            None => 0,
        }
    }
}

struct BTreeMapIndex {
    map: BTreeMap<u32, Vec<u64>>,
}

impl JoinIndex for BTreeMapIndex {
    fn build_from(tuples: &[(u32, u64)]) -> Self {
        let mut map: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
        for &(key, payload) in tuples {
            map.entry(key).or_default().push(payload);
        }
        Self { map }
    }

    #[inline]
    fn probe(&self, key: u32) -> u64 {
        match self.map.get(&key) {
            Some(payloads) => {
                let mut sum = 0u64;
                for &p in payloads {
                    sum = sum.wrapping_add(p);
                }
                sum
            }
            None => 0,
        }
    }
}

struct UnchainedIndex {
    table: UnchainedHashTable,
}

impl UnchainedIndex {
    #[inline]
    fn bloom_check(&self, key: u32) -> bool {
        self.table.bloom_check(key)
    }
}

impl JoinIndex for UnchainedIndex {
    fn build_from(tuples: &[(u32, u64)]) -> Self {
        let stride = 16; // key (u64) + payload (u64)
        let config = BuildConfig::new(stride);
        let mut collector = LocalCollector::new(&config);
        for &(key, payload) in tuples {
            collector.insert(key, &[payload]);
        }
        let table = build(vec![collector], &config);
        Self { table }
    }

    #[inline]
    fn probe(&self, key: u32) -> u64 {
        let mut sum = 0u64;
        self.table.probe(key, |tuple| {
            sum = sum.wrapping_add(tuple[1]);
        });
        sum
    }
}

fn bench_build<T: JoinIndex>(tuples: &[(u32, u64)]) -> T {
    T::build_from(tuples)
}

fn bench_probe<T: JoinIndex>(index: &T, probe_keys: &[u32]) -> u64 {
    let mut total = 0u64;
    for &key in probe_keys {
        total = total.wrapping_add(index.probe(key));
    }
    total
}

fn bench_build_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    group.measurement_time(Duration::from_secs(MEASURE_DURATION_SECS));

    for &build_size in &[1_000, 10_000, 100_000, 1_000_000] {
        let workload = JoinWorkload::generate(build_size, 1, 0, 0.0, 42);
        group.throughput(Throughput::Elements(build_size as u64));

        group.bench_with_input(
            BenchmarkId::new("HashMap", build_size),
            &workload.build_tuples,
            |b, tuples| b.iter(|| bench_build::<HashMapIndex>(black_box(tuples))),
        );

        group.bench_with_input(
            BenchmarkId::new("BTreeMap", build_size),
            &workload.build_tuples,
            |b, tuples| b.iter(|| bench_build::<BTreeMapIndex>(black_box(tuples))),
        );

        group.bench_with_input(
            BenchmarkId::new("Unchained", build_size),
            &workload.build_tuples,
            |b, tuples| b.iter(|| bench_build::<UnchainedIndex>(black_box(tuples))),
        );
    }

    group.finish();
}

fn bench_probe_selectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("probe_selectivity");
    group.measurement_time(Duration::from_secs(MEASURE_DURATION_SECS));

    let build_size = 100_000;
    let probe_count = 1_000_000;

    // Varying selectivity: 0% (all misses), 1%, 10%, 50%, 100% (all hits)
    for &selectivity in &[0.0, 0.01, 0.1, 0.5, 1.0] {
        let workload = JoinWorkload::generate(build_size, 1, probe_count, selectivity, 42);
        let sel_label = format!("{:.0}pct", selectivity * 100.0);

        group.throughput(Throughput::Elements(probe_count as u64));

        // Pre-build indices (not measured)
        let hm = HashMapIndex::build_from(&workload.build_tuples);
        let bt = BTreeMapIndex::build_from(&workload.build_tuples);
        let uc = UnchainedIndex::build_from(&workload.build_tuples);

        group.bench_with_input(
            BenchmarkId::new("HashMap", &sel_label),
            &workload.probe_keys,
            |b, keys| b.iter(|| bench_probe(&hm, black_box(keys))),
        );

        group.bench_with_input(
            BenchmarkId::new("BTreeMap", &sel_label),
            &workload.probe_keys,
            |b, keys| b.iter(|| bench_probe(&bt, black_box(keys))),
        );

        group.bench_with_input(
            BenchmarkId::new("Unchained", &sel_label),
            &workload.probe_keys,
            |b, keys| b.iter(|| bench_probe(&uc, black_box(keys))),
        );
    }

    group.finish();
}

fn bench_probe_multiplicity(c: &mut Criterion) {
    let mut group = c.benchmark_group("probe_multiplicity");
    group.measurement_time(Duration::from_secs(MEASURE_DURATION_SECS));

    let build_keys = 100_000;
    let probe_count = 1_000_000;
    let selectivity = 1.0; // all probes hit

    // Varying multiplicity: 1 (unique), 2, 5, 10, 50
    for &multiplicity in &[1, 2, 5, 10, 50] {
        let workload =
            JoinWorkload::generate(build_keys, multiplicity, probe_count, selectivity, 42);

        group.throughput(Throughput::Elements(probe_count as u64));

        let hm = HashMapIndex::build_from(&workload.build_tuples);
        let bt = BTreeMapIndex::build_from(&workload.build_tuples);
        let uc = UnchainedIndex::build_from(&workload.build_tuples);

        group.bench_with_input(
            BenchmarkId::new("HashMap", multiplicity),
            &workload.probe_keys,
            |b, keys| b.iter(|| bench_probe(&hm, black_box(keys))),
        );

        group.bench_with_input(
            BenchmarkId::new("BTreeMap", multiplicity),
            &workload.probe_keys,
            |b, keys| b.iter(|| bench_probe(&bt, black_box(keys))),
        );

        group.bench_with_input(
            BenchmarkId::new("Unchained", multiplicity),
            &workload.probe_keys,
            |b, keys| b.iter(|| bench_probe(&uc, black_box(keys))),
        );
    }

    group.finish();
}

fn bench_probe_table_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("probe_table_size");
    group.measurement_time(Duration::from_secs(MEASURE_DURATION_SECS));

    let probe_count = 1_000_000;
    let selectivity = 0.5;

    // Varying build size: fits in L1, L2, L3, exceeds LLC
    for &build_size in &[1_000, 10_000, 100_000, 1_000_000, 5_000_000] {
        let workload = JoinWorkload::generate(build_size, 1, probe_count, selectivity, 42);

        group.throughput(Throughput::Elements(probe_count as u64));

        let hm = HashMapIndex::build_from(&workload.build_tuples);
        let bt = BTreeMapIndex::build_from(&workload.build_tuples);
        let uc = UnchainedIndex::build_from(&workload.build_tuples);

        group.bench_with_input(
            BenchmarkId::new("HashMap", build_size),
            &workload.probe_keys,
            |b, keys| b.iter(|| bench_probe(&hm, black_box(keys))),
        );

        group.bench_with_input(
            BenchmarkId::new("BTreeMap", build_size),
            &workload.probe_keys,
            |b, keys| b.iter(|| bench_probe(&bt, black_box(keys))),
        );

        group.bench_with_input(
            BenchmarkId::new("Unchained", build_size),
            &workload.probe_keys,
            |b, keys| b.iter(|| bench_probe(&uc, black_box(keys))),
        );
    }

    group.finish();
}

fn bench_bloom_rejection(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_rejection");
    group.measurement_time(Duration::from_secs(MEASURE_DURATION_SECS));

    // This benchmark isolates the Bloom filter fast path.
    // 100% miss rate: every probe is rejected by the Bloom filter
    // before touching tuple storage. This is the best case for
    // the unchained design — the ~5 instruction hot path.
    let build_size = 100_000;
    let probe_count = 10_000_000;
    let workload = JoinWorkload::generate(build_size, 1, probe_count, 0.0, 42);

    group.throughput(Throughput::Elements(probe_count as u64));

    let uc = UnchainedIndex::build_from(&workload.build_tuples);

    // Full probe (Bloom check + would-scan, but Bloom rejects)
    group.bench_with_input(
        BenchmarkId::new("probe_all_miss", probe_count),
        &workload.probe_keys,
        |b, keys| b.iter(|| bench_probe(&uc, black_box(keys))),
    );

    // Bloom check only (the semi-join reducer path)
    group.bench_with_input(
        BenchmarkId::new("bloom_only_all_miss", probe_count),
        &workload.probe_keys,
        |b, keys| {
            b.iter(|| {
                let mut count = 0u64;
                for &key in keys {
                    if uc.bloom_check(black_box(key)) {
                        count += 1;
                    }
                }
                count
            })
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_build_throughput,
    bench_probe_selectivity,
    bench_probe_multiplicity,
    bench_probe_table_size,
    bench_bloom_rejection,
);
criterion_main!(benches);
