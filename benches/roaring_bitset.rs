use std::time::Instant;
use criterion::{black_box, Criterion, criterion_group, criterion_main};
use roaring_bitmap::roaring_bitset::RoaringBitmap;

pub fn criterion_add_million_elements(c: &mut Criterion) {
    let mut set = RoaringBitmap::new();
    c.bench_function("add 1 million elements", |b| b.iter_with_large_drop(|| {
        black_box((0..1_000_000).for_each(|x| set.add(x)));
    }));
}

pub fn criterion_lookup_from_set_of_billion_elements(c: &mut Criterion) {
    let mut set = RoaringBitmap::new();
    (0..1_000_000_000).for_each(|x| set.add(x));
    c.bench_function("search with 1 billion elements", |b| b.iter_with_large_drop(|| {
        black_box((0..1_000_000).for_each(|x| {set.contains(x);}));
    }));
}

pub fn criterion_union_of_two_sets_containing_ten_million_elements(c: &mut Criterion) {
    let mut set_a = RoaringBitmap::new();
    let mut set_b = RoaringBitmap::new();
    (0..10_000_000).for_each(|x| set_a.add(x));
    (10_000_000..20_000_000).for_each(|x| set_b.add(x));
    c.bench_function("union of disjoint sets with 10 million elements", |b| b.iter_custom(|iters| {
        let start = Instant::now();
        for _i in 0..iters {
            black_box(set_a.union(&set_b));
        }
        start.elapsed()
    }));
}

pub fn criterion_intersection_of_two_sets_containing_ten_million_elements(c: &mut Criterion) {
    let mut set_a = RoaringBitmap::new();
    let mut set_b = RoaringBitmap::new();
    (0..10_000_000).for_each(|x| set_a.add(x));
    (0..10_000_000).for_each(|x| set_b.add(x));
    c.bench_function("intersection of exact same sets with 10 million elements", |b| b.iter_with_large_drop(|| {
       black_box(set_a.intersection(&set_b));
    }));
}

pub fn criterion_difference_of_two_sets_containing_ten_million_elements(c: &mut Criterion) {
    let mut set_a = RoaringBitmap::new();
    let mut set_b = RoaringBitmap::new();
    (0..20_000_000).for_each(|x| set_a.add(x));
    (0..10_000_000).for_each(|x| set_b.add(x));
    c.bench_function("intersection of exact same sets with 10 million elements", |b| b.iter_with_large_drop(|| {
        black_box(set_a.difference(&set_b));
    }));
}

pub fn criterion_symmetric_difference_of_two_sets_containing_ten_million_elements(c: &mut Criterion) {
    let mut set_a = RoaringBitmap::new();
    let mut set_b = RoaringBitmap::new();
    (0..20_000_000).for_each(|x| set_a.add(x));
    (0..10_000_000).for_each(|x| set_b.add(x));
    c.bench_function("intersection of exact same sets with 10 million elements", |b| b.iter_with_large_drop(|| {
        black_box(set_a.symmetric_difference(&set_b));
    }));
}

criterion_group!(benches,
    criterion_add_million_elements,
    criterion_lookup_from_set_of_billion_elements,
    criterion_union_of_two_sets_containing_ten_million_elements,
    criterion_intersection_of_two_sets_containing_ten_million_elements,
    criterion_difference_of_two_sets_containing_ten_million_elements,
    criterion_symmetric_difference_of_two_sets_containing_ten_million_elements
);
criterion_main!(benches);