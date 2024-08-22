use criterion::{black_box, Criterion, criterion_group, criterion_main};
use roaring_bitmap::roaring_bitset::RoaringBitmap;

pub fn criterion_add_million_elements(c: &mut Criterion) {
    c.bench_function("add 1 million elements", |b| b.iter(|| {
        let mut set = RoaringBitmap::new();
        (1..1_000_000).for_each(|x| set.add(black_box(x)));
    }));
}

criterion_group!(benches, criterion_add_million_elements);
criterion_main!(benches);