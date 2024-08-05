#![deny(missing_docs)]

const MAX_POSSIBLE_CHUNKS: usize = 1<<16;
const CHUNK_BITSET_CONTAINER_SIZE: usize = (1<<16) >> 3;

/// `Container` holds the elements of the bitset in a chunk. All
/// elements in a chunk have their upper 16-bits in common.
#[derive(Debug)]
enum Container {
    Sparse(Vec<u16>),
    Dense([u8; CHUNK_BITSET_CONTAINER_SIZE]),
}

/// `RoaringBitmap` is the implementation of the bitmap supporting
/// efficient bitset operations. But note that it can only support
/// a maximum of 2^32 bits which is large enough in practice.
///
/// According to the paper, the `RoaringBitmap` consists of chunks
/// which are a 2^16 division of the space. This allows for avoiding
/// the allocation in case of sparse bitsets which is where this shines
/// over the naive Bitset implementations. Further, each `Chunk` contains
/// 2^16 contiguous elements which are stored in the different formats
/// depending on the cardinality. This allows for more space savings.
///
/// The paper outlines three formats for a chunk container representation
/// 1. As sorted set when the cardinality is <= 4096.
/// 2. As a full bitset when the cardinality is > 4096.
/// 3. Paper #2 introduces run-length encoding representation.
#[derive(Debug)]
pub struct RoaringBitmap {
    chunks: Vec<Container>,
}

impl RoaringBitmap {
    /// `new` creates a new roaring bitmap. This initializes an empty
    /// chunk vector which will then be added when elements are added
    /// further into it.
    pub fn new() -> Self {
        RoaringBitmap { chunks: vec![] }
    }

    /// `add` adds the given `item` into the roaring bitset. If the
    /// element already exists then the operation is a no-op.
    pub fn add(&mut self, item: u32) {
        unimplemented!()
    }

    /// `remove` removes the given `item` from the roaring bitset if it
    /// exists. If it is non-existent, then the operation is a no-op.
    pub fn remove(&mut self, item: u32) {
        unimplemented!()
    }

    /// `contains` returns true if the `item` is in the roaring bitset
    /// or false otherwise. This is for checking if a given item exists
    /// in the roaring bitset or not.
    pub fn contains(&self, item: u32) -> bool {
        unimplemented!()
    }

    /// `len` returns the cardinality of the roaring bitset.
    pub fn len(&self) -> usize {
        unimplemented!()
    }

    /// `union` performs the union of two roaring bitsets. More specifically,
    /// it constructs and returns another `RoaringBitmap` which contains elements
    /// such that for each element e, it belongs to either of the sets (or both)
    pub fn union(&self, other: &RoaringBitmap) -> RoaringBitmap {
        unimplemented!()
    }

    /// `intersection` performs the set intersection of two roaring bitsets.
    /// More specifically, it constructs and returns another `RoaringBitmap`
    /// which contains elements from both the sets.
    pub fn intersection(&self, other: &RoaringBitmap) -> RoaringBitmap {
        unimplemented!()
    }

    /// `difference` computes the set difference between this and the `other`
    /// bitset. If this bitset represented by self is `A` and the other bitset
    /// represented by the `other` parameter is `B` then this computes `A-B`.
    pub fn difference(&self, other: &RoaringBitmap) -> RoaringBitmap {
        unimplemented!()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_basic_set_operations() {
        let mut bm = RoaringBitmap::new();
        assert_eq!(bm.len(), 0);

        bm.add(10);
        bm.add(189079);
        assert_eq!(bm.len(), 2);
        assert!(bm.contains(10));
        assert!(bm.contains(189079));
        assert!(!bm.contains(20));

        bm.remove(10);
        assert!(!bm.contains(10));
    }

    #[test]
    fn test_set_union() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        a.add(10);
        b.add(20);
        b.add(10);

        let c = a.union(&b);
        assert_eq!(c.len(), 2);
        assert!(c.contains(10));
        assert!(c.contains(20));
    }

    #[test]
    fn test_set_intersection() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        a.add(10);
        b.add(20);
        b.add(10);

        let c = a.intersection(&b);
        assert_eq!(c.len(), 1);
        assert!(c.contains(10));
        assert!(!c.contains(20));
    }

    #[test]
    fn test_set_difference() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        a.add(10);
        b.add(20);
        b.add(10);

        let a_diff_b = a.difference(&b);
        let b_diff_a = b.difference(&a);

        assert_eq!(a_diff_b.len(), 0);
        assert_eq!(b_diff_a.len(), 1);
        assert!(b_diff_a.contains(20));
        assert!(!b_diff_a.contains(10));
    }
}