#![deny(missing_docs)]

use std::mem;

const MAX_SPARSE_CONTAINER_SIZE: usize = 4096;
const CHUNK_BITSET_CONTAINER_SIZE: usize = (1 << 16) >> 3;

/// `Container` holds the elements of the bitset in a chunk. All
/// elements in a chunk have their upper 16-bits in common.
#[derive(Debug)]
enum Container {
    Empty,
    Sparse(Vec<u16>),
    Dense(Box<[u8; CHUNK_BITSET_CONTAINER_SIZE]>),
}

impl Default for Container {
    fn default() -> Self {
        Self::Empty
    }
}

impl Container {
    fn len(&self) -> usize {
        match self {
            Container::Empty => 0,
            Container::Sparse(s) => s.len(),
            Container::Dense(d) => {
                d.iter().map(|b| (0..7_usize).map(|bit_pos| (b >> bit_pos) & 1).sum::<u8>())
                    .sum::<u8>() as usize
            },
        }
    }

    fn contains(&self, elem: u32) -> bool {
        let e = container_element(elem);
        match self {
            Container::Empty => false,
            Container::Sparse(s) => s.binary_search_by(|x| x.cmp(&e)).is_ok(),
            Container::Dense(d) => {
                let byte_pos = (e >> 3) as usize;
                let bit_pos = (e & 0b111) as usize;
                d[byte_pos] >> bit_pos > 0
            },
        }
    }

    fn into_sparse(self) -> Self {
        match self {
            Container::Empty => Container::Sparse(vec![]),
            Container::Dense(bitset) => {
                let mut bit_pos_vec = Vec::with_capacity(MAX_SPARSE_CONTAINER_SIZE);
                for byte_index in 0..bitset.len() {
                    for bit_pos in 0..8 {
                        if (bitset[byte_index] & (1<<bit_pos)) > 0 {
                            let byte_pos = (byte_index << 3) | bit_pos;
                            debug_assert!((byte_pos as u16) < u16::MAX);
                            bit_pos_vec.push(byte_pos as u16);
                        }
                    }
                }
                Container::Sparse(bit_pos_vec)
            },
            v => v,
        }
    }

    fn into_dense(self) -> Self {
        match self {
            Container::Empty => Container::Dense(Box::new([0_u8; CHUNK_BITSET_CONTAINER_SIZE])),
            Container::Sparse(v) => {
                let mut bitset = Box::new([0_u8; CHUNK_BITSET_CONTAINER_SIZE]);
                for bit_pos in v.into_iter() {
                    let byte_pos = ((bit_pos) >> 3) as usize;
                    let bit_pos = (bit_pos) & 0b111;
                    bitset[byte_pos] |= 1<<bit_pos;
                }
                Container::Dense(bitset)
            },
            v => v,
        }
    }

    fn is_sparse(&self) -> bool {
        match self {
            Container::Sparse(_) => true,
            _ => false,
        }
    }
}

#[inline]
fn chunk_index(item: u32) -> ChunkID {
    ((item & 0xffff_0000) >> 4) as u16
}

#[inline]
fn container_element(item: u32) -> u16 {
    (item & 0x0000_ffff) as u16
}

type ChunkID = u16;
type VectorIndex = usize;

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
    chunks: Vec<(ChunkID, Container)>,
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
        let chunk_idx = chunk_index(item);
        let vec_idx = self.maybe_allocate_chunk(chunk_idx);
        debug_assert!(vec_idx < self.chunks.len());
        self.add_item_to_chunk_container(item, vec_idx);
    }

    /// `remove` removes the given `item` from the roaring bitset if it
    /// exists. If it is non-existent, then the operation is a no-op.
    pub fn remove(&mut self, item: u32) {
        let chunk_idx = chunk_index(item);
        if let Some(vec_idx) = self.get_chunk(chunk_idx) {
            debug_assert!(vec_idx < self.chunks.len());
            self.remove_item_from_chunk_container(item, vec_idx);
        }
    }

    /// `contains` returns true if the `item` is in the roaring bitset
    /// or false otherwise. This is for checking if a given item exists
    /// in the roaring bitset or not.
    pub fn contains(&self, item: u32) -> bool {
        let chunk_idx = chunk_index(item);
        self.get_chunk(chunk_idx)
            .map(|idx| self.chunks[idx].1.contains(item))
            .unwrap_or(false)
    }

    /// `len` returns the cardinality of the roaring bitset.
    pub fn len(&self) -> usize {
        self.chunks.iter()
            .map(|c| c.1.len())
            .sum()
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

    fn maybe_allocate_chunk(&mut self, chunk_index: ChunkID) -> usize {
        self.chunks
            .binary_search_by(|chk| chk.0.cmp(&chunk_index))
            .unwrap_or_else(|pos_to_insert| {
                self.chunks.insert(pos_to_insert, (chunk_index, Container::Sparse(vec![])));
                pos_to_insert
            })
    }

    fn add_item_to_chunk_container(&mut self, item: u32, vec_idx: usize) {
        let chunk_idx = chunk_index(item);
        let elem = container_element(item);
        let mut should_convert_to_dense = false;

        // The scope is to make sure that the chunk_container
        // borrow is dropped as soon as we are done inserting
        // the element into the container.
        {
            let mut chunk_container = self.chunks.get_mut(vec_idx);
            if chunk_container.is_none() {
                return;
            }
            let (_, ref mut container) = chunk_container.unwrap();
            match container {
                Container::Empty => panic!("unexpected condition: empty container"),
                Container::Sparse(s) => {
                    s.binary_search_by(|ci| ci.cmp(&elem))
                        .unwrap_or_else(|pos_to_insert| {
                            s.insert(pos_to_insert, elem);
                            pos_to_insert
                        });
                    if s.len() > MAX_SPARSE_CONTAINER_SIZE {
                        should_convert_to_dense = true;
                    }
                },
                Container::Dense(d) => {
                    let byte_pos = (elem >> 3) as usize;
                    let bit_pos = (elem & 0b111) as usize;
                    d[byte_pos] |= 1<<bit_pos;
                },
            }
        }
        if should_convert_to_dense {
            let (prev_chunk_idx, prev_container) = mem::take(&mut self.chunks[vec_idx]);
            debug_assert!(prev_container.is_sparse());
            debug_assert!(prev_chunk_idx == chunk_idx);
            self.chunks[vec_idx] = (chunk_idx, prev_container.into_dense());
        }
    }

    fn remove_item_from_chunk_container(&mut self, item: u32, vec_idx: usize) {
        let chunk_idx = chunk_index(item);
        let elem = container_element(item);
        let mut needs_sparse_conversion = false;
        let mut can_free_slot = false;
        {
            let chunk_container = self.chunks.get_mut(vec_idx);
            if chunk_container.is_none() {
                return;
            }
            let container = chunk_container.unwrap();
            match container.1 {
                Container::Empty => {return;}
                Container::Sparse(ref mut v) => {
                    v.binary_search(&elem)
                        .into_iter().for_each(|idx| {v.remove(idx);});
                }
                Container::Dense(ref mut bitset) => {
                    let byte_pos = (elem >> 3) as usize;
                    let bit_pos = (elem & 0b111) as usize;
                    bitset[byte_pos] |= 1<<bit_pos;
                }
            }
            if !container.1.is_sparse() && container.1.len() < MAX_SPARSE_CONTAINER_SIZE {
                needs_sparse_conversion = true;
            } else if container.1.len() == 0 {
                can_free_slot = true;
            }
        }
        if can_free_slot {
            self.chunks.remove(vec_idx);
            return;
        }
        if needs_sparse_conversion {
            let (prev_chunk_id, prev_container) = mem::take(&mut self.chunks[vec_idx]);
            debug_assert!(!prev_container.is_sparse());
            self.chunks[vec_idx] = (prev_chunk_id, prev_container.into_sparse());
        }
    }

    fn do_insert_into_bitset(&mut self, elem: u16, dense_bitset: &mut [u8; CHUNK_BITSET_CONTAINER_SIZE]) {
        // it is simpler in case of dense bitset because the cardinality
        // can only go up and hence we don't need to re-allocate anything
        let byte_pos = (elem >> 3) as usize;
        let bit_pos = elem & 0b111;
        debug_assert!(byte_pos < dense_bitset.len());
        debug_assert!(bit_pos < 8 && bit_pos >= 0);
        dense_bitset[byte_pos] |= 1 << bit_pos;
    }

    fn get_chunk(&self, chunk_idx: u16) -> Option<usize> {
        self.chunks.binary_search_by(|chk| chk.0.cmp(&chunk_idx)).ok()
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