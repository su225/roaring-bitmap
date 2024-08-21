#![deny(missing_docs)]

use std::{cmp, mem};

use Container::{Dense, Empty, Sparse};

const MAX_SPARSE_CONTAINER_SIZE: usize = 4096;
const CHUNK_BITSET_CONTAINER_SIZE: usize = 8192;

/// `Container` holds the elements of the bitset in a chunk. All
/// elements in a chunk have their upper 16-bits in common.
#[derive(Debug, Clone, Eq, PartialEq)]
enum Container {
    Empty,
    Sparse(Vec<u16>),
    Dense(Box<[u8; CHUNK_BITSET_CONTAINER_SIZE]>),
}

impl Container {
    fn len(&self) -> usize {
        match self {
            Empty => 0,
            Sparse(s) => s.len(),
            Dense(d) => {
                d.iter().map(|b|
                (0..7_usize).map(|bit_pos| ((b >> bit_pos) & 1) as usize).sum::<usize>())
                    .sum::<usize>()
            }
        }
    }

    fn contains(&self, elem: u32) -> bool {
        let e = container_element(elem);
        match self {
            Empty => false,
            Sparse(s) => s.binary_search_by(|x| x.cmp(&e)).is_ok(),
            Dense(d) => {
                let byte_pos = (e >> 3) as usize;
                let bit_pos = (e & 0b111) as usize;
                d[byte_pos] >> bit_pos > 0
            }
        }
    }

    fn into_sparse(self) -> Self {
        match self {
            Empty => Sparse(vec![]),
            Dense(bitset) => {
                let mut bit_pos_vec = Vec::with_capacity(MAX_SPARSE_CONTAINER_SIZE);
                for byte_index in 0..bitset.len() {
                    for bit_pos in 0..8 {
                        if (bitset[byte_index] & (1 << bit_pos)) > 0 {
                            let byte_pos = (byte_index << 3) | bit_pos;
                            debug_assert!((byte_pos as u16) < u16::MAX);
                            bit_pos_vec.push(byte_pos as u16);
                        }
                    }
                }
                Sparse(bit_pos_vec)
            }
            v => v,
        }
    }

    fn into_dense(self) -> Self {
        match self {
            Empty => Dense(Box::new([0_u8; CHUNK_BITSET_CONTAINER_SIZE])),
            Sparse(v) => {
                let mut bitset = Box::new([0_u8; CHUNK_BITSET_CONTAINER_SIZE]);
                for bit_pos in v.into_iter() {
                    let byte_pos = ((bit_pos) >> 3) as usize;
                    let bit_pos = (bit_pos) & 0b111;
                    bitset[byte_pos] |= 1 << bit_pos;
                }
                Dense(bitset)
            }
            v => v,
        }
    }

    fn is_sparse(&self) -> bool {
        match self {
            Sparse(_) => true,
            _ => false,
        }
    }

    fn union(&self, right: &Container) -> Container {
        match (self, right) {
            (Empty, x) => x.clone(),
            (y, Empty) => y.clone(),
            (Dense(ref x), Dense(ref y)) => {
                let mut res = x.clone();
                for i in 0..CHUNK_BITSET_CONTAINER_SIZE {
                    res[i] |= y[i];
                }
                Dense(res)
            }
            (Sparse(ref x), Dense(ref y)) => {
                let mut res = y.clone();
                for bitpos in x {
                    let byte_idx = (bitpos >> 3) as usize;
                    let bit_idx = bitpos & 0b111;
                    res[byte_idx] |= 1 << bit_idx;
                }
                Dense(res)
            }
            (Dense(ref x), Sparse(ref y)) => {
                let mut res = x.clone();
                for bitpos in y {
                    let byte_idx = (bitpos >> 3) as usize;
                    let bit_idx = bitpos & 0b111;
                    res[byte_idx] |= 1 << bit_idx;
                }
                Dense(res)
            }
            (Sparse(ref x), Sparse(ref y)) => {
                // If we are not sure, we assign a dense bitmap
                // because we can be sure about the upper bound
                // on the allocated chunk. If it turns out to be
                // small, we can then allocate a sparse chunk.
                let mut bitset = [0_u8; CHUNK_BITSET_CONTAINER_SIZE];
                let mut set_bit = |pos| {
                    let byte_pos = (pos >> 3) as usize;
                    let bit_pos = (pos & 0b111) as u8;
                    bitset[byte_pos] |= 1 << bit_pos;
                };
                x.iter().for_each(&mut set_bit);
                y.iter().for_each(&mut set_bit);
                let mut bitset_len = 0;
                for b in bitset {
                    for i in 0..8 {
                        if (b & (1 << i)) > 0 {
                            bitset_len += 1;
                        }
                    }
                }
                // If the length of the resulting bitset is greater than
                // or equal to the length of the maximum sparse container
                // then we return as is because we started with dense.
                if bitset_len >= MAX_SPARSE_CONTAINER_SIZE {
                    Dense(Box::new(bitset))
                } else {
                    let mut sparse_pos = Vec::with_capacity(MAX_SPARSE_CONTAINER_SIZE);
                    for i in 0..bitset.len() {
                        for j in 0..8 {
                            if (bitset[i] & (1 << j)) > 0 {
                                let bitpos = ((i << 3) | j) as u16;
                                sparse_pos.push(bitpos);
                            }
                        }
                    }
                    Sparse(sparse_pos)
                }
            }
        }
    }

    fn intersection(&self, right: &Container) -> Container {
        match (self, right) {
            (Empty, _) | (_, Empty) => Empty,
            (Dense(ref x), Dense(ref y)) => {
                let mut intersected = x.clone();
                let mut intersect_length = 0;
                for i in 0..y.len() {
                    intersected[i] |= y[i];
                    for j in 0..8 {
                        if (intersected[i] & (1 << j)) > 0 {
                            intersect_length += 1;
                        }
                    }
                }
                let res = Dense(intersected);
                if intersect_length > MAX_SPARSE_CONTAINER_SIZE {
                    res
                } else {
                    res.into_sparse()
                }
            }
            (Sparse(ref x), Dense(ref y)) => {
                let mut intersected: Vec<u16> = Vec::with_capacity(x.len());
                for spos in x {
                    let byte_pos = (spos >> 3) as usize;
                    let bit_pos = spos & 0b111;
                    if (y[byte_pos] & (1 << bit_pos)) > 0 {
                        intersected.push(*spos);
                    }
                }
                Sparse(intersected)
            }
            (Dense(ref x), Sparse(ref y)) => {
                let mut intersected = Vec::with_capacity(y.len());
                for spos in y {
                    let byte_pos = (spos >> 3) as usize;
                    let bit_pos = spos & 0b111;
                    if (x[byte_pos] & (1 << bit_pos)) > 0 {
                        intersected.push(*spos);
                    }
                }
                Sparse(intersected)
            }
            (Sparse(ref x), Sparse(ref y)) => {
                let mut bitpos = Vec::with_capacity(cmp::min(x.len(), y.len()));
                let (mut idx1, mut idx2) = (0_usize, 0_usize);
                while idx1 < x.len() && idx2 < y.len() {
                    let mut candidate = None;
                    if x[idx1] < y[idx2] {
                        idx1 += 1;
                    } else if x[idx1] > y[idx2] {
                        idx2 += 1;
                    } else {
                        candidate = Some(x[idx1]);
                        idx1 += 1;
                        idx2 += 1;
                    }
                    if candidate.is_none() {
                        continue;
                    }
                    let potential_addition = candidate.unwrap();
                    let last_added = bitpos.last();
                    if last_added.is_none() || potential_addition > *last_added.unwrap() {
                        bitpos.push(potential_addition);
                    }
                }
                Sparse(bitpos)
            }
        }
    }

    fn difference(&self, right: &Container) -> Container {
        match (self, right) {
            (Empty, _) => Empty,
            (ref x, Empty) => (*x).clone(),
            (Sparse(ref x), Sparse(ref y)) => {
                let mut bitpos = Vec::with_capacity(x.len());
                let (mut idx1, mut idx2) = (0_usize, 0_usize);
                while idx1 < x.len() && idx2 < y.len() {
                    if x[idx1] < y[idx2] {
                        // At this point we are sure that there are no
                        // elements in y which can match x[idx1]. Hence,
                        // we can safely consider this part of the diff.
                        bitpos.push(x[idx1]);
                        idx1 += 1;
                    } else if x[idx1] > y[idx2] {
                        // We don't know if there exists some element in
                        // y which can equal the current element in x. So
                        // don't add it just yet. Imagine the following
                        // case where x=[10,20] and y=[1,2,10,20]. Here,
                        // adding 10 before going all the way to index 2
                        // (0-based indexing) in y would be incorrect.
                        idx2 += 1;
                    } else if x[idx1] == y[idx2] {
                        // The element is present in both the sets. Hence,
                        // this clearly doesn't belong to the difference.
                        idx1 += 1;
                        idx2 += 1;
                    }
                }
                // add the remaining elements (if any) to the difference
                // 1. If we ran out of elements in x, then we don't do anything
                // 2. If we have run out of elements in y, then we add the rest
                //    because we know for sure that they don't belong there.
                if idx1 < x.len() {
                    x[idx1..].iter().for_each(|xpos| bitpos.push(*xpos));
                }
                Sparse(bitpos)
            }
            (Sparse(ref x), Dense(ref y)) => {
                Sparse((*x).iter()
                    .filter(|&p| {
                        let byte_pos = (p >> 3) as usize;
                        let bit_pos = p & 0b111;
                        (y[byte_pos] & (1 << bit_pos)) == 0
                    })
                    .map(|x| *x)
                    .collect::<Vec<u16>>())
            }
            (Dense(ref x), Sparse(ref y)) => {
                let mut diff_bitset = x.clone();
                for yp in y {
                    let byte_pos = (yp >> 3) as usize;
                    let bit_pos = yp & 0b111;
                    let bit_mask = !(1 << bit_pos) as u8;
                    diff_bitset[byte_pos] &= bit_mask;
                }
                let res = Dense(diff_bitset);
                if res.len() < MAX_SPARSE_CONTAINER_SIZE {
                    res.into_sparse()
                } else {
                    res
                }
            }
            (Dense(ref x), Dense(ref y)) => {
                let mut diff = x.clone();
                for i in 0..y.len() {
                    diff[i] &= !y[i];
                }
                let res = Dense(diff);
                if res.len() < MAX_SPARSE_CONTAINER_SIZE {
                    res.into_sparse()
                } else {
                    res
                }
            }
        }
    }
}

enum ContainerIter<'a> {
    EmptyIter,
    SparseIter(std::slice::Iter<'a, u16>),
    DenseIter {
        bitset: &'a [u8; CHUNK_BITSET_CONTAINER_SIZE],
        byte_pos: usize,
        bit_pos: u8,
    },
}

impl<'a> ContainerIter<'a> {
    fn new(container: &'a Container) -> ContainerIter<'a> {
        match container {
            Empty => ContainerIter::EmptyIter,
            Sparse(ref v) => ContainerIter::SparseIter(v.iter()),
            Dense(ref bitset) => ContainerIter::DenseIter {
                bitset,
                byte_pos: 0,
                bit_pos: 0,
            }
        }
    }
}

impl<'a> Iterator for ContainerIter<'a> {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ContainerIter::EmptyIter => None,
            ContainerIter::SparseIter(ref mut iter) => iter.next().cloned(),
            ContainerIter::DenseIter {
                ref bitset,
                ref mut byte_pos,
                ref mut bit_pos,
            } => {
                if *byte_pos > bitset.len() {
                    return None;
                }
                let mut next_val = None;
                let mut found = false;
                for by in *byte_pos..bitset.len() {
                    let cur_byte = bitset[by];
                    for b in *bit_pos..8 {
                        if (cur_byte & (1 << b)) == 0 {
                            continue;
                        }
                        next_val = Some(((*byte_pos << 3) as u16) | (*bit_pos as u16));
                        *bit_pos = b + 1;
                        found = true;
                        break;
                    }
                    if found {
                        // we found the next set bit. However, we need
                        // to handle the edge case where the bit pos turns
                        // to be 8. In this case, we need to advance to the
                        // next byte for obvious reasons.
                        debug_assert!(*bit_pos <= 8);
                        if *bit_pos == 8 {
                            *bit_pos = 0;
                            *byte_pos += 1;
                        }
                        break;
                    }
                    *byte_pos = by;
                }
                return next_val;
            }
        }
    }
}

impl<'a> IntoIterator for &'a Container {
    type Item = u16;
    type IntoIter = ContainerIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ContainerIter::new(self)
    }
}

impl Default for Container {
    fn default() -> Self {
        Empty
    }
}

type ChunkID = u16;

#[inline]
fn chunk_index(item: u32) -> ChunkID {
    ((item & 0xffff_0000) >> 16) as u16
}

#[inline]
fn container_element(item: u32) -> u16 {
    (item & 0x0000_ffff) as u16
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
#[derive(Debug, Clone, Eq, PartialEq)]
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
        let mut idx1 = 0_usize;
        let mut idx2 = 0_usize;
        let mut unioned_chunks = vec![];
        while idx1 < self.chunks.len() && idx2 < other.chunks.len() {
            let chunk_idx1 = self.chunks[idx1].0;
            let chunk_idx2 = other.chunks[idx2].0;
            if chunk_idx1 < chunk_idx2 {
                unioned_chunks.push(self.chunks[idx1].clone());
                idx1 += 1;
            } else if chunk_idx1 > chunk_idx2 {
                unioned_chunks.push(other.chunks[idx2].clone());
                idx2 += 1;
            } else {
                let left: &Container = self.chunks.get(idx1).map(|(_, c)| c).unwrap();
                let right: &Container = other.chunks.get(idx2).map(|(_, c)| c).unwrap();
                unioned_chunks.push((chunk_idx1, left.union(right)));
                idx1 += 1;
                idx2 += 1;
            }
        }
        for x in idx1..self.chunks.len() {
            unioned_chunks.push(self.chunks[x].clone());
        }
        for y in idx2..other.chunks.len() {
            unioned_chunks.push(other.chunks[y].clone());
        }
        RoaringBitmap { chunks: unioned_chunks }
    }

    /// `intersection` performs the set intersection of two roaring bitsets.
    /// More specifically, it constructs and returns another `RoaringBitmap`
    /// which contains elements from both the sets.
    pub fn intersection(&self, other: &RoaringBitmap) -> RoaringBitmap {
        let mut chunks = vec![];
        let (mut idx1, mut idx2) = (0_usize, 0_usize);
        while idx1 < self.chunks.len() && idx2 < other.chunks.len() {
            let (chunk_idx1, c1) = self.chunks.get(idx1).unwrap();
            let (chunk_idx2, c2) = other.chunks.get(idx2).unwrap();
            if chunk_idx1 == chunk_idx2 {
                let intersection = c1.intersection(c2);
                if intersection.len() > 0 {
                    chunks.push((*chunk_idx1, intersection));
                }
                idx1 += 1;
                idx2 += 1;
            } else if chunk_idx1 < chunk_idx2 {
                idx1 += 1;
            } else if chunk_idx1 > chunk_idx2 {
                idx2 += 1;
            }
        }
        RoaringBitmap { chunks }
    }

    /// `difference` computes the set difference between this and the `other`
    /// bitset. If this bitset represented by self is `A` and the other bitset
    /// represented by the `other` parameter is `B` then this computes `A-B`.
    pub fn difference(&self, other: &RoaringBitmap) -> RoaringBitmap {
        let mut idx1 = 0_usize;
        let mut idx2 = 0_usize;
        let mut diff_chunks: Vec<(ChunkID, Container)> = vec![];
        while idx1 < self.chunks.len() && idx2 < other.chunks.len() {
            let (chunk_idx1, c1) = self.chunks.get(idx1).unwrap();
            let (chunk_idx2, c2) = other.chunks.get(idx2).unwrap();
            if chunk_idx2 < chunk_idx1 {
                idx2 += 1;
                continue;
            }
            debug_assert!(chunk_idx1 <= chunk_idx2);
            if chunk_idx1 < chunk_idx2 {
                // We know that there are no elements in the other set
                // which can exist in the current chunk. So we can add
                // the whole chunk into the result.
                diff_chunks.push((*chunk_idx1, c1.clone()));
                idx1 += 1;
            } else if chunk_idx1 == chunk_idx2 {
                diff_chunks.push((*chunk_idx1, c1.difference(c2)));
                idx1 += 1;
                idx2 += 1;
            } else {
                unreachable!()
            }
        }
        for (c, x) in &self.chunks[idx1..] {
            diff_chunks.push((*c, x.clone()));
        }
        RoaringBitmap { chunks: diff_chunks }
    }

    /// `symmetric_difference` computes the set difference symmetric difference
    /// between the two given roaring bitsets. Currently, it is computed as the
    /// difference between the union and the intersection of two sets.
    pub fn symmetric_difference(&self, other: &RoaringBitmap) -> RoaringBitmap {
        let set_union = self.union(&other);
        let set_intersection = self.intersection(&other);
        return set_union.difference(&set_intersection);
    }

    fn maybe_allocate_chunk(&mut self, chunk_index: ChunkID) -> usize {
        self.chunks
            .binary_search_by(|chk| chk.0.cmp(&chunk_index))
            .unwrap_or_else(|pos_to_insert| {
                self.chunks.insert(pos_to_insert, (chunk_index, Sparse(vec![])));
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
            let chunk_container = self.chunks.get_mut(vec_idx);
            if chunk_container.is_none() {
                return;
            }
            let (_, ref mut container) = chunk_container.unwrap();
            match container {
                Empty => panic!("unexpected condition: empty container"),
                Sparse(s) => {
                    s.binary_search_by(|ci| ci.cmp(&elem))
                        .unwrap_or_else(|pos_to_insert| {
                            s.insert(pos_to_insert, elem);
                            pos_to_insert
                        });
                    if s.len() > MAX_SPARSE_CONTAINER_SIZE {
                        should_convert_to_dense = true;
                    }
                }
                Dense(d) => {
                    let byte_pos = (elem >> 3) as usize;
                    let bit_pos = (elem & 0b111) as usize;
                    d[byte_pos] |= 1 << bit_pos;
                }
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
                Empty => { return; }
                Sparse(ref mut v) => {
                    v.binary_search(&elem)
                        .into_iter().for_each(|idx| { v.remove(idx); });
                }
                Dense(ref mut bitset) => {
                    let byte_pos = (elem >> 3) as usize;
                    let bit_pos = (elem & 0b111) as usize;
                    bitset[byte_pos] |= 1 << bit_pos;
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

    fn get_chunk(&self, chunk_idx: u16) -> Option<usize> {
        self.chunks.binary_search_by(|chk| chk.0.cmp(&chunk_idx)).ok()
    }
}

impl<'a> IntoIterator for &'a RoaringBitmap {
    type Item = u32;
    type IntoIter = RoaringBitmapIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        RoaringBitmapIter::new(self)
    }
}

/// `RoaringBitmapIter` creates an iterator over the underlying
/// `RoaringBitmap` structure. This allows iterating over the
/// elements of the bitmap in order.
pub struct RoaringBitmapIter<'a> {
    roaring_bitmap: &'a RoaringBitmap,
    vec_chunk_idx: usize,
    cur_chunk_iter: Option<ContainerIter<'a>>,
}

impl<'a> RoaringBitmapIter<'a> {
    fn new(rb: &'a RoaringBitmap) -> Self {
        RoaringBitmapIter {
            roaring_bitmap: rb,
            vec_chunk_idx: 0,
            cur_chunk_iter: rb.chunks.first().map(|(_, c)| c.into_iter()),
        }
    }
}

impl<'a> Iterator for RoaringBitmapIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let next_item = match self.cur_chunk_iter {
            None => None,
            Some(ref mut container_iter) => {
                container_iter.next()
            }
        };
        if let Some(nxt) = next_item {
            let cur_chunk_idx = self.roaring_bitmap.chunks[self.vec_chunk_idx].0;
            let elem = (cur_chunk_idx as u32) << 16 | (nxt as u32);
            return Some(elem);
        }
        // If it is none, then it might have been the end of the container.
        // So we move on to the next chunk container and so on until we either
        // reach the end or find the container.
        let mut next_elem = None;
        for nxt_vec_idx in self.vec_chunk_idx + 1..self.roaring_bitmap.chunks.len() {
            let (chunk_idx, container) = self.roaring_bitmap.chunks.get(nxt_vec_idx).unwrap();
            let mut container_iter = container.into_iter();
            if let Some(nxt_elem) = container_iter.next() {
                next_elem = Some((*chunk_idx as u32) << 16 | (nxt_elem as u32));
                self.vec_chunk_idx = nxt_vec_idx;
                self.cur_chunk_iter = Some(container_iter);
                break;
            }
        }
        return next_elem;
    }
}

#[cfg(test)]
mod roaring_bitset_test {
    use pretty_assertions::assert_eq;

    use crate::roaring_bitset::RoaringBitmap;

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
    fn test_iterator_sparse() {
        let mut bm = RoaringBitmap::new();
        bm.add(0x0000_1100);
        bm.add(0x0000_001f);
        bm.add(0x0010_dead);

        let mut iter = bm.into_iter();
        assert_eq!(iter.next(), Some(0x0000_001f));
        assert_eq!(iter.next(), Some(0x0000_1100));
        assert_eq!(iter.next(), Some(0x0010_dead));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterator_dense() {
        let mut bm = RoaringBitmap::new();
        (0..(1 << 16)).for_each(|x| bm.add(x));

        let mut iter = bm.into_iter();
        for expected_elem in 0..(1 << 16) {
            assert_eq!(iter.next(), Some(expected_elem));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterator_sparse_and_dense_mixed() {
        let mut bm = RoaringBitmap::new();
        (0..(1 << 16)).for_each(|x| bm.add(x));
        bm.add(0x0101_ffff);
        bm.add(0x0101_dead);
        bm.add(0x0101_beef);

        let mut iter = bm.into_iter();
        for expected_elem in 0..(1 << 16) {
            assert_eq!(iter.next(), Some(expected_elem));
        }
        assert_eq!(iter.next(), Some(0x0101_beef));
        assert_eq!(iter.next(), Some(0x0101_dead));
        assert_eq!(iter.next(), Some(0x0101_ffff));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_set_union_both_sparse() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        a.add(10);
        b.add(20);
        b.add(10);
        a.add(100_000);

        let c = a.union(&b);
        let elems = c.into_iter().collect::<Vec<u32>>();
        assert_eq!(vec![10, 20, 100_000], elems);
    }

    #[test]
    fn test_set_union_both_dense() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        (0..(1 << 16)).for_each(|x| a.add(x as u32));
        (0..(1 << 14)).for_each(|x| b.add(x as u32));

        let c = a.union(&b);
        assert_eq!((0..(1 << 16)).collect::<Vec<u32>>(),
                   c.into_iter().collect::<Vec<u32>>());
    }

    #[test]
    fn test_set_union_sparse_and_dense() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        b.add(0xffff_1111_u32);
        (0..(1 << 16)).for_each(|x| a.add(x as u32));

        let c = a.union(&b);
        let mut c_unioned = (0..(1 << 16)).collect::<Vec<u32>>();
        c_unioned.push(0xffff_1111);

        let actual_c_unioned = c.into_iter().collect::<Vec<u32>>();

        assert_eq!(c_unioned.len(), actual_c_unioned.len());
        assert_eq!(c_unioned, actual_c_unioned);
    }

    #[test]
    fn test_set_intersection_both_sparse() {
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
    fn test_set_intersection_both_dense() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        (0..(1 << 16)).for_each(|x| a.add(x));
        ((1 << 16)..2 * (1 << 16)).for_each(|x| b.add(x));

        let c = a.intersection(&b).into_iter().collect::<Vec<u32>>();
        assert!(c.is_empty());

        ((1 << 16)..(1 << 16) + 10).for_each(|x| a.add(x));
        let d = a.intersection(&b).into_iter().collect::<Vec<u32>>();
        assert_eq!(d, ((1 << 16)..(1 << 16) + 10).collect::<Vec<u32>>());
    }

    #[test]
    fn test_set_intersection_sparse_and_dense() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        (0..(1 << 16)).for_each(|x| a.add(x));
        b.add(0);
        b.add(0x0000_ff11);

        let c = a.intersection(&b).into_iter().collect::<Vec<u32>>();
        assert_eq!(c, vec![0, 0x0000_ff11]);
    }

    #[test]
    fn test_set_difference_both_sparse() {
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

    #[test]
    fn test_symmetric_difference_both_sparse() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        a.add(10);
        a.add(20);
        b.add(20);
        b.add(30);

        let a_symdiff_b = a.symmetric_difference(&b).into_iter().collect::<Vec<u32>>();
        let b_symdiff_b = b.symmetric_difference(&a).into_iter().collect::<Vec<u32>>();
        assert_eq!(a_symdiff_b, b_symdiff_b);
        assert_eq!(a_symdiff_b, vec![10, 30]);
    }

    #[test]
    fn test_symmetric_difference_with_disjoint_sets() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        a.add(10);
        a.add(20);
        b.add(30);
        b.add(40);

        let a_symdiff_b = a.symmetric_difference(&b).into_iter().collect::<Vec<u32>>();
        let b_symdiff_a = b.symmetric_difference(&a).into_iter().collect::<Vec<u32>>();
        assert_eq!(a_symdiff_b, b_symdiff_a);
        assert_eq!(a_symdiff_b, a.union(&b).into_iter().collect::<Vec<u32>>());
    }

    #[test]
    fn test_symmetric_difference_with_exactly_same_sets() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        a.add(10);
        b.add(10);
        a.add(20);
        b.add(20);

        let a_symdiff_b = a.symmetric_difference(&b).into_iter().collect::<Vec<u32>>();
        let b_symdiff_a = b.symmetric_difference(&a).into_iter().collect::<Vec<u32>>();
        assert_eq!(a_symdiff_b, b_symdiff_a);
        assert!(a_symdiff_b.is_empty());
    }

    #[test]
    fn test_symmetric_difference_with_one_being_subset_of_another() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();

        a.add(10);
        b.add(10);
        a.add(20);
        a.add(30);

        let a_symdiff_b = a.symmetric_difference(&b).into_iter().collect::<Vec<u32>>();
        let b_symdiff_a = b.symmetric_difference(&a).into_iter().collect::<Vec<u32>>();
        assert_eq!(a_symdiff_b, b_symdiff_a);
        assert_eq!(a_symdiff_b, vec![20, 30]);
    }
}

#[cfg(test)]
mod roaring_bitset_property_tests {
    use std::collections::HashSet;
    use std::fmt::Debug;
    use std::hash::Hash;

    use quickcheck::{Arbitrary, Gen};
    use quickcheck_macros::quickcheck;

    use crate::roaring_bitset::RoaringBitmap;

    #[derive(Debug, Clone)]
    struct SetAndSubset<T: Arbitrary + Clone + Debug + Eq + Hash> {
        set: HashSet<T>,
        subset: HashSet<T>,
    }

    impl<T> Arbitrary for SetAndSubset<T>
    where
        T: Clone + Arbitrary + Debug + Eq + Hash,
    {
        fn arbitrary(g: &mut Gen) -> Self {
            let set: HashSet<T> = Arbitrary::arbitrary(g);
            let subset: HashSet<T> = set.iter().cloned()
                .filter(|_| bool::arbitrary(g))
                .collect();
            SetAndSubset { set, subset }
        }
    }

    impl Arbitrary for RoaringBitmap {
        fn arbitrary(g: &mut Gen) -> Self {
            let set: HashSet<u32> = Arbitrary::arbitrary(g);
            let mut bitmap = RoaringBitmap::new();
            set.into_iter().for_each(|x| bitmap.add(x));
            bitmap
        }
    }

    #[quickcheck]
    fn union_must_have_elements_from_both_sets(a: RoaringBitmap, b: RoaringBitmap) -> bool {
        let a_union_b = a.union(&b);
        let elems_in_a_not_in_union = a.into_iter().all(|x| a_union_b.contains(x));
        let elems_in_b_not_in_union = b.into_iter().all(|x| a_union_b.contains(x));
        elems_in_a_not_in_union && elems_in_b_not_in_union
    }

    #[quickcheck]
    fn union_must_not_have_elements_other_than_from_its_constituents(a: RoaringBitmap, b: RoaringBitmap) -> bool {
        let a_union_b = a.union(&b);
        let union_elems = a_union_b.into_iter().collect::<Vec<u32>>();
        union_elems.into_iter().all(|e| a.contains(e) || b.contains(e))
    }

    #[quickcheck]
    fn union_must_be_commutative(a: RoaringBitmap, b: RoaringBitmap) -> bool {
        let a_union_b = a.union(&b);
        let b_union_a = b.union(&a);
        a_union_b == b_union_a
    }

    #[quickcheck]
    fn union_with_itself_is_the_same_set(a: RoaringBitmap) -> bool {
        a.union(&a) == a
    }

    #[quickcheck]
    fn union_with_an_empty_set_is_the_same_set(a: RoaringBitmap) -> bool {
        let empty_set = RoaringBitmap::new();
        a.union(&empty_set) == a
    }

    #[quickcheck]
    fn union_of_a_set_with_its_subset_is_the_same_set(x: SetAndSubset<u32>) -> bool {
        let mut set = RoaringBitmap::new();
        let mut subset = RoaringBitmap::new();
        x.set.iter().for_each(|&s| set.add(s));
        x.subset.iter().for_each(|&s| subset.add(s));

        set.union(&subset) == set
    }

    #[quickcheck]
    fn intersection_of_set_must_have_elements_in_both_sets(a: RoaringBitmap, b: RoaringBitmap) -> bool {
        a.intersection(&b).into_iter().all(|x| a.contains(x) && b.contains(x))
    }

    #[quickcheck]
    fn intersection_of_set_must_have_an_element_when_it_is_present_in_both_sets(a: RoaringBitmap, b: RoaringBitmap) -> bool {
        let a_intersection_b = a.intersection(&b);
        a.into_iter().all(|x| !b.contains(x) || a_intersection_b.contains(x)) &&
            b.into_iter().all(|x| !a.contains(x) || a_intersection_b.contains(x))
    }

    #[quickcheck]
    fn intersection_of_set_with_itself_is_the_same_set(a: RoaringBitmap) -> bool {
        a.intersection(&a) == a
    }

    #[quickcheck]
    fn intersection_of_set_with_empty_set_is_empty_set(a: RoaringBitmap) -> bool {
        let empty_set = RoaringBitmap::new();
        a.intersection(&empty_set) == empty_set
    }

    #[quickcheck]
    fn intersection_of_sets_is_commutative(a: RoaringBitmap, b: RoaringBitmap) -> bool {
        let a_intersection_b = a.intersection(&b);
        let b_intersection_a = b.intersection(&a);
        a_intersection_b == b_intersection_a
    }

    #[quickcheck]
    fn intersection_of_sets_is_associative(a: RoaringBitmap, b: RoaringBitmap, c: RoaringBitmap) -> bool {
        let ab_c = a.intersection(&b).intersection(&c);
        let a_bc = a.intersection(&b.intersection(&c));
        ab_c == a_bc
    }

    #[quickcheck]
    fn intersection_with_a_subset_is_the_subset(x: SetAndSubset<u32>) -> bool {
        let mut set = RoaringBitmap::new();
        let mut subset = RoaringBitmap::new();
        x.set.iter().for_each(|&s| set.add(s));
        x.subset.iter().for_each(|&s| subset.add(s));

        set.intersection(&subset) == subset
    }
}
