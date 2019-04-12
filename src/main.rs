#![feature(test)]
#![feature(const_fn_union)]
#![feature(stdsimd)]

extern crate test;
use aligned::{Aligned, A16};
use hibitset::BitSetLike;
use std::arch::x86_64::*;

const BITS_PER_PRIM: usize = std::mem::size_of::<u64>() * 8;

pub struct BitSet {
    level3: u64,
    level2: Vec<u64>,
    level1: Vec<u64>,
    level0: Vec<u64>,
}

#[inline(never)]
pub fn bitset_collect_old(bitset: &BitSet) -> Vec<u32> {
    bitset.iter().collect()
}

#[inline(never)]
pub fn bitset_collect_sse2(bitset: &BitSet) -> Vec<u32> {
    bitset.decode_iter::<Sse2Decoder>().collect()
}

fn next_layer(data: &[u64]) -> Vec<u64> {
    data.chunks(BITS_PER_PRIM)
        .map(|chunk| {
            let mut val = 0;
            for (i, &mask) in chunk.iter().enumerate() {
                if mask != 0 {
                    val = val | (1 << i);
                }
            }
            val
        })
        .collect()
}

impl BitSet {
    pub fn from_level0(level0: Vec<u64>) -> Self {
        let level1 = next_layer(&level0);
        let level2 = next_layer(&level1);
        let level3 = next_layer(&level2);
        BitSet {
            level3: level3.get(0).cloned().unwrap_or(0),
            level2,
            level1,
            level0,
        }
    }
}

trait IntoDecodeIter: BitSetLike + Sized {
    fn decode_iter<'a, D: Decoder>(&'a self) -> BitSetIter<'a, D, Self>;
}

impl<T: BitSetLike> IntoDecodeIter for T {
    // NOTE: The key here is obviously getting rid of vec allocations/clones.
    // This is obviously hard due to borrowing rules. I was able to reduce it to
    // one clone per chunk by providing buffers from outside.
    #[inline(always)]
    fn decode_iter<'a, D: Decoder>(&'a self) -> BitSetIter<'a, D, Self> {
        BitSetIter::<D, Self>::new(self)
    }
}

const LEVEL2_BATCH: usize = 16;
const LEVEL1_BATCH: usize = 8;
const LEVEL0_BATCH: usize = 24;

impl BitSetLike for BitSet {
    fn layer3(&self) -> usize {
        self.level3 as usize
    }

    fn layer2(&self, i: usize) -> usize {
        self.level2.get(i).cloned().unwrap_or(0) as usize
    }

    fn layer1(&self, i: usize) -> usize {
        self.level1.get(i).cloned().unwrap_or(0) as usize
    }

    fn layer0(&self, i: usize) -> usize {
        self.level0.get(i).cloned().unwrap_or(0) as usize
    }

    fn contains(&self, i: u32) -> bool {
        self.level0
            .get(i as usize / 64)
            .map_or(false, |b| b & (1 << i) != 0)
    }
}

fn test_indirect(b: BitSet) {
    // let values: Vec<u32> = iter.collect();
    for i in 0..10 {
        let iter = CoolBitSetIter::new(&b);
        test::black_box::<usize>(iter.count());
    }
}
#[inline(never)]
fn test_indirect_count(b: BitSet) -> usize {
    let iter = CoolBitSetIter::new(&b);
    iter.count()
}
#[inline(never)]
fn test_indirect_sum(b: BitSet) -> u32 {
    let iter = CoolBitSetIter::new(&b);
    iter.sum()
}
#[inline(never)]
fn test_indirect_sum_blackbox(b: BitSet) -> u32 {
    let iter = CoolBitSetIter::new(&b);
    iter.map(test::black_box).sum()
}

struct IndirectBase {
    pub input_ptr: *const u64,
    pub output_ptr: *mut u32,
    pub output_len: *mut usize,
    pub output_selector_tag: u32,
}

// We will produce a maximum of LEVEL2_MAX_GENERATED indices from L2 into the working buffer per batch
// These indices are used to scan the L1 index and are produced by a straight-forward method.
const PREFETCHING_LOOKAHEAD: usize = 4;
const LEVEL2_MAX_GENERATED: usize = 2;
const WORKING_BUFFER_STORAGE_REQUIRED: usize =
    BITS_PER_PRIM * LEVEL2_MAX_GENERATED + PREFETCHING_LOOKAHEAD;
const OUTPUT_BUFFER_STORAGE_REQUIRED: usize = WORKING_BUFFER_STORAGE_REQUIRED * BITS_PER_PRIM;

#[repr(align(16))]
pub struct CoolBitSetIter<'a> {
    working_buffer: Vec<u32>, //[u32; WORKING_BUFFER_STORAGE_REQUIRED],
    output_buffer: Vec<u32>,  //[u32; OUTPUT_BUFFER_STORAGE_REQUIRED],
    working_buffer_len: usize,
    output_buffer_len: usize,
    bitset: &'a BitSet,
    // iteration index for the working buffer
    working_iter_idx: usize,
    // iteration index for the output buffer
    output_iter_idx: usize,
    // iteration bit index of level 2 from the input bitset
    l2_idx: usize,
    // since the iter stores pointers into buffers in the indirect_bases we can't move it
    _pin: std::marker::PhantomPinned,
}

impl<'a> CoolBitSetIter<'a> {
    fn new(bitset: &'a BitSet) -> Self {
        Self {
            output_buffer: Vec::with_capacity(OUTPUT_BUFFER_STORAGE_REQUIRED),
            working_buffer: Vec::with_capacity(WORKING_BUFFER_STORAGE_REQUIRED),
            working_buffer_len: 0,
            output_buffer_len: 0,
            bitset: bitset,
            working_iter_idx: 0,
            output_iter_idx: 0,
            l2_idx: 0,
            _pin: std::marker::PhantomPinned,
        }
    }
    unsafe fn bases(&mut self) -> [IndirectBase; 2] {
        [
            IndirectBase {
                input_ptr: self.bitset.level1.get_unchecked(0),
                output_ptr: self.working_buffer.get_unchecked_mut(0),
                output_len: &mut self.working_buffer_len,
                output_selector_tag: 1 << 30,
            },
            IndirectBase {
                input_ptr: self.bitset.level0.get_unchecked(0),
                output_ptr: self.output_buffer.get_unchecked_mut(0),
                output_len: &mut self.output_buffer_len,
                output_selector_tag: 0,
            },
        ]
    }
    #[target_feature(enable = "sse2")]
    unsafe fn populate_working_buffer(&mut self) {
        // reset the working batch
        self.working_buffer_len = 0;
        self.working_iter_idx = 0;
        // produce new indices
        while self.l2_idx / 64 < self.bitset.level2.len()
            && self.working_buffer_len < LEVEL2_MAX_GENERATED
        {
            let l2_idx_element = self.l2_idx / 64;
            let mut bitset = self.bitset.layer2(l2_idx_element) as u64;
            // extract the bit index for the element from the lower 6 bits and create an inverted mask
            // this mask is used to remove bit indices we have already iterated through
            bitset = bitset & !((1u64 << (self.l2_idx & 0x3F)).checked_sub(1).unwrap_or(0));
            while bitset != 0 && self.working_buffer_len < LEVEL2_MAX_GENERATED {
                let t: u64 = bitset & bitset.overflowing_neg().0;
                let r = bitset.trailing_zeros();
                *self
                    .working_buffer
                    .get_unchecked_mut(self.working_buffer_len) = r + (l2_idx_element * 64) as u32;
                self.working_buffer_len += 1;
                self.l2_idx = (l2_idx_element * 64) + r as usize + 1;
                bitset ^= t;
            }
            if bitset == 0 {
                self.l2_idx = (l2_idx_element + 1) * 64;
            }
        }
    }

    // Takes an input array of tagged u32 indices and an array of (input_pointer, output_pointer, output_len, output_selector_tag)
    // The top 2 bits of a tagged u32 input index is used to select into the indirect_bases array.
    // The bottom 30 bits of the index is added onto the input_pointer to get a u64 value to be interpreted as a bitmap
    // For each bit set in the bitmap,
    // An output u32 index is generated by multiplying the input index by 64, adding the bit index and adding the output_selector_tag.
    // All output u32 indices are written to the output_pointer of the selected indirect_base offset by its output_len.
    // The output_len is then increased by the number of output u32 indices.
    #[target_feature(enable = "sse2")]
    unsafe fn bitmap_decode_sse2_indirect(&mut self) {
        let mut indirect_bases = self.bases();

        while self.working_iter_idx < self.working_buffer_len {
            let prefetch_idx = self
                .working_buffer
                .get_unchecked(self.working_iter_idx + PREFETCHING_LOOKAHEAD); // the buffer size is bigger than the potential length so we can always safely get future elements
            let prefetch_base = indirect_bases.get_unchecked_mut((prefetch_idx >> 30) as usize);
            let prefetch_addr = prefetch_base
                .input_ptr
                .offset((prefetch_idx & ((1 << 30) - 1)) as isize);
            _mm_prefetch(prefetch_addr as *const i8, 1);

            let idx = *self.working_buffer.get_unchecked_mut(self.working_iter_idx);
            self.working_iter_idx += 1;
            let indirect_base = indirect_bases.get_unchecked_mut((idx >> 30) as usize); // use the top 2 bits to get which indirect base this index is for
                                                                                        // Mask away the top 2 bits to get the untagged index
            let untagged_idx = idx & ((1 << 30) - 1);
            // offset the input ptr with the untagged index to get the input u64 bitmap
            let bits = *indirect_base.input_ptr.offset(untagged_idx as isize);

            // Multiply the untagged index by 64 to get the index in the next level of the hierarchy,
            // then add the output_selector_tag
            let mut base: __m128i =
                _mm_set1_epi32(((untagged_idx * 64) | indirect_base.output_selector_tag) as i32);

            for i in 0..4 {
                let move_mask = bits >> (i * 16);

                // pack the elements to the left using the move mask
                let movemask_a = move_mask & 0xF;
                let movemask_b = (move_mask >> 4) & 0xF;
                let movemask_c = (move_mask >> 8) & 0xF;
                let movemask_d = (move_mask >> 12) & 0xF;

                let mut a = lookup_index(movemask_a as isize);
                let mut b = lookup_index(movemask_b as isize);
                let mut c = lookup_index(movemask_c as isize);
                let mut d = lookup_index(movemask_d as isize);

                // offset by bit index
                a = _mm_add_epi32(base, a);
                b = _mm_add_epi32(base, b);
                c = _mm_add_epi32(base, c);
                d = _mm_add_epi32(base, d);
                // increase the base
                if i != 3 {
                    base = _mm_add_epi32(base, _mm_set1_epi32(16));
                }

                // correct lookups
                b = _mm_add_epi32(_mm_set1_epi32(4), b);
                c = _mm_add_epi32(_mm_set1_epi32(8), c);
                d = _mm_add_epi32(_mm_set1_epi32(12), d);

                let a_out = a;
                let b_out = b;
                let c_out = c;
                let d_out = d;

                // get the number of elements being output
                let adv_a = LUT_POPCNT.get_unchecked(movemask_a as usize);
                let adv_b = LUT_POPCNT.get_unchecked(movemask_b as usize);
                let adv_c = LUT_POPCNT.get_unchecked(movemask_c as usize);
                let adv_d = LUT_POPCNT.get_unchecked(movemask_d as usize);

                let adv_ab = adv_a + adv_b;
                let adv_abc = adv_ab + adv_c;
                let adv_abcd = adv_abc + adv_d;

                let len = *indirect_base.output_len;
                let out_ptr = indirect_base.output_ptr.offset(len as isize);
                *indirect_base.output_len += adv_abcd as usize;

                // perform the store
                _mm_storeu_si128(out_ptr as *mut _, a_out);
                _mm_storeu_si128(out_ptr.offset(*adv_a as isize) as _, b_out);
                _mm_storeu_si128(out_ptr.offset(adv_ab as isize) as _, c_out);
                _mm_storeu_si128(out_ptr.offset(adv_abc as isize) as _, d_out);
                // println!("to_advance {} pos {} base {}", to_advance, out_pos, _mm_extract_epi32(base, 0));
            }
        }
    }
}
impl<'a> Iterator for CoolBitSetIter<'a> {
    type Item = u32;

    // Here, we define the sequence using `.curr` and `.next`.
    // The return type is `Option<T>`:
    //     * When the `Iterator` is finished, `None` is returned.
    //     * Otherwise, the next value is wrapped in `Some` and returned.
    #[inline(always)]
    fn next(&mut self) -> Option<u32> {
        unsafe {
            if self.output_iter_idx >= self.output_buffer_len {
                if self.working_iter_idx >= self.working_buffer_len {
                    self.populate_working_buffer();
                    if self.working_iter_idx >= self.working_buffer_len {
                        return None;
                    }
                }
                self.output_iter_idx = 0;
                self.output_buffer_len = 0;
                self.bitmap_decode_sse2_indirect();
            }
            let val = *self.output_buffer.get_unchecked(self.output_iter_idx);
            self.output_iter_idx += 1;
            Some(val)
        }
    }
}

#[repr(align(16))]
pub struct BitSetIter<'a, D: Decoder, B: BitSetLike + 'a> {
    level2_buffer: [u32; BITS_PER_PRIM * LEVEL2_BATCH],
    level1_buffer: [u32; BITS_PER_PRIM * LEVEL1_BATCH],
    level0_buffer: [u32; BITS_PER_PRIM * LEVEL0_BATCH],
    level2_len: usize,
    level1_len: usize,
    level0_len: usize,
    level2_idx: usize,
    level1_idx: usize,
    level0_idx: usize,
    bitset: &'a B,
    marker: std::marker::PhantomData<D>,
}

impl<'a, D: Decoder, B: BitSetLike + 'a> BitSetIter<'a, D, B> {
    fn new(bitset: &'a B) -> Self {
        // debug_assert!(bitset.level2.len() <= LEVEL2_BATCH);
        let mut this = Self {
            level2_buffer: [0; BITS_PER_PRIM * LEVEL2_BATCH],
            level1_buffer: [0; BITS_PER_PRIM * LEVEL1_BATCH],
            level0_buffer: [0; BITS_PER_PRIM * LEVEL0_BATCH],
            level2_len: 0,
            level1_len: 0,
            level0_len: 0,
            level2_idx: LEVEL1_BATCH,
            level1_idx: LEVEL0_BATCH,
            level0_idx: 0,
            bitset,
            marker: std::marker::PhantomData,
        };

        if bitset.layer3() == 0 {
            return this;
        }

        unsafe {
            this.level2_len = D::decode_slice(
                (0..LEVEL2_BATCH).map(|i| bitset.layer2(i) as _),
                (0..).step_by(BITS_PER_PRIM),
                &mut this.level2_buffer,
            ) as _;

            let l2_buf = &this.level2_buffer[0..this.level2_len as _];
            this.level1_len = D::decode_slice(
                l2_buf
                    .iter()
                    .take(LEVEL1_BATCH)
                    .map(|b| bitset.layer1(*b as _) as _),
                l2_buf.iter().map(|b| b * BITS_PER_PRIM as u32),
                &mut this.level1_buffer,
            ) as _;

            let l1_buf = &this.level1_buffer[0..this.level1_len as _];
            this.level0_len = D::decode_slice(
                l1_buf
                    .iter()
                    .take(LEVEL0_BATCH)
                    .map(|b| bitset.layer0(*b as _) as _),
                l1_buf.iter().map(|b| b * BITS_PER_PRIM as u32),
                &mut this.level0_buffer,
            ) as _;
        }
        this
    }
}

unsafe fn populate_buf<'a, D: Decoder, B: BitSetLike>(b: &mut BitSetIter<'a, D, B>) -> usize {
    let bitset = b.bitset;

    if b.level1_idx >= b.level1_len {
        if b.level2_idx >= b.level2_len {
            b.level0_len = 0;
            return 0;
        }

        let l2_buf = &b.level2_buffer[b.level2_idx..b.level2_len];
        b.level1_len = D::decode_slice(
            l2_buf
                .iter()
                .take(LEVEL1_BATCH)
                .map(|b| bitset.layer1(*b as usize) as _),
            l2_buf.iter().map(|b| b * BITS_PER_PRIM as u32),
            &mut b.level1_buffer,
        ) as _;

        b.level2_idx += LEVEL1_BATCH;
        b.level1_idx = 0;
    }

    let l1_buf = &b.level1_buffer[b.level1_idx..b.level1_len];
    b.level0_len = D::decode_slice(
        l1_buf
            .iter()
            .take(LEVEL0_BATCH)
            .map(|b| bitset.layer0(*b as usize) as _),
        l1_buf.iter().map(|b| b * BITS_PER_PRIM as u32),
        &mut b.level0_buffer,
    ) as _;
    b.level1_idx += LEVEL0_BATCH;
    b.level0_idx = 0;
    b.level0_len
}

impl<'a, D: Decoder, B: BitSetLike> Iterator for BitSetIter<'a, D, B> {
    type Item = u32;
    #[inline(always)]
    fn next(&mut self) -> Option<u32> {
        if self.level0_len == 0 {
            return None;
        }

        unsafe {
            if self.level0_idx >= self.level0_len {
                if populate_buf(self) == 0 {
                    return None;
                }
            }
            let out = self.level0_buffer.get_unchecked(self.level0_idx);
            self.level0_idx += 1;
            return Some(*out);
        }
    }
}

pub trait Decoder {
    fn decode<I, O>(bitmap: I, offset: O, out: &mut Vec<u32>) -> usize
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: ExactSizeIterator,
        O: IntoIterator<Item = u32>,
    {
        let bitmap_iter = bitmap.into_iter();
        out.clear();
        out.reserve(bitmap_iter.len() * BITS_PER_PRIM);
        unsafe {
            let slice =
                std::slice::from_raw_parts_mut(out.as_mut_ptr(), bitmap_iter.len() * BITS_PER_PRIM);
            let len = Self::decode_slice(bitmap_iter, offset, slice);
            out.set_len(len);
            len
        }
    }

    unsafe fn decode_slice<I, O>(bitmap: I, offset: O, out: &mut [u32]) -> usize
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: ExactSizeIterator,
        O: IntoIterator<Item = u32>;
}

pub struct NaiveDecoder;
pub struct CtzDecoder;
pub struct Sse2Decoder;

impl Decoder for NaiveDecoder {
    #[inline(always)]
    unsafe fn decode_slice<I, O>(bitmap: I, offset: O, out: &mut [u32]) -> usize
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: ExactSizeIterator,
        O: IntoIterator<Item = u32>,
    {
        bitmap_decode_naive(bitmap, offset, out)
    }
}

impl Decoder for CtzDecoder {
    #[inline(always)]
    unsafe fn decode_slice<I, O>(bitmap: I, offset: O, out: &mut [u32]) -> usize
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: ExactSizeIterator,
        O: IntoIterator<Item = u32>,
    {
        bitmap_decode_ctz(bitmap, offset, out)
    }
}

impl Decoder for Sse2Decoder {
    #[inline(never)]
    unsafe fn decode_slice<I, O>(bitmap: I, offset: O, out: &mut [u32]) -> usize
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: ExactSizeIterator,
        O: IntoIterator<Item = u32>,
    {
        bitmap_decode_sse2(bitmap, offset, out)
    }
}

fn main() {
    // let mut bitset = BitSet::new();
    // bitset.add(1_048_575);
    // println!("bitset {:?}", bitset);

    // let values = [0x305u64];
    // let values = [0xABCDEFu64];
    // let values = [0xFFFFFFFFFFFFFFFFu64];

    let mut bitmap = vec![0xbbae187b00003b05, 0, 1];
    // let mut bitmap = vec![0xABCDEF];
    bitmap.resize(64 * 64 - 1, 0);
    bitmap.push(1);

    // let values = [
    //     0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    //     0x1, 0x1,
    // ];

    // let mut out = Vec::new();
    // bitmap_decode_naive(&values, &mut out);

    let bitset = BitSet::from_level0(bitmap.clone());
    // test_indirect(bitset);
    test_indirect(BitSet::from_level0(interleaved_buf()));
    test_indirect(BitSet::from_level0(random_num_buf()));
    test_indirect(BitSet::from_level0(blocky_buf()));
    // println!(
    //     "{}",
    //     test_indirect_count(BitSet::from_level0(interleaved_buf()))
    // );
    // println!(
    //     "{}",
    //     test_indirect_sum(BitSet::from_level0(interleaved_buf()))
    // );
    // println!(
    //     "{}",
    //     test_indirect_sum_blackbox(BitSet::from_level0(interleaved_buf()))
    // );
    test_indirect(BitSet::from_level0(interleaved_buf()));

    // bitset_collect_sse2(&bitset);
    // bitset_collect_old(&bitset);

    // let out: Vec<_> = bitset.decode_iter::<Sse2Decoder>().collect();
    // println!("{:b} -- {:?}", bitmap[0], out);
}

pub unsafe fn print_bytes(prefix: &str, x: __m128i) {
    println!(
        "{} x: {:x} y: {:x}, z: {:x}, w: {:x} ",
        prefix,
        _mm_extract_epi32(x, 0),
        _mm_extract_epi32(x, 1),
        _mm_extract_epi32(x, 2),
        _mm_extract_epi32(x, 3)
    );
}

const LUT_INDICES: Aligned<A16, [[u32; 4]; 16]> = {
    Aligned([
        [0, 0, 0, 0], // 0000 .... 0 0 0 0
        [0, 0, 0, 0], // 0001 X... 0 0 0 0
        [1, 0, 0, 0], // 0010 Y... 0 1 0 0
        [0, 1, 0, 0], // 0011 XY.. 0 0 0 0
        [2, 0, 0, 0], // 0100 Z... 0 0 2 0
        [0, 2, 0, 0], // 0101 XZ.. 0 0 1 0
        [1, 2, 0, 0], // 0110 YZ.. 0 1 1 0
        [0, 1, 2, 0], // 0111 XYZ. 0 0 0 0
        [3, 0, 0, 0], // 1000 W... 0 0 0 3
        [0, 3, 0, 0], // 1001 XW.. 0 0 0 2
        [1, 3, 0, 0], // 1010 YW.. 0 1 0 2
        [0, 1, 3, 0], // 1011 XYW. 0 0 0 1
        [2, 3, 0, 0], // 1100 ZW.. 0 0 2 2
        [0, 2, 3, 0], // 1101 XZW. 0 0 1 1
        [1, 2, 3, 0], // 1110 YZW. 0 1 1 1
        [0, 1, 2, 3], // 1111 XYZW 0 0 0 0
    ])
};

// NOTE: I'm not sure what primitive size fits here best, but clearly u8 is worse
const LUT_POPCNT: Aligned<A16, [u16; 16]> = {
    Aligned([
        0, // 0000 .... 0 0 0 0
        1, // 0001 X... 0 0 0 0
        1, // 0010 Y... 0 1 0 0
        2, // 0011 XY.. 0 0 0 0
        1, // 0100 Z... 0 0 2 0
        2, // 0101 XZ.. 0 0 1 0
        2, // 0110 YZ.. 0 1 1 0
        3, // 0111 XYZ. 0 0 0 0
        1, // 1000 W... 0 0 0 3
        2, // 1001 XW.. 0 0 0 2
        2, // 1010 YW.. 0 1 0 2
        3, // 1011 XYW. 0 0 0 1
        2, // 1100 ZW.. 0 0 2 2
        3, // 1101 XZW. 0 0 1 1
        3, // 1110 YZW. 0 1 1 1
        4, // 1111 XYZW 0 0 0 0
    ])
};

#[inline(always)]
unsafe fn lookup_index(mask: isize) -> __m128i {
    _mm_load_si128(LUT_INDICES.as_ptr().offset(mask) as _)
}

unsafe fn bitmap_decode_sse2<I, O>(bitmap: I, offsets: O, out: &mut [u32]) -> usize
where
    I: IntoIterator<Item = u64>,
    I::IntoIter: ExactSizeIterator,
    O: IntoIterator<Item = u32>,
{
    let mut out_pos = 0;

    let bitmap_iter = bitmap.into_iter();
    let mut offset_iter = offsets.into_iter();

    debug_assert!(out.len() >= bitmap_iter.len() * 64);

    for bits in bitmap_iter {
        let offset = offset_iter.next().unwrap();
        if bits == 0 {
            continue;
        }

        let mut base: __m128i = _mm_set1_epi32(offset as i32);

        for i in 0..4 {
            let move_mask = (bits >> (i * 16)) as u16;

            // pack the elements to the left using the move mask
            let movemask_a = move_mask & 0xF;
            let movemask_b = (move_mask >> 4) & 0xF;
            let movemask_c = (move_mask >> 8) & 0xF;
            let movemask_d = (move_mask >> 12) & 0xF;

            let mut a = lookup_index(movemask_a as isize);
            let mut b = lookup_index(movemask_b as isize);
            let mut c = lookup_index(movemask_c as isize);
            let mut d = lookup_index(movemask_d as isize);

            // offset by bit index
            a = _mm_add_epi32(base, a);
            b = _mm_add_epi32(base, b);
            c = _mm_add_epi32(base, c);
            d = _mm_add_epi32(base, d);
            // increase the base
            if i != 3 {
                base = _mm_add_epi32(base, _mm_set1_epi32(16));
            }

            // correct lookups
            b = _mm_add_epi32(_mm_set1_epi32(4), b);
            c = _mm_add_epi32(_mm_set1_epi32(8), c);
            d = _mm_add_epi32(_mm_set1_epi32(12), d);

            let a_out = a;
            let b_out = b;
            let c_out = c;
            let d_out = d;

            // get the number of elements being output
            let adv_a = LUT_POPCNT.get_unchecked(movemask_a as usize);
            let adv_b = LUT_POPCNT.get_unchecked(movemask_b as usize);
            let adv_c = LUT_POPCNT.get_unchecked(movemask_c as usize);
            let adv_d = LUT_POPCNT.get_unchecked(movemask_d as usize);

            let adv_ab = adv_a + adv_b;
            let adv_abc = adv_ab + adv_c;
            let adv_abcd = adv_abc + adv_d;

            let out_ptr = out.get_unchecked_mut(out_pos) as *mut u32;
            out_pos += adv_abcd as usize;

            // perform the store
            _mm_storeu_si128(out_ptr as *mut _, a_out);
            _mm_storeu_si128(out_ptr.offset(*adv_a as isize) as _, b_out);
            _mm_storeu_si128(out_ptr.offset(adv_ab as isize) as _, c_out);
            _mm_storeu_si128(out_ptr.offset(adv_abc as isize) as _, d_out);
            // println!("to_advance {} pos {} base {}", to_advance, out_pos, _mm_extract_epi32(base, 0));
        }
    }
    out_pos
}

#[inline(always)]
pub unsafe fn bitmap_decode_naive<I, O>(bitmap: I, offset: O, out: &mut [u32]) -> usize
where
    I: IntoIterator<Item = u64>,
    I::IntoIter: ExactSizeIterator,
    O: IntoIterator<Item = u32>,
{
    let mut pos = 0;
    let bitmap_iter = bitmap.into_iter();
    debug_assert!(out.len() >= bitmap_iter.len() * 64);
    for (v, offset) in bitmap_iter.zip(offset) {
        let mut bitset = v;
        let mut p = offset;
        while bitset != 0 {
            if (bitset & 0x1) != 0 {
                *out.get_unchecked_mut(pos) = p;
                pos += 1;
            }
            bitset >>= 1;
            p += 1;
        }
    }
    pos
}

pub unsafe fn bitmap_decode_ctz<I, O>(bitmap: I, offset: O, out: &mut [u32]) -> usize
where
    I: IntoIterator<Item = u64>,
    I::IntoIter: ExactSizeIterator,
    O: IntoIterator<Item = u32>,
{
    let mut pos = 0;
    let bitmap_iter = bitmap.into_iter();
    debug_assert!(out.len() >= bitmap_iter.len() * 64);

    for (v, offset) in bitmap_iter.zip(offset) {
        let mut bitset = v as i64;
        while bitset != 0 {
            let t: i64 = bitset & bitset.overflowing_neg().0;
            let r = bitset.trailing_zeros();
            *out.get_unchecked_mut(pos) = r + offset;
            pos += 1;
            bitset ^= t;
        }
    }
    return pos;
}

const TEST_BUF_SIZE: usize = 4 * 64 * 64;
// const TEST_BUF_SIZE: usize = (256 * 1024 * 1024) / 64;

fn empty_buf() -> Vec<u64> {
    vec![0; TEST_BUF_SIZE]
}

fn full_buff() -> Vec<u64> {
    vec![0xFFFFFFFFFFFFFFFFu64; TEST_BUF_SIZE]
}

fn blocky_buf() -> Vec<u64> {
    use std::iter::repeat;

    // a single long sequence of chains trips up the compiler :/
    let mut vec: Vec<u64> = repeat(0x5555555555555555u64)
        .take(1234)
        .chain(repeat(0u64).take(1000))
        .chain(repeat(0xFFFFFFFFFFFFFFFFu64).take(500))
        .chain(repeat(0xbbae187bfcdd3b05u64).take(1234))
        .chain(repeat(0xd7156e450545b7adu64).take(1456))
        .chain(repeat(0u64).take(7500))
        .chain(repeat(0x5555555555555555u64).take(55))
        .chain(repeat(0u64).take(50))
        .collect();
    vec.extend(
        repeat(0xFF00FF5500330210u64)
            .take(2500)
            .chain(repeat(0u64).take(7))
            .chain(repeat(0x1228b38cf8ef551bu64).take(1500))
            .chain(repeat(0x5555555555555555u64).take(1156))
            .chain(repeat(0u64).take(42)),
    );
    vec.resize(TEST_BUF_SIZE, 0xFFFFFFFF00000000u64);

    vec
}

fn interleaved_buf() -> Vec<u64> {
    vec![0x5555555555555555u64; TEST_BUF_SIZE]
}

fn random_num_buf() -> Vec<u64> {
    vec![0xbbae187bfcdd3b05u64; TEST_BUF_SIZE]
}
#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn test_sse2_equal_bitset() {
        // let bitmap = vec![0xbbae187b00003b05, 0, 1];
        // bitmap.resize(64 * 64 - 1, 0);
        // bitmap.push(1);
        // dbg!(bitmap.len());
        let bitmap = blocky_buf();
        let bitset = BitSet::from_level0(bitmap.clone());

        let mut sse2_values = Vec::with_capacity(bitmap.len() * 64);
        Sse2Decoder::decode(bitmap.iter().cloned(), (0..).step_by(64), &mut sse2_values);
        assert_eq!(
            sse2_values,
            bitset.decode_iter::<Sse2Decoder>().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_sse2_equal_naive() {
        let bitmap = vec![0xbbae187b00003b05u64; 10];
        let mut naive_values = Vec::with_capacity(bitmap.len() * BITS_PER_PRIM);
        let mut sse2_values = Vec::with_capacity(bitmap.len() * BITS_PER_PRIM);
        let naive_len =
            NaiveDecoder::decode(bitmap.iter().cloned(), (0..).step_by(64), &mut naive_values);
        let sse2_len =
            Sse2Decoder::decode(bitmap.iter().cloned(), (0..).step_by(64), &mut sse2_values);
        assert_eq!(naive_len, sse2_len);
        assert_eq!(naive_values, sse2_values);
    }

    fn bench_decode<D: Decoder>(b: &mut Bencher, bitmap: &[u64]) {
        let mut values = Vec::with_capacity(bitmap.len() * BITS_PER_PRIM);
        b.iter(|| {
            values.clear();
            test::black_box(D::decode(
                test::black_box(bitmap.iter().cloned()),
                (0..).step_by(64),
                &mut values,
            ));
        })
    }

    fn bench_bitset_old(b: &mut Bencher, bitmap: &[u64]) {
        let mut values = Vec::with_capacity(bitmap.len() * BITS_PER_PRIM);
        let bitset_conv = BitSet::from_level0(bitmap.iter().cloned().collect());
        let mut hibitset = hibitset::BitSet::new();
        hibitset.extend(bitset_conv.iter());

        b.iter(|| {
            values.clear();
            test::black_box(values.extend((&hibitset).iter()));
        })
    }

    fn bench_bitset<D: Decoder>(b: &mut Bencher, bitmap: &[u64]) {
        let bitset_conv = BitSet::from_level0(bitmap.iter().cloned().collect());
        let mut hibitset = hibitset::BitSet::new();
        hibitset.extend(bitset_conv.iter());

        b.iter(|| {
            test::black_box::<u32>(hibitset.decode_iter::<D>().sum());
        })
    }

    fn bench_indirect(b: &mut Bencher, bitmap: &[u64]) {
        let bitset = BitSet::from_level0(bitmap.iter().cloned().collect());

        b.iter(|| {
            test::black_box::<u32>(CoolBitSetIter::new(&bitset).sum());
        })
    }

    #[bench]
    fn bench_sse2_empty(b: &mut Bencher) {
        bench_decode::<Sse2Decoder>(b, &empty_buf());
    }
    #[bench]
    fn bench_sse2_full(b: &mut Bencher) {
        bench_decode::<Sse2Decoder>(b, &full_buff());
    }
    #[bench]
    fn bench_sse2_test(b: &mut Bencher) {
        bench_decode::<Sse2Decoder>(b, &VEC_DECODE_TABLE);
    }
    #[bench]
    fn bench_sse2_interleaved(b: &mut Bencher) {
        bench_decode::<Sse2Decoder>(b, &interleaved_buf());
    }
    #[bench]
    fn bench_sse2_random(b: &mut Bencher) {
        bench_decode::<Sse2Decoder>(b, &random_num_buf());
    }
    #[bench]
    fn bench_sse2_blocky(b: &mut Bencher) {
        bench_decode::<Sse2Decoder>(b, &blocky_buf());
    }

    #[bench]
    fn bench_old_bitset_empty(b: &mut Bencher) {
        bench_bitset_old(b, &empty_buf());
    }
    #[bench]
    fn bench_old_bitset_full(b: &mut Bencher) {
        bench_bitset_old(b, &full_buff());
    }
    #[bench]
    fn bench_old_bitset_test(b: &mut Bencher) {
        bench_bitset_old(b, &VEC_DECODE_TABLE);
    }
    #[bench]
    fn bench_old_bitset_interleaved(b: &mut Bencher) {
        bench_bitset_old(b, &interleaved_buf());
    }
    #[bench]
    fn bench_old_bitset_random(b: &mut Bencher) {
        bench_bitset_old(b, &random_num_buf());
    }
    #[bench]
    fn bench_old_bitset_blocky(b: &mut Bencher) {
        bench_bitset_old(b, &blocky_buf());
    }

    #[bench]
    fn bench_sse2_bitset_empty(b: &mut Bencher) {
        bench_bitset::<Sse2Decoder>(b, &empty_buf());
    }
    #[bench]
    fn bench_sse2_bitset_full(b: &mut Bencher) {
        bench_bitset::<Sse2Decoder>(b, &full_buff());
    }
    #[bench]
    fn bench_sse2_bitset_test(b: &mut Bencher) {
        bench_bitset::<Sse2Decoder>(b, &VEC_DECODE_TABLE);
    }
    #[bench]
    fn bench_sse2_bitset_interleaved(b: &mut Bencher) {
        bench_bitset::<Sse2Decoder>(b, &interleaved_buf());
    }
    #[bench]
    fn bench_sse2_bitset_random(b: &mut Bencher) {
        bench_bitset::<Sse2Decoder>(b, &random_num_buf());
    }
    #[bench]
    fn bench_sse2_bitset_blocky(b: &mut Bencher) {
        bench_bitset::<Sse2Decoder>(b, &blocky_buf());
    }

    #[bench]
    fn bench_naive_bitset_empty(b: &mut Bencher) {
        bench_bitset::<NaiveDecoder>(b, &empty_buf());
    }
    #[bench]
    fn bench_naive_bitset_full(b: &mut Bencher) {
        bench_bitset::<NaiveDecoder>(b, &full_buff());
    }
    #[bench]
    fn bench_naive_bitset_test(b: &mut Bencher) {
        bench_bitset::<NaiveDecoder>(b, &VEC_DECODE_TABLE);
    }
    #[bench]
    fn bench_naive_bitset_interleaved(b: &mut Bencher) {
        bench_bitset::<NaiveDecoder>(b, &interleaved_buf());
    }
    #[bench]
    fn bench_naive_bitset_random(b: &mut Bencher) {
        bench_bitset::<NaiveDecoder>(b, &random_num_buf());
    }
    #[bench]
    fn bench_naive_bitset_blocky(b: &mut Bencher) {
        bench_bitset::<NaiveDecoder>(b, &blocky_buf());
    }

    #[bench]
    fn bench_ctz_bitset_empty(b: &mut Bencher) {
        bench_bitset::<CtzDecoder>(b, &empty_buf());
    }
    #[bench]
    fn bench_ctz_bitset_full(b: &mut Bencher) {
        bench_bitset::<CtzDecoder>(b, &full_buff());
    }
    #[bench]
    fn bench_ctz_bitset_test(b: &mut Bencher) {
        bench_bitset::<CtzDecoder>(b, &VEC_DECODE_TABLE);
    }
    #[bench]
    fn bench_ctz_bitset_interleaved(b: &mut Bencher) {
        bench_bitset::<CtzDecoder>(b, &interleaved_buf());
    }
    #[bench]
    fn bench_ctz_bitset_random(b: &mut Bencher) {
        bench_bitset::<CtzDecoder>(b, &random_num_buf());
    }
    #[bench]
    fn bench_ctz_bitset_blocky(b: &mut Bencher) {
        bench_bitset::<CtzDecoder>(b, &blocky_buf());
    }

    #[bench]
    fn bench_naive_empty(b: &mut Bencher) {
        bench_decode::<NaiveDecoder>(b, &empty_buf());
    }
    #[bench]
    fn bench_naive_full(b: &mut Bencher) {
        bench_decode::<NaiveDecoder>(b, &full_buff());
    }
    #[bench]
    fn bench_naive_test(b: &mut Bencher) {
        bench_decode::<NaiveDecoder>(b, &VEC_DECODE_TABLE);
    }
    #[bench]
    fn bench_naive_interleaved(b: &mut Bencher) {
        bench_decode::<NaiveDecoder>(b, &interleaved_buf());
    }
    #[bench]
    fn bench_naive_random(b: &mut Bencher) {
        bench_decode::<NaiveDecoder>(b, &random_num_buf());
    }
    #[bench]
    fn bench_naive_blocky(b: &mut Bencher) {
        bench_decode::<NaiveDecoder>(b, &blocky_buf());
    }

    #[bench]
    fn bench_ctz_empty(b: &mut Bencher) {
        bench_decode::<CtzDecoder>(b, &empty_buf());
    }
    #[bench]
    fn bench_ctz_full(b: &mut Bencher) {
        bench_decode::<CtzDecoder>(b, &full_buff());
    }
    #[bench]
    fn bench_ctz_test(b: &mut Bencher) {
        bench_decode::<CtzDecoder>(b, &VEC_DECODE_TABLE);
    }
    #[bench]
    fn bench_ctz_interleaved(b: &mut Bencher) {
        bench_decode::<CtzDecoder>(b, &interleaved_buf());
    }
    #[bench]
    fn bench_ctz_random(b: &mut Bencher) {
        bench_decode::<CtzDecoder>(b, &random_num_buf());
    }
    #[bench]
    fn bench_ctz_blocky(b: &mut Bencher) {
        bench_decode::<CtzDecoder>(b, &blocky_buf());
    }
    #[bench]
    fn bench_indirect_empty(b: &mut Bencher) {
        bench_indirect(b, &empty_buf());
    }
    #[bench]
    fn bench_indirect_full(b: &mut Bencher) {
        bench_indirect(b, &full_buff());
    }
    #[bench]
    fn bench_indirect_test(b: &mut Bencher) {
        bench_indirect(b, &VEC_DECODE_TABLE);
    }
    #[bench]
    fn bench_indirect_interleaved(b: &mut Bencher) {
        bench_indirect(b, &interleaved_buf());
    }
    #[bench]
    fn bench_indirect_random(b: &mut Bencher) {
        bench_indirect(b, &random_num_buf());
    }
    #[bench]
    fn bench_indirect_blocky(b: &mut Bencher) {
        bench_indirect(b, &blocky_buf());
    }

    const VEC_DECODE_TABLE: [u64; 106] = [
        0x00010203, 0x04050607, 0x08090A0B, 0x0C, 0x0D, 0x0E, 0x0F101112, 0x13141516, 0x1718191A,
        0x1B, 0x1C, 0x1D, 0x1E1F2021, 0x22232425, 0x26272829, 0x2A, 0x2B, 0x2C, 0x2D2E2F30,
        0x31323334, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C3D3E3F, 0x40414243, 0x44454647,
        0x48, 0x49, 0x4A, 0x4B4C4D4E, 0x4F505152, 0x53545556, 0x57, 0x58, 0x59, 0x5A5B5C5D,
        0x5E5F6061, 0x62636465, 0x66, 0x67, 0x68, 0x696A6B6C, 0x6D6E6F70, 0x71727374, 0x75, 0x76,
        0x77, 0x78797A7B, 0x7C7D7E7F, 0x80818283, 0x84, 0x85, 0x86, 0x8788898A, 0x8B8C8D8E,
        0x8F909192, 0x93, 0x94, 0x95, 0x96979899, 0x9A9B9C9D, 0x9E9FA0A1, 0xA2, 0xA3, 0xA4,
        0xA5A6A7A8, 0xA9AAABAC, 0xADAEAFB0, 0xB1, 0xB2, 0xB3, 0xB4B5B6B7, 0xB8B9BABB, 0xBCBDBEBF,
        0xC0, 0xC1, 0xC2, 0xC3C4C5C6, 0xC7C8C9CA, 0xCBCCCDCE, 0xCF, 0xD0, 0xD1, 0xD2D3D4D5,
        0xD6D7D8D9, 0xDADBDCDD, 0xDE, 0xDF, 0xE0, 0xE1E2E3E4, 0xE5E6E7E8, 0xE9EAEBEC, 0xED, 0xEE,
        0xEF, 0xF0F1F2F3, 0xF4F5F6F7, 0xF8F9FAFB, 0xFC, 0xFD, 0xFE, 0xFF,
    ];
}
