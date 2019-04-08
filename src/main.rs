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
        assert!(level3.len() <= 1);
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
    fn decode_iter<'a, D: Decoder>(&'a self) -> BitSetIter<'a, D, Self> {
        BitSetIter::<D, Self>::new(self)
    }
}

const LEVEL2_BATCH: usize = 16;
const LEVEL1_BATCH: usize = 4;
const LEVEL0_BATCH: usize = 4;

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

impl<'a, D: Decoder, B: BitSetLike> Iterator for BitSetIter<'a, D, B> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.level0_len == 0 {
            return None;
        }

        unsafe {
            if self.level0_idx >= self.level0_len {
                let bitset = self.bitset;

                if self.level1_idx >= self.level1_len {
                    if self.level2_idx >= self.level2_len {
                        self.level0_len = 0;
                        return None;
                    }

                    let l2_buf = &self.level2_buffer[self.level2_idx..self.level2_len];
                    self.level1_len = D::decode_slice(
                        l2_buf
                            .iter()
                            .take(LEVEL1_BATCH)
                            .map(|b| bitset.layer1(*b as usize) as _),
                        l2_buf.iter().map(|b| b * BITS_PER_PRIM as u32),
                        &mut self.level1_buffer,
                    ) as _;

                    self.level2_idx += LEVEL1_BATCH;
                    self.level1_idx = 0;
                }

                let l1_buf = &self.level1_buffer[self.level1_idx..self.level1_len];
                self.level0_len = D::decode_slice(
                    l1_buf
                        .iter()
                        .take(LEVEL0_BATCH)
                        .map(|b| bitset.layer0(*b as usize) as _),
                    l1_buf.iter().map(|b| b * BITS_PER_PRIM as u32),
                    &mut self.level0_buffer,
                ) as _;
                self.level1_idx += LEVEL0_BATCH;
                self.level0_idx = 0;
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
    #[inline(always)]
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
    println!("Hello, world!");
    // let mut bitset = BitSet::new();
    // bitset.add(1_048_575);
    // println!("bitset {:?}", bitset);

    // let values = [0x305u64];
    // let values = [0xABCDEFu64];
    // let values = [0xFFFFFFFFFFFFFFFFu64];

    let mut bitmap = vec![0xbbae187b00003b05, 0, 1];
    bitmap.resize(64 * 64 - 1, 0);
    bitmap.push(1);

    // let values = [
    //     0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    //     0x1, 0x1,
    // ];

    // let mut out = Vec::new();
    // bitmap_decode_naive(&values, &mut out);

    let out: Vec<_> = BitSet::from_level0(bitmap.clone())
        .decode_iter::<Sse2Decoder>()
        .collect();
    println!("{:b} -- {:?}", bitmap[0], out);
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
unsafe fn lookup_index(mask: u16) -> __m128i {
    _mm_load_si128(LUT_INDICES.as_ptr().offset(mask as isize) as _)
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
        // if bits == 0 {
        //     continue;
        // }
        let mut base: __m128i = _mm_set1_epi32(offset as i32);

        for i in 0..4 {
            let move_mask = (bits >> (i * 16)) as u16;

            // pack the elements to the left using the move mask
            let movemask_a = move_mask & 0xF;
            let movemask_b = (move_mask >> 4) & 0xF;
            let movemask_c = (move_mask >> 8) & 0xF;
            let movemask_d = (move_mask >> 12) & 0xF;

            let mut a = lookup_index(movemask_a);
            let mut b = lookup_index(movemask_b);
            let mut c = lookup_index(movemask_c);
            let mut d = lookup_index(movemask_d);

            // offset by bit index
            a = _mm_add_epi32(base, a);
            b = _mm_add_epi32(base, b);
            c = _mm_add_epi32(base, c);
            d = _mm_add_epi32(base, d);
            base = _mm_add_epi32(base, _mm_set1_epi32(16));

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
            // increase the base
            // println!("to_advance {} pos {} base {}", to_advance, out_pos, _mm_extract_epi32(base, 0));
        }
    }
    out_pos
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    fn empty_buf() -> Vec<u64> {
        vec![0; 16384]
    }

    fn full_buff() -> Vec<u64> {
        vec![0xFFFFFFFFFFFFFFFFu64; 16384]
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
        vec.resize(16384, 0xFFFFFFFF00000000u64);

        vec
    }

    fn interleaved_buf() -> Vec<u64> {
        vec![0x5555555555555555u64; 16384]
    }

    fn random_num_buf() -> Vec<u64> {
        vec![0xbbae187bfcdd3b05u64; 16384]
    }

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
        let mut values = Vec::with_capacity(bitmap.len() * BITS_PER_PRIM);
        let bitset_conv = BitSet::from_level0(bitmap.iter().cloned().collect());
        let mut hibitset = hibitset::BitSet::new();
        hibitset.extend(bitset_conv.iter());

        b.iter(|| {
            values.clear();
            test::black_box(values.extend(hibitset.decode_iter::<D>()));
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
    // const VEC_DECODE_TABLE: [u64; 256] = [
    //     0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
    //     0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D,
    //     0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C,
    //     0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B,
    //     0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
    //     0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    //     0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    //     0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
    //     0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86,
    //     0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95,
    //     0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4,
    //     0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3,
    //     0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF, 0xC0, 0xC1, 0xC2,
    //     0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1,
    //     0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0xE0,
    //     0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF,
    //     0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE,
    //     0xFF,
    // ];

    // static uint8_t vecDecodeTableByte[256][8] ALIGNED(16)  = {
    //     { 0, 0, 0, 0, 0, 0, 0, 0 }, /* 0x00 (00000000) */
    //     { 1, 0, 0, 0, 0, 0, 0, 0 }, /* 0x01 (00000001) */
    //     { 2, 0, 0, 0, 0, 0, 0, 0 }, /* 0x02 (00000010) */
    //     { 1, 2, 0, 0, 0, 0, 0, 0 }, /* 0x03 (00000011) */
    //     { 3, 0, 0, 0, 0, 0, 0, 0 }, /* 0x04 (00000100) */
    //     { 1, 3, 0, 0, 0, 0, 0, 0 }, /* 0x05 (00000101) */
    //     { 2, 3, 0, 0, 0, 0, 0, 0 }, /* 0x06 (00000110) */
    //     { 1, 2, 3, 0, 0, 0, 0, 0 }, /* 0x07 (00000111) */
    //     { 4, 0, 0, 0, 0, 0, 0, 0 }, /* 0x08 (00001000) */
    //     { 1, 4, 0, 0, 0, 0, 0, 0 }, /* 0x09 (00001001) */
    //     { 2, 4, 0, 0, 0, 0, 0, 0 }, /* 0x0A (00001010) */
    //     { 1, 2, 4, 0, 0, 0, 0, 0 }, /* 0x0B (00001011) */
    //     { 3, 4, 0, 0, 0, 0, 0, 0 }, /* 0x0C (00001100) */
    //     { 1, 3, 4, 0, 0, 0, 0, 0 }, /* 0x0D (00001101) */
    //     { 2, 3, 4, 0, 0, 0, 0, 0 }, /* 0x0E (00001110) */
    //     { 1, 2, 3, 4, 0, 0, 0, 0 }, /* 0x0F (00001111) */
    //     { 5, 0, 0, 0, 0, 0, 0, 0 }, /* 0x10 (00010000) */
    //     { 1, 5, 0, 0, 0, 0, 0, 0 }, /* 0x11 (00010001) */
    //     { 2, 5, 0, 0, 0, 0, 0, 0 }, /* 0x12 (00010010) */
    //     { 1, 2, 5, 0, 0, 0, 0, 0 }, /* 0x13 (00010011) */
    //     { 3, 5, 0, 0, 0, 0, 0, 0 }, /* 0x14 (00010100) */
    //     { 1, 3, 5, 0, 0, 0, 0, 0 }, /* 0x15 (00010101) */
    //     { 2, 3, 5, 0, 0, 0, 0, 0 }, /* 0x16 (00010110) */
    //     { 1, 2, 3, 5, 0, 0, 0, 0 }, /* 0x17 (00010111) */
    //     { 4, 5, 0, 0, 0, 0, 0, 0 }, /* 0x18 (00011000) */
    //     { 1, 4, 5, 0, 0, 0, 0, 0 }, /* 0x19 (00011001) */
    //     { 2, 4, 5, 0, 0, 0, 0, 0 }, /* 0x1A (00011010) */
    //     { 1, 2, 4, 5, 0, 0, 0, 0 }, /* 0x1B (00011011) */
    //     { 3, 4, 5, 0, 0, 0, 0, 0 }, /* 0x1C (00011100) */
    //     { 1, 3, 4, 5, 0, 0, 0, 0 }, /* 0x1D (00011101) */
    //     { 2, 3, 4, 5, 0, 0, 0, 0 }, /* 0x1E (00011110) */
    //     { 1, 2, 3, 4, 5, 0, 0, 0 }, /* 0x1F (00011111) */
    //     { 6, 0, 0, 0, 0, 0, 0, 0 }, /* 0x20 (00100000) */
    //     { 1, 6, 0, 0, 0, 0, 0, 0 }, /* 0x21 (00100001) */
    //     { 2, 6, 0, 0, 0, 0, 0, 0 }, /* 0x22 (00100010) */
    //     { 1, 2, 6, 0, 0, 0, 0, 0 }, /* 0x23 (00100011) */
    //     { 3, 6, 0, 0, 0, 0, 0, 0 }, /* 0x24 (00100100) */
    //     { 1, 3, 6, 0, 0, 0, 0, 0 }, /* 0x25 (00100101) */
    //     { 2, 3, 6, 0, 0, 0, 0, 0 }, /* 0x26 (00100110) */
    //     { 1, 2, 3, 6, 0, 0, 0, 0 }, /* 0x27 (00100111) */
    //     { 4, 6, 0, 0, 0, 0, 0, 0 }, /* 0x28 (00101000) */
    //     { 1, 4, 6, 0, 0, 0, 0, 0 }, /* 0x29 (00101001) */
    //     { 2, 4, 6, 0, 0, 0, 0, 0 }, /* 0x2A (00101010) */
    //     { 1, 2, 4, 6, 0, 0, 0, 0 }, /* 0x2B (00101011) */
    //     { 3, 4, 6, 0, 0, 0, 0, 0 }, /* 0x2C (00101100) */
    //     { 1, 3, 4, 6, 0, 0, 0, 0 }, /* 0x2D (00101101) */
    //     { 2, 3, 4, 6, 0, 0, 0, 0 }, /* 0x2E (00101110) */
    //     { 1, 2, 3, 4, 6, 0, 0, 0 }, /* 0x2F (00101111) */
    //     { 5, 6, 0, 0, 0, 0, 0, 0 }, /* 0x30 (00110000) */
    //     { 1, 5, 6, 0, 0, 0, 0, 0 }, /* 0x31 (00110001) */
    //     { 2, 5, 6, 0, 0, 0, 0, 0 }, /* 0x32 (00110010) */
    //     { 1, 2, 5, 6, 0, 0, 0, 0 }, /* 0x33 (00110011) */
    //     { 3, 5, 6, 0, 0, 0, 0, 0 }, /* 0x34 (00110100) */
    //     { 1, 3, 5, 6, 0, 0, 0, 0 }, /* 0x35 (00110101) */
    //     { 2, 3, 5, 6, 0, 0, 0, 0 }, /* 0x36 (00110110) */
    //     { 1, 2, 3, 5, 6, 0, 0, 0 }, /* 0x37 (00110111) */
    //     { 4, 5, 6, 0, 0, 0, 0, 0 }, /* 0x38 (00111000) */
    //     { 1, 4, 5, 6, 0, 0, 0, 0 }, /* 0x39 (00111001) */
    //     { 2, 4, 5, 6, 0, 0, 0, 0 }, /* 0x3A (00111010) */
    //     { 1, 2, 4, 5, 6, 0, 0, 0 }, /* 0x3B (00111011) */
    //     { 3, 4, 5, 6, 0, 0, 0, 0 }, /* 0x3C (00111100) */
    //     { 1, 3, 4, 5, 6, 0, 0, 0 }, /* 0x3D (00111101) */
    //     { 2, 3, 4, 5, 6, 0, 0, 0 }, /* 0x3E (00111110) */
    //     { 1, 2, 3, 4, 5, 6, 0, 0 }, /* 0x3F (00111111) */
    //     { 7, 0, 0, 0, 0, 0, 0, 0 }, /* 0x40 (01000000) */
    //     { 1, 7, 0, 0, 0, 0, 0, 0 }, /* 0x41 (01000001) */
    //     { 2, 7, 0, 0, 0, 0, 0, 0 }, /* 0x42 (01000010) */
    //     { 1, 2, 7, 0, 0, 0, 0, 0 }, /* 0x43 (01000011) */
    //     { 3, 7, 0, 0, 0, 0, 0, 0 }, /* 0x44 (01000100) */
    //     { 1, 3, 7, 0, 0, 0, 0, 0 }, /* 0x45 (01000101) */
    //     { 2, 3, 7, 0, 0, 0, 0, 0 }, /* 0x46 (01000110) */
    //     { 1, 2, 3, 7, 0, 0, 0, 0 }, /* 0x47 (01000111) */
    //     { 4, 7, 0, 0, 0, 0, 0, 0 }, /* 0x48 (01001000) */
    //     { 1, 4, 7, 0, 0, 0, 0, 0 }, /* 0x49 (01001001) */
    //     { 2, 4, 7, 0, 0, 0, 0, 0 }, /* 0x4A (01001010) */
    //     { 1, 2, 4, 7, 0, 0, 0, 0 }, /* 0x4B (01001011) */
    //     { 3, 4, 7, 0, 0, 0, 0, 0 }, /* 0x4C (01001100) */
    //     { 1, 3, 4, 7, 0, 0, 0, 0 }, /* 0x4D (01001101) */
    //     { 2, 3, 4, 7, 0, 0, 0, 0 }, /* 0x4E (01001110) */
    //     { 1, 2, 3, 4, 7, 0, 0, 0 }, /* 0x4F (01001111) */
    //     { 5, 7, 0, 0, 0, 0, 0, 0 }, /* 0x50 (01010000) */
    //     { 1, 5, 7, 0, 0, 0, 0, 0 }, /* 0x51 (01010001) */
    //     { 2, 5, 7, 0, 0, 0, 0, 0 }, /* 0x52 (01010010) */
    //     { 1, 2, 5, 7, 0, 0, 0, 0 }, /* 0x53 (01010011) */
    //     { 3, 5, 7, 0, 0, 0, 0, 0 }, /* 0x54 (01010100) */
    //     { 1, 3, 5, 7, 0, 0, 0, 0 }, /* 0x55 (01010101) */
    //     { 2, 3, 5, 7, 0, 0, 0, 0 }, /* 0x56 (01010110) */
    //     { 1, 2, 3, 5, 7, 0, 0, 0 }, /* 0x57 (01010111) */
    //     { 4, 5, 7, 0, 0, 0, 0, 0 }, /* 0x58 (01011000) */
    //     { 1, 4, 5, 7, 0, 0, 0, 0 }, /* 0x59 (01011001) */
    //     { 2, 4, 5, 7, 0, 0, 0, 0 }, /* 0x5A (01011010) */
    //     { 1, 2, 4, 5, 7, 0, 0, 0 }, /* 0x5B (01011011) */
    //     { 3, 4, 5, 7, 0, 0, 0, 0 }, /* 0x5C (01011100) */
    //     { 1, 3, 4, 5, 7, 0, 0, 0 }, /* 0x5D (01011101) */
    //     { 2, 3, 4, 5, 7, 0, 0, 0 }, /* 0x5E (01011110) */
    //     { 1, 2, 3, 4, 5, 7, 0, 0 }, /* 0x5F (01011111) */
    //     { 6, 7, 0, 0, 0, 0, 0, 0 }, /* 0x60 (01100000) */
    //     { 1, 6, 7, 0, 0, 0, 0, 0 }, /* 0x61 (01100001) */
    //     { 2, 6, 7, 0, 0, 0, 0, 0 }, /* 0x62 (01100010) */
    //     { 1, 2, 6, 7, 0, 0, 0, 0 }, /* 0x63 (01100011) */
    //     { 3, 6, 7, 0, 0, 0, 0, 0 }, /* 0x64 (01100100) */
    //     { 1, 3, 6, 7, 0, 0, 0, 0 }, /* 0x65 (01100101) */
    //     { 2, 3, 6, 7, 0, 0, 0, 0 }, /* 0x66 (01100110) */
    //     { 1, 2, 3, 6, 7, 0, 0, 0 }, /* 0x67 (01100111) */
    //     { 4, 6, 7, 0, 0, 0, 0, 0 }, /* 0x68 (01101000) */
    //     { 1, 4, 6, 7, 0, 0, 0, 0 }, /* 0x69 (01101001) */
    //     { 2, 4, 6, 7, 0, 0, 0, 0 }, /* 0x6A (01101010) */
    //     { 1, 2, 4, 6, 7, 0, 0, 0 }, /* 0x6B (01101011) */
    //     { 3, 4, 6, 7, 0, 0, 0, 0 }, /* 0x6C (01101100) */
    //     { 1, 3, 4, 6, 7, 0, 0, 0 }, /* 0x6D (01101101) */
    //     { 2, 3, 4, 6, 7, 0, 0, 0 }, /* 0x6E (01101110) */
    //     { 1, 2, 3, 4, 6, 7, 0, 0 }, /* 0x6F (01101111) */
    //     { 5, 6, 7, 0, 0, 0, 0, 0 }, /* 0x70 (01110000) */
    //     { 1, 5, 6, 7, 0, 0, 0, 0 }, /* 0x71 (01110001) */
    //     { 2, 5, 6, 7, 0, 0, 0, 0 }, /* 0x72 (01110010) */
    //     { 1, 2, 5, 6, 7, 0, 0, 0 }, /* 0x73 (01110011) */
    //     { 3, 5, 6, 7, 0, 0, 0, 0 }, /* 0x74 (01110100) */
    //     { 1, 3, 5, 6, 7, 0, 0, 0 }, /* 0x75 (01110101) */
    //     { 2, 3, 5, 6, 7, 0, 0, 0 }, /* 0x76 (01110110) */
    //     { 1, 2, 3, 5, 6, 7, 0, 0 }, /* 0x77 (01110111) */
    //     { 4, 5, 6, 7, 0, 0, 0, 0 }, /* 0x78 (01111000) */
    //     { 1, 4, 5, 6, 7, 0, 0, 0 }, /* 0x79 (01111001) */
    //     { 2, 4, 5, 6, 7, 0, 0, 0 }, /* 0x7A (01111010) */
    //     { 1, 2, 4, 5, 6, 7, 0, 0 }, /* 0x7B (01111011) */
    //     { 3, 4, 5, 6, 7, 0, 0, 0 }, /* 0x7C (01111100) */
    //     { 1, 3, 4, 5, 6, 7, 0, 0 }, /* 0x7D (01111101) */
    //     { 2, 3, 4, 5, 6, 7, 0, 0 }, /* 0x7E (01111110) */
    //     { 1, 2, 3, 4, 5, 6, 7, 0 }, /* 0x7F (01111111) */
    //     { 8, 0, 0, 0, 0, 0, 0, 0 }, /* 0x80 (10000000) */
    //     { 1, 8, 0, 0, 0, 0, 0, 0 }, /* 0x81 (10000001) */
    //     { 2, 8, 0, 0, 0, 0, 0, 0 }, /* 0x82 (10000010) */
    //     { 1, 2, 8, 0, 0, 0, 0, 0 }, /* 0x83 (10000011) */
    //     { 3, 8, 0, 0, 0, 0, 0, 0 }, /* 0x84 (10000100) */
    //     { 1, 3, 8, 0, 0, 0, 0, 0 }, /* 0x85 (10000101) */
    //     { 2, 3, 8, 0, 0, 0, 0, 0 }, /* 0x86 (10000110) */
    //     { 1, 2, 3, 8, 0, 0, 0, 0 }, /* 0x87 (10000111) */
    //     { 4, 8, 0, 0, 0, 0, 0, 0 }, /* 0x88 (10001000) */
    //     { 1, 4, 8, 0, 0, 0, 0, 0 }, /* 0x89 (10001001) */
    //     { 2, 4, 8, 0, 0, 0, 0, 0 }, /* 0x8A (10001010) */
    //     { 1, 2, 4, 8, 0, 0, 0, 0 }, /* 0x8B (10001011) */
    //     { 3, 4, 8, 0, 0, 0, 0, 0 }, /* 0x8C (10001100) */
    //     { 1, 3, 4, 8, 0, 0, 0, 0 }, /* 0x8D (10001101) */
    //     { 2, 3, 4, 8, 0, 0, 0, 0 }, /* 0x8E (10001110) */
    //     { 1, 2, 3, 4, 8, 0, 0, 0 }, /* 0x8F (10001111) */
    //     { 5, 8, 0, 0, 0, 0, 0, 0 }, /* 0x90 (10010000) */
    //     { 1, 5, 8, 0, 0, 0, 0, 0 }, /* 0x91 (10010001) */
    //     { 2, 5, 8, 0, 0, 0, 0, 0 }, /* 0x92 (10010010) */
    //     { 1, 2, 5, 8, 0, 0, 0, 0 }, /* 0x93 (10010011) */
    //     { 3, 5, 8, 0, 0, 0, 0, 0 }, /* 0x94 (10010100) */
    //     { 1, 3, 5, 8, 0, 0, 0, 0 }, /* 0x95 (10010101) */
    //     { 2, 3, 5, 8, 0, 0, 0, 0 }, /* 0x96 (10010110) */
    //     { 1, 2, 3, 5, 8, 0, 0, 0 }, /* 0x97 (10010111) */
    //     { 4, 5, 8, 0, 0, 0, 0, 0 }, /* 0x98 (10011000) */
    //     { 1, 4, 5, 8, 0, 0, 0, 0 }, /* 0x99 (10011001) */
    //     { 2, 4, 5, 8, 0, 0, 0, 0 }, /* 0x9A (10011010) */
    //     { 1, 2, 4, 5, 8, 0, 0, 0 }, /* 0x9B (10011011) */
    //     { 3, 4, 5, 8, 0, 0, 0, 0 }, /* 0x9C (10011100) */
    //     { 1, 3, 4, 5, 8, 0, 0, 0 }, /* 0x9D (10011101) */
    //     { 2, 3, 4, 5, 8, 0, 0, 0 }, /* 0x9E (10011110) */
    //     { 1, 2, 3, 4, 5, 8, 0, 0 }, /* 0x9F (10011111) */
    //     { 6, 8, 0, 0, 0, 0, 0, 0 }, /* 0xA0 (10100000) */
    //     { 1, 6, 8, 0, 0, 0, 0, 0 }, /* 0xA1 (10100001) */
    //     { 2, 6, 8, 0, 0, 0, 0, 0 }, /* 0xA2 (10100010) */
    //     { 1, 2, 6, 8, 0, 0, 0, 0 }, /* 0xA3 (10100011) */
    //     { 3, 6, 8, 0, 0, 0, 0, 0 }, /* 0xA4 (10100100) */
    //     { 1, 3, 6, 8, 0, 0, 0, 0 }, /* 0xA5 (10100101) */
    //     { 2, 3, 6, 8, 0, 0, 0, 0 }, /* 0xA6 (10100110) */
    //     { 1, 2, 3, 6, 8, 0, 0, 0 }, /* 0xA7 (10100111) */
    //     { 4, 6, 8, 0, 0, 0, 0, 0 }, /* 0xA8 (10101000) */
    //     { 1, 4, 6, 8, 0, 0, 0, 0 }, /* 0xA9 (10101001) */
    //     { 2, 4, 6, 8, 0, 0, 0, 0 }, /* 0xAA (10101010) */
    //     { 1, 2, 4, 6, 8, 0, 0, 0 }, /* 0xAB (10101011) */
    //     { 3, 4, 6, 8, 0, 0, 0, 0 }, /* 0xAC (10101100) */
    //     { 1, 3, 4, 6, 8, 0, 0, 0 }, /* 0xAD (10101101) */
    //     { 2, 3, 4, 6, 8, 0, 0, 0 }, /* 0xAE (10101110) */
    //     { 1, 2, 3, 4, 6, 8, 0, 0 }, /* 0xAF (10101111) */
    //     { 5, 6, 8, 0, 0, 0, 0, 0 }, /* 0xB0 (10110000) */
    //     { 1, 5, 6, 8, 0, 0, 0, 0 }, /* 0xB1 (10110001) */
    //     { 2, 5, 6, 8, 0, 0, 0, 0 }, /* 0xB2 (10110010) */
    //     { 1, 2, 5, 6, 8, 0, 0, 0 }, /* 0xB3 (10110011) */
    //     { 3, 5, 6, 8, 0, 0, 0, 0 }, /* 0xB4 (10110100) */
    //     { 1, 3, 5, 6, 8, 0, 0, 0 }, /* 0xB5 (10110101) */
    //     { 2, 3, 5, 6, 8, 0, 0, 0 }, /* 0xB6 (10110110) */
    //     { 1, 2, 3, 5, 6, 8, 0, 0 }, /* 0xB7 (10110111) */
    //     { 4, 5, 6, 8, 0, 0, 0, 0 }, /* 0xB8 (10111000) */
    //     { 1, 4, 5, 6, 8, 0, 0, 0 }, /* 0xB9 (10111001) */
    //     { 2, 4, 5, 6, 8, 0, 0, 0 }, /* 0xBA (10111010) */
    //     { 1, 2, 4, 5, 6, 8, 0, 0 }, /* 0xBB (10111011) */
    //     { 3, 4, 5, 6, 8, 0, 0, 0 }, /* 0xBC (10111100) */
    //     { 1, 3, 4, 5, 6, 8, 0, 0 }, /* 0xBD (10111101) */
    //     { 2, 3, 4, 5, 6, 8, 0, 0 }, /* 0xBE (10111110) */
    //     { 1, 2, 3, 4, 5, 6, 8, 0 }, /* 0xBF (10111111) */
    //     { 7, 8, 0, 0, 0, 0, 0, 0 }, /* 0xC0 (11000000) */
    //     { 1, 7, 8, 0, 0, 0, 0, 0 }, /* 0xC1 (11000001) */
    //     { 2, 7, 8, 0, 0, 0, 0, 0 }, /* 0xC2 (11000010) */
    //     { 1, 2, 7, 8, 0, 0, 0, 0 }, /* 0xC3 (11000011) */
    //     { 3, 7, 8, 0, 0, 0, 0, 0 }, /* 0xC4 (11000100) */
    //     { 1, 3, 7, 8, 0, 0, 0, 0 }, /* 0xC5 (11000101) */
    //     { 2, 3, 7, 8, 0, 0, 0, 0 }, /* 0xC6 (11000110) */
    //     { 1, 2, 3, 7, 8, 0, 0, 0 }, /* 0xC7 (11000111) */
    //     { 4, 7, 8, 0, 0, 0, 0, 0 }, /* 0xC8 (11001000) */
    //     { 1, 4, 7, 8, 0, 0, 0, 0 }, /* 0xC9 (11001001) */
    //     { 2, 4, 7, 8, 0, 0, 0, 0 }, /* 0xCA (11001010) */
    //     { 1, 2, 4, 7, 8, 0, 0, 0 }, /* 0xCB (11001011) */
    //     { 3, 4, 7, 8, 0, 0, 0, 0 }, /* 0xCC (11001100) */
    //     { 1, 3, 4, 7, 8, 0, 0, 0 }, /* 0xCD (11001101) */
    //     { 2, 3, 4, 7, 8, 0, 0, 0 }, /* 0xCE (11001110) */
    //     { 1, 2, 3, 4, 7, 8, 0, 0 }, /* 0xCF (11001111) */
    //     { 5, 7, 8, 0, 0, 0, 0, 0 }, /* 0xD0 (11010000) */
    //     { 1, 5, 7, 8, 0, 0, 0, 0 }, /* 0xD1 (11010001) */
    //     { 2, 5, 7, 8, 0, 0, 0, 0 }, /* 0xD2 (11010010) */
    //     { 1, 2, 5, 7, 8, 0, 0, 0 }, /* 0xD3 (11010011) */
    //     { 3, 5, 7, 8, 0, 0, 0, 0 }, /* 0xD4 (11010100) */
    //     { 1, 3, 5, 7, 8, 0, 0, 0 }, /* 0xD5 (11010101) */
    //     { 2, 3, 5, 7, 8, 0, 0, 0 }, /* 0xD6 (11010110) */
    //     { 1, 2, 3, 5, 7, 8, 0, 0 }, /* 0xD7 (11010111) */
    //     { 4, 5, 7, 8, 0, 0, 0, 0 }, /* 0xD8 (11011000) */
    //     { 1, 4, 5, 7, 8, 0, 0, 0 }, /* 0xD9 (11011001) */
    //     { 2, 4, 5, 7, 8, 0, 0, 0 }, /* 0xDA (11011010) */
    //     { 1, 2, 4, 5, 7, 8, 0, 0 }, /* 0xDB (11011011) */
    //     { 3, 4, 5, 7, 8, 0, 0, 0 }, /* 0xDC (11011100) */
    //     { 1, 3, 4, 5, 7, 8, 0, 0 }, /* 0xDD (11011101) */
    //     { 2, 3, 4, 5, 7, 8, 0, 0 }, /* 0xDE (11011110) */
    //     { 1, 2, 3, 4, 5, 7, 8, 0 }, /* 0xDF (11011111) */
    //     { 6, 7, 8, 0, 0, 0, 0, 0 }, /* 0xE0 (11100000) */
    //     { 1, 6, 7, 8, 0, 0, 0, 0 }, /* 0xE1 (11100001) */
    //     { 2, 6, 7, 8, 0, 0, 0, 0 }, /* 0xE2 (11100010) */
    //     { 1, 2, 6, 7, 8, 0, 0, 0 }, /* 0xE3 (11100011) */
    //     { 3, 6, 7, 8, 0, 0, 0, 0 }, /* 0xE4 (11100100) */
    //     { 1, 3, 6, 7, 8, 0, 0, 0 }, /* 0xE5 (11100101) */
    //     { 2, 3, 6, 7, 8, 0, 0, 0 }, /* 0xE6 (11100110) */
    //     { 1, 2, 3, 6, 7, 8, 0, 0 }, /* 0xE7 (11100111) */
    //     { 4, 6, 7, 8, 0, 0, 0, 0 }, /* 0xE8 (11101000) */
    //     { 1, 4, 6, 7, 8, 0, 0, 0 }, /* 0xE9 (11101001) */
    //     { 2, 4, 6, 7, 8, 0, 0, 0 }, /* 0xEA (11101010) */
    //     { 1, 2, 4, 6, 7, 8, 0, 0 }, /* 0xEB (11101011) */
    //     { 3, 4, 6, 7, 8, 0, 0, 0 }, /* 0xEC (11101100) */
    //     { 1, 3, 4, 6, 7, 8, 0, 0 }, /* 0xED (11101101) */
    //     { 2, 3, 4, 6, 7, 8, 0, 0 }, /* 0xEE (11101110) */
    //     { 1, 2, 3, 4, 6, 7, 8, 0 }, /* 0xEF (11101111) */
    //     { 5, 6, 7, 8, 0, 0, 0, 0 }, /* 0xF0 (11110000) */
    //     { 1, 5, 6, 7, 8, 0, 0, 0 }, /* 0xF1 (11110001) */
    //     { 2, 5, 6, 7, 8, 0, 0, 0 }, /* 0xF2 (11110010) */
    //     { 1, 2, 5, 6, 7, 8, 0, 0 }, /* 0xF3 (11110011) */
    //     { 3, 5, 6, 7, 8, 0, 0, 0 }, /* 0xF4 (11110100) */
    //     { 1, 3, 5, 6, 7, 8, 0, 0 }, /* 0xF5 (11110101) */
    //     { 2, 3, 5, 6, 7, 8, 0, 0 }, /* 0xF6 (11110110) */
    //     { 1, 2, 3, 5, 6, 7, 8, 0 }, /* 0xF7 (11110111) */
    //     { 4, 5, 6, 7, 8, 0, 0, 0 }, /* 0xF8 (11111000) */
    //     { 1, 4, 5, 6, 7, 8, 0, 0 }, /* 0xF9 (11111001) */
    //     { 2, 4, 5, 6, 7, 8, 0, 0 }, /* 0xFA (11111010) */
    //     { 1, 2, 4, 5, 6, 7, 8, 0 }, /* 0xFB (11111011) */
    //     { 3, 4, 5, 6, 7, 8, 0, 0 }, /* 0xFC (11111100) */
    //     { 1, 3, 4, 5, 6, 7, 8, 0 }, /* 0xFD (11111101) */
    //     { 2, 3, 4, 5, 6, 7, 8, 0 }, /* 0xFE (11111110) */
    //     { 1, 2, 3, 4, 5, 6, 7, 8 }  /* 0xFF (11111111) */
    // };
}
