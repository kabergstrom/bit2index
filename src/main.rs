#![feature(test)]
#![feature(const_fn_union)]
#![feature(stdsimd)]

extern crate test;
use aligned::{Aligned, A16};
use std::arch::x86_64::*;
fn main() {
    println!("Hello, world!");
    let mut out = Vec::new();
    out.resize(64, 0);
    unsafe {
        // bitmap_decode_sse2(&[0xFFFFFFFFFFFFFFFFu64], &mut out);
        // let values = [0xABCDEFu64];
        let values = [0xFFFFFFFFFFFFFFFFu64];
        // let num_values = bitmap_decode_sse2(&values, &mut out);
        // let num_values = bitmap_decode_supernaive(&values, &mut out);
        // let num_values = bitmap_decode_naive(&values, &mut out);
        let num_values = bitmap_decode_ssse3(&values, &mut out);
        // let num_values = bitmap_decode_naive(&values, &mut out);
        // out.resize(num_values as usize, 0);
        println!("{:b} -- {:?}", values[0], out);
        // // bitmap_decode_sse2(&[0x0000000000000000u64], &mut out);
        // // let (x, y, z, w) = bits_to_bytes(84, 0x0F);
        // use std::mem::transmute;
        // let packed = pack_left_sse2(
        //     0x0101,
        //     _mm_set_ps(
        //         transmute(0xFF00FF),
        //         transmute(0xFF00),
        //         transmute(0x0F0F),
        //         transmute(0xF0F0),
        //     ),
        // );
        // print_bytes("result", _mm_castps_si128(packed));
    }
}

unsafe fn print_bytes(prefix: &str, x: __m128i) {
    println!(
        "{} x: {:x} y: {:x}, z: {:x}, w: {:x} ",
        prefix,
        _mm_extract_epi32(x, 0),
        _mm_extract_epi32(x, 1),
        _mm_extract_epi32(x, 2),
        _mm_extract_epi32(x, 3)
    );
}

unsafe fn print_bytes_u8(prefix: &str, x: __m128i) {
    println!(
        "{} x: {:x} y: {:x}, z: {:x}, w: {:x} a: {:x} b: {:x}, c: {:x}, d: {:x} e: {:x} f: {:x}, g: {:x}, h: {:x} i: {:x} j: {:x}, k: {:x}, l: {:x}",
        prefix,
        _mm_extract_epi8(x, 0),
        _mm_extract_epi8(x, 1),
        _mm_extract_epi8(x, 2),
        _mm_extract_epi8(x, 3),
        _mm_extract_epi8(x, 4),
        _mm_extract_epi8(x, 5),
        _mm_extract_epi8(x, 6),
        _mm_extract_epi8(x, 7),
        _mm_extract_epi8(x, 8),
        _mm_extract_epi8(x, 9),
        _mm_extract_epi8(x, 10),
        _mm_extract_epi8(x, 11),
        _mm_extract_epi8(x, 12),
        _mm_extract_epi8(x, 13),
        _mm_extract_epi8(x, 14),
        _mm_extract_epi8(x, 15),
    );
}

const unsafe fn create_ctrl_mask(x: u32, y: u32, z: u32, w: u32) -> [f32; 4] {
    union Transmute<T: Copy, U: Copy> {
        from: T,
        to: U,
    }
    [
        Transmute {
            from: x.swap_bytes(),
        }
        .to,
        Transmute {
            from: y.swap_bytes(),
        }
        .to,
        Transmute {
            from: z.swap_bytes(),
        }
        .to,
        Transmute {
            from: w.swap_bytes(),
        }
        .to,
    ]
}

const unsafe fn create_mask(x: u32, y: u32, z: u32, w: u32) -> [f32; 4] {
    union Transmute<T: Copy, U: Copy> {
        from: T,
        to: U,
    }
    [
        Transmute {
            from: x * std::u32::MAX,
        }
        .to,
        Transmute {
            from: y * std::u32::MAX,
        }
        .to,
        Transmute {
            from: z * std::u32::MAX,
        }
        .to,
        Transmute {
            from: w * std::u32::MAX,
        }
        .to,
    ]
}

const PACK_LEFT_MASKS: Aligned<A16, [[[f32; 4]; 2]; 16]> = unsafe {
    Aligned([
        //0000 .... 0 0 0 0
        [create_mask(0, 0, 0, 0), create_mask(0, 0, 0, 0)],
        // 0001 X... 0 0 0 0
        [create_mask(0, 0, 0, 0), create_mask(0, 0, 0, 0)],
        // 0010 Y... 0 1 0 0
        [create_mask(1, 0, 0, 0), create_mask(0, 0, 0, 0)],
        // 0011 XY.. 0 0 0 0
        [create_mask(0, 0, 0, 0), create_mask(0, 0, 0, 0)],
        // 0100 Z... 0 0 2 0
        [create_mask(0, 0, 0, 0), create_mask(1, 0, 0, 0)],
        // 0101 XZ.. 0 0 1 0
        [create_mask(0, 1, 0, 0), create_mask(0, 0, 0, 0)],
        // 0110 YZ.. 0 1 1 0
        [create_mask(1, 1, 0, 0), create_mask(0, 0, 0, 0)],
        // 0111 XYZ. 0 0 0 0
        [create_mask(0, 0, 0, 0), create_mask(0, 0, 0, 0)],
        // 1000 W... 0 0 0 3
        [create_mask(0, 0, 0, 0), create_mask(1, 0, 0, 0)],
        // 1001 XW.. 0 0 0 2
        [create_mask(0, 0, 0, 0), create_mask(0, 1, 0, 0)],
        // 1010 YW.. 0 1 0 2
        [create_mask(1, 0, 0, 0), create_mask(0, 1, 0, 0)],
        // 1011 XYW. 0 0 0 1
        [create_mask(0, 0, 1, 0), create_mask(0, 0, 0, 0)],
        // 1100 ZW.. 0 0 2 2
        [create_mask(0, 0, 0, 0), create_mask(1, 1, 0, 0)],
        // 1101 XZW. 0 0 1 1
        [create_mask(0, 1, 1, 0), create_mask(0, 0, 0, 0)],
        // 1110 YZW. 0 1 1 1
        [create_mask(1, 1, 1, 0), create_mask(0, 0, 0, 0)],
        // 1111 XYZW 0 0 0 0
        [create_mask(0, 0, 0, 0), create_mask(0, 0, 0, 0)],
    ])
};

unsafe fn pack_left_sse2(valid: i32, val: __m128) -> __m128 {
    let mask0: __m128 = _mm_load_ps(&PACK_LEFT_MASKS[valid as usize][0] as *const f32);
    let mask1: __m128 = _mm_load_ps(&PACK_LEFT_MASKS[valid as usize][1] as *const f32);
    let s0: __m128 = _mm_shuffle_ps(val, val, _MM_SHUFFLE(0, 3, 2, 1) as u32);
    let r0: __m128 = _mm_or_ps(_mm_and_ps(mask0, s0), _mm_andnot_ps(mask0, val));
    let s1: __m128 = _mm_shuffle_ps(r0, r0, _MM_SHUFFLE(1, 0, 3, 2) as u32);
    let r1: __m128 = _mm_or_ps(_mm_and_ps(mask1, s1), _mm_andnot_ps(mask1, r0));
    r1
}
const PACK_LEFT_MASKS_SSSE3: Aligned<A16, [[f32; 4]; 16]> = unsafe {
    Aligned([
        //0000 .... 0 0 0 0
        create_ctrl_mask(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F),
        // 0001 X... 0 0 0 0
        create_ctrl_mask(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F),
        // 0010 Y... 0 1 0 0
        create_ctrl_mask(0x04050607, 0x80808080, 0x80808080, 0x80808080),
        // 0011 XY.. 0 0 0 0
        create_ctrl_mask(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F),
        // 0100 Z... 0 0 2 0
        create_ctrl_mask(0x08090A0B, 0x80808080, 0x80808080, 0x80808080),
        // 0101 XZ.. 0 0 1 0
        create_ctrl_mask(0x00010203, 0x08090A0B, 0x80808080, 0x80808080),
        // 0110 YZ.. 0 1 1 0
        create_ctrl_mask(0x04050607, 0x08090A0B, 0x80808080, 0x80808080),
        // 0111 XYZ. 0 0 0 0
        create_ctrl_mask(0x00010203, 0x04050607, 0x08090A0B, 0x80808080),
        // 1000 W... 0 0 0 3
        create_ctrl_mask(0x0C0D0E0F, 0x80808080, 0x80808080, 0x80808080),
        // 1001 XW.. 0 0 0 2
        create_ctrl_mask(0x00010203, 0x0C0D0E0F, 0x80808080, 0x80808080),
        // 1010 YW.. 0 1 0 2
        create_ctrl_mask(0x04050607, 0x0C0D0E0F, 0x80808080, 0x80808080),
        // 1011 XYW. 0 0 0 1
        create_ctrl_mask(0x00010203, 0x04050607, 0x0C0D0E0F, 0x80808080),
        // 1100 ZW.. 0 0 2 2
        create_ctrl_mask(0x08090A0B, 0x0C0D0E0F, 0x80808080, 0x80808080),
        // 1101 XZW. 0 0 1 1
        create_ctrl_mask(0x00010203, 0x08090A0B, 0x0C0D0E0F, 0x80808080),
        // 1110 YZW. 0 1 1 1
        create_ctrl_mask(0x04050607, 0x08090A0B, 0x0C0D0E0F, 0x80808080),
        // 1111 XYZW 0 0 0 0
        create_ctrl_mask(0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F),
    ])
};

#[inline]
#[cfg(target_feature = "ssse3")]
unsafe fn pack_left_ssse3(mask: i32, val: __m128) -> __m128i
{
	// Select shuffle control data
	let shuf_ctrl: __m128i  = _mm_load_si128(std::mem::transmute(&PACK_LEFT_MASKS_SSSE3[mask as usize]));
	// Permute to move valid values to front of SIMD register
	let packed: __m128i = _mm_shuffle_epi8(_mm_castps_si128(val), shuf_ctrl);
	return packed;
}

#[cfg(target_feature = "ssse3")]
unsafe fn bitmap_decode_ssse3(bitmap: &[u64], out: &mut Vec<u32>) -> usize {
    let mut out_pos = 0;
    let mut base: __m128i = _mm_set1_epi32(0);
    let post_shuffle_mask = _mm_set_epi8(
        -128i8, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128i8, 0x40, 0x20, 0x10, 0x08, 0x04,
        0x02, 0x01,
    );
    let element_indices = _mm_set_epi8(
        0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
        0x0,
    );
    let shuffle_mask_a = _mm_set_epi8(
        -128i8, -128i8, -128i8, 0x03, -128i8, -128i8, -128i8, 0x02, -128i8, -128i8, -128i8, 0x01,
        -128i8, -128i8, -128i8, 0x00,
    );
    let shuffle_mask_b = _mm_set_epi8(
        -128i8, -128i8, -128i8, 0x07, -128i8, -128i8, -128i8, 0x06, -128i8, -128i8, -128i8, 0x05,
        -128i8, -128i8, -128i8, 0x04,
    );
    let shuffle_mask_c = _mm_set_epi8(
        -128i8, -128i8, -128i8, 0x0B, -128i8, -128i8, -128i8, 0x0A, -128i8, -128i8, -128i8, 0x09,
        -128i8, -128i8, -128i8, 0x08,
    );
    let shuffle_mask_d = _mm_set_epi8(
        -128i8, -128i8, -128i8, 0x0F, -128i8, -128i8, -128i8, 0x0E, -128i8, -128i8, -128i8, 0x0D,
        -128i8, -128i8, -128i8, 0x0C,
    );
	let lohi_mask = _mm_set_epi32(
		0,0,-1i32,-1i32
	);
    let zero_mask = _mm_castsi128_ps(_mm_set1_epi32(0));
    // reserve space in the out vec
    out.reserve(bitmap.len() * 64);
    for &bits in bitmap {
		// if bits == 0 {
		// 	continue
		// }
        for i in 0..4 {
            // broadcast the bits to all elements
            let truncated_bits = (bits >> (i * 16)) as i16;
            let high_bits = truncated_bits as i8;
            let low_bits = (truncated_bits >> 8) as i8;
            let mut x: __m128i = _mm_and_si128(_mm_set1_epi8(low_bits), lohi_mask);
			x = _mm_or_si128(x, _mm_andnot_si128(lohi_mask, _mm_set1_epi8(high_bits)));
				
            // print_bytes_u8("mask", post_shuffle_mask);
            // print_bytes_u8("pre-mask", x);

            // save only the relevant bit per 8 bit integer
            x = _mm_and_si128(x, post_shuffle_mask);
            // print_bytes_u8("post-mask", x);
            // fill the whole integer with FF if the relevant bit is set
            let mask = _mm_cmpeq_epi8(x, post_shuffle_mask);
            // cut away the irrelevant bits, leaving the element indices
            x = _mm_and_si128(mask, element_indices);
            // println!("bits {:b}", ((bits >> (i * 16)) & 0xFFFF) as i32);
            // print_bytes_u8("element indices", x);

            let mut a = _mm_shuffle_epi8(x, shuffle_mask_a);
            let mut b = _mm_shuffle_epi8(x, shuffle_mask_b);
            let mut c = _mm_shuffle_epi8(x, shuffle_mask_c);
            let mut d = _mm_shuffle_epi8(x, shuffle_mask_d);
            // print_bytes("post-shuffle a", a);
            // print_bytes("post-shuffle b", b);
            // print_bytes("post-shuffle c", c);
            // print_bytes("post-shuffle d", d);
            // offset by bit index
            a = _mm_add_epi32(base, a);
            b = _mm_add_epi32(base, b);
            c = _mm_add_epi32(base, c);
            d = _mm_add_epi32(base, d);
            // get a move mask from thej element mask
            let move_mask = _mm_movemask_epi8(mask);
            // println!("move mask {:b}", move_mask);
            // pack the elements to the left using the move mask
            let movemask_a = move_mask & 0xF;
            let movemask_b = (move_mask >> 4) & 0xF;
            let movemask_c = (move_mask >> 8) & 0xF;
            let movemask_d = (move_mask >> 12) & 0xF;
            // print_bytes("indices a", a);
            // print_bytes("indices b", b);
            // print_bytes("indices c", c);
            // print_bytes("indices d", d);

            let a_out = pack_left_ssse3(movemask_a, _mm_castsi128_ps(a));
            let b_out = pack_left_ssse3(movemask_b, _mm_castsi128_ps(b));
            let c_out = pack_left_ssse3(movemask_d, _mm_castsi128_ps(c));
            let d_out = pack_left_ssse3(movemask_d, _mm_castsi128_ps(d));
            // println!("movemask a {:b}", movemask_a);
            // println!("movemask b {:b}", movemask_b);
            // println!("movemask c {:b}", movemask_c);
            // println!("movemask d {:b}", movemask_d);
            // print_bytes("leftpacked a", a_out);
            // print_bytes("leftpacked b", b_out);
            // print_bytes("leftpacked c", c_out);
            // print_bytes("leftpacked d", d_out);
            // get the number of elements being output
            let advance_a = _popcnt32(movemask_a) as usize;
            let advance_b = _popcnt32(movemask_b) as usize;
            let advance_c = _popcnt32(movemask_c) as usize;
            let advance_d = _popcnt32(movemask_d) as usize;
            // perform the store
            _mm_storeu_si128(std::mem::transmute(out.get_unchecked(out_pos)), a_out);
            out_pos += advance_a;
            _mm_storeu_si128(std::mem::transmute(out.get_unchecked(out_pos)), b_out);
            out_pos += advance_b;
            _mm_storeu_si128(std::mem::transmute(out.get_unchecked(out_pos)), c_out);
            out_pos += advance_c;
            _mm_storeu_si128(std::mem::transmute(out.get_unchecked(out_pos)), d_out);
            out_pos += advance_d;
            // increase the base
            base = _mm_add_epi32(base, _mm_set1_epi32(16));
            // println!("to_advance {} pos {} base {}", to_advance, out_pos, _mm_extract_epi32(base, 0));
        }
    }
    out.set_len(out_pos);
    out_pos
}

unsafe fn bitmap_decode_sse2(bitmap: &[u64], out: &mut Vec<u32>) -> usize {
    let mut out_pos = 0;
    let mut base: __m128i = _mm_set1_epi32(0);
    let post_shuffle_mask = _mm_set_epi32(0x08, 0x04, 0x02, 0x01);
    let element_indices = _mm_set_epi32(0x03, 0x02, 0x01, 0x0);
    let zero_mask = _mm_castsi128_ps(_mm_set1_epi32(0));
    // reserve space in the out vec
    out.reserve(bitmap.len() * 64);
    for k in 0..bitmap.len() {
        let bits = bitmap[k];
        for i in 0..16 {
            // broadcast the bits to all elements
            let mut x: __m128i = _mm_set1_epi32((bits >> (i * 4)) as i32);
            // save only the relevant bit per 32 bit integer
            x = _mm_and_si128(x, post_shuffle_mask);
            // fill the whole integer with FFFFFFFF if the relevant bit is set
            let mask = _mm_cmpeq_epi32(x, post_shuffle_mask);
            if _mm_comieq_ss(_mm_castsi128_ps(mask), zero_mask) == 1 {
                continue;
            }
            // cut away the irrelevant bits, leaving the element indices
            x = _mm_and_si128(mask, element_indices);
            // offset by bit index
            x = _mm_add_epi32(base, x);
            // get a move mask from the element mask
            let move_mask = _mm_movemask_ps(_mm_castsi128_ps(mask));
            // pack the elements to the left using the move mask
            let result = pack_left_sse2(move_mask, _mm_castsi128_ps(x));
            // get the number of elements being output
            let to_advance = _popcnt32(move_mask) as usize;
            // perform the store
            _mm_storeu_ps(std::mem::transmute(out.get_unchecked(out_pos)), result);
            // increase the base and output index counters
            base = _mm_add_epi32(base, _mm_set1_epi32(4));
            out_pos += to_advance;
            // println!("to_advance {} pos {} base {}", to_advance, out_pos, _mm_extract_epi32(base, 0));
        }
    }
    out.set_len(out_pos);
    out_pos
}

fn bitmap_decode_supernaive(bitmap: &[u64], out: &mut Vec<u32>) -> usize {
    let mut pos = 0;
    out.reserve(bitmap.len() * 64);
    for k in 0..bitmap.len() {
        let bitset = bitmap[k];
        let p = k as u32 * 64;
        for i in 0..64 {
            if (bitset >> i) & 0x1 != 0 {
                unsafe {
                    *out.get_unchecked_mut(pos) = p + i;
                }
                pos += 1;
            }
        }
    }
    unsafe {
        out.set_len(pos);
    }
    pos
}

fn bitmap_decode_naive(bitmap: &[u64], out: &mut Vec<u32>) -> usize {
    let mut pos = 0;
    out.reserve(bitmap.len() * 64);
    for k in 0..bitmap.len() {
        let mut bitset = bitmap[k];
        let mut p = k as u32 * 64;
        while bitset != 0 {
            if (bitset & 0x1) != 0 {
                unsafe {
                    *out.get_unchecked_mut(pos) = p;
                }
                pos += 1;
            }
            bitset >>= 1;
            p += 1;
        }
    }
    return pos;
}

fn bitmap_decode_ctz(bitmap: &[u64], out: &mut Vec<u32>) -> usize {
    let mut pos = 0;
    out.reserve(bitmap.len() * 64);
    for k in 0..bitmap.len() {
        let mut bitset = unsafe { std::mem::transmute::<u64, i64>(bitmap[k]) };
        while bitset != 0 {
            let t: i64 = bitset & -bitset;
            let r = bitset.trailing_zeros();
            unsafe {
                *out.get_unchecked_mut(pos) = k as u32 * 64 + r;
            }
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
        let mut empty_buf = Vec::new();
        empty_buf.resize(8096, 0);
        empty_buf
    }

    fn full_buff() -> Vec<u64> {
        let mut full_buf = Vec::new();
        full_buf.resize(8096, 0xFFFFFFFFFFFFFFFFu64);
        full_buf
    }

    unsafe fn bench_decode(
        b: &mut Bencher,
        bitset: &[u64],
        fun: unsafe fn(&[u64], &mut Vec<u32>) -> usize,
    ) {
        let mut values = Vec::with_capacity(bitset.len() * 64);
        b.iter(|| {
            values.clear();
            fun(&bitset, &mut values);
        })
    }

    // #[bench]
    // fn bench_sse2_empty(b: &mut Bencher) {
    //     unsafe {
    //         bench_decode(b, &empty_buf(), bitmap_decode_sse2);
    //     }
    // }
    // #[bench]
    // fn bench_sse2_full(b: &mut Bencher) {
    //     unsafe {
    //         bench_decode(b, &full_buff(), bitmap_decode_sse2);
    //     }
    // }
    // #[bench]
    // fn bench_sse2_test(b: &mut Bencher) {
    //     unsafe {
    //         bench_decode(b, &VEC_DECODE_TABLE, bitmap_decode_sse2);
    //     }
    // }

    #[bench]
    fn bench_ssse3_empty(b: &mut Bencher) {
        unsafe {
            bench_decode(b, &empty_buf(), bitmap_decode_ssse3); 
        }
    }
    #[bench]
    fn bench_ssse3_full(b: &mut Bencher) {
        unsafe {
            bench_decode(b, &full_buff(), bitmap_decode_ssse3);
        }
    }
    #[bench]
    fn bench_ssse3_test(b: &mut Bencher) {
        unsafe {
            bench_decode(b, &VEC_DECODE_TABLE, bitmap_decode_ssse3);
        }
    }

    #[bench]
    fn bench_naive_empty(b: &mut Bencher) {
        unsafe {
            bench_decode(b, &empty_buf(), bitmap_decode_naive);
        }
    }
    #[bench]
    fn bench_naive_full(b: &mut Bencher) {
        unsafe {
            bench_decode(b, &full_buff(), bitmap_decode_naive);
        }
    }
    #[bench]
    fn bench_naive_test(b: &mut Bencher) {
        unsafe {
            bench_decode(b, &VEC_DECODE_TABLE, bitmap_decode_naive);
        }
    }

    #[bench]
    fn bench_ctz_empty(b: &mut Bencher) {
        unsafe {
            bench_decode(b, &empty_buf(), bitmap_decode_ctz);
        }
    }
    #[bench]
    fn bench_ctz_full(b: &mut Bencher) {
        unsafe {
            bench_decode(b, &full_buff(), bitmap_decode_ctz);
        }
    }
    #[bench]
    fn bench_ctz_test(b: &mut Bencher) {
        unsafe {
            bench_decode(b, &VEC_DECODE_TABLE, bitmap_decode_ctz);
        }
    }
    // static uint8_t lengthTable[256] = {
    //     0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    //     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    //     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    //     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    //     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    //     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    //     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    //     3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    //     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    //     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    //     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    //     3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    //     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    //     3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    //     3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    //     4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
    // };

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
