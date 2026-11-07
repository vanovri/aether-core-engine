use std::fs;
use std::io;
use std::path::Path;

pub const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

// ============================================================================
// IEEE 802.3 CRC-32 & RFC 1950 Adler-32 Checksums (Zero Dependencies)
// ============================================================================

const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            if (c & 1) != 0 {
                c = 0xEDB8_8320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

const CRC32_TABLE: [u32; 256] = build_crc32_table();

#[inline]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        let idx = ((crc ^ (b as u32)) & 0xFF) as usize;
        crc = CRC32_TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

#[inline]
pub fn crc32_chunk(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in chunk_type.iter().chain(data.iter()) {
        let idx = ((crc ^ (b as u32)) & 0xFF) as usize;
        crc = CRC32_TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

#[inline]
pub fn adler32(bytes: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    let mut s1 = 1u32;
    let mut s2 = 0u32;
    for chunk in bytes.chunks(5552) {
        for &b in chunk {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= MOD_ADLER;
        s2 %= MOD_ADLER;
    }
    (s2 << 16) | s1
}

// ============================================================================
// RFC 1951 DEFLATE Tables (Length & Distance)
// ============================================================================

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

struct BitWriter {
    buf: Vec<u8>,
    bit_buf: u64,
    bit_count: u8,
}

impl BitWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    /// Emit `nbits` LSB-first into output byte stream (RFC 1951 data elements).
    #[inline]
    fn write_bits_lsb(&mut self, val: u32, nbits: u8) {
        if nbits == 0 {
            return;
        }
        let mask = if nbits == 32 { u32::MAX } else { (1u32 << nbits) - 1 };
        self.bit_buf |= ((val & mask) as u64) << self.bit_count;
        self.bit_count += nbits;
        while self.bit_count >= 8 {
            self.buf.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf >>= 8;
            self.bit_count -= 8;
        }
    }

    /// Emit `nbits` Huffman codeword MSB-first into output byte stream (RFC 1951 Section 3.1.1).
    #[inline]
    fn write_huffman_msb(&mut self, code: u32, nbits: u8) {
        let mut rev = 0u32;
        for i in 0..nbits {
            if (code & (1u32 << (nbits - 1 - i))) != 0 {
                rev |= 1u32 << i;
            }
        }
        self.write_bits_lsb(rev, nbits);
    }

    #[inline]
    fn emit_fixed_lit_len(&mut self, sym: u16) {
        match sym {
            0..=143 => self.write_huffman_msb(0x30 + (sym as u32), 8),
            144..=255 => self.write_huffman_msb(0x190 + ((sym - 144) as u32), 9),
            256..=279 => self.write_huffman_msb((sym - 256) as u32, 7),
            280..=287 => self.write_huffman_msb(0xC0 + ((sym - 280) as u32), 8),
            _ => unreachable!(),
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.buf.push((self.bit_buf & 0xFF) as u8);
        }
        self.buf
    }
}

#[inline]
fn encode_length_symbol(length: usize) -> (u16, u8, u32) {
    if length == 258 {
        return (285, 0, 0);
    }
    let mut idx = 0usize;
    for i in 0..28 {
        let next_base = LENGTH_BASE[i + 1] as usize;
        if length < next_base {
            idx = i;
            break;
        }
    }
    let base = LENGTH_BASE[idx] as usize;
    let extra = LENGTH_EXTRA[idx];
    ((257 + idx) as u16, extra, (length - base) as u32)
}

#[inline]
fn encode_distance_symbol(dist: usize) -> (u8, u8, u32) {
    let mut idx = 29usize;
    for i in 0..29 {
        if dist < (DIST_BASE[i + 1] as usize) {
            idx = i;
            break;
        }
    }
    let base = DIST_BASE[idx] as usize;
    let extra = DIST_EXTRA[idx];
    (idx as u8, extra, (dist - base) as u32)
}

/// Compress `raw` into an RFC 1950 zlib stream using LZ77 32KB hash-chain matching + RFC 1951 Fixed Huffman codes.
pub fn zlib_compress_lz77(raw: &[u8]) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(raw.len() / 2 + 64);
    // RFC 1950 2-byte zlib header: CMF=0x78 (Deflate, 32K window), FLG=0x9C (default compression, FCHECK valid)
    bw.buf.push(0x78);
    bw.buf.push(0x9C);

    // Emit single final Fixed-Huffman DEFLATE block: BFINAL=1, BTYPE=01 (Fixed Huffman)
    bw.write_bits_lsb(1, 1);
    bw.write_bits_lsb(1, 2);

    const HASH_BITS: usize = 15;
    const HASH_SIZE: usize = 1 << HASH_BITS;
    const WINDOW_SIZE: usize = 32_768;
    const MAX_CHAIN: usize = 24;

    let mut head = vec![usize::MAX; HASH_SIZE];
    let mut prev = vec![usize::MAX; raw.len().max(1)];

    #[inline]
    fn hash3(b0: u8, b1: u8, b2: u8) -> usize {
        let v = ((b0 as u32) << 16) ^ ((b1 as u32) << 8) ^ (b2 as u32);
        (v.wrapping_mul(0x1E35_A7BD) >> (32 - HASH_BITS)) as usize
    }

    let mut pos = 0usize;
    let n = raw.len();
    while pos < n {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;

        if pos + 3 <= n {
            let h = hash3(raw[pos], raw[pos + 1], raw[pos + 2]);
            let mut cand = head[h];
            prev[pos] = cand;
            head[h] = pos;

            let max_len = (n - pos).min(258);
            let mut steps = 0usize;
            while cand != usize::MAX && steps < MAX_CHAIN {
                let dist = pos - cand;
                if dist == 0 || dist > WINDOW_SIZE {
                    break;
                }
                if raw[cand + best_len.min(max_len - 1)] == raw[pos + best_len.min(max_len - 1)]
                    && raw[cand] == raw[pos]
                    && raw[cand + 1] == raw[pos + 1]
                    && raw[cand + 2] == raw[pos + 2]
                {
                    let mut mlen = 3usize;
                    while mlen < max_len && raw[cand + mlen] == raw[pos + mlen] {
                        mlen += 1;
                    }
                    if mlen > best_len {
                        best_len = mlen;
                        best_dist = dist;
                        if best_len == max_len {
                            break;
                        }
                    }
                }
                cand = prev[cand];
                steps += 1;
            }
        }

        if best_len >= 3 {
            let (lsym, lextra, lval) = encode_length_symbol(best_len);
            bw.emit_fixed_lit_len(lsym);
            bw.write_bits_lsb(lval, lextra);

            let (dsym, dextra, dval) = encode_distance_symbol(best_dist);
            bw.write_huffman_msb(dsym as u32, 5);
            bw.write_bits_lsb(dval, dextra);

            // Insert skipped positions into hash table for high match quality
            for step in 1..best_len {
                let p = pos + step;
                if p + 3 <= n {
                    let h = hash3(raw[p], raw[p + 1], raw[p + 2]);
                    prev[p] = head[h];
                    head[h] = p;
                }
            }
            pos += best_len;
        } else {
            bw.emit_fixed_lit_len(raw[pos] as u16);
            pos += 1;
        }
    }

    // End-of-block symbol 256
    bw.emit_fixed_lit_len(256);

    let mut out = bw.finish();
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

/// Uncompressed RFC 1951 Stored-Block (`BTYPE=00`) zlib encoder (useful as reference & fallback).
pub fn zlib_compress_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + (raw.len() / 65535 + 1) * 5 + 6);
    out.push(0x78);
    out.push(0x01); // CMF=0x78, FLG=0x01 (no compression, valid FCHECK: 0x7801 % 31 == 0)

    if raw.is_empty() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xFF, 0xFF]);
    } else {
        let chunks: Vec<&[u8]> = raw.chunks(65_535).collect();
        for (idx, chunk) in chunks.iter().enumerate() {
            let is_final = u8::from(idx + 1 == chunks.len());
            out.push(is_final);
            let len = chunk.len() as u16;
            let nlen = !len;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&nlen.to_le_bytes());
            out.extend_from_slice(chunk);
        }
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

// ============================================================================
// PNG Scanline Paeth Predictor + Full PNG Encoder & Validator
// ============================================================================

#[inline]
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let ai = a as i32;
    let bi = b as i32;
    let ci = c as i32;
    let p = ai + bi - ci;
    let pa = (p - ai).abs();
    let pb = (p - bi).abs();
    let pc = (p - ci).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let crc = crc32_chunk(chunk_type, data);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Encode 8-bit RGB (`width * height * 3` bytes) into a complete PNG byte buffer with adaptive Paeth/Sub filtering + LZ77 DEFLATE.
pub fn encode_png_rgb8(width: u32, height: u32, rgb_bytes: &[u8]) -> Vec<u8> {
    let row_stride = (width as usize) * 3;
    assert_eq!(rgb_bytes.len(), row_stride * (height as usize));

    // Apply adaptive PNG row filtering (comparing Filter 1 Sub vs Filter 4 Paeth by minimum sum of absolute residuals)
    let mut filtered = Vec::with_capacity((row_stride + 1) * (height as usize));
    let zero_row = vec![0u8; row_stride];
    let mut sub_row = vec![0u8; row_stride];
    let mut paeth_row = vec![0u8; row_stride];

    for y in 0..(height as usize) {
        let cur = &rgb_bytes[y * row_stride..(y + 1) * row_stride];
        let prev = if y == 0 {
            &zero_row[..]
        } else {
            &rgb_bytes[(y - 1) * row_stride..y * row_stride]
        };

        let mut score_sub = 0u64;
        let mut score_paeth = 0u64;
        for x in 0..row_stride {
            let a = if x >= 3 { cur[x - 3] } else { 0 };
            let b = prev[x];
            let c = if x >= 3 { prev[x - 3] } else { 0 };

            let ds = cur[x].wrapping_sub(a);
            let dp = cur[x].wrapping_sub(paeth_predictor(a, b, c));
            sub_row[x] = ds;
            paeth_row[x] = dp;
            score_sub += (ds as i8).unsigned_abs() as u64;
            score_paeth += (dp as i8).unsigned_abs() as u64;
        }

        if score_paeth <= score_sub {
            filtered.push(4); // Paeth filter
            filtered.extend_from_slice(&paeth_row);
        } else {
            filtered.push(1); // Sub filter
            filtered.extend_from_slice(&sub_row);
        }
    }

    let compressed = zlib_compress_lz77(&filtered);

    let mut out = Vec::with_capacity(compressed.len() + 128);
    out.extend_from_slice(&PNG_SIGNATURE);

    // IHDR (13 bytes): width, height, bit_depth=8, color_type=2 (Truecolor RGB), comp=0, filter=0, interlace=0
    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8;
    ihdr[9] = 2;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    write_png_chunk(&mut out, b"IHDR", &ihdr);

    // sRGB chunk (rendering intent = 0 Perceptual) + gAMA (45455 = 1/2.2 * 100000)
    write_png_chunk(&mut out, b"sRGB", &[0]);
    write_png_chunk(&mut out, b"gAMA", &45_455u32.to_be_bytes());

    // IDAT chunk(s)
    for idat_part in compressed.chunks(1 << 20) {
        write_png_chunk(&mut out, b"IDAT", idat_part);
    }

    // IEND chunk
    write_png_chunk(&mut out, b"IEND", &[]);
    out
}

pub fn save_png_rgb8(path: &Path, width: u32, height: u32, rgb_bytes: &[u8]) -> io::Result<usize> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let png = encode_png_rgb8(width, height, rgb_bytes);
    fs::write(path, &png)?;
    Ok(png.len())
}

/// Binary P6 `.ppm` encoder (included for completeness alongside PNG).
pub fn encode_ppm_rgb8(width: u32, height: u32, rgb_bytes: &[u8]) -> Vec<u8> {
    let header = format!("P6\n{} {}\n255\n", width, height);
    let mut out = Vec::with_capacity(header.len() + rgb_bytes.len());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(rgb_bytes);
    out
}

/// Uncompressed 24-bit `.bmp` encoder (included for completeness alongside PNG).
pub fn encode_bmp_rgb8(width: u32, height: u32, rgb_bytes: &[u8]) -> Vec<u8> {
    let row_raw = (width as usize) * 3;
    let row_padded = (row_raw + 3) & !3;
    let pixel_bytes = row_padded * (height as usize);
    let file_size = 54 + pixel_bytes;

    let mut out = Vec::with_capacity(file_size);
    // BITMAPFILEHEADER (14 bytes)
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    // BITMAPINFOHEADER (40 bytes)
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(width as i32).to_le_bytes());
    out.extend_from_slice(&(height as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(pixel_bytes as u32).to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());

    let pad = [0u8; 3];
    for y in (0..(height as usize)).rev() {
        let row = &rgb_bytes[y * row_raw..(y + 1) * row_raw];
        for px in row.chunks_exact(3) {
            out.push(px[2]);
            out.push(px[1]);
            out.push(px[0]);
        }
        out.extend_from_slice(&pad[..(row_padded - row_raw)]);
    }
    out
}

// ============================================================================
// RFC 1950 / RFC 1951 / RFC 2083 Decompressor & Binary Verifier (For Self-Tests)
// ============================================================================

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bit(&mut self) -> Result<u32, String> {
        if self.byte_pos >= self.data.len() {
            return Err("Unexpected EOF in DEFLATE bitstream".to_string());
        }
        let bit = ((self.data[self.byte_pos] >> self.bit_pos) & 1) as u32;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit)
    }

    fn read_bits_lsb(&mut self, n: u8) -> Result<u32, String> {
        let mut v = 0u32;
        for i in 0..n {
            v |= self.read_bit()? << i;
        }
        Ok(v)
    }

    fn read_bits_msb(&mut self, n: u8) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit()?;
        }
        Ok(v)
    }

    fn align_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    fn read_fixed_lit_len(&mut self) -> Result<u16, String> {
        let c7 = self.read_bits_msb(7)?;
        if c7 <= 0x17 {
            return Ok((256 + c7) as u16);
        }
        let c8 = (c7 << 1) | self.read_bit()?;
        if (0x30..=0xBF).contains(&c8) {
            return Ok((c8 - 0x30) as u16);
        }
        if (0xC0..=0xC7).contains(&c8) {
            return Ok((280 + (c8 - 0xC0)) as u16);
        }
        let c9 = (c8 << 1) | self.read_bit()?;
        if (0x190..=0x1FF).contains(&c9) {
            return Ok((144 + (c9 - 0x190)) as u16);
        }
        Err(format!("Invalid Fixed Huffman code {:x}", c9))
    }
}

/// Standalone RFC 1950 + RFC 1951 zlib/DEFLATE decompressor verifying header FCHECK and Adler-32.
pub fn zlib_decompress(zbytes: &[u8]) -> Result<Vec<u8>, String> {
    if zbytes.len() < 6 {
        return Err("Zlib stream shorter than 6 bytes".to_string());
    }
    let cmf = zbytes[0];
    let flg = zbytes[1];
    if (cmf & 0x0F) != 8 {
        return Err(format!("Unsupported zlib CM={}", cmf & 0x0F));
    }
    if (((cmf as u16) << 8) | (flg as u16)) % 31 != 0 {
        return Err("Zlib header FCHECK failed modulo 31 check".to_string());
    }

    let deflate_payload = &zbytes[2..zbytes.len() - 4];
    let expected_adler = u32::from_be_bytes([
        zbytes[zbytes.len() - 4],
        zbytes[zbytes.len() - 3],
        zbytes[zbytes.len() - 2],
        zbytes[zbytes.len() - 1],
    ]);

    let mut br = BitReader::new(deflate_payload);
    let mut out = Vec::new();

    loop {
        let bfinal = br.read_bits_lsb(1)?;
        let btype = br.read_bits_lsb(2)?;
        match btype {
            0 => {
                br.align_byte();
                if br.byte_pos + 4 > deflate_payload.len() {
                    return Err("Truncated stored block header".to_string());
                }
                let len = u16::from_le_bytes([
                    deflate_payload[br.byte_pos],
                    deflate_payload[br.byte_pos + 1],
                ]);
                let nlen = u16::from_le_bytes([
                    deflate_payload[br.byte_pos + 2],
                    deflate_payload[br.byte_pos + 3],
                ]);
                br.byte_pos += 4;
                if len != !nlen {
                    return Err("Stored block LEN != ~NLEN".to_string());
                }
                let end = br.byte_pos + (len as usize);
                if end > deflate_payload.len() {
                    return Err("Truncated stored block data".to_string());
                }
                out.extend_from_slice(&deflate_payload[br.byte_pos..end]);
                br.byte_pos = end;
            }
            1 => loop {
                let sym = br.read_fixed_lit_len()?;
                if sym < 256 {
                    out.push(sym as u8);
                } else if sym == 256 {
                    break;
                } else if sym <= 285 {
                    let lidx = (sym - 257) as usize;
                    let length =
                        (LENGTH_BASE[lidx] as usize) + (br.read_bits_lsb(LENGTH_EXTRA[lidx])? as usize);
                    let dsym = br.read_bits_msb(5)? as usize;
                    if dsym >= 30 {
                        return Err(format!("Invalid distance symbol {}", dsym));
                    }
                    let dist =
                        (DIST_BASE[dsym] as usize) + (br.read_bits_lsb(DIST_EXTRA[dsym])? as usize);
                    if dist == 0 || dist > out.len() {
                        return Err(format!("Invalid LZ77 back-reference dist={} out_len={}", dist, out.len()));
                    }
                    let start = out.len() - dist;
                    for i in 0..length {
                        let b = out[start + i];
                        out.push(b);
                    }
                } else {
                    return Err(format!("Reserved literal/length symbol {}", sym));
                }
            },
            _ => return Err(format!("Unsupported DEFLATE block BTYPE={}", btype)),
        }
        if bfinal == 1 {
            break;
        }
    }

    let actual_adler = adler32(&out);
    if actual_adler != expected_adler {
        return Err(format!(
            "Adler32 mismatch: expected {:08X}, got {:08X}",
            expected_adler, actual_adler
        ));
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct DecodedPng {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_type: u8,
    pub chunk_count: usize,
    pub rgb_pixels: Vec<u8>,
}

/// Full binary PNG parser, CRC-32 verifier, DEFLATE decompressor, and scanline unfilterer.
pub fn decode_and_verify_png(png_bytes: &[u8]) -> Result<DecodedPng, String> {
    if png_bytes.len() < 33 || png_bytes[0..8] != PNG_SIGNATURE {
        return Err("Invalid 8-byte PNG signature".to_string());
    }

    let mut pos = 8usize;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut bit_depth = 0u8;
    let mut color_type = 0u8;
    let mut saw_ihdr = false;
    let mut saw_iend = false;
    let mut idat_stream = Vec::new();
    let mut chunk_count = 0usize;

    while pos < png_bytes.len() {
        if pos + 12 > png_bytes.len() {
            return Err("Truncated PNG chunk header".to_string());
        }
        let length = u32::from_be_bytes([
            png_bytes[pos],
            png_bytes[pos + 1],
            png_bytes[pos + 2],
            png_bytes[pos + 3],
        ]) as usize;
        let ctype: [u8; 4] = [
            png_bytes[pos + 4],
            png_bytes[pos + 5],
            png_bytes[pos + 6],
            png_bytes[pos + 7],
        ];
        let data_start = pos + 8;
        let data_end = data_start + length;
        if data_end + 4 > png_bytes.len() {
            return Err("Truncated PNG chunk payload".to_string());
        }
        let data = &png_bytes[data_start..data_end];
        let stored_crc = u32::from_be_bytes([
            png_bytes[data_end],
            png_bytes[data_end + 1],
            png_bytes[data_end + 2],
            png_bytes[data_end + 3],
        ]);
        let computed_crc = crc32_chunk(&ctype, data);
        if stored_crc != computed_crc {
            return Err(format!(
                "PNG chunk {:?} CRC32 mismatch: {:08X} != {:08X}",
                std::str::from_utf8(&ctype).unwrap_or("????"),
                stored_crc,
                computed_crc
            ));
        }

        chunk_count += 1;
        match &ctype {
            b"IHDR" => {
                if length != 13 || chunk_count != 1 {
                    return Err("Malformed IHDR chunk".to_string());
                }
                width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
                height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
                bit_depth = data[8];
                color_type = data[9];
                saw_ihdr = true;
            }
            b"IDAT" => {
                idat_stream.extend_from_slice(data);
            }
            b"IEND" => {
                if length != 0 {
                    return Err("Non-empty IEND chunk".to_string());
                }
                saw_iend = true;
                pos = data_end + 4;
                break;
            }
            _ => {}
        }
        pos = data_end + 4;
    }

    if !saw_ihdr || !saw_iend || pos != png_bytes.len() {
        return Err("Missing IHDR/IEND or trailing garbage after IEND".to_string());
    }
    if bit_depth != 8 || color_type != 2 {
        return Err(format!(
            "Unexpected PNG format: bit_depth={} color_type={}",
            bit_depth, color_type
        ));
    }

    let raw_filtered = zlib_decompress(&idat_stream)?;
    let row_stride = (width as usize) * 3;
    if raw_filtered.len() != (row_stride + 1) * (height as usize) {
        return Err("Decompressed IDAT scanline buffer size mismatch".to_string());
    }

    let mut rgb = vec![0u8; row_stride * (height as usize)];
    for y in 0..(height as usize) {
        let ftype = raw_filtered[y * (row_stride + 1)];
        let src = &raw_filtered[y * (row_stride + 1) + 1..(y + 1) * (row_stride + 1)];
        for x in 0..row_stride {
            let a = if x >= 3 { rgb[y * row_stride + x - 3] } else { 0 };
            let b = if y > 0 { rgb[(y - 1) * row_stride + x] } else { 0 };
            let c = if y > 0 && x >= 3 {
                rgb[(y - 1) * row_stride + x - 3]
            } else {
                0
            };
            rgb[y * row_stride + x] = match ftype {
                0 => src[x],
                1 => src[x].wrapping_add(a),
                2 => src[x].wrapping_add(b),
                3 => src[x].wrapping_add((((a as u16) + (b as u16)) / 2) as u8),
                4 => src[x].wrapping_add(paeth_predictor(a, b, c)),
                other => return Err(format!("Invalid PNG scanline filter type {}", other)),
            };
        }
    }

    Ok(DecodedPng {
        width,
        height,
        bit_depth,
        color_type,
        chunk_count,
        rgb_pixels: rgb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_and_adler32_rfc_vectors() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn test_png_lz77_roundtrip_lossless() {
        let w = 64u32;
        let h = 48u32;
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                pixels.push(((x * 4) & 0xFF) as u8);
                pixels.push(((y * 5) & 0xFF) as u8);
                pixels.push((((x ^ y) * 7) & 0xFF) as u8);
            }
        }
        let png = encode_png_rgb8(w, h, &pixels);
        assert!(png.len() < pixels.len(), "LZ77 + Paeth PNG should compress gradient!");
        let decoded = decode_and_verify_png(&png).expect("PNG must decode cleanly");
        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h);
        assert_eq!(decoded.rgb_pixels, pixels);
    }
}
