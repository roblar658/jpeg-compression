"""
Pure Python Baseline JPEG Encoder & Decoder (ITU-T T.81 / JFIF compliant).
Includes full pipeline implementation:
  - 8x8 2D-DCT and Inverse DCT
  - Standard Luminance & Chrominance Quantization with Quality Scaling
  - Zig-Zag Ordering & Run-Length Encoding (RLE)
  - Canonical Huffman Bitstream Writer/Reader with 0xFF/0x00 Byte Stuffing
  - Standard JFIF Marker Construction & Parsing (SOI, APP0, DQT, SOF0, DHT, SOS, EOI)
  - Comprehensive unit and integration test suite (unittest)
"""

import math
import struct
import io
import unittest


# =====================================================================
# 1. Standard JPEG Tables & Constants
# =====================================================================

STD_LUM_QUANT_TABLE = [
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99
]

STD_CHR_QUANT_TABLE = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
]

ZIGZAG = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
]

ZIGZAG_COORDS = [(idx // 8, idx % 8) for idx in ZIGZAG]

# Standard DHT Definitions (ITU-T T.81 Annex K)
STD_LU_DC_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
STD_LU_DC_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

STD_CH_DC_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
STD_CH_DC_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

STD_LU_AC_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D]
STD_LU_AC_VALS = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
]

STD_CH_AC_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77]
STD_CH_AC_VALS = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
    0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
    0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
]


# =====================================================================
# 2. Mathematical Utilities: 2D-DCT, IDCT, & Quantization
# =====================================================================

def _build_dct_matrix():
    matrix = [[0.0] * 8 for _ in range(8)]
    for u in range(8):
        alpha = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
        for x in range(8):
            matrix[u][x] = 0.5 * alpha * math.cos((2 * x + 1) * u * math.pi / 16.0)
    return matrix

DCT_MATRIX = _build_dct_matrix()
IDCT_MATRIX = [[DCT_MATRIX[x][u] for x in range(8)] for u in range(8)]


def matmul_8x8(A, B):
    """Multiplies two 8x8 matrices."""
    C = [[0.0] * 8 for _ in range(8)]
    for i in range(8):
        for k in range(8):
            a_ik = A[i][k]
            for j in range(8):
                C[i][j] += a_ik * B[k][j]
    return C


def dct_2d_8x8(block):
    """Transforms an 8x8 spatial block to frequency coefficients via T * block * T^T."""
    return matmul_8x8(matmul_8x8(DCT_MATRIX, block), IDCT_MATRIX)


def idct_2d_8x8(coeff):
    """Reconstructs an 8x8 spatial block via T^T * coeff * T."""
    return matmul_8x8(matmul_8x8(IDCT_MATRIX, coeff), DCT_MATRIX)


def scale_quant_table(table, quality: int):
    """Scales standard quantization table based on IJG quality factor [1..100]."""
    quality = max(1, min(100, quality))
    scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
    scaled = []
    for val in table:
        new_val = int((val * scale + 50) / 100)
        scaled.append(max(1, min(255, new_val)))
    return scaled


def block_to_zigzag(block):
    """Flattens an 8x8 block into a 64-element list in zig-zag order."""
    return [block[r][c] for r, c in ZIGZAG_COORDS]


def zigzag_to_block(zz):
    """Unpacks a 64-element zig-zag list into an 8x8 block."""
    block = [[0] * 8 for _ in range(8)]
    for k, (r, c) in enumerate(ZIGZAG_COORDS):
        block[r][c] = zz[k]
    return block


# =====================================================================
# 3. Huffman Tree & Entropy Bitstream Processing
# =====================================================================

class HuffmanDecoderNode:
    __slots__ = ('children', 'symbol')
    def __init__(self):
        self.children = [None, None]
        self.symbol = None


def build_huffman_table(bits, values):
    """Builds canonical Huffman code dictionary symbol -> (code, length)."""
    code = 0
    encode_dict = {}
    val_idx = 0
    for length in range(1, 17):
        count = bits[length - 1]
        for _ in range(count):
            sym = values[val_idx]
            encode_dict[sym] = (code, length)
            val_idx += 1
            code += 1
        code <<= 1
    return encode_dict


def build_decode_tree(encode_dict):
    """Builds binary prefix search tree for fast bitstream decoding."""
    root = HuffmanDecoderNode()
    for sym, (code, length) in encode_dict.items():
        node = root
        for i in range(length - 1, -1, -1):
            bit = (code >> i) & 1
            if node.children[bit] is None:
                node.children[bit] = HuffmanDecoderNode()
            node = node.children[bit]
        node.symbol = sym
    return root


def encode_amplitude(val: int):
    """Determines JPEG category and variable-length bits for an integer."""
    if val == 0:
        return 0, 0
    cat = abs(val).bit_length()
    bits = val if val > 0 else (val + (1 << cat) - 1)
    return cat, bits


def decode_amplitude(bits: int, cat: int) -> int:
    """Decodes amplitude from raw bit representation."""
    if cat == 0:
        return 0
    if bits < (1 << (cat - 1)):
        return bits - ((1 << cat) - 1)
    return bits


class BitStreamWriter:
    """Bit-packer with JPEG 0xFF/0x00 byte stuffing."""
    def __init__(self):
        self.bytes_out = bytearray()
        self.buf = 0
        self.bit_count = 0

    def write_bits(self, code: int, length: int):
        self.buf = (self.buf << length) | (code & ((1 << length) - 1))
        self.bit_count += length
        while self.bit_count >= 8:
            shift = self.bit_count - 8
            byte = (self.buf >> shift) & 0xFF
            self.bytes_out.append(byte)
            if byte == 0xFF:
                self.bytes_out.append(0x00)
            self.bit_count -= 8
            self.buf &= (1 << self.bit_count) - 1

    def flush(self):
        """Flushes remaining bits padded with 1s as per JPEG standard."""
        if self.bit_count > 0:
            pad = 8 - self.bit_count
            byte = ((self.buf << pad) | ((1 << pad) - 1)) & 0xFF
            self.bytes_out.append(byte)
            if byte == 0xFF:
                self.bytes_out.append(0x00)
            self.buf = 0
            self.bit_count = 0

    def get_bytes(self) -> bytes:
        return bytes(self.bytes_out)


class BitStreamReader:
    """Bitstream reader with byte unstuffing and marker detection."""
    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos
        self.buf = 0
        self.bit_count = 0

    def _next_byte(self):
        if self.pos >= len(self.data):
            return 0
        b = self.data[self.pos]
        self.pos += 1
        if b == 0xFF:
            while self.pos < len(self.data) and self.data[self.pos] == 0xFF:
                self.pos += 1
            if self.pos < len(self.data):
                b2 = self.data[self.pos]
                if b2 == 0x00:
                    self.pos += 1
                    return 0xFF
                else:
                    self.pos -= 1  # Hit marker (e.g., EOI), unwind
                    return 0
            return 0
        return b

    def read_bit(self) -> int:
        if self.bit_count == 0:
            self.buf = self._next_byte()
            self.bit_count = 8
        self.bit_count -= 1
        return (self.buf >> self.bit_count) & 1

    def read_bits(self, n: int) -> int:
        if n == 0:
            return 0
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val

    def decode_huffman(self, root: HuffmanDecoderNode) -> int:
        node = root
        while node.symbol is None:
            bit = self.read_bit()
            node = node.children[bit]
            if node is None:
                raise ValueError("Corrupted stream: invalid Huffman prefix code encountered")
        return node.symbol


# =====================================================================
# 4. Core Image Container & Color Transformation
# =====================================================================

class JPEGImage:
    """Represents an uncompressed image with width, height, and color channels."""
    def __init__(self, width: int, height: int, channels: int, pixels: list):
        self.width = width
        self.height = height
        self.channels = channels
        self.pixels = pixels  # 2D list: [y][x] = int (grayscale) or [R, G, B]

    def to_bytes(self) -> bytes:
        out = bytearray()
        if self.channels == 1:
            for row in self.pixels:
                out.extend(row)
        else:
            for row in self.pixels:
                for rgb in row:
                    out.extend(rgb)
        return bytes(out)

    @classmethod
    def from_bytes(cls, raw: bytes, width: int, height: int, channels: int = 3):
        pixels = []
        idx = 0
        if channels == 1:
            for _ in range(height):
                pixels.append(list(raw[idx : idx + width]))
                idx += width
        else:
            for _ in range(height):
                row = []
                for _ in range(width):
                    row.append([raw[idx], raw[idx + 1], raw[idx + 2]])
                    idx += 3
                pixels.append(row)
        return cls(width, height, channels, pixels)


# =====================================================================
# 5. JPEG Compressor
# =====================================================================

def _encode_mcu_block(writer: BitStreamWriter, zz: list, prev_dc: int,
                      dc_table: dict, ac_table: dict) -> int:
    """Encodes a single 8x8 block in zig-zag order using DPCM DC and RLE AC."""
    # DC coding
    dc = zz[0]
    diff = dc - prev_dc
    cat, bits = encode_amplitude(diff)
    code, length = dc_table[cat]
    writer.write_bits(code, length)
    if cat > 0:
        writer.write_bits(bits, cat)

    # AC coding
    zero_run = 0
    for k in range(1, 64):
        val = zz[k]
        if val == 0:
            zero_run += 1
        else:
            while zero_run >= 16:
                code, length = ac_table[0xF0]  # ZRL
                writer.write_bits(code, length)
                zero_run -= 16
            cat, bits = encode_amplitude(val)
            sym = (zero_run << 4) | cat
            code, length = ac_table[sym]
            writer.write_bits(code, length)
            writer.write_bits(bits, cat)
            zero_run = 0

    if zero_run > 0:
        code, length = ac_table[0x00]  # EOB
        writer.write_bits(code, length)

    return dc


def compress(image: JPEGImage, quality: int = 50) -> bytes:
    """Compresses a JPEGImage instance into a compliant JFIF JPEG byte stream."""
    width, height, channels = image.width, image.height, image.channels
    lum_q = scale_quant_table(STD_LUM_QUANT_TABLE, quality)
    chr_q = scale_quant_table(STD_CHR_QUANT_TABLE, quality)

    padded_w = ((width + 7) // 8) * 8
    padded_h = ((height + 7) // 8) * 8

    # Color space plane construction & edge padding
    if channels == 1:
        y_plane = [[0] * padded_w for _ in range(padded_h)]
        for y in range(padded_h):
            src_y = min(y, height - 1)
            for x in range(padded_w):
                src_x = min(x, width - 1)
                y_plane[y][x] = image.pixels[src_y][src_x]
    else:
        y_plane = [[0.0] * padded_w for _ in range(padded_h)]
        cb_plane = [[0.0] * padded_w for _ in range(padded_h)]
        cr_plane = [[0.0] * padded_w for _ in range(padded_h)]
        for y in range(padded_h):
            src_y = min(y, height - 1)
            for x in range(padded_w):
                src_x = min(x, width - 1)
                r, g, b = image.pixels[src_y][src_x]
                y_plane[y][x] = 0.299 * r + 0.587 * g + 0.114 * b
                cb_plane[y][x] = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
                cr_plane[y][x] = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    stream = bytearray()
    stream.extend(b'\xFF\xD8')  # SOI

    # APP0 (JFIF standard marker)
    app0_payload = (
        b'JFIF\x00' +
        b'\x01\x01' +  # v1.01
        b'\x00' +      # no units
        b'\x00\x01\x00\x01' +  # 1:1 aspect
        b'\x00\x00'    # no thumbnail
    )
    stream.extend(b'\xFF\xE0' + struct.pack('>H', len(app0_payload) + 2) + app0_payload)

    # DQT (Quantization tables in zigzag order)
    dqt0 = bytes([0]) + bytes([lum_q[ZIGZAG[k]] for k in range(64)])
    stream.extend(b'\xFF\xDB' + struct.pack('>H', len(dqt0) + 2) + dqt0)
    if channels == 3:
        dqt1 = bytes([1]) + bytes([chr_q[ZIGZAG[k]] for k in range(64)])
        stream.extend(b'\xFF\xDB' + struct.pack('>H', len(dqt1) + 2) + dqt1)

    # SOF0 (Baseline DCT frame)
    sof_payload = bytearray([8])  # 8-bit precision
    sof_payload.extend(struct.pack('>HH', height, width))
    sof_payload.append(channels)
    if channels == 1:
        sof_payload.extend([1, 0x11, 0])  # Y: 1x1 sampling, QTable 0
    else:
        sof_payload.extend([1, 0x11, 0])  # Y: 1x1, Q0
        sof_payload.extend([2, 0x11, 1])  # Cb: 1x1, Q1
        sof_payload.extend([3, 0x11, 1])  # Cr: 1x1, Q1
    stream.extend(b'\xFF\xC0' + struct.pack('>H', len(sof_payload) + 2) + sof_payload)

    # DHT (Huffman tables)
    def append_dht(t_class, t_id, bits, vals):
        payload = bytes([(t_class << 4) | t_id]) + bytes(bits) + bytes(vals)
        stream.extend(b'\xFF\xC4' + struct.pack('>H', len(payload) + 2) + payload)

    append_dht(0, 0, STD_LU_DC_BITS, STD_LU_DC_VALS)
    append_dht(1, 0, STD_LU_AC_BITS, STD_LU_AC_VALS)
    if channels == 3:
        append_dht(0, 1, STD_CH_DC_BITS, STD_CH_DC_VALS)
        append_dht(1, 1, STD_CH_AC_BITS, STD_CH_AC_VALS)

    # SOS (Start of Scan)
    sos_payload = bytearray([channels])
    if channels == 1:
        sos_payload.extend([1, 0x00])
    else:
        sos_payload.extend([1, 0x00, 2, 0x11, 3, 0x11])
    sos_payload.extend([0, 63, 0])  # Ss=0, Se=63, Ah=Al=0
    stream.extend(b'\xFF\xDA' + struct.pack('>H', len(sos_payload) + 2) + sos_payload)

    # Entropy Coding Pipeline
    writer = BitStreamWriter()
    lum_dc_enc = build_huffman_table(STD_LU_DC_BITS, STD_LU_DC_VALS)
    lum_ac_enc = build_huffman_table(STD_LU_AC_BITS, STD_LU_AC_VALS)
    chr_dc_enc = build_huffman_table(STD_CH_DC_BITS, STD_CH_DC_VALS) if channels == 3 else None
    chr_ac_enc = build_huffman_table(STD_CH_AC_BITS, STD_CH_AC_VALS) if channels == 3 else None

    prev_dc_Y = prev_dc_Cb = prev_dc_Cr = 0
    mcu_rows, mcu_cols = padded_h // 8, padded_w // 8

    def process_block(plane, y0, x0, q_table):
        # 1. Level shift
        blk = [[plane[y0 + r][x0 + c] - 128.0 for c in range(8)] for r in range(8)]
        # 2. 2D DCT
        dct_blk = dct_2d_8x8(blk)
        # 3. Quantize
        q_blk = [[int(round(dct_blk[r][c] / q_table[r * 8 + c])) for c in range(8)] for r in range(8)]
        # 4. Zig-zag reorder
        return block_to_zigzag(q_blk)

    for my in range(mcu_rows):
        for mx in range(mcu_cols):
            y0, x0 = my * 8, mx * 8
            zz_Y = process_block(y_plane, y0, x0, lum_q)
            prev_dc_Y = _encode_mcu_block(writer, zz_Y, prev_dc_Y, lum_dc_enc, lum_ac_enc)

            if channels == 3:
                zz_Cb = process_block(cb_plane, y0, x0, chr_q)
                prev_dc_Cb = _encode_mcu_block(writer, zz_Cb, prev_dc_Cb, chr_dc_enc, chr_ac_enc)

                zz_Cr = process_block(cr_plane, y0, x0, chr_q)
                prev_dc_Cr = _encode_mcu_block(writer, zz_Cr, prev_dc_Cr, chr_dc_enc, chr_ac_enc)

    writer.flush()
    stream.extend(writer.get_bytes())
    stream.extend(b'\xFF\xD9')  # EOI
    return bytes(stream)


# =====================================================================
# 6. JPEG Decompressor
# =====================================================================

def _decode_mcu_block(reader: BitStreamReader, prev_dc: int, dc_tree: HuffmanDecoderNode,
                      ac_tree: HuffmanDecoderNode, q_table_raster: list) -> tuple:
    """Decodes, dequantizes, and inversely transforms an 8x8 block."""
    cat = reader.decode_huffman(dc_tree)
    bits = reader.read_bits(cat)
    diff = decode_amplitude(bits, cat)
    dc = prev_dc + diff

    zz = [0] * 64
    zz[0] = dc
    k = 1
    while k < 64:
        sym = reader.decode_huffman(ac_tree)
        if sym == 0x00:  # EOB
            break
        elif sym == 0xF0:  # ZRL: 16 consecutive zero coefficients
            k += 16
            continue
        run = sym >> 4
        cat = sym & 0x0F
        k += run
        if k < 64:
            bits = reader.read_bits(cat)
            zz[k] = decode_amplitude(bits, cat)
            k += 1

    # De-zigzag
    q_block = zigzag_to_block(zz)

    # Dequantize
    dct_block = [[float(q_block[r][c] * q_table_raster[r * 8 + c]) for c in range(8)] for r in range(8)]

    # 2D Inverse DCT & unshift
    spatial = idct_2d_8x8(dct_block)
    pixel_block = [[max(0, min(255, int(round(spatial[r][c] + 128.0)))) for c in range(8)] for r in range(8)]

    return pixel_block, dc


def decompress(jpeg_bytes: bytes) -> JPEGImage:
    """Decompresses standard baseline JFIF JPEG bytes into a JPEGImage instance."""
    if len(jpeg_bytes) < 4 or jpeg_bytes[:2] != b'\xFF\xD8':
        raise ValueError("Invalid JPEG data: SOI marker (0xFFD8) not found.")

    pos = 2
    quant_tables = {}
    huffman_trees = {}
    width = height = num_components = 0
    components = {}
    scan_components = []
    scan_start = 0

    # Header & Marker Parsing Loop
    while pos < len(jpeg_bytes):
        if jpeg_bytes[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(jpeg_bytes) and jpeg_bytes[pos] == 0xFF:
            pos += 1
        if pos >= len(jpeg_bytes):
            break
        marker = jpeg_bytes[pos]
        pos += 1

        if marker in (0xD8, 0xD9):  # SOI / EOI
            continue
        if 0xD0 <= marker <= 0xD7 or marker == 0x00:  # RST / stuffed byte
            continue

        if pos + 2 > len(jpeg_bytes):
            break
        seg_len = struct.unpack('>H', jpeg_bytes[pos:pos + 2])[0]
        payload = jpeg_bytes[pos + 2 : pos + seg_len]
        pos += seg_len

        if marker == 0xDB:  # DQT
            p = 0
            while p < len(payload):
                info = payload[p]
                qid = info & 0x0F
                p += 1
                q_raw = payload[p : p + 64]
                p += 64
                q_raster = [0] * 64
                for k in range(64):
                    q_raster[ZIGZAG[k]] = q_raw[k]
                quant_tables[qid] = q_raster

        elif marker == 0xC4:  # DHT
            p = 0
            while p < len(payload):
                info = payload[p]
                t_class = info >> 4
                t_id = info & 0x0F
                p += 1
                bits = list(payload[p : p + 16])
                p += 16
                n_vals = sum(bits)
                vals = list(payload[p : p + n_vals])
                p += n_vals
                enc_dict = build_huffman_table(bits, vals)
                huffman_trees[(t_class, t_id)] = build_decode_tree(enc_dict)

        elif marker == 0xC0:  # SOF0
            prec, height, width, num_components = struct.unpack('>BHHB', payload[:6])
            if prec != 8:
                raise ValueError(f"Unsupported sample precision: {prec} bits")
            for i in range(num_components):
                cid, samp, qid = struct.unpack('>BBB', payload[6 + i * 3 : 9 + i * 3])
                components[cid] = {'h': samp >> 4, 'v': samp & 0x0F, 'qid': qid}

        elif marker == 0xDA:  # SOS
            n_comps = payload[0]
            scan_components = []
            for i in range(n_comps):
                cid, t_info = struct.unpack('>BB', payload[1 + i * 2 : 3 + i * 2])
                scan_components.append({
                    'cid': cid,
                    'dc_tid': t_info >> 4,
                    'ac_tid': t_info & 0x0F
                })
            scan_start = pos
            break

    if not scan_components or width == 0 or height == 0:
        raise ValueError("Corrupt JPEG: missing necessary headers or image dimensions.")

    # MCU Layout Configuration
    ordered_comps = []
    for sc in scan_components:
        cid = sc['cid']
        c_info = components[cid]
        ordered_comps.append({
            'cid': cid,
            'h': c_info['h'],
            'v': c_info['v'],
            'qid': c_info['qid'],
            'dc_tree': huffman_trees[(0, sc['dc_tid'])],
            'ac_tree': huffman_trees[(1, sc['ac_tid'])],
            'q_table': quant_tables[c_info['qid']]
        })

    max_h = max(c['h'] for c in ordered_comps)
    max_v = max(c['v'] for c in ordered_comps)
    mcu_w = max_h * 8
    mcu_h = max_v * 8

    mcu_cols = (width + mcu_w - 1) // mcu_w
    mcu_rows = (height + mcu_h - 1) // mcu_h

    reader = BitStreamReader(jpeg_bytes, scan_start)
    prev_dc = {c['cid']: 0 for c in ordered_comps}

    # Initialize destination image buffer
    if num_components == 1:
        img_pixels = [[0] * width for _ in range(height)]
    else:
        img_pixels = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]

    for my in range(mcu_rows):
        for mx in range(mcu_cols):
            mcu_blocks = {c['cid']: [] for c in ordered_comps}
            for c in ordered_comps:
                cid = c['cid']
                for v in range(c['v']):
                    for h in range(c['h']):
                        blk, prev_dc[cid] = _decode_mcu_block(
                            reader, prev_dc[cid],
                            c['dc_tree'], c['ac_tree'],
                            c['q_table']
                        )
                        mcu_blocks[cid].append((v, h, blk))

            # Rasterize reconstructed MCU to coordinate grid
            if num_components == 1:
                cid = ordered_comps[0]['cid']
                v_scale = mcu_h // (ordered_comps[0]['v'] * 8)
                h_scale = mcu_w // (ordered_comps[0]['h'] * 8)
                for v, h, blk in mcu_blocks[cid]:
                    for r in range(8):
                        for c in range(8):
                            py = my * mcu_h + (v * 8 + r) * v_scale
                            px = mx * mcu_w + (h * 8 + c) * h_scale
                            val = blk[r][c]
                            for dy in range(v_scale):
                                for dx in range(h_scale):
                                    if py + dy < height and px + dx < width:
                                        img_pixels[py + dy][px + dx] = val
            else:
                mcu_planes = {}
                for comp in ordered_comps:
                    cid = comp['cid']
                    plane = [[0] * mcu_w for _ in range(mcu_h)]
                    v_scale = mcu_h // (comp['v'] * 8)
                    h_scale = mcu_w // (comp['h'] * 8)
                    for v, h, blk in mcu_blocks[cid]:
                        for r in range(8):
                            for c in range(8):
                                sy = (v * 8 + r) * v_scale
                                sx = (h * 8 + c) * h_scale
                                val = blk[r][c]
                                for dy in range(v_scale):
                                    for dx in range(h_scale):
                                        plane[sy + dy][sx + dx] = val
                    mcu_planes[cid] = plane

                cid_Y, cid_Cb, cid_Cr = ordered_comps[0]['cid'], ordered_comps[1]['cid'], ordered_comps[2]['cid']
                for r in range(mcu_h):
                    py = my * mcu_h + r
                    if py >= height:
                        continue
                    for c in range(mcu_w):
                        px = mx * mcu_w + c
                        if px >= width:
                            continue
                        Y = mcu_planes[cid_Y][r][c]
                        Cb = mcu_planes[cid_Cb][r][c]
                        Cr = mcu_planes[cid_Cr][r][c]

                        # YCbCr to sRGB conversion
                        R = int(round(Y + 1.402 * (Cr - 128.0)))
                        G = int(round(Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)))
                        B = int(round(Y + 1.772 * (Cb - 128.0)))

                        img_pixels[py][px] = [
                            max(0, min(255, R)),
                            max(0, min(255, G)),
                            max(0, min(255, B))
                        ]

    return JPEGImage(width, height, num_components, img_pixels)


# =====================================================================
# 7. Unit and Integration Test Suite
# =====================================================================

class TestJPEGCodec(unittest.TestCase):

    def test_dct_idct_invertibility(self):
        """Validates that 2D-DCT followed by IDCT recovers the source block accurately."""
        orig_block = [
            [52 + (r * 8 + c) * 2 for c in range(8)]
            for r in range(8)
        ]
        freq = dct_2d_8x8(orig_block)
        rec = idct_2d_8x8(freq)
        max_error = max(abs(orig_block[r][c] - rec[r][c]) for r in range(8) for c in range(8))
        self.assertLess(max_error, 1e-4)

    def test_zigzag_ordering_roundtrip(self):
        """Verifies bijective mapping of zig-zag scan and inverse."""
        orig = [[r * 8 + c for c in range(8)] for r in range(8)]
        zz = block_to_zigzag(orig)
        self.assertEqual(len(zz), 64)
        rec = zigzag_to_block(zz)
        self.assertEqual(orig, rec)

    def test_amplitude_encoding_and_decoding(self):
        """Verifies symmetrical amplitude representation for positive and negative values."""
        test_values = [-255, -128, -63, -15, -7, -1, 0, 1, 7, 15, 63, 128, 255]
        for val in test_values:
            cat, bits = encode_amplitude(val)
            decoded = decode_amplitude(bits, cat)
            self.assertEqual(val, decoded, f"Failed for value {val}")

    def test_bitstream_byte_stuffing(self):
        """Ensures that 0xFF bytes inside the entropy payload are stuffed with 0x00."""
        writer = BitStreamWriter()
        # Emit eight 1-bits to force an 0xFF byte
        writer.write_bits(0xFF, 8)
        writer.write_bits(0xAA, 8)
        writer.flush()
        raw = writer.get_bytes()
        self.assertEqual(raw[:2], b'\xFF\x00')

        reader = BitStreamReader(raw)
        self.assertEqual(reader.read_bits(8), 0xFF)
        self.assertEqual(reader.read_bits(8), 0xAA)

    def test_huffman_encoding_decoding(self):
        """Ensures all standard Huffman symbols can be coded and parsed correctly."""
        enc = build_huffman_table(STD_LU_DC_BITS, STD_LU_DC_VALS)
        root = build_decode_tree(enc)

        for sym in STD_LU_DC_VALS:
            code, length = enc[sym]
            writer = BitStreamWriter()
            writer.write_bits(code, length)
            writer.flush()

            reader = BitStreamReader(writer.get_bytes())
            decoded_sym = reader.decode_huffman(root)
            self.assertEqual(sym, decoded_sym)

    def test_grayscale_codec_roundtrip(self):
        """Compresses and decompresses a 16x16 grayscale gradient image."""
        w, h = 16, 16
        pixels = [[(r * 16 + c * 8) % 256 for c in range(w)] for r in range(h)]
        img = JPEGImage(w, h, 1, pixels)

        compressed = compress(img, quality=85)
        self.assertTrue(compressed.startswith(b'\xFF\xD8'))
        self.assertTrue(compressed.endswith(b'\xFF\xD9'))

        decompressed = decompress(compressed)
        self.assertEqual(decompressed.width, w)
        self.assertEqual(decompressed.height, h)
        self.assertEqual(decompressed.channels, 1)

        # Check Reconstruction Mean Squared Error (MSE)
        mse = sum(
            (pixels[r][c] - decompressed.pixels[r][c]) ** 2
            for r in range(h) for c in range(w)
        ) / (w * h)
        self.assertLess(mse, 10.0)

    def test_color_codec_roundtrip(self):
        """Compresses and decompresses an RGB 16x16 color block."""
        w, h = 16, 16
        pixels = [
            [[r * 15, c * 15, (r + c) * 8] for c in range(w)]
            for r in range(h)
        ]
        img = JPEGImage(w, h, 3, pixels)

        compressed = compress(img, quality=75)
        decompressed = decompress(compressed)

        self.assertEqual(decompressed.width, w)
        self.assertEqual(decompressed.height, h)
        self.assertEqual(decompressed.channels, 3)

        mse = sum(
            (pixels[r][c][ch] - decompressed.pixels[r][c][ch]) ** 2
            for r in range(h) for c in range(w) for ch in range(3)
        ) / (w * h * 3)
        self.assertLess(mse, 25.0)

    def test_arbitrary_dimension_padding(self):
        """Checks handling of image dimensions that are not multiples of 8 (e.g. 13x19)."""
        w, h = 13, 19
        pixels = [[(x * 17 + y * 11) % 256 for x in range(w)] for y in range(h)]
        img = JPEGImage(w, h, 1, pixels)

        compressed = compress(img, quality=70)
        decompressed = decompress(compressed)

        self.assertEqual(decompressed.width, 13)
        self.assertEqual(decompressed.height, 19)

    def test_quality_scale_dynamics(self):
        """Verifies that higher quality results in a larger file size and lower MSE."""
        w, h = 32, 32
        pixels = [[(x ^ y) * 8 for x in range(w)] for y in range(h)]
        img = JPEGImage(w, h, 1, pixels)

        low_q = compress(img, quality=15)
        high_q = compress(img, quality=90)
        self.assertLess(len(low_q), len(high_q))

        dec_low = decompress(low_q)
        dec_high = decompress(high_q)

        mse_low = sum((pixels[y][x] - dec_low.pixels[y][x]) ** 2 for y in range(h) for x in range(w))
        mse_high = sum((pixels[y][x] - dec_high.pixels[y][x]) ** 2 for y in range(h) for x in range(w))
        self.assertLess(mse_high, mse_low)

    def test_invalid_header_error_handling(self):
        """Confirms that non-JPEG data raises a descriptive ValueError."""
        with self.assertRaises(ValueError):
            decompress(b"NOT_A_JPEG_FILE")


# =====================================================================
# 8. Command Line & Test Runner Interface
# =====================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Running Pure-Python JPEG Codec Test Suite (ITU-T T.81 / JFIF)")
    print("=" * 70)
    unittest.main(verbosity=2)
