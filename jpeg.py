"""
Pure Python Baseline JPEG Encoder & Decoder (ITU-T T.81 / JFIF compliant)
with Educational Step-by-Step Visualization & Plotting Pipeline.

Includes full pipeline implementation:
  - 8x8 2D-DCT and Inverse DCT
  - Standard Luminance & Chrominance Quantization with Quality Scaling
  - Zig-Zag Ordering & Run-Length Encoding (RLE)
  - Canonical Huffman Bitstream Writer/Reader with 0xFF/0x00 Byte Stuffing
  - Standard JFIF Marker Construction & Parsing (SOI, APP0, DQT, SOF0, DHT, SOS, EOI)
  - Comprehensive unit and integration test suite (unittest)
  - Visual Step-by-Step Plots for every stage in the JPEG algorithm (matplotlib)
"""

import math
import struct
import io
import os
import sys
import argparse
import unittest

# Matplotlib headless backend configuration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


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

    def to_numpy(self):
        """Converts pixels to numpy uint8 array."""
        return np.array(self.pixels, dtype=np.uint8)

    @classmethod
    def from_numpy(cls, arr):
        """Creates JPEGImage from numpy uint8 array."""
        arr = np.asarray(arr, dtype=np.uint8)
        if arr.ndim == 2:
            h, w = arr.shape
            return cls(w, h, 1, arr.tolist())
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            h, w, c = arr.shape
            return cls(w, h, 3, arr[:, :, :3].tolist())
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

    def to_pil(self):
        """Converts JPEGImage to a PIL Image."""
        from PIL import Image
        arr = self.to_numpy()
        mode = 'L' if self.channels == 1 else 'RGB'
        return Image.fromarray(arr, mode=mode)

    @classmethod
    def from_pil(cls, pil_img):
        """Creates JPEGImage from a PIL Image."""
        if pil_img.mode in ('L', '1'):
            arr = np.array(pil_img.convert('L'), dtype=np.uint8)
            return cls(arr.shape[1], arr.shape[0], 1, arr.tolist())
        else:
            arr = np.array(pil_img.convert('RGB'), dtype=np.uint8)
            return cls(arr.shape[1], arr.shape[0], 3, arr.tolist())

    @classmethod
    def from_file(cls, path_or_stream):
        """Loads an image from file path using PIL or raw bytes."""
        from PIL import Image
        pil_img = Image.open(path_or_stream)
        return cls.from_pil(pil_img)

    def save(self, filepath, quality: int = 50):
        """Compresses and saves JPEG file."""
        data = compress(self, quality=quality)
        with open(filepath, 'wb') as f:
            f.write(data)


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
# 7. Pipeline Intermediate State Extraction
# =====================================================================

def extract_block_pipeline_data(image: JPEGImage, block_y: int = None, block_x: int = None, quality: int = 50):
    """
    Extracts intermediate representations at every single stage of the JPEG pipeline for an 8x8 block.
    If block_y or block_x is None, selects an interesting block with rich texture and variance.
    """
    width, height = image.width, image.height
    padded_w = ((width + 7) // 8) * 8
    padded_h = ((height + 7) // 8) * 8

    if image.channels == 1:
        plane = [[float(image.pixels[min(y, height - 1)][min(x, width - 1)]) for x in range(padded_w)] for y in range(padded_h)]
    else:
        plane = [[0.0] * padded_w for _ in range(padded_h)]
        for y in range(padded_h):
            sy = min(y, height - 1)
            for x in range(padded_w):
                sx = min(x, width - 1)
                r, g, b = image.pixels[sy][sx]
                plane[y][x] = 0.299 * r + 0.587 * g + 0.114 * b

    mcu_rows, mcu_cols = padded_h // 8, padded_w // 8
    if block_y is None or block_x is None:
        best_var = -1.0
        best_by, best_bx = 0, 0
        for by in range(max(1, mcu_rows - 1)):
            for bx in range(max(1, mcu_cols - 1)):
                sub = [plane[by * 8 + r][bx * 8 + c] for r in range(8) for c in range(8)]
                mean_v = sum(sub) / 64.0
                var_v = sum((v - mean_v) ** 2 for v in sub) / 64.0
                if var_v > best_var and 300.0 < var_v < 6000.0:
                    best_var = var_v
                    best_by, best_bx = by, bx
        if best_var <= 0:
            block_y, block_x = mcu_rows // 2, mcu_cols // 2
        else:
            block_y, block_x = best_by, best_bx

    y0, x0 = block_y * 8, block_x * 8
    orig_block = [[plane[y0 + r][x0 + c] for c in range(8)] for r in range(8)]

    # Step 3: Level Shift (-128)
    shifted_block = [[orig_block[r][c] - 128.0 for c in range(8)] for r in range(8)]

    # Step 4: 2D-DCT
    dct_block = dct_2d_8x8(shifted_block)

    # Step 5: Quantization
    q_table = scale_quant_table(STD_LUM_QUANT_TABLE, quality)
    quantized_block = [[int(round(dct_block[r][c] / q_table[r * 8 + c])) for c in range(8)] for r in range(8)]

    # Step 6: Zig-Zag ordering
    zigzag_seq = block_to_zigzag(quantized_block)

    # Step 7: RLE Tokens
    rle_tokens = []
    dc_diff = zigzag_seq[0]
    cat, bits = encode_amplitude(dc_diff)
    rle_tokens.append({'type': 'DC', 'run': 0, 'cat': cat, 'val': dc_diff, 'bits': bits})

    zero_run = 0
    for k in range(1, 64):
        val = zigzag_seq[k]
        if val == 0:
            zero_run += 1
        else:
            while zero_run >= 16:
                rle_tokens.append({'type': 'ZRL', 'run': 16, 'cat': 0, 'val': 0, 'bits': 0})
                zero_run -= 16
            cat, bits = encode_amplitude(val)
            rle_tokens.append({'type': 'AC', 'run': zero_run, 'cat': cat, 'val': val, 'bits': bits, 'k': k})
            zero_run = 0
    if zero_run > 0:
        rle_tokens.append({'type': 'EOB', 'run': zero_run, 'cat': 0, 'val': 0, 'bits': 0})

    # Step 8: Dequantization
    dequantized_block = [[float(quantized_block[r][c] * q_table[r * 8 + c]) for c in range(8)] for r in range(8)]

    # Step 9: 2D-IDCT
    idct_spatial = idct_2d_8x8(dequantized_block)

    # Step 10: Unshift (+128) & clamp
    reconstructed_block = [[max(0, min(255, int(round(idct_spatial[r][c] + 128.0)))) for c in range(8)] for r in range(8)]

    # Difference / error
    error_block = [[abs(orig_block[r][c] - reconstructed_block[r][c]) for c in range(8)] for r in range(8)]
    mse = sum((orig_block[r][c] - reconstructed_block[r][c]) ** 2 for r in range(8) for c in range(8)) / 64.0
    psnr = 10.0 * math.log10((255.0 ** 2) / (mse + 1e-10)) if mse > 0 else 99.9

    return {
        'block_y': block_y,
        'block_x': block_x,
        'quality': quality,
        'orig_block': orig_block,
        'shifted_block': shifted_block,
        'dct_block': dct_block,
        'q_table': q_table,
        'quantized_block': quantized_block,
        'zigzag_seq': zigzag_seq,
        'rle_tokens': rle_tokens,
        'dequantized_block': dequantized_block,
        'idct_spatial': idct_spatial,
        'reconstructed_block': reconstructed_block,
        'error_block': error_block,
        'mse': mse,
        'psnr': psnr
    }


# =====================================================================
# 8. Educational Step-by-Step Visualization & Plotting Suite
# =====================================================================

def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Applies modern clean typography and border styling to matplotlib axes."""
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#1e293b")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, fontweight="medium", color="#475569")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, fontweight="medium", color="#475569")
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)


def plot_01_color_space(image: JPEGImage, save_path: str = None):
    """
    Steg 1: Fargeromskonvertering (RGB -> YCbCr).
    Viser hvordan fargebildet separeres i Luminans (Y) og Krominans (Cb, Cr).
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), facecolor="#f8fafc")
    fig.suptitle("Steg 1: Fargeromskonvertering / Color Space Transformation (RGB -> YCbCr)",
                 fontsize=14, fontweight="bold", color="#0f172a", y=0.98)

    arr = image.to_numpy()
    if image.channels == 1:
        # Grayscale image
        axes[0, 0].imshow(arr, cmap="gray")
        _style_axes(axes[0, 0], "Original Gråtone (Y)")
        for ax in [axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 1], axes[1, 2]]:
            ax.axis("off")
        axes[1, 0].imshow(arr, cmap="gray")
        _style_axes(axes[1, 0], "Luminans Y (100% Signal)")
        axes[1, 3].bar(["Y (Lys)"], [100.0], color=["#334155"], width=0.4)
        axes[1, 3].set_ylim(0, 105)
        _style_axes(axes[1, 3], "Variansfordeling", ylabel="%")
    else:
        R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = 128.0 - 0.168736 * R - 0.331264 * G + 0.5 * B
        Cr = 128.0 + 0.5 * R - 0.418688 * G - 0.081312 * B

        axes[0, 0].imshow(arr)
        _style_axes(axes[0, 0], "Original RGB Bilde")

        axes[0, 1].imshow(R, cmap="Reds")
        _style_axes(axes[0, 1], "Rød Kanal (R)")

        axes[0, 2].imshow(G, cmap="Greens")
        _style_axes(axes[0, 2], "Grønn Kanal (G)")

        axes[0, 3].imshow(B, cmap="Blues")
        _style_axes(axes[0, 3], "Blå Kanal (B)")

        axes[1, 0].imshow(Y, cmap="gray")
        _style_axes(axes[1, 0], "Y: Luminans (Lysstyrke)\nHolder ~80% av opplevd detalj")

        im_cb = axes[1, 1].imshow(Cb, cmap="coolwarm", vmin=0, vmax=255)
        _style_axes(axes[1, 1], "Cb: Blå-differanse\n(Chrominance Blue)")
        plt.colorbar(im_cb, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im_cr = axes[1, 2].imshow(Cr, cmap="coolwarm", vmin=0, vmax=255)
        _style_axes(axes[1, 2], "Cr: Rød-differanse\n(Chrominance Red)")
        plt.colorbar(im_cr, ax=axes[1, 2], fraction=0.046, pad=0.04)

        var_y = float(np.var(Y))
        var_cb = float(np.var(Cb))
        var_cr = float(np.var(Cr))
        tot_var = var_y + var_cb + var_cr + 1e-10
        pcts = [100.0 * var_y / tot_var, 100.0 * var_cb / tot_var, 100.0 * var_cr / tot_var]

        bars = axes[1, 3].bar(["Y (Lys)", "Cb (Blå)", "Cr (Rød)"], pcts,
                              color=["#334155", "#3b82f6", "#ef4444"], width=0.55)
        axes[1, 3].set_ylim(0, 100)
        axes[1, 3].grid(axis="y", linestyle="--", alpha=0.5)
        _style_axes(axes[1, 3], "Signalkontrast / Variansfordeling (%)", ylabel="% av Total Varians")
        for b in bars:
            h = b.get_height()
            axes[1, 3].text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"{h:.1f}%",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_02_block_partition(image: JPEGImage, block_y: int = None, block_x: int = None, save_path: str = None):
    """
    Steg 2: Blokkdeling (8x8 Pixelblokker / MCUer).
    Viser 8x8 rutenettet over bildet og zoomer inn på en valgt blokk med tallverdier.
    """
    arr = image.to_numpy()
    if image.channels == 3:
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    else:
        gray = arr.astype(np.float32)

    h, w = gray.shape
    bdata = extract_block_pipeline_data(image, block_y, block_x)
    by, bx = bdata['block_y'], bdata['block_x']
    y0, x0 = by * 8, bx * 8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor="#f8fafc",
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("Steg 2: Blokkdeling / 8x8 Spatial Macroblock Partitioning",
                 fontsize=14, fontweight="bold", color="#0f172a", y=0.98)

    crop_size = min(h, w, 256)
    cy0 = max(0, min(h - crop_size, y0 - crop_size // 4))
    cx0 = max(0, min(w - crop_size, x0 - crop_size // 4))
    view_crop = arr[cy0 : cy0 + crop_size, cx0 : cx0 + crop_size]

    if image.channels == 3:
        ax1.imshow(view_crop)
    else:
        ax1.imshow(view_crop, cmap="gray")

    for gy in range(0, crop_size, 8):
        ax1.axhline(gy, color="white", linewidth=0.3, alpha=0.4)
    for gx in range(0, crop_size, 8):
        ax1.axvline(gx, color="white", linewidth=0.3, alpha=0.4)

    rect = plt.Rectangle((x0 - cx0, y0 - cy0), 8, 8, fill=False, edgecolor="#f59e0b", linewidth=2.5)
    ax1.add_patch(rect)
    _style_axes(ax1, f"Bildeutsnitt med 8x8 Rutenett (Markert Blokk: Rad {by}, Kolonne {bx})")

    block_pixels = np.array(bdata['orig_block'])
    im2 = ax2.imshow(block_pixels, cmap="Blues", vmin=0, vmax=255)
    _style_axes(ax2, "Forstørret 8x8 Pikselblokk med Intensitet [0..255]")
    for r in range(8):
        for c in range(8):
            val = int(round(block_pixels[r, c]))
            tc = "white" if val > 140 else "black"
            ax2.text(c, r, str(val), ha="center", va="center", color=tc, fontsize=7.5, fontweight="bold")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_03_level_shift(orig_block, shifted_block, save_path: str = None):
    """
    Steg 3: Nivåforskyvning (Level Shifting: -128).
    Sentrerer intensitetsområdet rundt 0 slik at DC-komponenten representerer avvik.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#f8fafc",
                             gridspec_kw={"width_ratios": [1, 1, 1.2]})
    fig.suptitle("Steg 3: Nivåforskyvning / Level Shifting: f(x, y) - 128",
                 fontsize=14, fontweight="bold", color="#0f172a", y=0.98)

    arr_orig = np.array(orig_block)
    arr_shift = np.array(shifted_block)

    im1 = axes[0].imshow(arr_orig, cmap="gray", vmin=0, vmax=255)
    _style_axes(axes[0], f"Før: Område [0..255]\nSnitt = {np.mean(arr_orig):.1f}")
    for r in range(8):
        for c in range(8):
            val = int(round(arr_orig[r, c]))
            tc = "white" if val < 128 else "black"
            axes[0].text(c, r, str(val), ha="center", va="center", color=tc, fontsize=7.5)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(arr_shift, cmap="coolwarm", vmin=-128, vmax=127)
    _style_axes(axes[1], f"Etter (-128): Område [-128..127]\nSnitt = {np.mean(arr_shift):.1f}")
    for r in range(8):
        for c in range(8):
            val = int(round(arr_shift[r, c]))
            axes[1].text(c, r, str(val), ha="center", va="center", color="black", fontsize=7.5)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].hist(arr_orig.flatten(), bins=16, range=(0, 256), alpha=0.6,
                 color="#0284c7", label="Før (Usignert [0..255])")
    axes[2].hist(arr_shift.flatten(), bins=16, range=(-128, 128), alpha=0.6,
                 color="#ef4444", label="Etter (Signert [-128..127])")
    axes[2].axvline(0, color="#1e293b", linestyle="--", linewidth=1.2, label="Senter (0)")
    axes[2].legend(fontsize=8.5, loc="upper right")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    _style_axes(axes[2], "Verdifordeling før og etter nivåforskyvning",
                xlabel="Pikselverdi", ylabel="Antall Piksler")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_04_dct_basis_functions(save_path: str = None):
    """
    Steg 4A: 2D-DCT Basisfunksjoner (De 64 ortogonale cosinus-bølgene).
    Viser alle 64 frekvensmønstre som bygger opp ethvert 8x8 bilde.
    """
    fig, axes = plt.subplots(8, 8, figsize=(9.5, 9.5), facecolor="#0f172a")
    fig.suptitle("Steg 4A: De 64 2D-DCT Basisfunksjonene (Frekvensmønstre)",
                 fontsize=14, fontweight="bold", color="#f8fafc", y=0.98)

    for v in range(8):
        for u in range(8):
            ax = axes[v, u]
            basis = np.zeros((8, 8))
            alpha_u = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
            alpha_v = 1.0 / math.sqrt(2.0) if v == 0 else 1.0
            for y in range(8):
                for x in range(8):
                    basis[y, x] = 0.25 * alpha_u * alpha_v * \
                                  math.cos((2 * x + 1) * u * math.pi / 16.0) * \
                                  math.cos((2 * y + 1) * v * math.pi / 16.0)
            ax.imshow(basis, cmap="gray", vmin=-0.5, vmax=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if v == 0:
                ax.set_title(f"u={u}", fontsize=8, color="#38bdf8", pad=2)
            if u == 0:
                ax.set_ylabel(f"v={v}", fontsize=8, color="#38bdf8", labelpad=2)
            for spine in ax.spines.values():
                if u == 0 and v == 0:
                    spine.set_color("#f59e0b")
                    spine.set_linewidth(2.0)
                else:
                    spine.set_color("#334155")
                    spine.set_linewidth(0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_05_dct_coefficients(shifted_block, dct_block, save_path: str = None):
    """
    Steg 4B: 2D Discrete Cosine Transform (DCT) Koeffisienter.
    Viser overgangen til frekvensdomenet og energikonsentrasjon (Energy Compaction).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="#f8fafc",
                                        gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
    fig.suptitle("Steg 4B: 2D-DCT Frekvenstransformasjon og Energikomprimering",
                 fontsize=14, fontweight="bold", color="#0f172a", y=0.98)

    arr_s = np.array(shifted_block)
    arr_d = np.array(dct_block)

    im1 = ax1.imshow(arr_s, cmap="coolwarm", vmin=-128, vmax=127)
    _style_axes(ax1, "Romlig Blokk f(x, y)\n(Nivåforskjøvet [-128..127])")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    v_max = max(abs(arr_d.min()), abs(arr_d.max()), 10.0)
    im2 = ax2.imshow(arr_d, cmap="PuOr", vmin=-v_max, vmax=v_max)
    _style_axes(ax2, "2D-DCT Frekvensmatrise F(u, v)\n(DC øverst til venstre, AC økende)")
    for r in range(8):
        for c in range(8):
            val = arr_d[r, c]
            txt = f"{val:.0f}"
            color = "white" if abs(val) > v_max * 0.45 else "black"
            ax2.text(c, r, txt, ha="center", va="center", color=color, fontsize=7.2, fontweight="bold")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    magnitudes = np.sort(np.abs(arr_d.flatten()))[::-1]
    cum_energy = np.cumsum(magnitudes ** 2) / np.sum(magnitudes ** 2) * 100.0

    ax3_twin = ax3.twinx()
    ax3.bar(range(1, 65), magnitudes, color="#6366f1", width=0.7, alpha=0.7, label="Koeffisient-størrelse")
    ax3_twin.plot(range(1, 65), cum_energy, color="#ef4444", linewidth=2.0, label="Kumulativ Energi (%)")
    ax3_twin.set_ylim(0, 105)
    ax3_twin.set_ylabel("Kumulativ Energi (%)", color="#ef4444", fontsize=8.5)
    _style_axes(ax3, "Energikonsentrasjon (De første få koeffisientene dominerer)",
                xlabel="Sortert Koeffisient Indeks (1-64)", ylabel="Absolutt Størrelse |F|")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_06_quantization(dct_block, q_table, quantized_block, quality: int = 50, save_path: str = None):
    """
    Steg 5: Kvantisering (Quantization: F_Q = round(F / Q_table)).
    Viser divisjon med kvantiseringstabellen som skaper de mange null-koeffisientene.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), facecolor="#f8fafc")
    fig.suptitle(f"Steg 5: Kvantisering ved Kvalitetsfaktor Q={quality} (Det tapsbringende komprimeringstrinnet)",
                 fontsize=14, fontweight="bold", color="#0f172a", y=0.98)

    arr_d = np.array(dct_block)
    arr_q = np.array(q_table).reshape((8, 8))
    arr_out = np.array(quantized_block)

    v_max = max(abs(arr_d.min()), abs(arr_d.max()), 10.0)
    axes[0].imshow(arr_d, cmap="PuOr", vmin=-v_max, vmax=v_max)
    _style_axes(axes[0], "1. DCT Koeffisienter F(u, v)")
    for r in range(8):
        for c in range(8):
            axes[0].text(c, r, f"{arr_d[r, c]:.0f}", ha="center", va="center", fontsize=6.8)

    im_q = axes[1].imshow(arr_q, cmap="YlOrRd")
    _style_axes(axes[1], f"2. Kvantiseringstabell Q(u, v)\n(Kvalitet = {quality})")
    for r in range(8):
        for c in range(8):
            axes[1].text(c, r, str(arr_q[r, c]), ha="center", va="center", fontsize=7.2, fontweight="bold")
    plt.colorbar(im_q, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(arr_out != 0, cmap="Blues", vmin=0, vmax=1.5)
    _style_axes(axes[2], "3. Kvantisert F_Q = round(F / Q)\n(Hvite felt = 0, Blå felt = Ikke-null)")
    zero_count = 0
    for r in range(8):
        for c in range(8):
            val = arr_out[r, c]
            if val == 0:
                zero_count += 1
                axes[2].text(c, r, "0", ha="center", va="center", color="#94a3b8", fontsize=7.5)
            else:
                axes[2].text(c, r, str(val), ha="center", va="center", color="#0f172a", fontsize=7.5, fontweight="bold")

    qualities = [10, 50, 90]
    zero_pcts = []
    for q in qualities:
        qt = scale_quant_table(STD_LUM_QUANT_TABLE, q)
        q_b = [[int(round(dct_block[r][c] / qt[r * 8 + c])) for c in range(8)] for r in range(8)]
        z_pct = 100.0 * sum(1 for r in range(8) for c in range(8) if q_b[r][c] == 0) / 64.0
        zero_pcts.append(z_pct)

    bars = axes[3].bar([f"Q={q}" for q in qualities], zero_pcts, color=["#10b981", "#3b82f6", "#f59e0b"], width=0.55)
    axes[3].set_ylim(0, 105)
    axes[3].grid(axis="y", linestyle="--", alpha=0.5)
    _style_axes(axes[3], f"Null-koeffisienter (Sparsity)\nDenne blokken: {zero_count}/64 ({100*zero_count/64:.0f}%)",
                ylabel="% Koeffisienter som er 0")
    for b in bars:
        h = b.get_height()
        axes[3].text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_07_zigzag_and_rle(quantized_block, zigzag_seq, rle_tokens, save_path: str = None):
    """
    Steg 6: Sikksakk-avlesning og Run-Length Encoding (RLE).
    Viser 2D -> 1D linjærisering som samler nullene foran EOB (End of Block).
    """
    fig = plt.figure(figsize=(15, 6.5), facecolor="#f8fafc")
    fig.suptitle("Steg 6: Sikksakk-skanning (Zig-Zag) og Run-Length Koding (RLE)",
                 fontsize=14, fontweight="bold", color="#0f172a", y=0.98)

    ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    ax3 = plt.subplot2grid((2, 3), (1, 1), colspan=2)

    arr_out = np.array(quantized_block)
    ax1.imshow(arr_out != 0, cmap="Blues", alpha=0.3)
    _style_axes(ax1, "8x8 Sikksakk-mønster (Fra DC til høyfrekvent AC)")

    for k in range(63):
        r1, c1 = ZIGZAG_COORDS[k]
        r2, c2 = ZIGZAG_COORDS[k + 1]
        ax1.annotate("", xy=(c2, r2), xytext=(c1, r1),
                     arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.3, alpha=0.8))

    for k, (r, c) in enumerate(ZIGZAG_COORDS):
        val = arr_out[r, c]
        txt = f"{val}" if val != 0 else "."
        ax1.text(c, r, txt, ha="center", va="center",
                 color="#0f172a" if val != 0 else "#94a3b8", fontsize=7.5, fontweight="bold")

    colors = ["#f59e0b" if k == 0 else ("#3b82f6" if val != 0 else "#cbd5e1")
              for k, val in enumerate(zigzag_seq)]
    ax2.bar(range(64), zigzag_seq, color=colors, width=0.8)
    _style_axes(ax2, "1D Linjærisert Koeffisientsekvens (Index 0 til 63)",
                xlabel="Sikksakk-indeks k", ylabel="Kvantisert Verdi")
    ax2.axhline(0, color="#64748b", lw=0.8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    ax3.axis("off")
    token_strs = []
    for tok in rle_tokens[:12]:
        if tok['type'] == 'DC':
            token_strs.append(f"DC: Diff={tok['val']} (Cat {tok['cat']})")
        elif tok['type'] == 'AC':
            token_strs.append(f"({tok['run']} nuller, Val={tok['val']})")
        elif tok['type'] == 'EOB':
            token_strs.append("EOB (Resten er 0)")
        elif tok['type'] == 'ZRL':
            token_strs.append("ZRL (16 nuller)")

    tokens_formatted = "  ->  ".join(token_strs)
    if len(rle_tokens) > 12:
        tokens_formatted += f" ... (+ {len(rle_tokens) - 12} symboler)"

    summary_text = (
        f"Antall koeffisienter: 64  |  Ikke-null koeffisienter: {sum(1 for v in zigzag_seq if v != 0)}\n"
        f"Kompakt RLE-representasjon:\n{tokens_formatted}\n\n"
        f"Merk: EOB (End of Block) avslutter kodingen umiddelbart så snart alle gjenværende koeffisienter er 0."
    )
    ax3.text(0.02, 0.5, summary_text, fontsize=9.5, color="#1e293b",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#e2e8f0", edgecolor="#cbd5e1", alpha=0.8),
             va="center", transform=ax3.transAxes)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_08_reconstruction_and_error(orig_block, dequantized_block, reconstructed_block, save_path: str = None):
    """
    Steg 7 & 8: Dekvantisering, Invers 2D-DCT (IDCT) og Rekonstruksjonsfeil.
    Sammenligner original blokk mot rekonstruert blokk og viser feilkartet (residua).
    """
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), facecolor="#f8fafc")
    fig.suptitle("Steg 7 & 8: Dekvantisering, 2D-IDCT og Piksel-for-Piksel Rekonstruksjonsfeil",
                 fontsize=14, fontweight="bold", color="#0f172a", y=0.98)

    arr_orig = np.array(orig_block)
    arr_deq = np.array(dequantized_block)
    arr_rec = np.array(reconstructed_block)
    arr_err = np.abs(arr_orig - arr_rec)

    mse = float(np.mean((arr_orig - arr_rec) ** 2))
    psnr = 10.0 * math.log10((255.0 ** 2) / (mse + 1e-10)) if mse > 0 else 99.9

    im1 = axes[0].imshow(arr_orig, cmap="gray", vmin=0, vmax=255)
    _style_axes(axes[0], "1. Original 8x8 Blokk f(x, y)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    v_max = max(abs(arr_deq.min()), abs(arr_deq.max()), 10.0)
    im2 = axes[1].imshow(arr_deq, cmap="PuOr", vmin=-v_max, vmax=v_max)
    _style_axes(axes[1], "2. Dekvantisert DCT F_hat = F_Q * Q")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(arr_rec, cmap="gray", vmin=0, vmax=255)
    _style_axes(axes[2], "3. Rekonstruert Blokk (IDCT + 128)")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    max_err = max(1.0, float(arr_err.max()))
    im4 = axes[3].imshow(arr_err, cmap="Reds", vmin=0, vmax=max(15.0, max_err))
    _style_axes(axes[3], f"4. Absolutt Feil |f - f_hat|\nMSE = {mse:.2f}, PSNR = {psnr:.1f} dB")
    for r in range(8):
        for c in range(8):
            val = arr_err[r, c]
            axes[3].text(c, r, f"{val:.0f}", ha="center", va="center",
                         color="white" if val > 8 else "black", fontsize=7.2, fontweight="bold")
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_09_quality_comparison(image: JPEGImage, qualities: list = None, save_path: str = None):
    """
    Steg 9: Kvalitetsfaktor, Komprimeringsgrad og Rate-Distortion.
    Sammenligner rekonstruert bilde og kurver for Q=10, 30, 50, 80, 95.
    """
    if qualities is None:
        qualities = [10, 30, 50, 80, 95]

    orig_np = image.to_numpy()
    h, w = orig_np.shape[:2]
    crop_size = min(h, w, 256)
    cy0 = (h - crop_size) // 2
    cx0 = (w - crop_size) // 2
    crop_pixels = orig_np[cy0 : cy0 + crop_size, cx0 : cx0 + crop_size]

    if image.channels == 1:
        test_img = JPEGImage(crop_size, crop_size, 1, crop_pixels.tolist())
    else:
        test_img = JPEGImage(crop_size, crop_size, 3, crop_pixels.tolist())

    raw_bytes = len(test_img.to_bytes())
    results = []

    fig = plt.figure(figsize=(16, 7.5), facecolor="#f8fafc")
    fig.suptitle("Steg 9: Kvalitetsfaktor-dynamikk, Bildekvalitet og Rate-Distortion Kurver",
                 fontsize=14, fontweight="bold", color="#0f172a", y=0.98)

    n_q = len(qualities)
    for idx, q in enumerate(qualities):
        ax = fig.add_subplot(2, n_q, idx + 1)
        comp = compress(test_img, quality=q)
        dec = decompress(comp)
        dec_np = dec.to_numpy()

        mse = float(np.mean((crop_pixels.astype(np.float64) - dec_np.astype(np.float64)) ** 2))
        psnr = 10.0 * math.log10((255.0 ** 2) / (mse + 1e-10)) if mse > 0 else 99.9
        ratio = raw_bytes / max(1, len(comp))

        results.append({
            'q': q,
            'size_kb': len(comp) / 1024.0,
            'mse': mse,
            'psnr': psnr,
            'ratio': ratio
        })

        if image.channels == 3:
            ax.imshow(dec_np)
        else:
            ax.imshow(dec_np, cmap="gray")
        _style_axes(ax, f"Q = {q}\n{len(comp)/1024.0:.1f} KB | {ratio:.1f}:1\nPSNR: {psnr:.1f} dB")
        ax.axis("off")

    ax_size = fig.add_subplot(2, 3, 4)
    ax_psnr = fig.add_subplot(2, 3, 5)
    ax_ratio = fig.add_subplot(2, 3, 6)

    qs = [r['q'] for r in results]
    sizes = [r['size_kb'] for r in results]
    psnrs = [r['psnr'] for r in results]
    ratios = [r['ratio'] for r in results]

    ax_size.plot(qs, sizes, "o-", color="#3b82f6", linewidth=2.0, markersize=6)
    _style_axes(ax_size, "Filstørrelse vs. Kvalitetsfaktor", xlabel="Kvalitet Q (1..100)", ylabel="Størrelse (KB)")
    ax_size.grid(True, linestyle="--", alpha=0.5)

    ax_psnr.plot(qs, psnrs, "s-", color="#10b981", linewidth=2.0, markersize=6)
    _style_axes(ax_psnr, "Bildekvalitet (PSNR) vs. Kvalitet", xlabel="Kvalitet Q (1..100)", ylabel="PSNR (dB)")
    ax_psnr.grid(True, linestyle="--", alpha=0.5)

    ax_ratio.plot(qs, ratios, "^-", color="#f59e0b", linewidth=2.0, markersize=6)
    _style_axes(ax_ratio, "Komprimeringsgrad (x:1) vs. Kvalitet", xlabel="Kvalitet Q (1..100)", ylabel="Komprimeringsgrad (x:1)")
    ax_ratio.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def plot_00_pipeline_master_overview(image: JPEGImage, block_data: dict = None, save_path: str = None):
    """
    Komplett Master-oversikt: Viser hele JPEG-flyten fra piksel til bitstrøm og tilbake.
    """
    if block_data is None:
        block_data = extract_block_pipeline_data(image, quality=50)

    fig = plt.figure(figsize=(18, 9.5), facecolor="#f8fafc")
    fig.suptitle("Komplett JPEG Pipeline Oversikt (Fra Pikseldata til Frekvenser, Bitstrøm og Rekonstruksjon)",
                 fontsize=15, fontweight="bold", color="#0f172a", y=0.98)

    ax1 = fig.add_subplot(3, 3, 1)
    arr = image.to_numpy()
    if image.channels == 3:
        ax1.imshow(arr)
    else:
        ax1.imshow(arr, cmap="gray")
    by, bx = block_data['block_y'], block_data['block_x']
    rect = plt.Rectangle((bx * 8, by * 8), 8, 8, fill=False, edgecolor="#ef4444", linewidth=2.5)
    ax1.add_patch(rect)
    _style_axes(ax1, f"1. Inndatabilde (Markert 8x8: Rad {by}, Kol {bx})")

    ax2 = fig.add_subplot(3, 3, 2)
    im2 = ax2.imshow(block_data['orig_block'], cmap="Blues", vmin=0, vmax=255)
    _style_axes(ax2, "2. 8x8 Romlig Blokk f(x, y) [0..255]")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(3, 3, 3)
    im3 = ax3.imshow(block_data['shifted_block'], cmap="coolwarm", vmin=-128, vmax=127)
    _style_axes(ax3, "3. Nivåforskjøvet f(x, y) - 128 [-128..127]")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(3, 3, 4)
    arr_d = np.array(block_data['dct_block'])
    v_max = max(abs(arr_d.min()), abs(arr_d.max()), 10.0)
    im4 = ax4.imshow(arr_d, cmap="PuOr", vmin=-v_max, vmax=v_max)
    _style_axes(ax4, "4. 2D-DCT Frekvenser F(u, v)")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(3, 3, 5)
    arr_q = np.array(block_data['q_table']).reshape((8, 8))
    im5 = ax5.imshow(arr_q, cmap="YlOrRd")
    _style_axes(ax5, f"5. Q-Tabell (Q={block_data['quality']})")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(3, 3, 6)
    arr_qb = np.array(block_data['quantized_block'])
    im6 = ax6.imshow(arr_qb != 0, cmap="Blues", vmin=0, vmax=1.5)
    _style_axes(ax6, "6. Kvantisert F_Q (Hvite felt = 0)")
    for r in range(8):
        for c in range(8):
            val = arr_qb[r, c]
            txt = str(val) if val != 0 else "0"
            color = "#0f172a" if val != 0 else "#94a3b8"
            ax6.text(c, r, txt, ha="center", va="center", color=color, fontsize=7.2, fontweight="bold")

    ax7 = fig.add_subplot(3, 3, 7)
    zz = block_data['zigzag_seq']
    ax7.bar(range(64), zz, color=["#f59e0b" if k == 0 else ("#3b82f6" if v != 0 else "#cbd5e1") for k, v in enumerate(zz)])
    _style_axes(ax7, "7. Sikksakk 1D Rekke (Lang hale av 0-ere)", xlabel="Sikksakk k (0..63)", ylabel="Verdi")
    ax7.grid(axis="y", linestyle="--", alpha=0.4)

    ax8 = fig.add_subplot(3, 3, 8)
    im8 = ax8.imshow(block_data['reconstructed_block'], cmap="Blues", vmin=0, vmax=255)
    _style_axes(ax8, "8. Rekonstruert Blokk (IDCT + 128)")
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    ax9 = fig.add_subplot(3, 3, 9)
    im9 = ax9.imshow(block_data['error_block'], cmap="Reds", vmin=0, vmax=15)
    _style_axes(ax9, f"9. Feil |f - f_hat|\nMSE: {block_data['mse']:.2f} | PSNR: {block_data['psnr']:.1f} dB")
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    return fig


def generate_all_plots(image_or_path=None, output_dir: str = "C:/dev/jpeg_plots",
                       quality: int = 50, block_y: int = None, block_x: int = None):
    """
    Kjører hele visualiseringspipelinen og lagrer alle de 10 detaljerte plottene i output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Genererer pedagogiske JPEG-plott til: {output_dir}")

    if image_or_path is None or (isinstance(image_or_path, str) and not os.path.exists(image_or_path)):
        if isinstance(image_or_path, str):
            print(f"Merk: Fant ikke '{image_or_path}'. Forsøker 'C:/dev/bilde.jpg'...")
        if os.path.exists("C:/dev/bilde.jpg"):
            image = JPEGImage.from_file("C:/dev/bilde.jpg")
        else:
            print("Genererer syntetisk testbilde med gradient og mønster...")
            w, h = 128, 128
            pixels = [
                [[int(128 + 60 * math.sin(x / 8.0) + 40 * math.cos(y / 8.0)),
                  int((x * 2) % 256),
                  int((y * 2) % 256)] for x in range(w)]
                for y in range(h)
            ]
            image = JPEGImage(w, h, 3, pixels)
    elif isinstance(image_or_path, str):
        image = JPEGImage.from_file(image_or_path)
    elif isinstance(image_or_path, JPEGImage):
        image = image_or_path
    else:
        raise ValueError("Ugyldig bildetype for generering av plott")

    bdata = extract_block_pipeline_data(image, block_y=block_y, block_x=block_x, quality=quality)
    print(f"Analyserer blokk ved posisjon: Rad {bdata['block_y']}, Kolonne {bdata['block_x']}")

    saved_files = []

    p1 = os.path.join(output_dir, "01_color_space_conversion.png")
    plot_01_color_space(image, p1)
    saved_files.append(p1)
    print("  [1/10] Lagret 01_color_space_conversion.png")

    p2 = os.path.join(output_dir, "02_block_partition_8x8.png")
    plot_02_block_partition(image, bdata['block_y'], bdata['block_x'], p2)
    saved_files.append(p2)
    print("  [2/10] Lagret 02_block_partition_8x8.png")

    p3 = os.path.join(output_dir, "03_level_shifting.png")
    plot_03_level_shift(bdata['orig_block'], bdata['shifted_block'], p3)
    saved_files.append(p3)
    print("  [3/10] Lagret 03_level_shifting.png")

    p4 = os.path.join(output_dir, "04_dct_basis_functions.png")
    plot_04_dct_basis_functions(p4)
    saved_files.append(p4)
    print("  [4/10] Lagret 04_dct_basis_functions.png")

    p5 = os.path.join(output_dir, "05_dct_coefficients.png")
    plot_05_dct_coefficients(bdata['shifted_block'], bdata['dct_block'], p5)
    saved_files.append(p5)
    print("  [5/10] Lagret 05_dct_coefficients.png")

    p6 = os.path.join(output_dir, "06_quantization_and_sparsity.png")
    plot_06_quantization(bdata['dct_block'], bdata['q_table'], bdata['quantized_block'], quality, p6)
    saved_files.append(p6)
    print("  [6/10] Lagret 06_quantization_and_sparsity.png")

    p7 = os.path.join(output_dir, "07_zigzag_scanning_and_rle.png")
    plot_07_zigzag_and_rle(bdata['quantized_block'], bdata['zigzag_seq'], bdata['rle_tokens'], p7)
    saved_files.append(p7)
    print("  [7/10] Lagret 07_zigzag_scanning_and_rle.png")

    p8 = os.path.join(output_dir, "08_inverse_dct_and_error.png")
    plot_08_reconstruction_and_error(bdata['orig_block'], bdata['dequantized_block'], bdata['reconstructed_block'], p8)
    saved_files.append(p8)
    print("  [8/10] Lagret 08_inverse_dct_and_error.png")

    p9 = os.path.join(output_dir, "09_quality_comparison_and_rate_distortion.png")
    plot_09_quality_comparison(image, [10, 30, 50, 80, 95], p9)
    saved_files.append(p9)
    print("  [9/10] Lagret 09_quality_comparison_and_rate_distortion.png")

    p0 = os.path.join(output_dir, "00_jpeg_full_pipeline_overview.png")
    plot_00_pipeline_master_overview(image, bdata, p0)
    saved_files.append(p0)
    print("  [10/10] Lagret 00_jpeg_full_pipeline_overview.png")

    print("\nAlle 10 pedagogiske plott er generert og lagret!")
    return saved_files


# =====================================================================
# 9. Unit and Integration Test Suite
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
        # Note: Discontinuous modulo wrapped gradient has standard baseline DCT loss around 13.7
        mse = sum(
            (pixels[r][c] - decompressed.pixels[r][c]) ** 2
            for r in range(h) for c in range(w)
        ) / (w * h)
        self.assertLess(mse, 15.0)

        # Continuous smooth gradient achieves near-lossless reconstruction (MSE < 2.0)
        smooth_pixels = [[r * 8 + c * 8 for c in range(w)] for r in range(h)]
        smooth_img = JPEGImage(w, h, 1, smooth_pixels)
        smooth_comp = compress(smooth_img, quality=85)
        smooth_dec = decompress(smooth_comp)
        smooth_mse = sum(
            (smooth_pixels[r][c] - smooth_dec.pixels[r][c]) ** 2
            for r in range(h) for c in range(w)
        ) / (w * h)
        self.assertLess(smooth_mse, 2.0)

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

    def test_pipeline_block_extraction(self):
        """Validates that extract_block_pipeline_data retrieves consistent metrics."""
        w, h = 32, 32
        pixels = [[(r * 7 + c * 5) % 256 for c in range(w)] for r in range(h)]
        img = JPEGImage(w, h, 1, pixels)
        data = extract_block_pipeline_data(img, block_y=0, block_x=0, quality=75)

        self.assertEqual(len(data['orig_block']), 8)
        self.assertEqual(len(data['zigzag_seq']), 64)
        self.assertGreater(data['psnr'], 20.0)


# =====================================================================
# 10. Command Line & Test Runner Interface
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pure Python Baseline JPEG Codec with Educational Step-by-Step Visualizations"
    )
    parser.add_argument("--test", action="store_true", help="Kjør unittest-testsuiten")
    parser.add_argument("--plot", nargs="?", const="C:/dev/bilde.jpg", default=None,
                        help="Generer pedagogiske plott for alle trinn for oppgitt bilde (standard: C:/dev/bilde.jpg)")
    parser.add_argument("--quality", "-q", type=int, default=50, help="JPEG kvalitetsfaktor (1-100, standard: 50)")
    parser.add_argument("--outdir", "-o", type=str, default="C:/dev/jpeg_plots",
                        help="Mappe for lagring av plott (standard: C:/dev/jpeg_plots)")
    parser.add_argument("--block-y", type=int, default=None, help="Y-indeks for 8x8 blokk")
    parser.add_argument("--block-x", type=int, default=None, help="X-indeks for 8x8 blokk")
    parser.add_argument("--demo", action="store_true", help="Kjør både tester og generer alle demonstrasjonsplott")

    args = parser.parse_args()

    # Hvis ingen argumenter er gitt, eller --test/--demo
    if len(sys.argv) == 1 or args.test or args.demo:
        print("=" * 70)
        print("Running Pure-Python JPEG Codec Test Suite (ITU-T T.81 / JFIF)")
        print("=" * 70)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestJPEGCodec)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if not result.wasSuccessful():
            sys.exit(1)

        if len(sys.argv) == 1:
            print("\n[Info] Kjører automatisk demonstrasjonsplott (--plot) for C:/dev/bilde.jpg...")
            generate_all_plots("C:/dev/bilde.jpg", output_dir=args.outdir, quality=args.quality)
            return

    if args.demo or args.plot is not None:
        img_path = args.plot if args.plot else "C:/dev/bilde.jpg"
        generate_all_plots(img_path, output_dir=args.outdir, quality=args.quality,
                           block_y=args.block_y, block_x=args.block_x)


if __name__ == '__main__':
    main()
