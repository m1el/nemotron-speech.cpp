#!/usr/bin/env python3
"""
Convert NeMo ASR checkpoint to GGUF-like binary format for C++ inference.

Usage:
    python scripts/convert_weights.py \
        ../nemotron-speech-streaming-en-0.6b/model_weights.ckpt \
        weights/model.bin

Format:
    Header:
        magic: "NEMO" (4 bytes)
        version: uint32 (4 bytes)
        n_tensors: uint32 (4 bytes)

    For each tensor:
        name_len: uint32 (4 bytes)
        name: char[name_len] (name_len bytes)
        n_dims: uint32 (4 bytes)
        dims: uint32[n_dims] (n_dims * 4 bytes)
        dtype: uint32 (4 bytes) - 0=f32, 1=f16
        data: float[product(dims)] (product(dims) * sizeof(dtype) bytes)
"""

import argparse
import struct
import numpy as np
import torch
from pathlib import Path


DTYPE_F32 = 0
DTYPE_F16 = 1


def write_tensor(f, name: str, tensor: np.ndarray, dtype: int = DTYPE_F32):
    """Write a single tensor to binary file."""
    name_bytes = name.encode('utf-8')

    # Write name
    f.write(struct.pack('<I', len(name_bytes)))
    f.write(name_bytes)

    # Write dimensions
    f.write(struct.pack('<I', len(tensor.shape)))
    for dim in tensor.shape:
        f.write(struct.pack('<I', dim))

    # Write dtype
    f.write(struct.pack('<I', dtype))

    # Write data
    if dtype == DTYPE_F32:
        data = tensor.astype(np.float32)
    else:  # DTYPE_F16
        data = tensor.astype(np.float16)

    f.write(data.tobytes())


def convert_checkpoint(ckpt_path: str, output_path: str, use_f16: bool = False):
    """Convert PyTorch checkpoint to binary format."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    print(f"Found {len(ckpt)} tensors")

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    dtype = DTYPE_F16 if use_f16 else DTYPE_F32

    with open(output_path, 'wb') as f:
        # Write header
        f.write(b'NEMO')  # magic
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', len(ckpt)))  # n_tensors

        # Write each tensor
        total_params = 0
        for name, tensor in ckpt.items():
            arr = tensor.numpy()
            write_tensor(f, name, arr, dtype)
            total_params += arr.size

        print(f"Total parameters: {total_params:,}")
        print(f"Written to: {output_path}")

    # Print file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


def print_checkpoint_info(ckpt_path: str):
    """Print detailed information about checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu')

    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Total tensors: {len(ckpt)}")
    print(f"{'='*60}\n")

    # Group by component
    groups = {}
    for name, tensor in ckpt.items():
        prefix = name.split('.')[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append((name, tensor.shape, tensor.numel()))

    total_params = 0
    for prefix in sorted(groups.keys()):
        tensors = groups[prefix]
        params = sum(t[2] for t in tensors)
        total_params += params
        print(f"\n{prefix} ({len(tensors)} tensors, {params:,} params)")
        print("-" * 40)
        for name, shape, numel in tensors[:5]:
            print(f"  {name}: {list(shape)}")
        if len(tensors) > 5:
            print(f"  ... and {len(tensors) - 5} more")

    print(f"\n{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Size (f32): {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"Size (f16): {total_params * 2 / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo checkpoint to binary format")
    parser.add_argument("checkpoint", help="Path to model_weights.ckpt")
    parser.add_argument("output", nargs="?", default="weights/model.bin",
                        help="Output path (default: weights/model.bin)")
    parser.add_argument("--f16", action="store_true", help="Use float16 (half precision)")
    parser.add_argument("--info", action="store_true", help="Only print checkpoint info")

    args = parser.parse_args()

    if args.info:
        print_checkpoint_info(args.checkpoint)
    else:
        convert_checkpoint(args.checkpoint, args.output, use_f16=args.f16)


if __name__ == "__main__":
    main()
