#!/usr/bin/env python3
"""
Compare binary tensor dumps from NeMo (Python) and ggml (C++) implementations.

Usage:
    python compare_tensors.py <file1> <file2>

Examples:
    python compare_tensors.py my_bin/nemo_pre_encode.bin my_bin/ggml_pre_encode.bin
    python compare_tensors.py my_bin/nemo_encoder_out.bin my_bin/ggml_encoder_out.bin

Both files should contain raw float32 binary data.
"""

import sys
import numpy as np
import argparse
import struct

def load_tensor(filepath: str) -> np.ndarray:
    """Load a binary file as a numpy array with the given shape."""
    with open(filepath, "rb") as f:
        header = f.read(32)
        ne = list(struct.unpack('4q', header))
        while ne[-1] == 1 and len(ne) > 1: ne.pop()
        shape = tuple(reversed(ne))
        data = np.fromfile(f, dtype=np.float32)

    expected_size = np.prod(shape)

    if len(data) < expected_size:
        raise ValueError(
            f"File {filepath} has {len(data)} elements, "
            f"but shape {shape} requires {expected_size} elements"
        )

    if len(data) > expected_size:
        # File may contain multiple chunks, compare first chunk only
        print(f"Note: {filepath} has {len(data)} elements, using first {expected_size} for shape {shape}")
        data = data[:expected_size]

    return data.reshape(shape)


def compare_tensors(
    tensor1: np.ndarray,
    tensor2: np.ndarray,
    name1: str = "tensor1",
    name2: str = "tensor2",
    rtol: float = 1e-4,
    atol: float = 1e-5
) -> dict:
    """
    Compare two tensors and return statistics.

    Args:
        tensor1: First tensor (e.g., NeMo output)
        tensor2: Second tensor (e.g., ggml output)
        name1: Name for first tensor
        name2: Name for second tensor
        rtol: Relative tolerance for np.allclose
        atol: Absolute tolerance for np.allclose

    Returns:
        Dictionary with comparison statistics
    """
    # Flatten for comparison
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()

    # Compute differences
    abs_diff = np.abs(flat1 - flat2)
    rel_diff = np.abs(flat1 - flat2) / (np.abs(flat1) + 1e-10)

    stats = {
        "shape": tensor1.shape,
        "n_elements": len(flat1),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "median_abs_diff": float(np.median(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "all_close": bool(np.allclose(flat1, flat2, rtol=rtol, atol=atol)),
        f"{name1}_min": float(np.min(flat1)),
        f"{name1}_max": float(np.max(flat1)),
        f"{name1}_mean": float(np.mean(flat1)),
        f"{name1}_std": float(np.std(flat1)),
        f"{name2}_min": float(np.min(flat2)),
        f"{name2}_max": float(np.max(flat2)),
        f"{name2}_mean": float(np.mean(flat2)),
        f"{name2}_std": float(np.std(flat2)),
    }

    # Find worst indices
    worst_idx = np.argmax(abs_diff)
    stats["worst_idx"] = int(worst_idx)
    stats["worst_val1"] = float(flat1[worst_idx])
    stats["worst_val2"] = float(flat2[worst_idx])

    return stats


def print_comparison(stats: dict, name1: str, name2: str) -> None:
    """Pretty print comparison statistics."""
    print("=" * 60)
    print(f"Tensor Comparison: {name1} vs {name2}")
    print("=" * 60)

    print(f"\nShape: {stats['shape']}")
    print(f"Elements: {stats['n_elements']}")

    print(f"\n--- Difference Statistics ---")
    print(f"Max absolute diff:    {stats['max_abs_diff']:.6e}")
    print(f"Mean absolute diff:   {stats['mean_abs_diff']:.6e}")
    print(f"Median absolute diff: {stats['median_abs_diff']:.6e}")
    print(f"Max relative diff:    {stats['max_rel_diff']:.6e}")
    print(f"Mean relative diff:   {stats['mean_rel_diff']:.6e}")

    print(f"\n--- {name1} Statistics ---")
    print(f"Min:  {stats[f'{name1}_min']:.6f}")
    print(f"Max:  {stats[f'{name1}_max']:.6f}")
    print(f"Mean: {stats[f'{name1}_mean']:.6f}")
    print(f"Std:  {stats[f'{name1}_std']:.6f}")

    print(f"\n--- {name2} Statistics ---")
    print(f"Min:  {stats[f'{name2}_min']:.6f}")
    print(f"Max:  {stats[f'{name2}_max']:.6f}")
    print(f"Mean: {stats[f'{name2}_mean']:.6f}")
    print(f"Std:  {stats[f'{name2}_std']:.6f}")

    print(f"\n--- Worst Mismatch ---")
    print(f"Index: {stats['worst_idx']}")
    print(f"{name1} value: {stats['worst_val1']:.6f}")
    print(f"{name2} value: {stats['worst_val2']:.6f}")

    print(f"\n--- Result ---")
    if stats['all_close']:
        print("PASS: Tensors are close (within tolerance)")
    else:
        print("FAIL: Tensors differ beyond tolerance")

    print("=" * 60)


def log_histogram(tensor1: np.ndarray, tensor2: np.ndarray) -> None:
    """
    Print a log-scale histogram of absolute differences.
    Shows what percentage of values fall into each order of magnitude bin.
    """
    abs_diff = np.abs(tensor1.flatten() - tensor2.flatten())
    n_total = len(abs_diff)

    # Handle exact matches (diff = 0)
    n_exact = np.sum(abs_diff == 0)
    nonzero_diff = abs_diff[abs_diff > 0]

    print("\n--- Log-scale Histogram of Differences ---")

    if n_exact > 0:
        print(f"Exact match (diff=0):  {n_exact:>8} ({100*n_exact/n_total:>6.2f}%)")

    if len(nonzero_diff) == 0:
        print("All values match exactly!")
        return

    # Determine range of log10 values
    log_diffs = np.log10(nonzero_diff)
    min_log = int(np.floor(np.min(log_diffs)))
    max_log = int(np.ceil(np.max(log_diffs)))

    # Create bins from 10^min_log to 10^max_log
    # Bins: [0, 10^-10), [10^-10, 10^-9), ..., [10^-1, 10^0), [10^0, inf)
    print(f"\n{'Diff Range':<25} {'Count':>10} {'Percent':>10} {'Cumulative':>12}")
    print("-" * 57)

    cumulative = n_exact  # Start with exact matches
    cumulative_pct = 100 * n_exact / n_total

    for exp in range(min_log, max_log + 1):
        lower = 10.0 ** exp
        upper = 10.0 ** (exp + 1)

        if exp == min_log:
            # First bin: [0, 10^(min_log+1))
            count = np.sum(nonzero_diff < upper)
            range_str = f"< 1e{exp+1}"
        elif exp == max_log:
            # Last bin: [10^max_log, inf)
            count = np.sum(nonzero_diff >= lower)
            range_str = f">= 1e{exp}"
        else:
            # Middle bins: [10^exp, 10^(exp+1))
            count = np.sum((nonzero_diff >= lower) & (nonzero_diff < upper))
            range_str = f"[1e{exp}, 1e{exp+1})"

        pct = 100 * count / n_total
        cumulative += count
        cumulative_pct = 100 * cumulative / n_total

        if count > 0:
            print(f"{range_str:<25} {count:>10} {pct:>9.2f}% {cumulative_pct:>10.2f}%")

    # Summary statistics
    print(f"\n--- Summary ---")
    percentiles = [50, 90, 95, 99, 99.9]
    pct_values = np.percentile(abs_diff, percentiles)
    for p, v in zip(percentiles, pct_values):
        print(f"{p:>5.1f}% of diffs <= {v:.2e}")


def visualize_diff(tensor1: np.ndarray, tensor2: np.ndarray, max_show: int = 10) -> None:
    """Show first few elements side by side."""
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()

    print(f"\nFirst {max_show} elements comparison:")
    print(f"{'Index':<8} {'File1':<15} {'File2':<15} {'Diff':<15}")
    print("-" * 53)
    for i in range(min(max_show, len(flat1))):
        diff = flat1[i] - flat2[i]
        print(f"{i:<8} {flat1[i]:<15.6f} {flat2[i]:<15.6f} {diff:<15.6e}")


def parse_shape(shape_str: str) -> tuple[int, ...]:
    """Parse shape string like '1,3,1024' into tuple (1, 3, 1024)."""
    return tuple(int(x.strip()) for x in shape_str.split(','))


def main():
    parser = argparse.ArgumentParser(
        description="Compare binary tensor dumps from NeMo and ggml implementations"
    )
    parser.add_argument("file1", help="First binary file (e.g., NeMo output)")
    parser.add_argument("file2", help="Second binary file (e.g., ggml output)")
    parser.add_argument("--name1", default="nemo", help="Name for first tensor (default: nemo)")
    parser.add_argument("--name2", default="ggml", help="Name for second tensor (default: ggml)")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance (default: 1e-4)")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance (default: 1e-5)")
    parser.add_argument("--show", type=int, default=10, help="Number of elements to show (default: 10)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show element-by-element comparison")

    args = parser.parse_args()

    try:
        tensor1 = load_tensor(args.file1)
        tensor2 = load_tensor(args.file2)
        assert tensor1.shape == tensor2.shape, \
            f"Shape mismatch between {args.file1} and {args.file2}: {tensor1.shape} vs {tensor2.shape}"
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    stats = compare_tensors(tensor1, tensor2, args.name1, args.name2, args.rtol, args.atol)
    print_comparison(stats, args.name1, args.name2)

    # Always show log histogram - it's very useful
    log_histogram(tensor1, tensor2)

    if args.verbose:
        visualize_diff(tensor1, tensor2, args.show)

    # Return exit code based on comparison result
    sys.exit(0 if stats['all_close'] else 1)


if __name__ == "__main__":
    main()
