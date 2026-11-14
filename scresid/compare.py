"""Lightweight metric comparison table (no torch / scanpy dependency)."""

from __future__ import annotations

COMPARISON_KEYS = (
    "r2_mean_all", "r2_mean_top", "r2_var_all", "r2_var_top",
    "common_top_degs", "top_response_pearson", "top_response_cosine",
    "offtarget_ratio", "cell_level_delta_std", "content_response_abs_corr",
)


def print_comparison(baseline, calibrated, keys=COMPARISON_KEYS):
    """Print the baseline vs calibrated table used by the benchmark runner."""
    print("=" * 76)
    print(f"{'metric':<36}{'baseline':>8}{'calibrated':>14}{'delta':>14}")
    print("-" * 76)
    for k in keys:
        b = float(baseline.get(k, 0.0))
        c = float(calibrated.get(k, 0.0))
        print(f"{k:<36}{b:>8.4f}{c:>14.4f}{c - b:>+14.4f}")
    print("=" * 76)
