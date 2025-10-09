"""Produce summary plots for preprocessing variant evaluation.

Reads `preprocessing_metrics.csv` in the same directory and generates
publication-friendly figures summarizing N1/P3b peak amplitudes and SNR
across preprocessing pipelines. The script does not re-run any analysis; it
simply visualizes existing results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
CSV_PATH = THIS_DIR / "preprocessing_metrics.csv"


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _format_dataset_labels(df: pd.DataFrame) -> pd.Series:
    return df["dataset"].str.replace("_", " ").str.replace("-", "-")


def plot_rank_scores(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = _format_dataset_labels(df)
    ax.bar(labels, df["rank_score"], color="#4E79A7")
    ax.set_ylabel("Mean ROI SNR (Rank Score)")
    ax.set_title("Preprocessing Comparison: Combined N1/P3b SNR")
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_component_snrs(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    width = 0.35
    ax.bar(
        [xi - width / 2 for xi in x],
        df["mean_N1_snr"],
        width,
        label="N1 SNR",
        color="#59A14F",
    )
    ax.bar(
        [xi + width / 2 for xi in x],
        df["mean_P3b_snr"],
        width,
        label="P3b SNR",
        color="#E15759",
    )
    labels = _format_dataset_labels(df)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("SNR (Peak / Baseline SD)")
    ax.set_title("Component-specific SNR by Preprocessing Pipeline")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_peak_amplitudes(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    width = 0.35
    ax.bar(
        [xi - width / 2 for xi in x],
        df["mean_N1_peak_uv"],
        width,
        label="N1 Peak (µV)",
        color="#76B7B2",
    )
    ax.bar(
        [xi + width / 2 for xi in x],
        df["mean_P3b_peak_uv"],
        width,
        label="P3b Peak (µV)",
        color="#F28E2B",
    )
    labels = _format_dataset_labels(df)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Mean Peak Amplitude (µV)")
    ax.set_title("Component Peak Amplitudes by Preprocessing Pipeline")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Cannot find {CSV_PATH}. Run evaluator first.")

    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("rank_score", ascending=False)

    _setup_style()

    plot_rank_scores(df, THIS_DIR / "preprocessing_rank_score.png")
    plot_component_snrs(df, THIS_DIR / "preprocessing_snrs.png")
    plot_peak_amplitudes(df, THIS_DIR / "preprocessing_peaks.png")

    print("Saved plots to:")
    print(THIS_DIR / "preprocessing_rank_score.png")
    print(THIS_DIR / "preprocessing_snrs.png")
    print(THIS_DIR / "preprocessing_peaks.png")


if __name__ == "__main__":
    main()

