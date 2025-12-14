#!/usr/bin/env python3
"""
Generate presentation-ready visualizations for the multilingual NLU system.

The script supports both intent classification and slot filling views. When the
relevant artifacts are present it produces PNG figures that summarise:

Intent classification (requires ``--model-dir``):
1. Overall evaluation metrics (Accuracy, Micro/Macro F1)
2. Per-language accuracy (lowest 10 vs highest 10)
3. Per-intent F1 (lowest 10 vs highest 10)
4. (optional, needs ``--dataset-dir``) Intent frequency across the MASSIVE test split

Slot filling (optional):
1. Overall evaluation metrics (requires ``--slot-results`` test_results.json)
2. Slot tag frequency from the MASSIVE test split (requires ``--dataset-dir``)
3. Coarse slot type frequency (BIO prefixes stripped, requires ``--dataset-dir``)

Usage:
    python generate_visualizations.py \
        --model-dir "/path/to/xlm-roberta-intent-classifier-final" \
        --output-dir "/path/to/visualizations" \
        --slot-results "/path/to/slot_filling_model/final_model/test_results.json" \
        --dataset-dir "/path/to/data"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create intent classifier visualizations.")
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing evaluation_summary.json, per_language_accuracy.csv, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory to store generated PNG charts.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output resolution (dots-per-inch) for saved figures.",
    )
    parser.add_argument(
        "--slot-results",
        type=pathlib.Path,
        help="Optional slot filling test_results.json to chart overall slot metrics.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        help="Optional MASSIVE dataset directory (*.jsonl) to compute intent/slot frequency charts.",
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["test"],
        help="Dataset partitions to include when --dataset-dir is supplied (default: test).",
    )
    parser.add_argument(
        "--slot-top-n",
        type=int,
        default=20,
        help="Number of slot tags to display in frequency plots when dataset data is available.",
    )
    return parser.parse_args()


def load_overall_metrics(model_dir: pathlib.Path) -> pd.DataFrame:
    test_results_path = model_dir / "test_results.json"
    if not test_results_path.exists():
        raise FileNotFoundError(f"Missing {test_results_path}")

    with test_results_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metrics = {
        "Accuracy": data["eval_accuracy"],
        "F1 (Micro)": data["eval_f1_micro"],
        "F1 (Macro)": data["eval_f1_macro"],
    }
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])
    df["Percentage"] = df["Score"] * 100
    return df


def plot_overall_metrics(
    df: pd.DataFrame, output_path: pathlib.Path, dpi: int, title: str
) -> None:
    plt.figure(figsize=(6, 4))
    palette = sns.color_palette("viridis", n_colors=len(df))
    sns.barplot(data=df, x="Percentage", y="Metric", palette=palette)
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Score (%)")
    plt.ylabel("")
    plt.xlim(0, 100)

    for index, row in df.iterrows():
        plt.text(
            row["Percentage"] + 1,
            index,
            f"{row['Percentage']:.1f}%",
            va="center",
            fontsize=11,
            weight="semibold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def load_per_language(model_dir: pathlib.Path) -> pd.DataFrame:
    csv_path = model_dir / "per_language_accuracy.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    df["Accuracy_pct"] = df["Accuracy"] * 100
    return df.sort_values("Accuracy_pct")


def plot_language_performance(df: pd.DataFrame, output_path: pathlib.Path, dpi: int) -> None:
    lowest = df.head(10)
    highest = df.tail(10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    sns.barplot(
        data=lowest,
        x="Accuracy_pct",
        y="locale",
        ax=axes[0],
        palette="Reds",
    )
    axes[0].set_title("Lowest 10 Languages", weight="bold")
    axes[0].set_xlabel("Accuracy (%)")
    axes[0].set_ylabel("")

    sns.barplot(
        data=highest,
        x="Accuracy_pct",
        y="locale",
        ax=axes[1],
        palette="Greens",
    )
    axes[1].set_title("Highest 10 Languages", weight="bold")
    axes[1].set_xlabel("Accuracy (%)")
    axes[1].set_ylabel("")

    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_xlim(0, 100)

    fig.suptitle("Per-Language Accuracy", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def load_per_intent(model_dir: pathlib.Path) -> pd.DataFrame:
    csv_path = model_dir / "per_class_f1_scores.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    df["f1_pct"] = df["f1_score"] * 100
    return df.sort_values("f1_pct")


def plot_intent_performance(df: pd.DataFrame, output_path: pathlib.Path, dpi: int) -> None:
    lowest = df.head(10)
    highest = df.tail(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    sns.barplot(
        data=lowest,
        x="f1_pct",
        y="intent",
        ax=axes[0],
        palette="Reds",
    )
    axes[0].set_title("Lowest 10 Intents (F1)", weight="bold")
    axes[0].set_xlabel("F1 Score (%)")
    axes[0].set_ylabel("")

    sns.barplot(
        data=highest,
        x="f1_pct",
        y="intent",
        ax=axes[1],
        palette="Blues",
    )
    axes[1].set_title("Highest 10 Intents (F1)", weight="bold")
    axes[1].set_xlabel("F1 Score (%)")
    axes[1].set_ylabel("")

    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_xlim(0, 100)

    fig.suptitle("Per-Intent F1 Performance", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def load_slot_overall_metrics(slot_results_path: pathlib.Path) -> pd.DataFrame:
    if not slot_results_path.exists():
        raise FileNotFoundError(f"Missing {slot_results_path}")

    with slot_results_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metrics = {
        "Precision": data.get("eval_precision"),
        "Recall": data.get("eval_recall"),
        "Accuracy": data.get("eval_accuracy"),
        "F1": data.get("eval_f1"),
    }
    filtered = {k: v for k, v in metrics.items() if v is not None}
    df = pd.DataFrame(list(filtered.items()), columns=["Metric", "Score"])
    df["Percentage"] = df["Score"] * 100
    return df


def collect_intent_support(
    dataset_dir: pathlib.Path, partitions: Tuple[str, ...]
) -> Tuple[pd.DataFrame, int]:
    counts: dict[str, int] = {}
    total = 0
    for path in sorted(dataset_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if partitions and record.get("partition") not in partitions:
                    continue
                intent = record.get("intent")
                if not intent:
                    continue
                counts[intent] = counts.get(intent, 0) + 1
                total += 1

    if not counts:
        return pd.DataFrame(columns=["intent", "count", "percentage"]), total

    df = pd.DataFrame(
        {"intent": list(counts.keys()), "count": list(counts.values())}
    )
    df = df.sort_values("count", ascending=False)
    df["percentage"] = df["count"] / total * 100
    return df, total


def plot_intent_support(
    df: pd.DataFrame, output_path: pathlib.Path, dpi: int, top_n: int = 20
) -> None:
    if df.empty:
        return
    subset = df.head(top_n).copy()
    subset = subset.iloc[::-1].reset_index(drop=True)

    plt.figure(figsize=(8, max(6, top_n * 0.35)))
    sns.barplot(data=subset, x="percentage", y="intent", palette="mako")
    plt.title("Intent Distribution (Test Partition)", fontsize=14, weight="bold")
    plt.xlabel("Share of utterances (%)")
    plt.ylabel("")

    for idx, row in subset.iterrows():
        plt.text(
            row["percentage"] + 0.5,
            idx,
            f"{row['percentage']:.1f}% ({row['count']:,})",
            va="center",
            fontsize=9,
        )

    plt.xlim(0, subset["percentage"].max() * 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def collect_slot_counts(
    dataset_dir: pathlib.Path, partitions: Tuple[str, ...]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_counts: dict[str, int] = {}
    coarse_counts: dict[str, int] = {}
    pattern = re.compile(r"\[([A-Za-z0-9_-]+)\s*:")

    for path in sorted(dataset_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if partitions and record.get("partition") not in partitions:
                    continue
                annot = record.get("annot_utt") or ""
                for label in pattern.findall(annot):
                    raw_counts[label] = raw_counts.get(label, 0) + 1
                    if "-" in label:
                        coarse = label.split("-", 1)[1]
                    else:
                        coarse = label
                    coarse_counts[coarse] = coarse_counts.get(coarse, 0) + 1

    raw_df = (
        pd.DataFrame({"slot": list(raw_counts.keys()), "count": list(raw_counts.values())})
        if raw_counts
        else pd.DataFrame(columns=["slot", "count"])
    )
    if not raw_df.empty:
        raw_df = raw_df.sort_values("count", ascending=False)
        raw_df["percentage"] = raw_df["count"] / raw_df["count"].sum() * 100

    coarse_df = (
        pd.DataFrame(
            {"slot": list(coarse_counts.keys()), "count": list(coarse_counts.values())}
        )
        if coarse_counts
        else pd.DataFrame(columns=["slot", "count"])
    )
    if not coarse_df.empty:
        coarse_df = coarse_df.sort_values("count", ascending=False)
        coarse_df["percentage"] = coarse_df["count"] / coarse_df["count"].sum() * 100

    return raw_df, coarse_df


def plot_slot_frequency(
    df: pd.DataFrame,
    output_path: pathlib.Path,
    dpi: int,
    title: str,
    top_n: int,
    palette: str,
) -> None:
    if df.empty:
        return
    subset = df.head(top_n).copy()
    subset = subset.iloc[::-1].reset_index(drop=True)

    plt.figure(figsize=(8, max(6, top_n * 0.35)))
    sns.barplot(data=subset, x="percentage", y="slot", palette=palette)
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Share of entities (%)")
    plt.ylabel("")

    for idx, row in subset.iterrows():
        plt.text(
            row["percentage"] + 0.3,
            idx,
            f"{row['percentage']:.1f}% ({row['count']:,})",
            va="center",
            fontsize=9,
        )

    plt.xlim(0, subset["percentage"].max() * 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    print("Generating overall metrics chart...")
    overall_df = load_overall_metrics(model_dir)
    plot_overall_metrics(
        overall_df,
        output_dir / "overall_metrics.png",
        args.dpi,
        title="Overall Intent Classification Performance",
    )

    print("Generating per-language accuracy chart...")
    language_df = load_per_language(model_dir)
    plot_language_performance(language_df, output_dir / "per_language_accuracy.png", args.dpi)

    print("Generating per-intent F1 chart...")
    intent_df = load_per_intent(model_dir)
    plot_intent_performance(intent_df, output_dir / "per_intent_f1.png", args.dpi)

    if args.dataset_dir:
        dataset_dir = args.dataset_dir.expanduser().resolve()
        if not dataset_dir.is_dir():
            raise NotADirectoryError(f"--dataset-dir {dataset_dir} is not a directory")

        partitions = tuple(args.partitions) if args.partitions else tuple()

        print("Computing intent distribution from dataset...")
        intent_support_df, total_intents = collect_intent_support(dataset_dir, partitions)
        if not intent_support_df.empty:
            plot_intent_support(
                intent_support_df,
                output_dir / "intent_distribution.png",
                args.dpi,
                top_n=20,
            )
            print(f"  Included {total_intents:,} labelled intents.")
        else:
            print("  No intents found for requested partitions; skipping chart.")

        print("Computing slot tag distribution from dataset...")
        raw_slot_df, coarse_slot_df = collect_slot_counts(dataset_dir, partitions)
        if not raw_slot_df.empty:
            plot_slot_frequency(
                raw_slot_df,
                output_dir / "slot_tag_frequency_bio.png",
                args.dpi,
                title=f"Slot Tag Frequency (BIO, top {args.slot_top_n})",
                top_n=args.slot_top_n,
                palette="rocket",
            )
        else:
            print("  No BIO slot tags found for requested partitions; skipping chart.")

        if not coarse_slot_df.empty:
            plot_slot_frequency(
                coarse_slot_df,
                output_dir / "slot_tag_frequency_coarse.png",
                args.dpi,
                title=f"Slot Tag Frequency (Coarse, top {args.slot_top_n})",
                top_n=args.slot_top_n,
                palette="crest",
            )

    if args.slot_results:
        slot_results_path = args.slot_results.expanduser().resolve()
        print("Generating slot overall metrics chart...")
        slot_overall_df = load_slot_overall_metrics(slot_results_path)
        plot_overall_metrics(
            slot_overall_df,
            output_dir / "slot_overall_metrics.png",
            args.dpi,
            title="Overall Slot Filling Performance",
        )

    print("\n✅ Visualizations saved to:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

