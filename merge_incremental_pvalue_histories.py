import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

COLOR_MAP = {
    "Deduped": "#4c72b0",
    "Non-Deduped": "#dd8452",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple incremental p-value history files.")
    parser.add_argument(
        "--history-files",
        required=True,
        help="Comma-separated list of history.json paths.",
    )
    parser.add_argument("--output-dir", default="results/incremental_pvalue_figure_merged")
    parser.add_argument("--history-file", default="history.json")
    parser.add_argument("--table-file", default="history_table.csv")
    parser.add_argument("--figure-file", default="figure.png")
    return parser.parse_args()


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_history(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_histories(history_files):
    merged_runs = []
    seen_keys = set()
    source_files = []
    for history_file in history_files:
        history = load_history(history_file)
        source_files.append(str(Path(history_file).resolve()))
        for run in history.get("runs", []):
            key = (
                run.get("model_name"),
                run.get("model_label"),
                float(run.get("model_size_b")),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_runs.append(run)
    return {"runs": merged_runs, "source_history_files": source_files}


def history_to_dataframe(history):
    rows = []
    for run in history.get("runs", []):
        for row in run.get("dataset_results", []):
            rows.append(
                {
                    "model_name": run["model_name"],
                    "model_label": run["model_label"],
                    "model_size_b": float(run["model_size_b"]),
                    "dataset": row["dataset"],
                    "random_index": row.get("random_index", 0),
                    "p_value": float(row["p_value"]),
                }
            )
    return pd.DataFrame(rows)


def plot_history(history_df, figure_path):
    if history_df.empty:
        return

    sizes = sorted(history_df["model_size_b"].unique())
    labels = []
    if "Deduped" in history_df["model_label"].unique():
        labels.append("Deduped")
    if "Non-Deduped" in history_df["model_label"].unique():
        labels.append("Non-Deduped")
    remaining = sorted(set(history_df["model_label"].unique()) - set(labels))
    labels.extend(remaining)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.32
    offsets = {}
    if len(labels) == 1:
        offsets[labels[0]] = 0.0
    elif len(labels) == 2:
        offsets[labels[0]] = -0.18
        offsets[labels[1]] = 0.18
    else:
        spread = np.linspace(-0.24, 0.24, num=len(labels))
        offsets = {label: float(offset) for label, offset in zip(labels, spread)}

    for x_index, size in enumerate(sizes):
        for label in labels:
            subset = history_df[
                (history_df["model_size_b"] == size) & (history_df["model_label"] == label)
            ]
            if subset.empty:
                continue
            xpos = x_index + offsets[label]
            violin = ax.violinplot(
                subset["p_value"].to_list(),
                positions=[xpos],
                widths=width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            color = COLOR_MAP.get(label, "#6c757d")
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.85)

            ax.scatter(
                np.full(len(subset), xpos),
                subset["p_value"],
                color=color,
                s=9,
                alpha=0.65,
                zorder=3,
            )

    ax.axhline(0.1, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(size) for size in sizes], fontsize=12)
    ax.set_xlabel("Model Size (Billions Parameters)", fontsize=14)
    ax.set_ylabel("p-value", fontsize=14)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.2)

    legend_handles = [Patch(facecolor=COLOR_MAP.get(label, "#6c757d"), label=label) for label in labels]
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, loc="upper right")

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    history_files = parse_csv_list(args.history_files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_history = merge_histories(history_files)
    history_path = output_dir / args.history_file
    history_path.write_text(json.dumps(merged_history, indent=2), encoding="utf-8")

    history_df = history_to_dataframe(merged_history)
    table_path = output_dir / args.table_file
    history_df.to_csv(table_path, index=False)

    figure_path = output_dir / args.figure_file
    plot_history(history_df, figure_path)

    print(
        json.dumps(
            {
                "history_path": str(history_path.resolve()),
                "table_path": str(table_path.resolve()),
                "figure_path": str(figure_path.resolve()),
                "num_runs": len(merged_history.get("runs", [])),
                "num_rows": int(len(history_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
