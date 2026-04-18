import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export dataset-by-model p-values as LaTeX and a heatmap figure."
    )
    parser.add_argument(
        "--history-file",
        default="results/incremental_pvalue_figure_selected_merged/history.json",
        help="Path to merged or single history.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/pvalue_report_assets",
        help="Directory for LaTeX and figure outputs.",
    )
    parser.add_argument(
        "--aggregate",
        choices=["median", "mean"],
        default="median",
        help="How to combine repeated p-values for the same dataset/model.",
    )
    parser.add_argument(
        "--split-by-label",
        action="store_true",
        help="Write separate tables/figures for Deduped and Non-Deduped instead of one combined table.",
    )
    return parser.parse_args()


def load_history(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def history_to_dataframe(history):
    rows = []
    for run in history.get("runs", []):
        for row in run.get("dataset_results", []):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "model_name": run["model_name"],
                    "model_label": run["model_label"],
                    "model_size_b": float(run["model_size_b"]),
                    "p_value": float(row["p_value"]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_frame(df, mode):
    agg_fn = "median" if mode == "median" else "mean"
    grouped = (
        df.groupby(["dataset", "model_label", "model_size_b"], as_index=False)["p_value"]
        .agg(agg_fn)
        .sort_values(["dataset", "model_size_b", "model_label"])
    )
    return grouped


def make_column_name(model_size_b, model_label):
    short_label = "D" if model_label.lower().startswith("deduped") else "ND"
    return f"{model_size_b:g} {short_label}"


def pivot_combined(df):
    frame = df.copy()
    frame["column_name"] = frame.apply(
        lambda row: make_column_name(row["model_size_b"], row["model_label"]),
        axis=1,
    )
    pivot = frame.pivot(index="dataset", columns="column_name", values="p_value")
    ordered_columns = sorted(
        pivot.columns,
        key=lambda name: (float(name.split()[0]), 0 if name.endswith("D") else 1),
    )
    pivot = pivot.reindex(columns=ordered_columns)
    return pivot


def pivot_by_label(df, model_label):
    frame = df[df["model_label"] == model_label].copy()
    pivot = frame.pivot(index="dataset", columns="model_size_b", values="p_value")
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    pivot.columns = [f"{column:g}" for column in pivot.columns]
    return pivot


def dataframe_to_latex_table(df, caption, label):
    return df.to_latex(
        float_format=lambda value: f"{value:.4g}",
        na_rep="--",
        escape=False,
        caption=caption,
        label=label,
    )


def draw_heatmap(df, title, output_path):
    values = df.to_numpy(dtype=float)
    fig_width = max(8, 0.9 * len(df.columns) + 4)
    fig_height = max(6, 0.4 * len(df.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # image = ax.imshow(values, cmap="coolwarm_r", vmin=0.0, vmax=1.0, aspect="auto")
    image = ax.imshow(values, cmap="coolwarm", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title(title)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                text = "--"
                color = "black"
            else:
                text = f"{value:.2f}"
                color = "white" if value < 0.35 or value > 0.75 else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("p-value")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(df, output_dir, stem, caption, label, title):
    tex_path = output_dir / f"{stem}.tex"
    png_path = output_dir / f"{stem}.png"
    csv_path = output_dir / f"{stem}.csv"

    csv_path.write_text(df.to_csv(), encoding="utf-8")
    tex_path.write_text(dataframe_to_latex_table(df, caption, label), encoding="utf-8")
    draw_heatmap(df, title, png_path)
    return tex_path, png_path, csv_path


def main():
    args = parse_args()
    history = load_history(args.history_file)
    raw_df = history_to_dataframe(history)
    if raw_df.empty:
        raise ValueError("No dataset results found in the provided history file.")

    aggregated_df = aggregate_frame(raw_df, args.aggregate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if args.split_by_label:
        for model_label in sorted(aggregated_df["model_label"].unique()):
            pivot = pivot_by_label(aggregated_df, model_label)
            stem = f"pvalues_{model_label.lower().replace('-', '_').replace(' ', '_')}"
            caption = f"P-values by dataset and model size for {model_label} models."
            latex_label = f"tab:{stem}"
            title = f"P-values by Dataset and Model Size ({model_label})"
            outputs.append(write_outputs(pivot, output_dir, stem, caption, latex_label, title))
    else:
        pivot = pivot_combined(aggregated_df)
        outputs.append(
            write_outputs(
                pivot,
                output_dir,
                "pvalues_combined",
                "P-values by dataset and model size.",
                "tab:pvalues_combined",
                "P-values by Dataset and Model Size",
            )
        )

    summary = {
        "history_file": str(Path(args.history_file).resolve()),
        "output_dir": str(output_dir.resolve()),
        "outputs": [
            {
                "tex": str(tex_path.resolve()),
                "png": str(png_path.resolve()),
                "csv": str(csv_path.resolve()),
            }
            for tex_path, png_path, csv_path in outputs
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
