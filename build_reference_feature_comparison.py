import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline and enhanced dataset inference runs.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--enhanced-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten_summary(summary):
    return {
        "best_single_feature_name": summary["best_single_feature"]["name"],
        "best_single_feature_auc": summary["best_single_feature"]["median_heldout_auc"],
        "dataset_mean_p_value": summary["dataset_level"]["mean_p_value"],
        "dataset_median_p_value": summary["dataset_level"]["median_p_value"],
        "dataset_combined_p_value": summary["dataset_level"]["sidak_combined_p_value"],
        "dataset_mean_gap": summary["dataset_level"]["mean_gap_nonmember_minus_member"],
        "runtime_seconds": summary.get("runtime_seconds"),
        "peak_gpu_memory_gb": summary.get("peak_gpu_memory_gb"),
    }


def build_metric_rows(baseline_summary, enhanced_summary):
    baseline_flat = flatten_summary(baseline_summary)
    enhanced_flat = flatten_summary(enhanced_summary)
    rows = []
    for metric in [
        "best_single_feature_auc",
        "dataset_mean_p_value",
        "dataset_median_p_value",
        "dataset_combined_p_value",
        "dataset_mean_gap",
        "runtime_seconds",
        "peak_gpu_memory_gb",
    ]:
        rows.append(
            {
                "category": "summary",
                "metric": metric,
                "baseline_value": baseline_flat[metric],
                "enhanced_value": enhanced_flat[metric],
                "delta_enhanced_minus_baseline": enhanced_flat[metric] - baseline_flat[metric],
            }
        )
    rows.append(
        {
            "category": "summary",
            "metric": "best_single_feature_name",
            "baseline_value": baseline_flat["best_single_feature_name"],
            "enhanced_value": enhanced_flat["best_single_feature_name"],
            "delta_enhanced_minus_baseline": "",
        }
    )
    return rows


def build_feature_auc_rows(baseline_feature_df, enhanced_feature_df):
    baseline_auc = (
        baseline_feature_df.groupby("feature", as_index=False)["sample_auc_heldout"]
        .median()
        .rename(columns={"sample_auc_heldout": "baseline_value"})
    )
    enhanced_auc = (
        enhanced_feature_df.groupby("feature", as_index=False)["sample_auc_heldout"]
        .median()
        .rename(columns={"sample_auc_heldout": "enhanced_value"})
    )
    merged = baseline_auc.merge(enhanced_auc, on="feature", how="outer").fillna("")
    rows = []
    for row in merged.itertuples(index=False):
        baseline_value = row.baseline_value if row.baseline_value != "" else ""
        enhanced_value = row.enhanced_value if row.enhanced_value != "" else ""
        if baseline_value == "" or enhanced_value == "":
            delta = ""
        else:
            delta = enhanced_value - baseline_value
        rows.append(
            {
                "category": "feature_auc",
                "metric": row.feature,
                "baseline_value": baseline_value,
                "enhanced_value": enhanced_value,
                "delta_enhanced_minus_baseline": delta,
            }
        )
    return rows, merged


def write_summary_md(output_path, baseline_dir, enhanced_dir, baseline_summary, enhanced_summary, feature_auc_table):
    top_enhanced = feature_auc_table.copy()
    top_enhanced["enhanced_numeric"] = pd.to_numeric(top_enhanced["enhanced_value"], errors="coerce")
    top_enhanced = top_enhanced.sort_values("enhanced_numeric", ascending=False).head(10)

    reference_rows = top_enhanced[top_enhanced["feature"].astype(str).str.startswith("ref_ppl_ratio_")]
    lines = [
        "# Reference Feature Comparison",
        "",
        "## Runs",
        f"- Baseline dir: `{baseline_dir}`",
        f"- Enhanced dir: `{enhanced_dir}`",
        "",
        "## Summary",
        f"- Baseline best single feature: `{baseline_summary['best_single_feature']['name']}` ({baseline_summary['best_single_feature']['median_heldout_auc']:.4f})",
        f"- Enhanced best single feature: `{enhanced_summary['best_single_feature']['name']}` ({enhanced_summary['best_single_feature']['median_heldout_auc']:.4f})",
        f"- Baseline median dataset-level p-value: `{baseline_summary['dataset_level']['median_p_value']:.6f}`",
        f"- Enhanced median dataset-level p-value: `{enhanced_summary['dataset_level']['median_p_value']:.6f}`",
        f"- Baseline mean score gap: `{baseline_summary['dataset_level']['mean_gap_nonmember_minus_member']:.6f}`",
        f"- Enhanced mean score gap: `{enhanced_summary['dataset_level']['mean_gap_nonmember_minus_member']:.6f}`",
        f"- Baseline runtime: `{baseline_summary.get('runtime_seconds', 0.0):.2f}` seconds",
        f"- Enhanced runtime: `{enhanced_summary.get('runtime_seconds', 0.0):.2f}` seconds",
        f"- Baseline peak GPU memory: `{baseline_summary.get('peak_gpu_memory_gb', 0.0):.2f}` GiB",
        f"- Enhanced peak GPU memory: `{enhanced_summary.get('peak_gpu_memory_gb', 0.0):.2f}` GiB",
        "",
        "## Top Enhanced Features",
    ]
    for row in top_enhanced.itertuples(index=False):
        lines.append(
            f"- `{row.feature}`: baseline `{row.baseline_value}`, enhanced `{row.enhanced_value}`"
        )

    lines.extend(["", "## Reference Feature AUCs"])
    if reference_rows.empty:
        lines.append("- No reference features were present in the enhanced run.")
    else:
        for row in reference_rows.itertuples(index=False):
            lines.append(
                f"- `{row.feature}`: enhanced median AUC `{row.enhanced_value}`"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = Path(args.baseline_dir)
    enhanced_dir = Path(args.enhanced_dir)

    baseline_summary = load_json(baseline_dir / "summary.json")
    enhanced_summary = load_json(enhanced_dir / "summary.json")
    baseline_feature_df = pd.read_csv(baseline_dir / "sample_level_feature_metrics.csv")
    enhanced_feature_df = pd.read_csv(enhanced_dir / "sample_level_feature_metrics.csv")

    rows = build_metric_rows(baseline_summary, enhanced_summary)
    feature_rows, feature_auc_table = build_feature_auc_rows(baseline_feature_df, enhanced_feature_df)
    comparison_df = pd.DataFrame(rows + feature_rows)
    comparison_df.to_csv(output_dir / "comparison_table.csv", index=False)

    write_summary_md(
        output_dir / "summary.md",
        baseline_dir,
        enhanced_dir,
        baseline_summary,
        enhanced_summary,
        feature_auc_table,
    )


if __name__ == "__main__":
    main()
