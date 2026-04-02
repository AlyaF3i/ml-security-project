import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import get_dataset_config_names, load_dataset
from matplotlib.patches import Patch
from scipy.stats import ttest_ind
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
OFFICIAL_REPO_DIR = PROJECT_ROOT / "official_repo"
if str(OFFICIAL_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO_DIR))

from linear_di import get_predictions, normalize_and_stack, remove_outliers, train_model  # noqa: E402
import metrics as metrics_module  # noqa: E402
from metrics import aggregate_metrics  # noqa: E402
from selected_features import feature_list as selected_feature_list  # noqa: E402


REPO_DATASET_NAMES = [
    "stackexchange", # <=
    "wikipedia", # <=
    "cc", # <=
    "github", # <=
    # "pubmed_abstracts", # <= got removed
    "openwebtext2", # <=
    "freelaw", # <=
    "math", # <=
    # "nih",
    "uspto", # <=
    "hackernews", # <=
    # "enron",
    "books3", # <=
    # "pubmed_central", # <= got removed
    "gutenberg", # <=
    "arxiv", # <=
    "bookcorpus2", # <=
    "opensubtitles", # <=
    "youtubesubtitles", # <=
    "ubuntu", # <=
    "europarl", # <=
    "philpapers", # <=
]

DEFAULT_DATASET_NAMES = [
    "wikipedia",
    "github",
    "openwebtext2",
    "books3",
    "stackexchange",
    "freelaw",
    "hackernews",
]

BASIC_METRIC_LIST = ["k_min_probs", "ppl", "zlib_ratio", "k_max_probs"]
FULL_METRIC_DRIVERS = ["k_min_probs", "ppl", "zlib_ratio", "k_max_probs", "perturbation", "reference_model"]
COLOR_MAP = {
    "Deduped": "#4c72b0",
    "Non-Deduped": "#dd8452",
}
REFERENCE_MODEL_REGISTRY = {
    "tinystories-33M": "roneneldan/TinyStories-33M",
    "tinystories-1M": "roneneldan/TinyStories-1M",
    "phi-1_5": "microsoft/phi-1_5",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incrementally build a paper-style p-value violin plot one model at a time."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-label", required=True, help="Legend label, e.g. Deduped or Non-Deduped.")
    parser.add_argument("--model-size-b", type=float, required=True, help="Model size in billions, e.g. 0.41.")
    parser.add_argument(
        "--dataset-names",
        default=",".join(DEFAULT_DATASET_NAMES),
        help="Comma-separated list, or 'all'.",
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-epochs", type=int, default=1000)
    parser.add_argument("--num-random", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--feature-mode", choices=["basic", "selected", "full"], default="basic")
    parser.add_argument("--normalize", choices=["no", "train", "combined"], default="train")
    parser.add_argument(
        "--outliers",
        default="clip",
        choices=["randomize", "keep", "zero", "mean", "clip", "mean+p-value", "p-value"],
    )
    parser.add_argument(
        "--reference-model-aliases",
        default=",".join(REFERENCE_MODEL_REGISTRY.keys()),
        help="Comma-separated supported reference aliases.",
    )
    parser.add_argument("--trim-heldout-frac", type=float, default=0.0)
    parser.add_argument("--output-dir", default="results/incremental_pvalue_figure")
    parser.add_argument("--history-file", default="history.json")
    parser.add_argument("--figure-file", default="figure.png")
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanitize_name(name):
    return name.replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")


def build_run_filename(model_name, model_label, model_size_b):
    return (
        f"{sanitize_name(model_name)}"
        f"__{sanitize_name(model_label)}"
        f"__{str(model_size_b).replace('.', '_')}b.json"
    )


def parse_dataset_names(dataset_arg):
    if dataset_arg.strip().lower() == "all":
        return REPO_DATASET_NAMES
    return [name.strip() for name in dataset_arg.split(",") if name.strip()]


def resolve_supported_dataset_names(requested_names):
    available = set(get_dataset_config_names("pratyushmaini/llm_dataset_inference"))
    supported = [name for name in requested_names if name in available]
    skipped = [name for name in requested_names if name not in available]
    return supported, skipped, sorted(available)


def load_model_and_tokenizer(model_name, max_length, cache_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def load_dataset_split_pair(dataset_name):
    ds = load_dataset("pratyushmaini/llm_dataset_inference", dataset_name)
    val_key = "val" if "val" in ds else "validation"
    return ds["train"], ds[val_key]


def get_metric_drivers(feature_mode):
    if feature_mode in {"selected", "full"}:
        return list(FULL_METRIC_DRIVERS)
    return list(BASIC_METRIC_LIST)


def prepare_metrics(
    members_metrics,
    nonmembers_metrics,
    outliers="clip",
    normalize="train",
    allowed_keys=None,
    return_tensors=False,
):
    keys = list(members_metrics.keys())
    if allowed_keys is not None:
        keys = [key for key in keys if key in allowed_keys]
    np_members_metrics = []
    np_nonmembers_metrics = []
    for key in keys:
        members_metric_key = np.array(members_metrics[key])
        nonmembers_metric_key = np.array(nonmembers_metrics[key])

        if outliers is not None and outliers != "keep":
            members_metric_key = remove_outliers(members_metric_key, remove_frac=0.05, outliers=outliers)
            nonmembers_metric_key = remove_outliers(nonmembers_metric_key, remove_frac=0.05, outliers=outliers)

        np_members_metrics.append(members_metric_key)
        np_nonmembers_metrics.append(nonmembers_metric_key)

    np_members_metrics, np_nonmembers_metrics = normalize_and_stack(
        np_members_metrics,
        np_nonmembers_metrics,
        normalize=normalize,
    )
    if return_tensors:
        np_members_metrics = torch.tensor(np_members_metrics, dtype=torch.float32)
        np_nonmembers_metrics = torch.tensor(np_nonmembers_metrics, dtype=torch.float32)
    return np_members_metrics, np_nonmembers_metrics


def trim_scores(scores, frac):
    scores = np.asarray(scores, dtype=np.float64)
    if frac <= 0:
        return scores
    k = int(len(scores) * frac / 2)
    if k == 0 or len(scores) <= 2 * k:
        return scores
    ids = np.argsort(scores)
    keep_ids = ids[k : len(scores) - k]
    return scores[keep_ids]


def get_dataset_splits(train_metrics, val_metrics, num_samples):
    for_train_train_metrics = train_metrics[:num_samples]
    for_train_val_metrics = val_metrics[:num_samples]
    for_val_train_metrics = train_metrics[num_samples:]
    for_val_val_metrics = val_metrics[num_samples:]

    train_x = np.concatenate((for_train_train_metrics, for_train_val_metrics), axis=0)
    train_y = np.concatenate((-1 * np.zeros(for_train_train_metrics.shape[0]), np.ones(for_train_val_metrics.shape[0])))
    val_x = np.concatenate((for_val_train_metrics, for_val_val_metrics), axis=0)
    val_y = np.concatenate((-1 * np.zeros(for_val_train_metrics.shape[0]), np.ones(for_val_val_metrics.shape[0])))

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)
    return (train_x, train_y), (val_x, val_y)


def split_A_B(train_split, val_split, requested_sample_size):
    max_supported = min(len(train_split), len(val_split)) // 2
    if max_supported < 1:
        raise ValueError(
            f"Dataset is too small to form A/B splits: train={len(train_split)}, val={len(val_split)}."
        )

    effective_sample_size = min(requested_sample_size, max_supported)

    A_members = train_split.select(range(0, effective_sample_size))
    A_nonmembers = val_split.select(range(0, effective_sample_size))
    B_members = train_split.select(range(effective_sample_size, effective_sample_size * 2))
    B_nonmembers = val_split.select(range(effective_sample_size, effective_sample_size * 2))
    return A_members, A_nonmembers, B_members, B_nonmembers, effective_sample_size


def parse_reference_aliases(alias_string):
    aliases = [alias.strip() for alias in alias_string.split(",") if alias.strip()]
    unknown = [alias for alias in aliases if alias not in REFERENCE_MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown reference aliases: {unknown}")
    return aliases


def configure_reference_registry(args):
    aliases = parse_reference_aliases(args.reference_model_aliases)
    metrics_module.reference_model_registry = {
        alias: REFERENCE_MODEL_REGISTRY[alias] for alias in aliases
    }
    return aliases


def ensure_reference_metric_files(dataset_name, split_name, dataset_split, args, device):
    if "reference_model" not in get_metric_drivers(args.feature_mode):
        return

    for alias in parse_reference_aliases(args.reference_model_aliases):
        model_name = REFERENCE_MODEL_REGISTRY[alias]
        output_path = PROJECT_ROOT / "results" / model_name / f"{dataset_name}_{split_name}_metrics.json"
        if output_path.exists():
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ref_model, ref_tokenizer = load_model_and_tokenizer(model_name, args.max_length, args.cache_dir, device)
        ref_metrics = aggregate_metrics(
            ref_model,
            ref_tokenizer,
            dataset_split,
            ["ppl"],
            None,
            batch_size=args.batch_size,
        )
        output_path.write_text(json.dumps(ref_metrics), encoding="utf-8")
        del ref_model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def get_allowed_feature_keys(feature_mode):
    if feature_mode == "selected":
        return set(selected_feature_list)
    return None


def run_one_dataset(model, tokenizer, dataset_name, args, device):
    train_split, val_split = load_dataset_split_pair(dataset_name)
    A_members, A_nonmembers, B_members, B_nonmembers, effective_sample_size = split_A_B(
        train_split,
        val_split,
        args.sample_size,
    )
    print(f"Finished Splitting {dataset_name}")
    metric_drivers = get_metric_drivers(args.feature_mode)
    allowed_keys = get_allowed_feature_keys(args.feature_mode)
    if "reference_model" in metric_drivers:
        configure_reference_registry(args)
        ensure_reference_metric_files(dataset_name, "A_members", A_members, args, device)
        ensure_reference_metric_files(dataset_name, "A_nonmembers", A_nonmembers, args, device)
        ensure_reference_metric_files(dataset_name, "B_members", B_members, args, device)
        ensure_reference_metric_files(dataset_name, "B_nonmembers", B_nonmembers, args, device)

    A_members_metrics = aggregate_metrics(
        model,
        tokenizer,
        A_members,
        metric_drivers,
        SimpleNamespace(dataset_name=dataset_name, split="A_members"),
        batch_size=args.batch_size,
    )
    print("="*30)
    print(len(A_members_metrics))
    print(A_members_metrics.keys())
    print("="*30)
    A_nonmembers_metrics = aggregate_metrics(
        model,
        tokenizer,
        A_nonmembers,
        metric_drivers,
        SimpleNamespace(dataset_name=dataset_name, split="A_nonmembers"),
        batch_size=args.batch_size,
    )
    B_members_metrics = aggregate_metrics(
        model,
        tokenizer,
        B_members,
        metric_drivers,
        SimpleNamespace(dataset_name=dataset_name, split="B_members"),
        batch_size=args.batch_size,
    )
    B_nonmembers_metrics = aggregate_metrics(
        model,
        tokenizer,
        B_nonmembers,
        metric_drivers,
        SimpleNamespace(dataset_name=dataset_name, split="B_nonmembers"),
        batch_size=args.batch_size,
    )

    base_train_metrics, base_val_metrics = prepare_metrics(
        A_members_metrics,
        A_nonmembers_metrics,
        outliers=args.outliers,
        normalize=args.normalize,
        allowed_keys=allowed_keys,
        return_tensors=False,
    )
    B_members_metrics_tensor, B_nonmembers_metrics_tensor = prepare_metrics(
        B_members_metrics,
        B_nonmembers_metrics,
        outliers=None,
        normalize=args.normalize,
        allowed_keys=allowed_keys,
        return_tensors=True,
    )

    dataset_results = []
    for random_index in range(args.num_random):
        train_metrics = np.array(base_train_metrics, copy=True)
        val_metrics = np.array(base_val_metrics, copy=True)
        np.random.shuffle(train_metrics)
        np.random.shuffle(val_metrics)

        probe_sample_size = max(1, train_metrics.shape[0] // 2)
        (train_x, train_y), _ = get_dataset_splits(train_metrics, val_metrics, probe_sample_size)
        probe = train_model(train_x, train_y, num_epochs=args.train_epochs)

        B_members_preds, _ = get_predictions(
            probe,
            B_members_metrics_tensor,
            torch.zeros(B_members_metrics_tensor.shape[0]),
        )
        B_nonmembers_preds, _ = get_predictions(
            probe,
            B_nonmembers_metrics_tensor,
            torch.ones(B_nonmembers_metrics_tensor.shape[0]),
        )

        if args.trim_heldout_frac > 0:
            B_members_preds = trim_scores(B_members_preds, args.trim_heldout_frac)
            B_nonmembers_preds = trim_scores(B_nonmembers_preds, args.trim_heldout_frac)

        p_value = ttest_ind(B_members_preds, B_nonmembers_preds, alternative="less").pvalue
        dataset_results.append(
            {
                "dataset": dataset_name,
                "random_index": int(random_index),
                "requested_sample_size": int(args.sample_size),
                "effective_sample_size": int(effective_sample_size),
                "train_split_size": int(len(train_split)),
                "val_split_size": int(len(val_split)),
                "p_value": float(p_value),
                "member_mean_score": float(np.mean(B_members_preds)),
                "nonmember_mean_score": float(np.mean(B_nonmembers_preds)),
                "mean_gap_nonmember_minus_member": float(np.mean(B_nonmembers_preds) - np.mean(B_members_preds)),
                "num_member_scores": int(len(B_members_preds)),
                "num_nonmember_scores": int(len(B_nonmembers_preds)),
            }
        )

    return dataset_results


def build_run_record(args, dataset_results, runtime_seconds, peak_gpu_memory_gb):
    p_values = [row["p_value"] for row in dataset_results]
    return {
        "model_name": args.model_name,
        "model_label": args.model_label,
        "model_size_b": args.model_size_b,
        "dataset_names": sorted(set(row["dataset"] for row in dataset_results)),
        "metric_list": get_metric_drivers(args.feature_mode),
        "feature_mode": args.feature_mode,
        "sample_size": args.sample_size,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_epochs": args.train_epochs,
        "num_random": args.num_random,
        "normalize": args.normalize,
        "outliers": args.outliers,
        "trim_heldout_frac": args.trim_heldout_frac,
        "runtime_seconds": float(runtime_seconds),
        "peak_gpu_memory_gb": float(peak_gpu_memory_gb),
        "summary": {
            "num_datasets": len(sorted(set(row["dataset"] for row in dataset_results))),
            "num_p_values": len(dataset_results),
            "mean_p_value": float(np.mean(p_values)),
            "median_p_value": float(np.median(p_values)),
            "num_below_0_1": int(sum(p < 0.1 for p in p_values)),
        },
        "dataset_results": dataset_results,
    }


def load_history(history_path):
    if not history_path.exists():
        return {"runs": []}
    return json.loads(history_path.read_text(encoding="utf-8"))


def load_partial_run(partial_run_path):
    if not partial_run_path.exists():
        return None
    return json.loads(partial_run_path.read_text(encoding="utf-8"))


def write_partial_run(partial_run_path, partial_run):
    partial_run_path.parent.mkdir(parents=True, exist_ok=True)
    partial_run_path.write_text(json.dumps(partial_run, indent=2), encoding="utf-8")


def upsert_run(history, run_record):
    runs = history.get("runs", [])
    replacement_key = (
        run_record["model_name"],
        run_record["model_label"],
        float(run_record["model_size_b"]),
    )
    new_runs = []
    replaced = False
    for run in runs:
        run_key = (run["model_name"], run["model_label"], float(run["model_size_b"]))
        if run_key == replacement_key:
            new_runs.append(run_record)
            replaced = True
        else:
            new_runs.append(run)
    if not replaced:
        new_runs.append(run_record)
    history["runs"] = new_runs
    return history


def write_history_files(output_dir, history_filename, run_record):
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / history_filename
    history = load_history(history_path)
    history = upsert_run(history, run_record)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_filename = build_run_filename(
        run_record["model_name"],
        run_record["model_label"],
        run_record["model_size_b"],
    )
    run_path = runs_dir / run_filename
    run_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    return history, history_path, run_path


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
    requested_dataset_names = parse_dataset_names(args.dataset_names)
    dataset_names, skipped_datasets, available_datasets = resolve_supported_dataset_names(requested_dataset_names)
    if skipped_datasets:
        print(
            "Skipping unsupported dataset configs from the local HF release:",
            ", ".join(skipped_datasets),
        )
    if not dataset_names:
        raise ValueError(
            "None of the requested dataset names are available. "
            f"Available configs: {available_datasets}"
        )
    output_dir = PROJECT_ROOT / args.output_dir
    figure_path = output_dir / args.figure_file
    runs_dir = output_dir / "runs"
    run_filename = build_run_filename(args.model_name, args.model_label, args.model_size_b)
    partial_run_path = runs_dir / (run_filename.replace(".json", ".partial.json"))
    device = choose_device(args.device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.max_length,
        args.cache_dir,
        device,
    )

    partial_run = load_partial_run(partial_run_path)
    if partial_run is None:
        partial_run = {
            "model_name": args.model_name,
            "model_label": args.model_label,
            "model_size_b": args.model_size_b,
            "requested_dataset_names": requested_dataset_names,
            "skipped_dataset_names": skipped_datasets,
            "available_dataset_names": available_datasets,
            "dataset_results": [],
            "failed_datasets": [],
        }
    completed_datasets = {row["dataset"] for row in partial_run.get("dataset_results", [])}

    dataset_results = list(partial_run.get("dataset_results", []))
    failed_datasets = list(partial_run.get("failed_datasets", []))
    for dataset_name in dataset_names:
        if dataset_name in completed_datasets:
            print(f"Skipping already completed dataset {dataset_name}")
            continue
        print(f"Running {args.model_name} on {dataset_name}")
        try:
            result_rows = run_one_dataset(model, tokenizer, dataset_name, args, device)
            dataset_results.extend(result_rows)
            partial_run["dataset_results"] = dataset_results
            write_partial_run(partial_run_path, partial_run)
        except Exception as exc:
            failed_entry = {"dataset": dataset_name, "error": str(exc)}
            failed_datasets.append(failed_entry)
            partial_run["failed_datasets"] = failed_datasets
            write_partial_run(partial_run_path, partial_run)
            print(f"Failed on {dataset_name}: {exc}")

    peak_gpu_memory_gb = 0.0
    if device.type == "cuda":
        peak_gpu_memory_gb = float(torch.cuda.max_memory_allocated() / (1024**3))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    runtime_seconds = time.perf_counter() - start
    run_record = build_run_record(args, dataset_results, runtime_seconds, peak_gpu_memory_gb)
    run_record["requested_dataset_names"] = requested_dataset_names
    run_record["skipped_dataset_names"] = skipped_datasets
    run_record["available_dataset_names"] = available_datasets
    run_record["failed_datasets"] = failed_datasets
    history, history_path, run_path = write_history_files(output_dir, args.history_file, run_record)
    if partial_run_path.exists():
        partial_run_path.unlink()

    history_df = history_to_dataframe(history)
    history_df.to_csv(output_dir / "history_table.csv", index=False)
    plot_history(history_df, figure_path)

    print(json.dumps(
        {
            "history_path": str(history_path),
            "run_path": str(run_path),
            "figure_path": str(figure_path),
            "num_datasets": len(dataset_results),
            "median_p_value": run_record["summary"]["median_p_value"],
            "mean_p_value": run_record["summary"]["mean_p_value"],
            "num_below_0_1": run_record["summary"]["num_below_0_1"],
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
            "runtime_seconds": runtime_seconds,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
