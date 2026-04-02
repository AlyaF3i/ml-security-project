import argparse
import json
import random
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy.stats import rankdata, ttest_ind
from transformers import AutoModelForCausalLM, AutoTokenizer


PERTURBATION_COLUMNS = [
    "synonym_substitution",
    "butter_fingers",
    "random_deletion",
    "change_char_case",
    "whitespace_perturbation",
    "underscore_trick",
]

REFERENCE_MODEL_REGISTRY = {
    "silo": "kernelmachine/silo-pdswby-1.3b",
    "tinystories-33M": "roneneldan/TinyStories-33M",
    "tinystories-1M": "roneneldan/TinyStories-1M",
    "phi-1_5": "microsoft/phi-1_5",
}

DEFAULT_REFERENCE_MODEL_ALIASES = ["tinystories-1M", "tinystories-33M", "phi-1_5"]
DEFAULT_K_VALUES = [0.05, 0.1, 0.2]


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Windows-friendly dataset inference run.")
    parser.add_argument("--model-name", default="EleutherAI/pythia-410m")
    parser.add_argument("--dataset-name", default="wikipedia")
    parser.add_argument("--member-split", default="train")
    parser.add_argument("--nonmember-split", default="val")
    parser.add_argument("--num-member", type=int, default=40)
    parser.add_argument("--num-nonmember", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=11)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--aggregation-method", choices=["linear", "effect"], default="linear")
    parser.add_argument("--train-epochs", type=int, default=500)
    parser.add_argument("--include-reference-features", action="store_true")
    parser.add_argument(
        "--reference-model-aliases",
        default=",".join(DEFAULT_REFERENCE_MODEL_ALIASES),
        help="Comma-separated aliases from the official repo registry.",
    )
    parser.add_argument("--reference-cache-dir", default="results/reference_model_cache")
    parser.add_argument("--output-dir", default="results/minimal_pythia410m_wikipedia")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg):
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_reference_aliases(alias_string):
    aliases = [alias.strip() for alias in alias_string.split(",") if alias.strip()]
    unknown = [alias for alias in aliases if alias not in REFERENCE_MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown reference aliases: {unknown}")
    return aliases


def sanitize_name(name):
    return name.replace("/", "__").replace("\\", "__").replace(":", "_")


def reset_gpu_peak_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def load_model_and_tokenizer(model_name, max_length, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def load_member_nonmember_dataset(args):
    member = load_dataset(
        "pratyushmaini/llm_dataset_inference",
        args.dataset_name,
        split=f"{args.member_split}[:{args.num_member}]",
    )
    nonmember = load_dataset(
        "pratyushmaini/llm_dataset_inference",
        args.dataset_name,
        split=f"{args.nonmember_split}[:{args.num_nonmember}]",
    )
    return member, nonmember


def raw_token_losses(model, tokenizer, texts, batch_size, device):
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)

        labels = encoded["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        shifted_labels = labels[:, 1:].contiguous()
        shifted_logits = outputs.logits[:, :-1, :].contiguous()

        losses = loss_fct(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        ).view(shifted_labels.size(0), shifted_labels.size(1))

        valid_mask = shifted_labels != -100
        for row_losses, row_mask in zip(losses, valid_mask):
            values = row_losses[row_mask].detach().float().cpu().numpy().tolist()
            all_losses.append(values)
    return all_losses


def mean_of_fraction(values, fraction, largest=False):
    ordered = sorted(values, reverse=largest)
    count = max(1, int(len(ordered) * fraction))
    return float(np.mean(ordered[:count]))


def perplexity(losses):
    return float(np.exp(np.mean(losses)))


def zlib_ratio(losses, text):
    return float(np.mean(losses) / max(1, len(zlib.compress(text.encode("utf-8")))))


def ppl_ratio(base_losses, perturbed_losses):
    return float(np.mean(base_losses) / np.mean(perturbed_losses))


def compute_metric_frame(model, tokenizer, dataset, batch_size, device, k_values):
    texts = dataset["text"]
    base_losses = raw_token_losses(model, tokenizer, texts, batch_size, device)

    metrics = {
        "ppl": [perplexity(losses) for losses in base_losses],
        "zlib_ratio": [zlib_ratio(losses, text) for losses, text in zip(base_losses, texts)],
    }
    for k in k_values:
        metrics[f"k_min_probs_{k}"] = [mean_of_fraction(losses, k, largest=False) for losses in base_losses]
        metrics[f"k_max_probs_{k}"] = [mean_of_fraction(losses, k, largest=True) for losses in base_losses]

    for column in PERTURBATION_COLUMNS:
        perturbed_losses = raw_token_losses(model, tokenizer, dataset[column], batch_size, device)
        metrics[f"ppl_ratio_{column}"] = [
            ppl_ratio(base, perturbed)
            for base, perturbed in zip(base_losses, perturbed_losses)
        ]

    frame = pd.DataFrame(metrics)
    frame.insert(0, "text", texts)
    return frame, base_losses


def compute_reference_ppl(reference_model_name, texts, split_name, args, device):
    cache_dir = Path(args.reference_cache_dir) / sanitize_name(reference_model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (
        f"{args.dataset_name}_{split_name}_n{len(texts)}_l{args.max_length}_ppl.csv"
    )

    if cache_file.exists():
        cached = pd.read_csv(cache_file)
        return cached["ppl"].tolist(), {
            "model_name": reference_model_name,
            "split_name": split_name,
            "cache_hit": True,
            "runtime_seconds": 0.0,
            "peak_gpu_memory_gb": 0.0,
            "cache_file": str(cache_file),
        }

    reset_gpu_peak_stats()
    start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(reference_model_name, args.max_length, device)
    losses = raw_token_losses(model, tokenizer, texts, args.batch_size, device)
    ppl_values = [perplexity(losses_per_sample) for losses_per_sample in losses]
    runtime_seconds = time.perf_counter() - start
    peak_gpu_memory_gb = get_peak_gpu_memory_gb()
    pd.DataFrame({"ppl": ppl_values}).to_csv(cache_file, index=False)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ppl_values, {
        "model_name": reference_model_name,
        "split_name": split_name,
        "cache_hit": False,
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "cache_file": str(cache_file),
    }


def add_reference_features(member_scores, nonmember_scores, member_losses, nonmember_losses, member_ds, nonmember_ds, args, device):
    runtime_details = []
    aliases = parse_reference_aliases(args.reference_model_aliases)
    split_payload = {
        "member": (member_scores, member_losses, member_ds["text"]),
        "nonmember": (nonmember_scores, nonmember_losses, nonmember_ds["text"]),
    }

    for alias in aliases:
        ref_model_name = REFERENCE_MODEL_REGISTRY[alias]
        column_name = f"ref_ppl_ratio_{alias}"
        for split_name, (frame, base_losses, texts) in split_payload.items():
            ref_ppl, detail = compute_reference_ppl(ref_model_name, texts, split_name, args, device)
            runtime_details.append({"alias": alias, **detail})
            frame[column_name] = [
                float(np.mean(losses) / ref_ppl_value)
                for losses, ref_ppl_value in zip(base_losses, ref_ppl)
            ]
    return runtime_details


def clip_feature(values, fraction=0.05):
    values = np.asarray(values, dtype=np.float64).copy()
    if len(values) < 8:
        return values
    trim_each_side = max(1, int(round(len(values) * fraction / 2)))
    order = np.argsort(values)
    low_ids = order[:trim_each_side]
    high_ids = order[-trim_each_side:]
    low_bound = values[order[trim_each_side]]
    high_bound = values[order[-trim_each_side - 1]]
    values[low_ids] = low_bound
    values[high_ids] = high_bound
    return values


def auc_from_scores(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int(labels.sum())
    neg = int((1 - labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    pos_ranks = ranks[labels == 1].sum()
    auc = (pos_ranks - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


def sidak_combine(p_values):
    clipped = np.clip(np.asarray(p_values, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(1 - np.exp(np.sum(np.log(1 - clipped))))


def fit_linear_probe(train_x, train_y, epochs):
    model = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(train_x).squeeze(-1)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        weights = model.weight.detach().cpu().numpy().reshape(-1)
    return model, weights


def build_dataset_level_results(
    member_df,
    nonmember_df,
    num_seeds,
    base_seed,
    aggregation_method,
    train_epochs,
):
    feature_cols = [c for c in member_df.columns if c != "text"]
    sample_rows = []
    dataset_rows = []

    for seed_offset in range(num_seeds):
        seed = base_seed + seed_offset
        rng = np.random.default_rng(seed)
        member_ids = rng.permutation(len(member_df))
        nonmember_ids = rng.permutation(len(nonmember_df))

        half_member = len(member_ids) // 2
        half_nonmember = len(nonmember_ids) // 2
        member_a, member_b = member_ids[:half_member], member_ids[half_member:]
        nonmember_a, nonmember_b = nonmember_ids[:half_nonmember], nonmember_ids[half_nonmember:]

        weights = {}
        matrices = {
            "member_a": [],
            "nonmember_a": [],
            "member_b": [],
            "nonmember_b": [],
        }

        for feature in feature_cols:
            member_values = member_df[feature].to_numpy(dtype=np.float64)
            nonmember_values = nonmember_df[feature].to_numpy(dtype=np.float64)

            member_a_vals = clip_feature(member_values[member_a])
            nonmember_a_vals = clip_feature(nonmember_values[nonmember_a])
            member_b_vals = clip_feature(member_values[member_b])
            nonmember_b_vals = clip_feature(nonmember_values[nonmember_b])

            combined_a = np.concatenate([member_a_vals, nonmember_a_vals])
            mean_a = combined_a.mean()
            std_a = combined_a.std() if combined_a.std() > 1e-8 else 1.0

            member_a_norm = (member_a_vals - mean_a) / std_a
            nonmember_a_norm = (nonmember_a_vals - mean_a) / std_a
            member_b_norm = (member_b_vals - mean_a) / std_a
            nonmember_b_norm = (nonmember_b_vals - mean_a) / std_a

            direction = 1.0 if member_a_norm.mean() <= nonmember_a_norm.mean() else -1.0
            member_a_score = direction * member_a_norm
            nonmember_a_score = direction * nonmember_a_norm
            member_b_score = direction * member_b_norm
            nonmember_b_score = direction * nonmember_b_norm

            effect = max(0.0, float(nonmember_a_score.mean() - member_a_score.mean()))
            weights[feature] = effect

            sample_auc = auc_from_scores(
                np.concatenate([np.zeros(len(member_b_score)), np.ones(len(nonmember_b_score))]),
                np.concatenate([member_b_score, nonmember_b_score]),
            )
            sample_rows.append(
                {
                    "seed": seed,
                    "feature": feature,
                    "sample_auc_heldout": sample_auc,
                    "member_mean_a": float(member_a_score.mean()),
                    "nonmember_mean_a": float(nonmember_a_score.mean()),
                }
            )
            matrices["member_a"].append(member_a_score)
            matrices["nonmember_a"].append(nonmember_a_score)
            matrices["member_b"].append(member_b_score)
            matrices["nonmember_b"].append(nonmember_b_score)

        member_a_matrix = np.stack(matrices["member_a"], axis=1)
        nonmember_a_matrix = np.stack(matrices["nonmember_a"], axis=1)
        member_b_matrix = np.stack(matrices["member_b"], axis=1)
        nonmember_b_matrix = np.stack(matrices["nonmember_b"], axis=1)

        if aggregation_method == "linear":
            train_x = np.concatenate([member_a_matrix, nonmember_a_matrix], axis=0)
            train_y = np.concatenate(
                [np.zeros(len(member_a_matrix)), np.ones(len(nonmember_a_matrix))],
                axis=0,
            )
            train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
            train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
            probe, probe_weights = fit_linear_probe(train_x_tensor, train_y_tensor, train_epochs)
            with torch.no_grad():
                member_train_scores = (
                    probe(torch.tensor(member_a_matrix, dtype=torch.float32))
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )
                nonmember_train_scores = (
                    probe(torch.tensor(nonmember_a_matrix, dtype=torch.float32))
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )
                member_agg = (
                    probe(torch.tensor(member_b_matrix, dtype=torch.float32))
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )
                nonmember_agg = (
                    probe(torch.tensor(nonmember_b_matrix, dtype=torch.float32))
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )
            orientation = 1.0 if member_train_scores.mean() <= nonmember_train_scores.mean() else -1.0
            member_agg = orientation * member_agg
            nonmember_agg = orientation * nonmember_agg
            for feature, weight in zip(feature_cols, probe_weights):
                weights[feature] = float(orientation * weight)
        else:
            weight_vector = np.array([weights[f] for f in feature_cols], dtype=np.float64)
            if np.allclose(weight_vector.sum(), 0.0):
                weight_vector = np.ones_like(weight_vector)
            weight_vector = weight_vector / weight_vector.sum()
            member_agg = member_b_matrix @ weight_vector
            nonmember_agg = nonmember_b_matrix @ weight_vector

        for row in sample_rows[-len(feature_cols) :]:
            row["weight"] = weights[row["feature"]]

        t_result = ttest_ind(member_agg, nonmember_agg, alternative="less", equal_var=False)
        dataset_rows.append(
            {
                "seed": seed,
                "heldout_member_count": int(len(member_agg)),
                "heldout_nonmember_count": int(len(nonmember_agg)),
                "member_mean_score": float(member_agg.mean()),
                "nonmember_mean_score": float(nonmember_agg.mean()),
                "mean_gap_nonmember_minus_member": float(nonmember_agg.mean() - member_agg.mean()),
                "t_statistic": float(t_result.statistic),
                "p_value": float(t_result.pvalue),
            }
        )

    sample_df = pd.DataFrame(sample_rows)
    dataset_df = pd.DataFrame(dataset_rows)
    return sample_df, dataset_df


def build_summary(args, device, sample_df, dataset_df):
    feature_rank = (
        sample_df.groupby("feature", as_index=False)["sample_auc_heldout"]
        .median()
        .sort_values("sample_auc_heldout", ascending=False)
    )
    best_feature = feature_rank.iloc[0].to_dict()
    summary = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "member_split": args.member_split,
        "nonmember_split": args.nonmember_split,
        "num_member": args.num_member,
        "num_nonmember": args.num_nonmember,
        "num_seeds": args.num_seeds,
        "device_used": str(device),
        "aggregation_method": args.aggregation_method,
        "include_reference_features": bool(args.include_reference_features),
        "reference_model_aliases": parse_reference_aliases(args.reference_model_aliases)
        if args.include_reference_features
        else [],
        "best_single_feature": {
            "name": best_feature["feature"],
            "median_heldout_auc": float(best_feature["sample_auc_heldout"]),
        },
        "dataset_level": {
            "mean_p_value": float(dataset_df["p_value"].mean()),
            "median_p_value": float(dataset_df["p_value"].median()),
            "sidak_combined_p_value": sidak_combine(dataset_df["p_value"].tolist()),
            "mean_gap_nonmember_minus_member": float(
                dataset_df["mean_gap_nonmember_minus_member"].mean()
            ),
        },
    }
    return summary, feature_rank


def write_summary_markdown(path, summary, feature_rank, dataset_df):
    top_features = feature_rank.head(8)
    lines = [
        "# Minimal Dataset Inference Summary",
        "",
        "## Run",
        f"- Model: `{summary['model_name']}`",
        f"- Dataset subset: `{summary['dataset_name']}`",
        f"- Member split: `{summary['member_split']}`",
        f"- Non-member split: `{summary['nonmember_split']}`",
        f"- Samples per side: `{summary['num_member']}` / `{summary['num_nonmember']}`",
        f"- Seeds: `{summary['num_seeds']}`",
        f"- Device: `{summary['device_used']}`",
        f"- Aggregation method: `{summary['aggregation_method']}`",
        f"- Reference features enabled: `{summary['include_reference_features']}`",
    ]
    if summary["reference_model_aliases"]:
        lines.append(f"- Reference models: `{', '.join(summary['reference_model_aliases'])}`")
    if "runtime_seconds" in summary:
        lines.append(f"- Total runtime: `{summary['runtime_seconds']:.2f}` seconds")
    if "peak_gpu_memory_gb" in summary:
        lines.append(f"- Peak GPU memory observed: `{summary['peak_gpu_memory_gb']:.2f}` GiB")

    lines.extend(
        [
            "",
            "## What Was Reproduced",
            "- Inference-only feature extraction from a pretrained causal LM.",
            "- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.",
            "- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.",
            "- Combined p-value across dependent tests using the Sidak-style formula from the paper.",
            "",
            "## Key Result",
            f"- Best single feature on held-out samples: `{summary['best_single_feature']['name']}` with median AUC `{summary['best_single_feature']['median_heldout_auc']:.4f}`.",
            f"- Mean dataset-level p-value across seeds: `{summary['dataset_level']['mean_p_value']:.6f}`.",
            f"- Median dataset-level p-value across seeds: `{summary['dataset_level']['median_p_value']:.6f}`.",
            f"- Combined dataset-level p-value: `{summary['dataset_level']['sidak_combined_p_value']:.6f}`.",
            f"- Mean held-out score gap (non-member minus member): `{summary['dataset_level']['mean_gap_nonmember_minus_member']:.6f}`.",
            "",
            "## Top Held-Out Features",
        ]
    )
    for row in top_features.itertuples(index=False):
        lines.append(f"- `{row.feature}`: median held-out AUC `{row.sample_auc_heldout:.4f}`")

    lines.extend(["", "## Seed-Level P-Values"])
    for row in dataset_df.itertuples(index=False):
        lines.append(
            f"- Seed `{row.seed}`: p-value `{row.p_value:.6f}`, member mean `{row.member_mean_score:.4f}`, non-member mean `{row.nonmember_mean_score:.4f}`"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.base_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    member_ds, nonmember_ds = load_member_nonmember_dataset(args)

    overall_start = time.perf_counter()
    reset_gpu_peak_stats()
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.max_length, device)
    victim_after_load_peak = get_peak_gpu_memory_gb()

    member_start = time.perf_counter()
    member_scores, member_losses = compute_metric_frame(
        model,
        tokenizer,
        member_ds,
        args.batch_size,
        device,
        DEFAULT_K_VALUES,
    )
    member_runtime = time.perf_counter() - member_start

    nonmember_start = time.perf_counter()
    nonmember_scores, nonmember_losses = compute_metric_frame(
        model,
        tokenizer,
        nonmember_ds,
        args.batch_size,
        device,
        DEFAULT_K_VALUES,
    )
    nonmember_runtime = time.perf_counter() - nonmember_start
    victim_peak_gpu_memory_gb = get_peak_gpu_memory_gb()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reference_runtime_details = []
    if args.include_reference_features:
        reference_runtime_details = add_reference_features(
            member_scores,
            nonmember_scores,
            member_losses,
            nonmember_losses,
            member_ds,
            nonmember_ds,
            args,
            device,
        )

    member_scores.insert(0, "label", "member")
    nonmember_scores.insert(0, "label", "nonmember")
    member_scores.insert(1, "source_split", args.member_split)
    nonmember_scores.insert(1, "source_split", args.nonmember_split)

    all_scores = pd.concat([member_scores, nonmember_scores], ignore_index=True)
    sample_df, dataset_df = build_dataset_level_results(
        member_scores.drop(columns=["label", "source_split"]),
        nonmember_scores.drop(columns=["label", "source_split"]),
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
        aggregation_method=args.aggregation_method,
        train_epochs=args.train_epochs,
    )
    summary, feature_rank = build_summary(args, device, sample_df, dataset_df)
    summary["runtime_seconds"] = float(time.perf_counter() - overall_start)
    summary["peak_gpu_memory_gb"] = max(
        victim_after_load_peak,
        victim_peak_gpu_memory_gb,
        max((detail["peak_gpu_memory_gb"] for detail in reference_runtime_details), default=0.0),
    )
    summary["runtime_breakdown_seconds"] = {
        "victim_member_metrics": float(member_runtime),
        "victim_nonmember_metrics": float(nonmember_runtime),
        "reference_models_total": float(sum(detail["runtime_seconds"] for detail in reference_runtime_details)),
    }
    summary["gpu_memory_breakdown_gb"] = {
        "victim_after_load": float(victim_after_load_peak),
        "victim_peak": float(victim_peak_gpu_memory_gb),
        **{
            f"reference_{detail['alias']}_{detail['split_name']}": float(detail["peak_gpu_memory_gb"])
            for detail in reference_runtime_details
        },
    }
    summary["reference_runtime_details"] = reference_runtime_details

    all_scores.to_csv(output_dir / "per_sample_scores.csv", index=False)
    sample_df.to_csv(output_dir / "sample_level_feature_metrics.csv", index=False)
    dataset_df.to_csv(output_dir / "dataset_level_results.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(output_dir / "summary.md", summary, feature_rank, dataset_df)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
