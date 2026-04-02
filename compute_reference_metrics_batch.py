import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"

REFERENCE_MODEL_REGISTRY = {
    "tinystories-33M": "roneneldan/TinyStories-33M",
    "tinystories-1M": "roneneldan/TinyStories-1M",
    "phi-1_5": "microsoft/phi-1_5",
}

DEFAULT_DATASETS = [
    "stackexchange",
    "wikipedia",
    "cc",
    "github",
    "openwebtext2",
    "freelaw",
    "math",
    "uspto",
    "hackernews",
    "books3",
    "gutenberg",
    "arxiv",
    "bookcorpus2",
    "opensubtitles",
    "youtubesubtitles",
    "ubuntu",
    "europarl",
    "philpapers",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute reference-model PPL metrics for multiple datasets.")
    parser.add_argument(
        "--reference-model-aliases",
        default=",".join(REFERENCE_MODEL_REGISTRY.keys()),
        help="Comma-separated aliases to compute.",
    )
    parser.add_argument(
        "--dataset-names",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--manifest-path", default="results/reference_metrics_manifest.json")
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


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


def raw_values_batch(model, tokenizer, texts):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    if model.device.type == "cuda":
        encoded = {k: v.cuda() for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    labels = encoded["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    shifted_labels = labels[..., 1:].contiguous().view(-1)
    shifted_logits = outputs.logits[..., :-1, :].contiguous().view(-1, outputs.logits.size(-1))
    losses = torch.nn.functional.cross_entropy(shifted_logits, shifted_labels, reduction="none")
    losses = losses.view(labels.size(0), labels.size(1) - 1)

    result = []
    for row in losses.tolist():
        row = [value for value in row if value != 0]
        if row:
            result.append(row)
    return result


def raw_values(model, tokenizer, texts, batch_size):
    all_losses = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        all_losses.extend(raw_values_batch(model, tokenizer, batch_texts))
    return all_losses


def perplexity(losses_per_sample):
    mean_loss = sum(losses_per_sample) / len(losses_per_sample)
    return torch.exp(torch.tensor(mean_loss)).item()


def split_a_b(dataset_name, sample_size):
    ds = load_dataset("pratyushmaini/llm_dataset_inference", dataset_name)
    val_key = "val" if "val" in ds else "validation"
    train_split = ds["train"]
    val_split = ds[val_key]
    max_supported = min(len(train_split), len(val_split)) // 2
    effective_sample_size = min(sample_size, max_supported)
    if effective_sample_size < 1:
        raise ValueError(f"Dataset too small for A/B split: {dataset_name}")
    return {
        "A_members": train_split.select(range(0, effective_sample_size)),
        "A_nonmembers": val_split.select(range(0, effective_sample_size)),
        "B_members": train_split.select(range(effective_sample_size, effective_sample_size * 2)),
        "B_nonmembers": val_split.select(range(effective_sample_size, effective_sample_size * 2)),
        "_effective_sample_size": effective_sample_size,
        "_train_split_size": len(train_split),
        "_val_split_size": len(val_split),
    }


def load_manifest(path):
    if not path.exists():
        return {"runs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    device = choose_device(args.device)
    aliases = parse_csv_list(args.reference_model_aliases)
    datasets = parse_csv_list(args.dataset_names)
    manifest_path = PROJECT_ROOT / args.manifest_path
    manifest = load_manifest(manifest_path)

    run_entry = {
        "started_at_epoch": time.time(),
        "device": device.type,
        "sample_size": args.sample_size,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "reference_model_aliases": aliases,
        "dataset_names": datasets,
        "completed": [],
        "skipped_existing": [],
        "failed": [],
    }
    manifest["runs"].append(run_entry)
    save_manifest(manifest_path, manifest)

    for alias in aliases:
        model_name = REFERENCE_MODEL_REGISTRY[alias]
        print(f"Loading reference model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name, args.max_length, args.cache_dir, device)
        try:
            for dataset_name in datasets:
                split_map = split_a_b(dataset_name, args.sample_size)
                for split_name in ["A_members", "A_nonmembers", "B_members", "B_nonmembers"]:
                    output_path = RESULTS_ROOT / model_name / f"{dataset_name}_{split_name}_metrics.json"
                    if output_path.exists():
                        run_entry["skipped_existing"].append(
                            {"alias": alias, "dataset": dataset_name, "split": split_name, "path": str(output_path)}
                        )
                        save_manifest(manifest_path, manifest)
                        continue

                    print(f"Computing {alias} on {dataset_name} {split_name}")
                    texts = split_map[split_name]["text"]
                    losses = raw_values(model, tokenizer, texts, args.batch_size)
                    metrics = {"ppl": [perplexity(sample_losses) for sample_losses in losses]}
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(json.dumps(metrics), encoding="utf-8")
                    run_entry["completed"].append(
                        {
                            "alias": alias,
                            "dataset": dataset_name,
                            "split": split_name,
                            "path": str(output_path),
                            "effective_sample_size": split_map["_effective_sample_size"],
                            "train_split_size": split_map["_train_split_size"],
                            "val_split_size": split_map["_val_split_size"],
                        }
                    )
                    save_manifest(manifest_path, manifest)
        except Exception as exc:
            run_entry["failed"].append({"alias": alias, "error": str(exc)})
            save_manifest(manifest_path, manifest)
            raise
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    run_entry["finished_at_epoch"] = time.time()
    save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
