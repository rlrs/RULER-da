#!/usr/bin/env python3
"""
Plot NIAH performance from RULER outputs.

Generates:
- Accuracy vs context length (per task)
- Accuracy vs relative needle depth (for a selected length)

Inputs (created by scripts/launch.py or the legacy pipeline):
- Predictions: benchmark_root/<model>/<benchmark>/<seq>/pred/<task>.jsonl
- Data:        benchmark_root/<model>/<benchmark>/<seq>/data/<task>/validation.jsonl

Usage example:
  python scripts/plot/plot_niah.py \
    --root benchmark_root/<model>/synthetic \
    --niah_tasks niah_single_2,niah_multikey_1 \
    --seqs 4096,8192,16384,32768 \
    --out_dir plots
"""

import argparse
import json
import os
import unicodedata
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


# Keep in sync with scripts/data/synthetic/constants.py for 'niah'
TOKENS_TO_GENERATE = 128


def norm(s: str) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", s).casefold().strip()


def load_jsonl(p: Path):
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def correctness_niah(pred: str, refs):
    # string_match_all: all gold values must appear in prediction
    p = norm(pred)
    rlist = [norm(r) for r in refs]
    return 1.0 if all(r in p for r in rlist) else 0.0


def plot_acc_vs_length(root: Path, tasks, seqs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        xs, ys = [], []
        for L in seqs:
            pred_file = root / str(L) / "pred" / f"{task}.jsonl"
            data_file = root / str(L) / "data" / task / "validation.jsonl"
            if not pred_file.exists() or not data_file.exists():
                continue
            preds = load_jsonl(pred_file)
            total, correct = 0, 0
            for row in preds:
                pred_text = row.get("pred", "")
                refs = row.get("outputs", [])
                if not isinstance(refs, list) or not refs:
                    continue
                total += 1
                if correctness_niah(pred_text, refs):
                    correct += 1
            if total > 0:
                xs.append(L)
                ys.append(100.0 * correct / total)
        if xs:
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.title(f"{task} — Accuracy vs Context Length")
            plt.xlabel("Max sequence length (tokens)")
            plt.ylabel("Accuracy (%)")
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir / f"{task}_acc_vs_length.png", dpi=150, bbox_inches="tight")
            plt.close()


def plot_acc_vs_depth(root: Path, task: str, seq_len: int, out_dir: Path, bins: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = root / str(seq_len) / "pred" / f"{task}.jsonl"
    data_file = root / str(seq_len) / "data" / task / "validation.jsonl"
    if not pred_file.exists() or not data_file.exists():
        return
    preds = {r["index"]: r for r in load_jsonl(pred_file)}
    data = load_jsonl(data_file)

    buckets = defaultdict(lambda: [0, 0])  # bin -> [correct, total]
    for row in data:
        idx = row.get("index")
        if idx not in preds:
            continue
        pred_text = preds[idx].get("pred", "")
        refs = preds[idx].get("outputs", [])
        length = row.get("length", 0)
        token_pos = row.get("token_position_answer", None)
        if token_pos is None or length <= TOKENS_TO_GENERATE:
            continue
        input_len = max(1, length - TOKENS_TO_GENERATE)
        rel = max(0.0, min(0.999, token_pos / input_len))  # 0..1
        bin_id = min(bins - 1, int(rel * bins))
        correct = correctness_niah(pred_text, refs)
        buckets[bin_id][0] += correct
        buckets[bin_id][1] += 1

    xs, ys = [], []
    for b in range(bins):
        c, t = buckets[b]
        xs.append((b + 0.5) * (100.0 / bins))  # mid of bin in %
        ys.append(100.0 * c / t if t else 0.0)

    if any(t for _, t in buckets.values()):
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.title(f"{task} — Accuracy vs Needle Depth @ {seq_len}")
        plt.xlabel("Relative depth in input (%)")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / f"{task}_acc_vs_depth_{seq_len}.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="benchmark_root/<model>/synthetic")
    ap.add_argument("--niah_tasks", default="niah_single_2,niah_multikey_1")
    ap.add_argument("--seqs", default="4096,8192,16384,32768")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    tasks = [t.strip() for t in args.niah_tasks.split(",") if t.strip()]
    seqs = [int(s) for s in args.seqs.split(",") if s.strip()]

    plot_acc_vs_length(root, tasks, seqs, out_dir)
    if tasks and seqs:
        plot_acc_vs_depth(root, tasks[0], max(seqs), out_dir, bins=10)


if __name__ == "__main__":
    main()

