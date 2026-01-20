#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_eval_results(model: str, extraction: str) -> dict | None:
    path = Path(f"artifacts/evaluations/direction_eval_{model}_{extraction}.pt")
    if not path.exists():
        return None
    return torch.load(path, weights_only=True)


def plot_direction_discovery(output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    configs = [
        ("qwen", "last_token"),
        ("qwen", "response_mean"),
        ("llama", "last_token"),
        ("llama", "response_mean"),
    ]

    for ax, (model, extraction) in zip(axes.flat, configs):
        results = load_eval_results(model, extraction)
        if results is None:
            ax.text(0.5, 0.5, f"No data for {model}/{extraction}", ha="center", va="center")
            ax.set_title(f"{model} - {extraction}")
            continue

        all_results = results["all_results"]
        layers = sorted(all_results.keys())
        cohens_d = [all_results[l]["cohens_d"] for l in layers]
        accuracy = [all_results[l]["accuracy"] for l in layers]

        ax.plot(layers, cohens_d, "b-", label="Cohen's d", linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cohen's d", color="b")
        ax.tick_params(axis="y", labelcolor="b")

        ax2 = ax.twinx()
        ax2.plot(layers, accuracy, "r--", label="Accuracy", linewidth=2)
        ax2.set_ylabel("Accuracy", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_ylim(0, 1.1)

        best_layer = results["best_layer"]
        ax.axvline(x=best_layer, color="green", linestyle=":", alpha=0.7, label=f"Best (layer {best_layer})")

        ax.set_title(f"{model} - {extraction}")
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "direction_discovery.png", dpi=150)
    plt.close()
    print(f"Saved direction_discovery.png")


def plot_layer_comparison(output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["cohens_d", "accuracy"]):
        for model in ["qwen", "llama"]:
            for extraction, style in [("last_token", "-"), ("response_mean", "--")]:
                results = load_eval_results(model, extraction)
                if results is None:
                    continue

                all_results = results["all_results"]
                layers = sorted(all_results.keys())
                values = [all_results[l][metric] for l in layers]

                label = f"{model} ({extraction.replace('_', ' ')})"
                ax.plot(layers, values, style, label=label, linewidth=2)

        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        ax.set_title(f"{metric.replace('_', ' ').title()} by Layer")

    plt.tight_layout()
    plt.savefig(output_dir / "layer_comparison.png", dpi=150)
    plt.close()
    print(f"Saved layer_comparison.png")


def plot_projection_stats(output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    configs = [
        ("qwen", "last_token"),
        ("qwen", "response_mean"),
        ("llama", "last_token"),
        ("llama", "response_mean"),
    ]

    for ax, (model, extraction) in zip(axes.flat, configs):
        results = load_eval_results(model, extraction)
        if results is None:
            ax.text(0.5, 0.5, f"No data for {model}/{extraction}", ha="center", va="center")
            ax.set_title(f"{model} - {extraction}")
            continue

        all_results = results["all_results"]
        layers = sorted(all_results.keys())

        assistant_mean = [all_results[l]["assistant_mean"] for l in layers]
        non_assistant_mean = [all_results[l]["non_assistant_mean"] for l in layers]
        assistant_std = [all_results[l]["assistant_std"] for l in layers]
        non_assistant_std = [all_results[l]["non_assistant_std"] for l in layers]

        ax.fill_between(
            layers,
            [m - s for m, s in zip(assistant_mean, assistant_std)],
            [m + s for m, s in zip(assistant_mean, assistant_std)],
            alpha=0.3,
            color="blue",
        )
        ax.plot(layers, assistant_mean, "b-", label="Assistant", linewidth=2)

        ax.fill_between(
            layers,
            [m - s for m, s in zip(non_assistant_mean, non_assistant_std)],
            [m + s for m, s in zip(non_assistant_mean, non_assistant_std)],
            alpha=0.3,
            color="red",
        )
        ax.plot(layers, non_assistant_mean, "r-", label="Non-assistant", linewidth=2)

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Projection onto direction")
        ax.set_title(f"{model} - {extraction}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "projection_stats.png", dpi=150)
    plt.close()
    print(f"Saved projection_stats.png")


def plot_intervention_results(output_dir: Path):
    scores_path = Path("artifacts/evaluations/intervention_scores.json")
    if not scores_path.exists():
        print("Skipping intervention plots (no intervention_scores.json)")
        return

    with open(scores_path) as f:
        data = json.load(f)

    experiments = data["experiments"]

    # Steering plot
    steering = [e for e in experiments if e["intervention"] == "steering"]
    if steering:
        scales = sorted(set(e["scale"] for e in steering))
        assistant_scores = []
        coherence_scores = []
        baseline_assistant = None
        baseline_coherence = None

        for scale in scales:
            subset = [e for e in steering if e["scale"] == scale]
            valid = [e for e in subset if e["scores"]["assistant_score"] >= 0]
            if valid:
                avg_assistant = sum(e["scores"]["assistant_score"] for e in valid) / len(valid)
                avg_coherence = sum(e["scores"]["coherence_score"] for e in valid) / len(valid)
            else:
                avg_assistant = 0
                avg_coherence = 0
            assistant_scores.append(avg_assistant)
            coherence_scores.append(avg_coherence)
            if scale == 0:
                baseline_assistant = avg_assistant
                baseline_coherence = avg_coherence

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(scales))

        ax.plot(x, assistant_scores, "o-", label="Assistant-likeness", color="steelblue", linewidth=2, markersize=8)
        ax.plot(x, coherence_scores, "s-", label="Coherence", color="coral", linewidth=2, markersize=8)

        if baseline_assistant is not None:
            ax.axhline(y=baseline_assistant, color="steelblue", linestyle="--", alpha=0.5, label=f"Baseline assistant ({baseline_assistant:.1f})")
            ax.axhline(y=baseline_coherence, color="coral", linestyle="--", alpha=0.5, label=f"Baseline coherence ({baseline_coherence:.1f})")

        ax.set_xlabel("Steering Scale")
        ax.set_ylabel("Score (0-10)")
        ax.set_title("Additive Steering: LLM Judge Scores (Qwen 3B)")
        ax.set_xticks(x)
        ax.set_xticklabels(scales)
        ax.legend(loc="upper left")
        ax.set_ylim(-0.5, 10.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "steering_evaluation.png", dpi=150)
        plt.close()
        print("Saved steering_evaluation.png")

    # Capping plot
    capping = [e for e in experiments if e["intervention"] == "capping"]
    if capping:
        thresholds = sorted(set(e["threshold"] for e in capping))
        assistant_scores = []
        coherence_scores = []
        baseline_assistant = None
        baseline_coherence = None

        for threshold in thresholds:
            subset = [e for e in capping if e["threshold"] == threshold]
            valid = [e for e in subset if e["scores"]["assistant_score"] >= 0]
            if valid:
                avg_assistant = sum(e["scores"]["assistant_score"] for e in valid) / len(valid)
                avg_coherence = sum(e["scores"]["coherence_score"] for e in valid) / len(valid)
            else:
                avg_assistant = 0
                avg_coherence = 0
            assistant_scores.append(avg_assistant)
            coherence_scores.append(avg_coherence)
            if threshold == 0:
                baseline_assistant = avg_assistant
                baseline_coherence = avg_coherence

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(thresholds))

        ax.plot(x, assistant_scores, "o-", label="Assistant-likeness", color="steelblue", linewidth=2, markersize=8)
        ax.plot(x, coherence_scores, "s-", label="Coherence", color="coral", linewidth=2, markersize=8)

        if baseline_assistant is not None:
            ax.axhline(y=baseline_assistant, color="steelblue", linestyle="--", alpha=0.5, label=f"Baseline assistant ({baseline_assistant:.1f})")
            ax.axhline(y=baseline_coherence, color="coral", linestyle="--", alpha=0.5, label=f"Baseline coherence ({baseline_coherence:.1f})")

        ax.set_xlabel("Capping Threshold")
        ax.set_ylabel("Score (0-10)")
        ax.set_title("Activation Capping: LLM Judge Scores (Qwen 3B)")
        ax.set_xticks(x)
        ax.set_xticklabels(thresholds)
        ax.legend(loc="upper left")
        ax.set_ylim(-0.5, 10.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "capping_evaluation.png", dpi=150)
        plt.close()
        print("Saved capping_evaluation.png")


def main():
    parser = argparse.ArgumentParser(description="Generate analysis figures from artifacts")
    parser.add_argument("--output", type=str, default="artifacts/figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    plot_direction_discovery(output_dir)
    plot_layer_comparison(output_dir)
    plot_projection_stats(output_dir)
    plot_intervention_results(output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
