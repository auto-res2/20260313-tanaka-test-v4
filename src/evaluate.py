"""
Evaluation script for comparing multiple runs.
Fetches metrics from WandB and generates comparison plots.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON string list of run IDs"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="airas", help="WandB entity"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="20260313-tanaka-test-v4",
        help="WandB project",
    )
    return parser.parse_args()


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB API.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with run config, summary, and history
    """
    api = wandb.Api()

    # Query runs by display name (most recent)
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        print(f"Warning: No runs found for {run_id}")
        return None

    run = runs[0]  # Most recent run with that name

    print(f"Fetching data for {run_id} (WandB run: {run.id})")

    # Get run data
    config = run.config
    summary = run.summary._json_dict
    history = run.history()

    return {
        "id": run.id,
        "name": run_id,
        "config": config,
        "summary": summary,
        "history": history,
        "url": run.url,
    }


def export_run_metrics(run_data: Dict[str, Any], results_dir: str, run_id: str):
    """
    Export per-run metrics to JSON file.

    Args:
        run_data: Run data from WandB
        results_dir: Base results directory
        run_id: Run ID
    """
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    metrics = {
        "run_id": run_id,
        "wandb_url": run_data["url"],
        "summary": run_data["summary"],
        "config": run_data["config"],
    }

    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Exported metrics to {metrics_file}")


def create_run_figures(run_data: Dict[str, Any], results_dir: str, run_id: str):
    """
    Create per-run figures.

    Args:
        run_data: Run data from WandB
        results_dir: Base results directory
        run_id: Run ID
    """
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    history = run_data["history"]

    if history.empty:
        print(f"No history data for {run_id}, skipping plots")
        return

    # Set style
    sns.set_style("whitegrid")

    # Plot correctness over examples (if available)
    if "correct" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute running accuracy
        history["running_accuracy"] = history["correct"].expanding().mean()

        ax.plot(history["example_idx"], history["running_accuracy"], linewidth=2)
        ax.set_xlabel("Example Index")
        ax.set_ylabel("Running Accuracy")
        ax.set_title(f"{run_id}: Running Accuracy")
        ax.grid(True, alpha=0.3)

        fig_path = os.path.join(run_dir, "running_accuracy.pdf")
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Created figure: {fig_path}")

    # Plot answer variance (if available)
    if "answer_variance" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            history["example_idx"], history["answer_variance"], linewidth=2, alpha=0.7
        )
        ax.set_xlabel("Example Index")
        ax.set_ylabel("Answer Variance")
        ax.set_title(f"{run_id}: Answer Variance Across Samples")
        ax.grid(True, alpha=0.3)

        fig_path = os.path.join(run_dir, "answer_variance.pdf")
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Created figure: {fig_path}")


def create_comparison_figures(all_run_data: List[Dict[str, Any]], results_dir: str):
    """
    Create comparison figures across all runs.

    Args:
        all_run_data: List of run data dictionaries
        results_dir: Base results directory
    """
    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Prepare data for comparison
    run_names = [rd["name"] for rd in all_run_data]
    accuracies = [rd["summary"].get("accuracy", 0.0) for rd in all_run_data]

    # 1. Accuracy comparison bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(run_names))
    bars = ax.bar(run_names, accuracies, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Run ID", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Comparison Across Runs", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(accuracies) * 1.2 if accuracies else 1.0)
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    fig_path = os.path.join(comparison_dir, "comparison_accuracy.pdf")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Created comparison figure: {fig_path}")

    # 2. Running accuracy comparison (if history available)
    has_history = all(
        not rd["history"].empty and "correct" in rd["history"].columns
        for rd in all_run_data
    )

    if has_history:
        fig, ax = plt.subplots(figsize=(12, 6))

        for rd, color in zip(all_run_data, colors):
            history = rd["history"]
            history["running_accuracy"] = history["correct"].expanding().mean()
            ax.plot(
                history["example_idx"],
                history["running_accuracy"],
                label=rd["name"],
                linewidth=2,
                color=color,
                alpha=0.8,
            )

        ax.set_xlabel("Example Index", fontsize=12)
        ax.set_ylabel("Running Accuracy", fontsize=12)
        ax.set_title("Running Accuracy Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        fig_path = os.path.join(comparison_dir, "comparison_running_accuracy.pdf")
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Created comparison figure: {fig_path}")

    # 3. Answer variance comparison (if available)
    has_variance = all(
        not rd["history"].empty and "answer_variance" in rd["history"].columns
        for rd in all_run_data
    )

    if has_variance:
        fig, ax = plt.subplots(figsize=(12, 6))

        for rd, color in zip(all_run_data, colors):
            history = rd["history"]
            ax.plot(
                history["example_idx"],
                history["answer_variance"],
                label=rd["name"],
                linewidth=2,
                color=color,
                alpha=0.7,
            )

        ax.set_xlabel("Example Index", fontsize=12)
        ax.set_ylabel("Answer Variance", fontsize=12)
        ax.set_title("Answer Variance Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        fig_path = os.path.join(comparison_dir, "comparison_answer_variance.pdf")
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Created comparison figure: {fig_path}")


def export_aggregated_metrics(all_run_data: List[Dict[str, Any]], results_dir: str):
    """
    Export aggregated metrics across all runs.

    Args:
        all_run_data: List of run data dictionaries
        results_dir: Base results directory
    """
    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Collect metrics by run
    metrics_by_run = {}
    for rd in all_run_data:
        run_id = rd["name"]
        summary = rd["summary"]

        metrics_by_run[run_id] = {
            "accuracy": summary.get("accuracy", 0.0),
            "correct": summary.get("correct", 0),
            "total": summary.get("total", 0),
            "n_samples_processed": summary.get("n_samples_processed", 0),
        }

    # Identify proposed vs baseline
    proposed_runs = [rid for rid in metrics_by_run.keys() if "proposed" in rid.lower()]
    baseline_runs = [
        rid
        for rid in metrics_by_run.keys()
        if "comparative" in rid.lower() or "baseline" in rid.lower()
    ]

    best_proposed = None
    best_proposed_acc = -1
    if proposed_runs:
        for rid in proposed_runs:
            acc = metrics_by_run[rid]["accuracy"]
            if acc > best_proposed_acc:
                best_proposed_acc = acc
                best_proposed = rid

    best_baseline = None
    best_baseline_acc = -1
    if baseline_runs:
        for rid in baseline_runs:
            acc = metrics_by_run[rid]["accuracy"]
            if acc > best_baseline_acc:
                best_baseline_acc = acc
                best_baseline = rid

    # Compute gap
    gap = (
        best_proposed_acc - best_baseline_acc
        if (best_proposed and best_baseline)
        else 0.0
    )

    # Create aggregated metrics
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_proposed_accuracy": best_proposed_acc,
        "best_baseline": best_baseline,
        "best_baseline_accuracy": best_baseline_acc,
        "gap": gap,
    }

    # Export to JSON
    agg_file = os.path.join(comparison_dir, "aggregated_metrics.json")
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregated Metrics:")
    print(f"  Best Proposed: {best_proposed} (accuracy={best_proposed_acc:.4f})")
    print(f"  Best Baseline: {best_baseline} (accuracy={best_baseline_acc:.4f})")
    print(f"  Gap: {gap:+.4f}")
    print(f"\nExported aggregated metrics to {agg_file}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")

    # Fetch data for all runs
    all_run_data = []
    for run_id in run_ids:
        run_data = fetch_run_data(args.wandb_entity, args.wandb_project, run_id)
        if run_data:
            all_run_data.append(run_data)

            # Export per-run metrics
            export_run_metrics(run_data, args.results_dir, run_id)

            # Create per-run figures
            create_run_figures(run_data, args.results_dir, run_id)

    if not all_run_data:
        print("Error: No run data found")
        sys.exit(1)

    # Create comparison figures
    print("\nCreating comparison figures...")
    create_comparison_figures(all_run_data, args.results_dir)

    # Export aggregated metrics
    print("\nExporting aggregated metrics...")
    export_aggregated_metrics(all_run_data, args.results_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
