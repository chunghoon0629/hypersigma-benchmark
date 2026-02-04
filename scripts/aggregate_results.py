#!/usr/bin/env python3
"""
Aggregate HyperSIGMA benchmark results from multiple seeds.

Collects JSON results from all tasks and computes mean ± std statistics.
Outputs a summary table and updated RESULTS.md.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# Seeds used for experiments
ALL_SEEDS = [42, 123, 456, 789, 2024, 1234, 5678, 9999, 1111, 7777]


def load_json_results(results_dir: Path) -> dict:
    """Load all JSON result files from results directory."""
    results = defaultdict(lambda: defaultdict(list))

    # Classification (already has 10 runs in single file)
    cls_files = list((results_dir / "classification").glob("*.json"))
    for f in cls_files:
        with open(f) as fp:
            data = json.load(fp)
            if "per_run_results" in data:
                # Multi-run format
                dataset = data["dataset"]
                for run in data["per_run_results"]:
                    results["classification"][dataset].append(run)

    # Anomaly Detection
    ad_dir = results_dir / "anomaly_detection" / "ss"
    if ad_dir.exists():
        for f in ad_dir.glob("*.json"):
            with open(f) as fp:
                data = json.load(fp)
                dataset = data.get("dataset", "unknown")
                results["anomaly_detection"][dataset].append(data.get("metrics", {}))

    # Change Detection
    cd_dir = results_dir / "change_detection" / "ss"
    if cd_dir.exists():
        seen_seeds = set()
        for f in sorted(cd_dir.glob("result_*.json")):
            with open(f) as fp:
                data = json.load(fp)
                dataset = data.get("dataset", "unknown")
                seed = data.get("seed", 0)
                # Avoid duplicates (old format without seed + new with seed)
                key = (dataset, seed)
                if key not in seen_seeds:
                    seen_seeds.add(key)
                    metrics = data.get("metrics", {})
                    metrics["seed"] = seed
                    results["change_detection"][dataset].append(metrics)

    # Target Detection (both sa and ss modes)
    for mode in ["sa", "ss"]:
        td_dir = results_dir / "target_detection" / mode
        if td_dir.exists():
            seen_seeds = set()
            for f in sorted(td_dir.glob("result_*.json")):
                with open(f) as fp:
                    data = json.load(fp)
                    dataset = data.get("dataset", "unknown")
                    seed = data.get("seed", 0)
                    key = (dataset, seed)
                    if key not in seen_seeds:
                        seen_seeds.add(key)
                        metrics = data.get("metrics", {})
                        metrics["seed"] = seed
                        results[f"target_detection_{mode}"][dataset].append(metrics)

    # Denoising
    dn_dir = results_dir / "denoising"
    if dn_dir.exists():
        seen_seeds = set()
        for f in sorted(dn_dir.glob("result_*.json")):
            with open(f) as fp:
                data = json.load(fp)
                dataset = data.get("dataset", "unknown")
                seed = data.get("seed", 0)
                key = (dataset, seed)
                if key not in seen_seeds:
                    seen_seeds.add(key)
                    metrics = data.get("metrics", {})
                    metrics["seed"] = seed
                    results["denoising"][dataset].append(metrics)

    # Unmixing (both sa and ss modes)
    for mode in ["sa", "ss"]:
        um_dir = results_dir / "unmixing" / mode
        if um_dir.exists():
            seen_seeds = set()
            for f in sorted(um_dir.glob("result_*.json")):
                with open(f) as fp:
                    data = json.load(fp)
                    dataset = data.get("dataset", "unknown")
                    seed = data.get("seed", 0)
                    key = (dataset, seed)
                    if key not in seen_seeds:
                        seen_seeds.add(key)
                        metrics = data.get("metrics", {})
                        metrics["seed"] = seed
                        results[f"unmixing_{mode}"][dataset].append(metrics)

    return results


def compute_stats(values: list) -> tuple:
    """Compute mean and std of a list of values."""
    if not values:
        return 0.0, 0.0
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def format_metric(mean: float, std: float, precision: int = 2, multiply: float = 1.0) -> str:
    """Format metric as 'mean ± std'."""
    m = mean * multiply
    s = std * multiply
    if s > 0:
        return f"{m:.{precision}f} ± {s:.{precision}f}"
    else:
        return f"{m:.{precision}f}"


def aggregate_task_results(task_results: list, metric_keys: list) -> dict:
    """Aggregate results for a single task/dataset combination."""
    aggregated = {"n_runs": len(task_results)}

    for key in metric_keys:
        values = []
        for r in task_results:
            if key in r:
                values.append(r[key])
            elif key.lower() in r:
                values.append(r[key.lower()])

        mean, std = compute_stats(values)
        aggregated[key] = {"mean": mean, "std": std, "n": len(values)}

    return aggregated


def print_summary(results: dict):
    """Print summary of all results."""
    print("\n" + "=" * 80)
    print("HyperSIGMA BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Classification
    if "classification" in results:
        print("\n## Classification (50-shot)")
        print("-" * 60)
        print(f"{'Dataset':<20} {'OA (%)':<20} {'Kappa':<20} {'Runs':<10}")
        print("-" * 60)

        for dataset, runs in results["classification"].items():
            oa_vals = [r.get("overall_accuracy", r.get("seg_overall_accuracy", 0)) for r in runs]
            kappa_vals = [r.get("kappa", r.get("seg_kappa", 0)) for r in runs]

            oa_mean, oa_std = compute_stats(oa_vals)
            kappa_mean, kappa_std = compute_stats(kappa_vals)

            oa_str = format_metric(oa_mean, oa_std, 2, 100)
            kappa_str = format_metric(kappa_mean, kappa_std, 3)

            print(f"{dataset:<20} {oa_str:<20} {kappa_str:<20} {len(runs):<10}")

    # Anomaly Detection
    if "anomaly_detection" in results:
        print("\n## Anomaly Detection")
        print("-" * 60)
        print(f"{'Dataset':<20} {'AUC-ROC (%)':<20} {'Runs':<10}")
        print("-" * 60)

        for dataset, runs in results["anomaly_detection"].items():
            # Handle both uppercase and lowercase metric names
            auc_vals = []
            for r in runs:
                val = r.get("AUC_ROC") or r.get("auc_roc") or 0
                auc_vals.append(val)
            auc_mean, auc_std = compute_stats(auc_vals)
            auc_str = format_metric(auc_mean, auc_std, 2, 100)
            print(f"{dataset:<20} {auc_str:<20} {len(runs):<10}")

    # Change Detection
    if "change_detection" in results:
        print("\n## Change Detection (SS mode)")
        print("-" * 60)
        print(f"{'Dataset':<20} {'OA (%)':<20} {'Kappa':<20} {'F1 (%)':<20} {'Runs':<10}")
        print("-" * 60)

        for dataset, runs in results["change_detection"].items():
            oa_vals = [r.get("OA", 0) for r in runs]
            kappa_vals = [r.get("kappa", 0) for r in runs]
            f1_vals = [r.get("F1", 0) for r in runs]

            oa_mean, oa_std = compute_stats(oa_vals)
            kappa_mean, kappa_std = compute_stats(kappa_vals)
            f1_mean, f1_std = compute_stats(f1_vals)

            oa_str = format_metric(oa_mean, oa_std, 2, 100)
            kappa_str = format_metric(kappa_mean, kappa_std, 3)
            f1_str = format_metric(f1_mean, f1_std, 2, 100)

            print(f"{dataset:<20} {oa_str:<20} {kappa_str:<20} {f1_str:<20} {len(runs):<10}")

    # Target Detection
    for mode in ["sa", "ss"]:
        key = f"target_detection_{mode}"
        if key in results:
            print(f"\n## Target Detection ({mode.upper()} mode)")
            print("-" * 60)
            print(f"{'Dataset':<20} {'AUC-ROC (%)':<20} {'F1 (%)':<20} {'Runs':<10}")
            print("-" * 60)

            for dataset, runs in results[key].items():
                auc_vals = [r.get("AUC_ROC", 0) for r in runs]
                f1_vals = [r.get("F1", 0) for r in runs]

                auc_mean, auc_std = compute_stats(auc_vals)
                f1_mean, f1_std = compute_stats(f1_vals)

                auc_str = format_metric(auc_mean, auc_std, 2, 100)
                f1_str = format_metric(f1_mean, f1_std, 2, 100)

                print(f"{dataset:<20} {auc_str:<20} {f1_str:<20} {len(runs):<10}")

    # Denoising
    if "denoising" in results:
        print("\n## Denoising")
        print("-" * 60)
        print(f"{'Dataset':<20} {'PSNR (dB)':<20} {'SSIM':<20} {'SAM':<20} {'Runs':<10}")
        print("-" * 60)

        for dataset, runs in results["denoising"].items():
            psnr_vals = [r.get("PSNR", 0) for r in runs]
            ssim_vals = [r.get("SSIM", 0) for r in runs]
            sam_vals = [r.get("SAM", 0) for r in runs]

            psnr_mean, psnr_std = compute_stats(psnr_vals)
            ssim_mean, ssim_std = compute_stats(ssim_vals)
            sam_mean, sam_std = compute_stats(sam_vals)

            psnr_str = format_metric(psnr_mean, psnr_std, 2)
            ssim_str = format_metric(ssim_mean, ssim_std, 4)
            sam_str = format_metric(sam_mean, sam_std, 2)

            print(f"{dataset:<20} {psnr_str:<20} {ssim_str:<20} {sam_str:<20} {len(runs):<10}")

    # Unmixing
    for mode in ["sa", "ss"]:
        key = f"unmixing_{mode}"
        if key in results:
            print(f"\n## Unmixing ({mode.upper()} mode)")
            print("-" * 60)
            print(f"{'Dataset':<20} {'RMSE':<20} {'SAM':<20} {'Runs':<10}")
            print("-" * 60)

            for dataset, runs in results[key].items():
                rmse_vals = [r.get("abundance_RMSE", 0) for r in runs]
                sam_vals = [r.get("reconstruction_SAM", 0) for r in runs]

                rmse_mean, rmse_std = compute_stats(rmse_vals)
                sam_mean, sam_std = compute_stats(sam_vals)

                rmse_str = format_metric(rmse_mean, rmse_std, 4)
                sam_str = format_metric(sam_mean, sam_std, 2)

                print(f"{dataset:<20} {rmse_str:<20} {sam_str:<20} {len(runs):<10}")

    print("\n" + "=" * 80)


def generate_markdown_table(results: dict) -> str:
    """Generate markdown summary table."""
    lines = []
    lines.append("# HyperSIGMA Benchmark Results (10-Run Statistics)")
    lines.append("")
    lines.append(f"**Seeds used**: {', '.join(map(str, ALL_SEEDS))}")
    lines.append("")

    # Summary table
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Task | Dataset | Mode | Primary Metric | Result (mean ± std) | Runs |")
    lines.append("|------|---------|------|----------------|---------------------|------|")

    # Classification
    if "classification" in results:
        for dataset, runs in results["classification"].items():
            oa_vals = [r.get("overall_accuracy", 0) for r in runs]
            oa_mean, oa_std = compute_stats(oa_vals)
            oa_str = format_metric(oa_mean, oa_std, 2, 100)
            lines.append(f"| Classification | {dataset} | - | OA (%) | **{oa_str}** | {len(runs)} |")

    # Anomaly Detection
    if "anomaly_detection" in results:
        for dataset, runs in results["anomaly_detection"].items():
            auc_vals = [r.get("AUC_ROC") or r.get("auc_roc") or 0 for r in runs]
            auc_mean, auc_std = compute_stats(auc_vals)
            auc_str = format_metric(auc_mean, auc_std, 2, 100)
            lines.append(f"| Anomaly Detection | {dataset} | SS | AUC-ROC (%) | **{auc_str}** | {len(runs)} |")

    # Change Detection
    if "change_detection" in results:
        for dataset, runs in results["change_detection"].items():
            oa_vals = [r.get("OA", 0) for r in runs]
            oa_mean, oa_std = compute_stats(oa_vals)
            oa_str = format_metric(oa_mean, oa_std, 2, 100)
            lines.append(f"| Change Detection | {dataset} | SS | OA (%) | **{oa_str}** | {len(runs)} |")

    # Target Detection
    for mode in ["sa", "ss"]:
        key = f"target_detection_{mode}"
        if key in results:
            for dataset, runs in results[key].items():
                auc_vals = [r.get("AUC_ROC", 0) for r in runs]
                auc_mean, auc_std = compute_stats(auc_vals)
                auc_str = format_metric(auc_mean, auc_std, 2, 100)
                lines.append(f"| Target Detection | {dataset} | {mode.upper()} | AUC-ROC (%) | **{auc_str}** | {len(runs)} |")

    # Denoising
    if "denoising" in results:
        for dataset, runs in results["denoising"].items():
            psnr_vals = [r.get("PSNR", 0) for r in runs]
            psnr_mean, psnr_std = compute_stats(psnr_vals)
            psnr_str = format_metric(psnr_mean, psnr_std, 2)
            lines.append(f"| Denoising | {dataset} | - | PSNR (dB) | **{psnr_str}** | {len(runs)} |")

    # Unmixing
    for mode in ["sa", "ss"]:
        key = f"unmixing_{mode}"
        if key in results:
            for dataset, runs in results[key].items():
                rmse_vals = [r.get("abundance_RMSE", 0) for r in runs]
                rmse_mean, rmse_std = compute_stats(rmse_vals)
                rmse_str = format_metric(rmse_mean, rmse_std, 4)
                lines.append(f"| Unmixing | {dataset} | {mode.upper()} | RMSE | **{rmse_str}** | {len(runs)} |")

    return "\n".join(lines)


def save_aggregated_json(results: dict, output_file: Path):
    """Save aggregated results to JSON file."""
    aggregated = {}

    # Process each task
    for task, datasets in results.items():
        aggregated[task] = {}
        for dataset, runs in datasets.items():
            # Get all metric keys from first run
            if runs:
                metric_keys = [k for k in runs[0].keys() if k != "seed"]
                agg = aggregate_task_results(runs, metric_keys)
                aggregated[task][dataset] = agg

    with open(output_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregated results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate HyperSIGMA benchmark results")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for aggregated results")
    parser.add_argument("--markdown", action="store_true",
                        help="Generate markdown summary")
    args = parser.parse_args()

    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / "results"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Loading results from: {results_dir}")

    # Load all results
    results = load_json_results(results_dir)

    # Print summary
    print_summary(results)

    # Save aggregated JSON
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = results_dir / "aggregated_10seed_results.json"

    save_aggregated_json(results, output_file)

    # Generate markdown
    if args.markdown:
        md_content = generate_markdown_table(results)
        md_file = results_dir / "RESULTS_10SEED.md"
        with open(md_file, "w") as f:
            f.write(md_content)
        print(f"Markdown summary saved to: {md_file}")


if __name__ == "__main__":
    main()
