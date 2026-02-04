#!/usr/bin/env python
"""
Collect results from all experiments and generate RESULTS.md

Usage:
    python scripts/collect_results.py
"""

import json
import os
from datetime import datetime
from pathlib import Path


def find_result_files(results_dir: str) -> list:
    """Find all JSON result files."""
    results = []
    for path in Path(results_dir).rglob('*.json'):
        if 'result' in path.name:
            results.append(path)
    return results


def load_results(result_files: list) -> dict:
    """Load all result files."""
    all_results = {
        'anomaly_detection': [],
        'classification': [],
        'change_detection': [],
        'denoising': [],
        'unmixing': [],
    }

    for path in result_files:
        try:
            with open(path) as f:
                data = json.load(f)

            task = data.get('task', 'unknown')
            if task in all_results:
                all_results[task].append(data)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    return all_results


def generate_markdown(all_results: dict) -> str:
    """Generate RESULTS.md content."""
    lines = [
        "# HyperSIGMA Benchmark Results",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]

    # Anomaly Detection
    if all_results['anomaly_detection']:
        lines.extend([
            "## Anomaly Detection",
            "",
            "| Dataset | Mode | AUC-ROC |",
            "|---------|------|---------|",
        ])

        for r in sorted(all_results['anomaly_detection'],
                       key=lambda x: (x['dataset'], x['mode'])):
            auc = r['metrics'].get('auc_roc', 0)
            if isinstance(auc, dict):
                auc = auc.get('mean', 0)
            lines.append(f"| {r['dataset']} | {r['mode']} | {auc*100:.2f}% |")

        lines.append("")

    # Classification
    if all_results['classification']:
        lines.extend([
            "## Classification",
            "",
            "| Dataset | Samples | OA | AA | Kappa |",
            "|---------|---------|----|----|-------|",
        ])

        for r in sorted(all_results['classification'],
                       key=lambda x: (x['dataset'], x.get('samples_per_class', 0))):
            m = r['metrics']
            oa = m['overall_accuracy']
            aa = m['average_accuracy']
            kappa = m['kappa']

            if isinstance(oa, dict):
                oa_str = f"{oa['mean']*100:.2f}%"
            else:
                oa_str = f"{oa*100:.2f}%"

            if isinstance(aa, dict):
                aa_str = f"{aa['mean']*100:.2f}%"
            else:
                aa_str = f"{aa*100:.2f}%"

            if isinstance(kappa, dict):
                kappa_str = f"{kappa['mean']:.4f}"
            else:
                kappa_str = f"{kappa:.4f}"

            lines.append(
                f"| {r['dataset']} | {r.get('samples_per_class', 'N/A')} | "
                f"{oa_str} | {aa_str} | {kappa_str} |"
            )

        lines.append("")

    # Footer
    lines.extend([
        "---",
        "",
        "## Comparison with SpectralFM",
        "",
        "*To be filled after running SpectralFM benchmarks*",
        "",
    ])

    return "\n".join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(repo_dir, 'results')

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        print("Run some experiments first!")
        return

    result_files = find_result_files(results_dir)
    print(f"Found {len(result_files)} result files")

    all_results = load_results(result_files)
    markdown = generate_markdown(all_results)

    output_path = os.path.join(repo_dir, 'RESULTS.md')
    with open(output_path, 'w') as f:
        f.write(markdown)

    print(f"Results written to {output_path}")


if __name__ == '__main__':
    main()
