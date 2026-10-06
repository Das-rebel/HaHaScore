#!/usr/bin/env python3
"""
handle_v1_result.py — Post-processing pipeline for v1 Kaggle kernel
================================================================
Run this AFTER the Kaggle kernel completes. It:
1. Pulls the v1_result.json from Kaggle
2. Validates against the pre-registered gate (rho >= 0.35 AND CI excludes 0.30)
3. Commits result_*.json to HaHaScore repo
4. Updates README + paper outline with actual numbers
5. Decides: publish v1 paper, retry with roberta-large, or methodology-only

Usage:
    python3 handle_v1_result.py --kaggle-slug subhajitdas/v1-jester-roberta-5x3-cv
"""
import argparse
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime


KAGGLE_SLUG = 'subhajitdas/v1-jester-roberta-5x3-cv'
OUTPUT_DIR = Path('/Users/Subho/funny-strength-predictor/experiments/v1_jester_regression')


def pull_kaggle_output(slug, dest_dir):
    """Pull kernel output from Kaggle."""
    import os
    os.environ['PATH'] = '/Users/Subho/Library/Python/3.12/bin:' + os.environ.get('PATH', '')
    result_dir = '/tmp/.k_output_pull'
    os.makedirs(result_dir, exist_ok=True)
    cmd = f'kaggle kernels output {slug} -p {result_dir}'
    print(f'Running: {cmd}')
    r = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=60)
    print(r.stdout)
    if r.returncode != 0:
        print(f'ERROR: {r.stderr}')
        return None

    # Find the result JSON in the pulled output
    for path in Path(result_dir).rglob('*.json'):
        with open(path) as f:
            return json.load(f)
    return None


def validate_gate(result):
    """Check pre-registered gate: rho >= 0.35 AND CI excludes 0.30."""
    rho = result.get('rho_mean', 0)
    ci_low = result.get('rho_ci_low', 0)
    ci_high = result.get('rho_ci_high', 0)
    mae = result.get('mae_mean', 999)
    rmse = result.get('rmse_mean', 999)

    rho_pass = rho >= 0.35
    ci_pass = ci_low >= 0.30
    verdict = result.get('verdict', '?')

    print('\n=== Pre-registered gate check ===')
    print(f'  rho = {rho:.4f} (target >= 0.35): {"PASS" if rho_pass else "FAIL"}')
    print(f'  CI lower = {ci_low:.4f} (target >= 0.30): {"PASS" if ci_pass else "FAIL"}')
    print(f'  MAE = {mae:.4f}')
    print(f'  RMSE = {rmse:.4f}')

    if rho_pass and ci_pass:
        return 'PUBLISHABLE'
    elif rho >= 0.30 and ci_low >= 0.20:
        return 'MODEST_PUBLISH'
    elif ci_low < 0 < ci_high:
        return 'METHODOLOGY_ONLY'
    else:
        return 'NEGATIVE'


def commit_result(result, gate_verdict):
    """Commit result to HaHaScore repo."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUTPUT_DIR / f'result_kaggle_{timestamp}.json'

    enriched = {
        **result,
        'pre_registered_gate': gate_verdict,
        'gate_criteria': {
            'rho_min': 0.35,
            'ci_lower_min': 0.30,
            'rho_pass': result.get('rho_mean', 0) >= 0.35,
            'ci_pass': result.get('rho_ci_low', 0) >= 0.30,
        },
        'kaggle_slug': KAGGLE_SLUG,
        'retrieved_at': timestamp,
    }
    out_path.write_text(json.dumps(enriched, indent=2))
    print(f'\nSaved: {out_path}')

    # Commit
    repo_dir = Path('/Users/Subho/funny-strength-predictor')
    subprocess.run(['git', 'add', str(out_path)], cwd=repo_dir, check=True)

    # Update RESULT.md with the actual numbers
    result_md = OUTPUT_DIR / 'RESULT.md'
    result_md.write_text(f"""# v1 HaHaScore Result (Kaggle Run, {timestamp})

## Outcome: {gate_verdict}

| Metric | Value | Pre-Registered Gate |
|---|---|---|
| Spearman rho | {result.get('rho_mean', 0):.4f} ± {result.get('rho_std', 0):.4f} | >= 0.35 |
| 95% CI | [{result.get('rho_ci_low', 0):.4f}, {result.get('rho_ci_high', 0):.4f}] | lower >= 0.30 |
| MAE | {result.get('mae_mean', 0):.4f} | <= 22 |
| RMSE | {result.get('rmse_mean', 0):.4f} | - |
| N measurements | {result.get('n_measurements', 0)} | >= 15 |

## Decision

| Gate | Action |
|---|---|
| PUBLISHABLE (rho >= 0.35 AND CI excludes 0.30) | Draft v1 paper using PAPER_OUTLINE.md, arXiv submission |
| MODEST_PUBLISH (rho >= 0.30 AND CI excludes 0.20) | Publish methodology with modest v1 result |
| METHODOLOGY_ONLY (CI spans 0.30) | 5×3 repeated CV paper, NO v1 metric claim |
| NEGATIVE (rho <= 0) | Try roberta-large or LoRA; document negative result |

## Source

- Kaggle kernel: {KAGGLE_SLUG}
- v1 pipeline: train_jester_regression.py + laugho_cv.repeated_joke_disjoint_cv_regression
- Commit: see `git log` for the result commit

## Configuration used

{json.dumps(result.get('config', {}), indent=2)}
""")
    subprocess.run(['git', 'add', str(result_md)], cwd=repo_dir, check=True)

    commit_msg = f"v1 Jester result (Kaggle {timestamp}): {gate_verdict}\n\n"
    commit_msg += f"  rho = {result.get('rho_mean', 0):.4f} +/- {result.get('rho_std', 0):.4f}\n"
    commit_msg += f"  95% CI = [{result.get('rho_ci_low', 0):.4f}, {result.get('rho_ci_high', 0):.4f}]\n"
    commit_msg += f"  MAE = {result.get('mae_mean', 0):.4f}\n"
    commit_msg += f"  Verdict: {gate_verdict}\n"

    subprocess.run(['git', 'commit', '-m', commit_msg], cwd=repo_dir, check=True)
    subprocess.run(['git', 'push', 'origin', 'main'], cwd=repo_dir, check=True)
    print(f'\nCommitted and pushed: {commit_msg}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kaggle-slug', default=KAGGLE_SLUG)
    ap.add_argument('--auto-commit', action='store_true',
                    help='auto-commit result to HaHaScore repo')
    args = ap.parse_args()

    print('=== v1 Jester Result Handler ===')
    print(f'Kaggle slug: {args.kaggle_slug}')

    # Pull result
    result = pull_kaggle_output(args.kaggle_slug, OUTPUT_DIR)
    if result is None:
        print('ERROR: Could not pull kernel output. Is the kernel still running?')
        return 1

    print(f'\n=== Raw result ===')
    print(json.dumps(result, indent=2))

    # Validate gate
    verdict = validate_gate(result)

    print(f'\n=== Decision ===')
    print(f'Gate verdict: {verdict}')

    if args.auto_commit:
        commit_result(result, verdict)

    return 0


if __name__ == '__main__':
    exit(main() or 0)