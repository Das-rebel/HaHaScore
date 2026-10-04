#!/usr/bin/env python3
"""
laugho_cv.py — Reusable 5×3 Repeated Speaker-Disjoint CV Harness
================================================================
For low-positive-class speech classification tasks (laughter, sarcasm,
deception detection). Per the HaHaScore 5×3 CV falsification paper
recommendation: always 5×3 repeated speaker-disjoint CV with bootstrap
confidence intervals before publishing any headline metric.

Usage:
    from laugho_cv import repeated_speaker_disjoint_cv, bootstrap_ci, paired_t

    # data: features(X), groups=speakers, labels=y, fold_fn=callable
    results = repeated_speaker_disjoint_cv(
        X, y, groups,
        fold_fn=lambda tr_X, tr_y, va_X, va_y, seed: my_model(tr_X, tr_y, va_X, seed),
        n_seeds=3, n_folds=5,
    )

    print(f"Mean AUC: {results['mean']:.4f} ± {results['std']:.4f}")
    print(f"95% CI:   {results['ci_low']:.4f} - {results['ci_high']:.4f}")
    print(f"verdict:  {results['verdict']}")
"""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel


def repeated_speaker_disjoint_cv(
    X, y, groups,
    fold_fn=None,
    n_seeds=3,
    n_folds=5,
    seed_base=42,
    shuffle_within_seed=True,
    return_per_fold=False,
):
    """
    Run 5×3 repeated speaker-disjoint cross-validation.

    Parameters
    ----------
    X : array-like, shape (n_samples, ...)
        Feature matrix
    y : array-like, shape (n_samples,)
        Binary or continuous labels
    groups : array-like, shape (n_samples,)
        Speaker IDs (or any grouping variable to keep disjoint across folds)
    fold_fn : callable
        Function (tr_X, tr_y, va_X, va_y, seed) -> float AUC
        Must return AUC value (0.5 = chance, 1.0 = perfect)
    n_seeds : int
        Number of random seeds (default 3, giving 3 × 5 = 15 measurements)
    n_folds : int
        Number of folds (default 5)
    seed_base : int
        Base random seed
    shuffle_within_seed : bool
        If True, reshuffle groups within each seed for different splits
    return_per_fold : bool
        If True, return per-fold AUC values for plotting

    Returns
    -------
    dict with keys:
        mean : float — mean AUC across all measurements
        std : float — standard deviation
        ci_low, ci_high : float — 95% bootstrap CI
        per_fold : list — per-measurement AUC values (if return_per_fold=True)
        n_measurements : int
        verdict : str — interpretation hint
    """
    assert len(X) == len(y) == len(groups), "X, y, groups must have same length"
    assert fold_fn is not None, "fold_fn must be provided"
    assert n_seeds >= 1, "n_seeds must be >= 1"
    assert n_folds >= 2, "n_folds must be >= 2"

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    all_aucs = []

    for seed_idx in range(n_seeds):
        seed = seed_base + seed_idx * 100
        rng = np.random.RandomState(seed)

        if shuffle_within_seed:
            unique_groups = np.unique(groups)
            shuffled_groups = rng.permutation(unique_groups)
            group_map = {g: shuffled_groups[i] for i, g in enumerate(unique_groups)}
            groups_shuffled = np.array([group_map[g] for g in groups])
        else:
            groups_shuffled = groups.copy()

        gkf = GroupKFold(n_splits=n_folds)
        for fold_idx, (tr, va) in enumerate(gkf.split(X, y, groups_shuffled)):
            auc = fold_fn(X[tr], y[tr], X[va], y[va], seed=seed)
            all_aucs.append(auc)

    all_aucs = np.array(all_aucs)
    mean = float(np.mean(all_aucs))
    std = float(np.std(all_aucs))

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(10000):
        idx = rng.choice(len(all_aucs), len(all_aucs), replace=True)
        boot_means.append(np.mean(all_aucs[idx]))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # Verdict
    if mean > 0.7 and (ci_low > 0.55):
        verdict = "STRONG: AUC > 0.7 with CI excluding 0.55. Safe to publish."
    elif mean > 0.55 and (ci_low > 0.5):
        verdict = "MODEST: AUC > 0.55 with CI excluding chance. Publishable with caveat."
    elif ci_low < 0.5 < ci_high:
        verdict = "WEAK: CI includes 0.5 (chance). Need more data or better features."
    else:
        verdict = "NEGATIVE: AUC below chance. Likely reversed or label leakage."

    result = {
        'mean': mean,
        'std': std,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'n_measurements': len(all_aucs),
        'verdict': verdict,
    }
    if return_per_fold:
        result['per_fold'] = all_aucs.tolist()
    return result


def paired_comparison(results_a, results_b):
    """
    Paired t-test comparing two CV runs on the same folds.
    Use when comparing preprocessing variants (e.g., raw vs normalized).

    Parameters
    ----------
    results_a, results_b : dict
        Output from repeated_speaker_disjoint_cv with return_per_fold=True

    Returns
    -------
    dict with paired_t, paired_p, delta_mean, delta_std, verdict
    """
    assert 'per_fold' in results_a, "Need per_fold AUC values; pass return_per_fold=True"
    assert 'per_fold' in results_b, "Need per_fold AUC values; pass return_per_fold=True"
    assert len(results_a['per_fold']) == len(results_b['per_fold']), \
        "Two runs must have same number of measurements for pairing"

    a = np.array(results_a['per_fold'])
    b = np.array(results_b['per_fold'])
    delta = b - a  # positive means b is better

    t, p = ttest_rel(b, a)

    # Bootstrap CI on delta
    rng = np.random.RandomState(42)
    boot_deltas = []
    for _ in range(10000):
        idx = rng.choice(len(delta), len(delta), replace=True)
        boot_deltas.append(np.mean(delta[idx]))
    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])

    if p < 0.01 and ci_low > 0:
        verdict = "B significantly better than A (p<0.01, CI excludes 0)"
    elif p < 0.05 and ci_low > 0:
        verdict = "B significantly better than A (p<0.05)"
    elif p < 0.01 and ci_high < 0:
        verdict = "A significantly better than B (B is harmful)"
    elif p < 0.05 and ci_high < 0:
        verdict = "A significantly better than B"
    else:
        verdict = "No significant difference between A and B"

    return {
        'delta_mean': float(np.mean(delta)),
        'delta_std': float(np.std(delta)),
        'paired_t': float(t),
        'paired_p': float(p),
        'bootstrap_ci_low': float(ci_low),
        'bootstrap_ci_high': float(ci_high),
        'verdict': verdict,
    }


# Convenience function for sklearn-compatible models
def sklearn_fold_fn(model_factory):
    """
    Wraps an sklearn-compatible estimator factory into a fold_fn.

    Example:
        fold_fn = sklearn_fold_fn(lambda: LogisticRegression(class_weight='balanced'))
        results = repeated_speaker_disjoint_cv(X, y, groups, fold_fn=fold_fn)
    """
    def fold_fn(tr_X, tr_y, va_X, va_y, seed):
        model = model_factory()
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        model.fit(tr_X, tr_y)
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(va_X)[:, 1]
        else:
            scores = model.decision_function(va_X)
        if len(np.unique(va_y)) < 2:
            return 0.5
        return float(roc_auc_score(va_y, scores))
    return fold_fn


if __name__ == "__main__":
    # Demo: simulate two preprocessing variants with fold-luck
    print("=== laugho_cv demo ===")
    np.random.seed(0)
    n_samples, n_features, n_groups = 600, 50, 12
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples) > 0.85).astype(int)
    groups = np.repeat(np.arange(n_groups), n_samples // n_groups)

    from sklearn.linear_model import LogisticRegression
    fold_fn = sklearn_fold_fn(lambda: LogisticRegression(max_iter=200))

    raw = repeated_speaker_disjoint_cv(X, y, groups, fold_fn=fold_fn, return_per_fold=True)
    print(f"Raw:    {raw['mean']:.4f} ± {raw['std']:.4f}, CI [{raw['ci_low']:.4f}, {raw['ci_high']:.4f}]")

    # Try a normalization that doesn't help
    X_norm = X - X.mean(axis=0, keepdims=True)
    norm = repeated_speaker_disjoint_cv(X_norm, y, groups, fold_fn=fold_fn, return_per_fold=True)
    print(f"Norm:   {norm['mean']:.4f} ± {norm['std']:.4f}, CI [{norm['ci_low']:.4f}, {norm['ci_high']:.4f}]")

    cmp = paired_comparison(raw, norm)
    print(f"Delta:  {cmp['delta_mean']:+.4f}, t={cmp['paired_t']:.2f}, p={cmp['paired_p']:.4f}")
    print(f"Verdict: {cmp['verdict']}")