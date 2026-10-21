"""Holdout utility. Training labels must always be supplied explicitly."""
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             mean_squared_error, explained_variance_score, r2_score)


def mean_squared_error_metric(real, synthetic):
    return mean_squared_error(real, synthetic)


def average_categorical_jsd(real, synthetic, categorical_indices):
    scores = []
    for index in categorical_indices:
        categories = np.unique(np.concatenate([real[:, index], synthetic[:, index]]))
        p = [np.mean(real[:, index] == c) for c in categories]
        q = [np.mean(synthetic[:, index] == c) for c in categories]
        scores.append(jensenshannon(p, q) ** 2)
    return float(np.mean(scores)) if scores else None


def classification_utility(train_real, train_synthetic, test_real, test_labels,
                           *, real_labels, synthetic_labels, seed=42):
    result = {}
    for name, x, y in [("real", train_real, real_labels),
                       ("synthetic", train_synthetic, synthetic_labels)]:
        model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
        model.fit(x, y)
        pred = model.predict(test_real)
        result[f"{name}_train_accuracy"] = float(accuracy_score(test_labels, pred))
        result[f"{name}_train_f1_macro"] = float(f1_score(test_labels, pred, average="macro", zero_division=0))
        result[f"{name}_confusion_matrix"] = confusion_matrix(test_labels, pred).tolist()
    return result


def regression_utility(train_real, train_synthetic, test_real, test_labels,
                       *, real_labels, synthetic_labels, seed=42):
    result = {}
    for name, x, y in [("real", train_real, real_labels),
                       ("synthetic", train_synthetic, synthetic_labels)]:
        model = RandomForestRegressor(random_state=seed, n_jobs=1).fit(x, y)
        pred = model.predict(test_real)
        result[f"{name}_train_mse"] = float(mean_squared_error(test_labels, pred))
        result[f"{name}_train_evs"] = float(explained_variance_score(test_labels, pred))
        result[f"{name}_train_r2"] = float(r2_score(test_labels, pred))
    return result


def correlation_difference(real, synthetic):
    """Mean absolute correlation error over unique pairs, excluding constants.

    Return None when fewer than two columns vary in both tables. Constant-column
    counts should be reported separately; dropping them is not a fidelity pass.
    """
    real, synthetic = np.asarray(real), np.asarray(synthetic)
    valid = (real.std(0) > 1e-10) & (synthetic.std(0) > 1e-10)
    if valid.sum() < 2:
        return None
    delta = np.abs(np.corrcoef(real[:, valid], rowvar=False) -
                   np.corrcoef(synthetic[:, valid], rowvar=False))
    return float(delta[np.triu_indices(int(valid.sum()), k=1)].mean())
