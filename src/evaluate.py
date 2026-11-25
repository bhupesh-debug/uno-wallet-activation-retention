"""Evaluation helpers: classification metrics, lift vs a naive baseline,
and permutation driver ranking."""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_score):
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def lift_vs_naive(y_true, y_score, top_frac=0.2):
    """How much better is targeting the top 20% by score than the base rate?"""
    order = np.argsort(-y_score)
    k = max(1, int(len(y_true) * top_frac))
    top = y_true[order[:k]]
    return {
        "top_frac": top_frac,
        "top_retention": float(np.mean(top)),
        "base_rate": float(np.mean(y_true)),
        "lift": float(np.mean(top) / max(np.mean(y_true), 1e-9)),
    }


def driver_ranking(model, X_test, y_test, feature_names, n_repeats=10, seed=42):
    """Permutation importance (AUC drop) on the test set."""
    res = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=seed,
        scoring="roc_auc", n_jobs=-1,
    )
    order = np.argsort(-res.importances_mean)
    return [
        {"feature": feature_names[i],
         "importance": float(res.importances_mean[i]),
         "std": float(res.importances_std[i])}
        for i in order
    ]
