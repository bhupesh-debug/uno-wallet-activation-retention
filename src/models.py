"""Models for D30 retention: a simple rule baseline, logistic regression,
and a random forest. The rule is the bar — if it beats the forest, that is
the honest finding, not a failure to report."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def rule_predict(df):
    """Simple rule: predict retained if activated AND >= 2 cards linked."""
    return ((df["activated"] == 1) & (df["cards_linked"] >= 2)).astype(int).values


def rule_predict_proba(df):
    """Rule as a score (0.85 / 0.15) so AUC is computable."""
    pred = rule_predict(df)
    return np.where(pred == 1, 0.85, 0.15)


def make_models():
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        ),
    }
