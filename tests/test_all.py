"""Test suite. Prints a runtime 'N passed' line — quote that line,
never a count of check() call sites (checks inside loops run many times).
Run: python -m tests.test_all  (from the project root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cohorts import channel_table, cohort_table, retention_by_activation_cards
from src.data_gen import CHANNELS, generate
from src.evaluate import classification_metrics, lift_vs_naive
from src.experiment import design_experiment, sample_size_per_arm
from src.features import FEATURES, build_user_frame, daily_active_share, model_columns
from src.models import make_models, rule_predict, rule_predict_proba

PASSED = 0
FAILED = 0


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"FAIL: {name}")


def main():
    global PASSED, FAILED
    users, events = generate(n_users=1200)  # 1200 keeps the suite fast; run_all.py uses 2400

    # --- Data generation ---
    check("1200 users", len(users) == 1200)
    check("12 signup weeks", set(users["signup_week"].unique()) == set(range(12)))
    check("100 users per week", (users.groupby("signup_week").size() == 100).all())
    check("channels are the 5 known ones", set(users["channel"].unique()) <= set(CHANNELS))
    check("cards_linked in 0..3", users["cards_linked"].between(0, 3).all())
    check("activation rate between 40% and 70%", 0.40 < users["activated"].mean() < 0.70)
    check("retention decays: D7 > D30 > D60",
          users["retained_d7"].mean() > users["retained_d30"].mean() > users["retained_d60"].mean())
    check("D30 retention between 10% and 35%", 0.10 < users["retained_d30"].mean() < 0.35)
    check("activated users retain better at D30",
          users.loc[users["activated"], "retained_d30"].mean()
          > users.loc[~users["activated"], "retained_d30"].mean())
    check("more cards -> higher D30 (activated users)",
          users.loc[users["activated"]].groupby("cards_linked")["retained_d30"].mean().is_monotonic_increasing)
    check("days_to_activation only for activated users",
          users.loc[users["activated"], "days_to_activation"].notna().all()
          and users.loc[~users["activated"], "days_to_activation"].isna().all())
    check("days_to_activation within 1..7",
          users.loc[users["activated"], "days_to_activation"].between(1, 7).all())
    check("first_week_txns non-negative", (users["first_week_txns"] >= 0).all())
    check("reward non-negative", (users["reward_earned_wk1"] >= 0).all())
    check("reward is 0 when no first-week txns",
          (users.loc[users["first_week_txns"] == 0, "reward_earned_wk1"] == 0).all())
    check("events only for active days", set(events["day"].unique()) <= set(range(1, 61)))
    check("events reference known users", set(events["user_id"].unique()) <= set(users["user_id"]))
    check("n_active_days_60 matches events",
          users.set_index("user_id")["n_active_days_60"].equals(
              events.groupby("user_id").size().reindex(users["user_id"]).fillna(0).astype(int)))
    u2, e2 = generate(n_users=1200)
    check("seed reproducible", u2.equals(users) and e2.equals(events))
    check("different seed differs", not generate(n_users=1200, seed=7)[0].equals(users))
    check("referral activates more than paid",
          users.loc[users["channel"] == "referral", "activated"].mean()
          > users.loc[users["channel"] == "paid", "activated"].mean())
    check("referral links more cards than paid",
          users.loc[users["channel"] == "referral", "cards_linked"].mean()
          > users.loc[users["channel"] == "paid", "cards_linked"].mean())

    # --- Features ---
    frame = build_user_frame(users)
    check("frame has all FEATURES", all(c in frame.columns for c in FEATURES))
    check("frame has target", "retained_d30" in frame.columns)
    check("no NaN in model columns", not frame[model_columns()].isna().any().any())
    check("days_to_activation filled with 8 for non-activated",
          (frame.loc[~users["activated"].values, "days_to_activation"] == 8).all())
    check("channel dummies sum to 1 per row",
          frame[[f"channel_{c}" for c in CHANNELS]].sum(axis=1).eq(1).all())
    daily = daily_active_share(events, users)
    check("daily share has 60 rows", len(daily) == 60)
    check("daily share in [0,1]",
          daily[["share_all", "share_activated", "share_not_activated"]].apply(
              lambda s: s.between(0, 1).all()).all())
    check("activated share above not-activated share on day 7",
          daily.loc[daily["day"] == 7, "share_activated"].iloc[0]
          > daily.loc[daily["day"] == 7, "share_not_activated"].iloc[0])
    check("daily share decays: day 55 below day 5",
          daily.loc[daily["day"] == 55, "share_all"].iloc[0]
          < daily.loc[daily["day"] == 5, "share_all"].iloc[0])

    # --- Cohorts ---
    cohorts = cohort_table(users)
    check("cohort table has 12 rows", len(cohorts) == 12)
    check("cohort sizes sum to N", cohorts["cohort_size"].sum() == len(users))
    check("cohort rates in [0,1]",
          cohorts[["activation_rate", "d7", "d30", "d60"]].apply(
              lambda s: s.between(0, 1).all()).all())
    channels = channel_table(users)
    check("channel table has 5 rows", len(channels) == 5)
    check("channel sizes sum to N", channels["n"].sum() == len(users))
    ac = retention_by_activation_cards(users)
    check("activation x cards table has 8 rows", len(ac) == 8)
    check("activation x cards sizes sum to N", ac["n"].sum() == len(users))

    # --- Models ---
    cols = model_columns()
    X = frame[cols].values
    y = frame["retained_d30"].values
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, frame, test_size=0.30, random_state=42, stratify=y)
    rule_pred = rule_predict(df_test)
    check("rule predicts only 0/1", set(np.unique(rule_pred)) <= {0, 1})
    check("rule predicts 1 only for activated users",
          (df_test.loc[rule_pred == 1, "activated"] == 1).all())
    check("rule predicts 1 only for >=2 cards",
          (df_test.loc[rule_pred == 1, "cards_linked"] >= 2).all())
    rule_score = rule_predict_proba(df_test)
    check("rule scores are 0.15 or 0.85", set(np.unique(rule_score)) == {0.15, 0.85})
    rule_m = classification_metrics(y_test, rule_pred, rule_score)
    check("rule AUC above 0.60", rule_m["auc"] > 0.60)
    check("rule AUC below 0.75 (it is simple)", rule_m["auc"] < 0.75)
    lift = lift_vs_naive(y_test, rule_score)
    check("rule lift above 1.0", lift["lift"] > 1.0)
    check("rule lift below 3.0 (no magic)", lift["lift"] < 3.0)
    models = make_models()
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        score = m.predict_proba(X_test)[:, 1]
        met = classification_metrics(y_test, pred, score)
        check(f"{name}: AUC above 0.60", met["auc"] > 0.60)
        check(f"{name}: AUC below 0.90 (no leakage)", met["auc"] < 0.90)
        check(f"{name}: predictions are 0/1", set(np.unique(pred)) <= {0, 1})
        check(f"{name}: scores in [0,1]", (score >= 0).all() and (score <= 1).all())
        check(f"{name}: lift above 1.0", lift_vs_naive(y_test, score)["lift"] > 1.0)
    check("logistic beats rule on AUC (models add ranking value)",
          classification_metrics(y_test, models["logistic"].predict(X_test),
                                 models["logistic"].predict_proba(X_test)[:, 1])["auc"]
          > rule_m["auc"])

    # --- Experiment ---
    check("sample size positive", sample_size_per_arm(0.20, 0.03) > 0)
    check("smaller MDE needs more users",
          sample_size_per_arm(0.20, 0.03) > sample_size_per_arm(0.20, 0.05))
    check("higher baseline needs more users",
          sample_size_per_arm(0.30, 0.03) > sample_size_per_arm(0.10, 0.03))
    exp = design_experiment(0.20, mde=0.03, signups_per_week=200)
    check("experiment total = 2x per-arm", exp["total_n"] == 2 * exp["n_per_arm"])
    check("experiment weeks math",
          exp["weeks_to_recruit"] == int(np.ceil(exp["total_n"] / 200)))
    check("decision rule mentions SHIP", "SHIP" in exp["decision_rule"])
    check("decision rule mentions KILL", "KILL" in exp["decision_rule"])
    check("decision rule mentions guardrail", "guardrail" in exp["decision_rule"].lower())
    check("causation note present", "causation" in exp["causation_note"].lower()
          or "causal" in exp["causation_note"].lower())

    print(f"{PASSED} passed, {FAILED} failed")
    if FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
