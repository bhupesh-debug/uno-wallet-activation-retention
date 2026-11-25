"""Run the whole pipeline: data -> features -> cohorts -> models -> charts
+ metrics JSON. Charts regenerate with this script (PNGs are gitignored and
excluded from any GitHub push).
Usage: python run_all.py   (from the project root)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cohorts import channel_table, cohort_table, retention_by_activation_cards
from src.data_gen import SEED, generate, save_synthetic
from src.evaluate import classification_metrics, driver_ranking, lift_vs_naive
from src.experiment import design_experiment
from src.features import FEATURES, build_user_frame, daily_active_share, model_columns
from src.models import make_models, rule_predict, rule_predict_proba

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "reports" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

BLUE = "#1A3A6B"
TEAL = "#2E7D5B"
AMBER = "#E07B39"
RED = "#B03A2E"
GREY = "#777777"


def main():
    users, events = generate()
    save_synthetic(users, events,
                   ROOT / "data" / "synthetic" / "synthetic_users.csv",
                   ROOT / "data" / "synthetic" / "synthetic_events.csv")
    frame = build_user_frame(users)
    frame.to_csv(ROOT / "data" / "processed" / "user_features.csv", index=False)

    cohorts = cohort_table(users)
    channels = channel_table(users)
    act_cards = retention_by_activation_cards(users)
    daily = daily_active_share(events, users)

    # --- Modelling: D30 retention from week-1 features only ---
    cols = model_columns()
    X = frame[cols].values
    y = frame["retained_d30"].values
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, frame, test_size=0.30, random_state=42, stratify=y)
    models = make_models()
    results = {}

    # Rule baseline (no training).
    rule_pred = rule_predict(df_test)
    rule_score = rule_predict_proba(df_test)
    results["rule"] = classification_metrics(y_test, rule_pred, rule_score)
    results["rule"]["lift"] = lift_vs_naive(y_test, rule_score)

    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        score = m.predict_proba(X_test)[:, 1]
        results[name] = classification_metrics(y_test, pred, score)
        results[name]["lift"] = lift_vs_naive(y_test, score)
        # Train AUC (overfit check).
        train_score = m.predict_proba(X_train)[:, 1]
        from sklearn.metrics import f1_score, roc_auc_score
        results[name]["train_auc"] = float(roc_auc_score(y_train, train_score))
        # The 0.5 threshold is meaningless at a 20% base rate (it predicts
        # almost no positives): pick the F1-maximising threshold ON TRAIN,
        # then apply it to test. Report both so the artefact is visible.
        grid = np.arange(0.05, 0.96, 0.05)
        f1s = [f1_score(y_train, (train_score >= t).astype(int), zero_division=0) for t in grid]
        best_t = float(grid[int(np.argmax(f1s))])
        results[name]["threshold_best_f1"] = best_t
        results[name]["f1_at_best_threshold"] = float(
            f1_score(y_test, (score >= best_t).astype(int), zero_division=0))
        results[name]["f1_at_0.5"] = results[name].pop("f1")

    rf = models["random_forest"]
    drivers = driver_ranking(rf, X_test, y_test, cols)
    # Logistic coefficients (standardised features, so magnitudes compare).
    logreg = models["logistic"].named_steps["logisticregression"]
    coefs = sorted(
        [{"feature": c, "coef": float(v)} for c, v in zip(cols, logreg.coef_[0])],
        key=lambda r: -abs(r["coef"]))

    # --- Experiment design from the OBSERVED D30 baseline ---
    baseline_d30 = float(frame["retained_d30"].mean())
    experiment = design_experiment(baseline_d30, mde=0.03, signups_per_week=200)
    # A 3pp effect needs ~30 weeks of signups at 200/week — too slow for a
    # 2-person startup. Price the faster alternative honestly alongside it.
    from src.experiment import sample_size_per_arm
    n5 = sample_size_per_arm(baseline_d30, 0.05)
    experiment["mde_5pp_alternative"] = {
        "mde_pp": 5.0, "n_per_arm": n5, "total_n": 2 * n5,
        "weeks_to_recruit": int(np.ceil(2 * n5 / 200)),
        "note": ("Same design, larger minimum effect: powered for +5pp instead "
                 "of +3pp. The 3pp design stays the primary; the 5pp variant is "
                 "the pragmatic fallback if the team cannot wait 30 weeks."),
    }

    metrics = {
        "synthetic_data": True,
        "seed": SEED,
        "n_users": int(len(users)),
        "n_events": int(len(events)),
        "n_weeks": 12,
        "headline": {
            "activation_rate": float(frame["activated"].mean()),
            "d7_retention": float(frame["retained_d7"].mean()),
            "d30_retention": baseline_d30,
            "d60_retention": float(frame["retained_d60"].mean()),
            "mean_cards_linked": float(frame["cards_linked"].mean()),
            "mean_first_week_txns": float(frame["first_week_txns"].mean()),
            "mean_reward_wk1": float(frame["reward_earned_wk1"].mean()),
        },
        "cohorts": cohorts.to_dict(orient="records"),
        "channels": channels.to_dict(orient="records"),
        "activation_cards": act_cards.to_dict(orient="records"),
        "models": results,
        "drivers": drivers,
        "logistic_coefs": coefs,
        "experiment": experiment,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------------- Charts ----------------
    # 1. Cohort retention curves (D7 / D30 by signup week).
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(cohorts["signup_week"], cohorts["d7"] * 100, marker="o", color=TEAL, label="D7 retention")
    ax.plot(cohorts["signup_week"], cohorts["d30"] * 100, marker="s", color=BLUE, label="D30 retention")
    ax.plot(cohorts["signup_week"], cohorts["activation_rate"] * 100, marker="^", color=AMBER, label="Activation rate")
    ax.set_xlabel("Signup week"); ax.set_ylabel("% of cohort")
    ax.set_title("Cohort retention is flat — no week is special, the drivers are user-level")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(CHARTS / "cohort_retention.png"); plt.close(fig)

    # 2. Retention by channel.
    ch = channels.set_index("channel").loc[["organic", "referral", "paid", "campus", "waitlist"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(ch)); w = 0.26
    ax.bar(x - w, ch["activation_rate"] * 100, w, label="Activation", color=AMBER)
    ax.bar(x, ch["d7"] * 100, w, label="D7", color=TEAL)
    ax.bar(x + w, ch["d30"] * 100, w, label="D30", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(ch.index)
    ax.set_ylabel("% of cohort"); ax.set_title("Channel gaps are mostly a cards-linked story, not a channel story")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(CHARTS / "channel_retention.png"); plt.close(fig)

    # 3. D30 retention by activated x cards linked.
    ac = act_cards.pivot(index="cards_linked", columns="activated", values="d30") * 100
    lo = float(act_cards["d30"].min() * 100)
    hi = float(act_cards["d30"].max() * 100)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(4); w = 0.35
    ax.bar(x - w / 2, ac[False], w, label="Not activated", color=GREY)
    ax.bar(x + w / 2, ac[True], w, label="Activated", color=TEAL)
    ax.set_xticks(x); ax.set_xticklabels(["0 cards", "1 card", "2 cards", "3 cards"])
    ax.set_ylabel("D30 retention (%)")
    ax.set_title(f"Activation + cards linked: D30 retention from {lo:.0f}% to {hi:.0f}%")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(CHARTS / "activation_retention.png"); plt.close(fig)

    # 4. Model comparison (AUC + accuracy).
    names = ["rule", "logistic", "random_forest"]
    labels = ["Simple rule", "Logistic regression", "Random forest"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w / 2, [results[n]["auc"] for n in names], w, label="AUC", color=BLUE)
    ax.bar(x + w / 2, [results[n]["accuracy"] for n in names], w, label="Accuracy", color=TEAL)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("Score")
    ax.set_title("Model comparison — the simple rule is competitive with the forest")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(CHARTS / "model_comparison.png"); plt.close(fig)

    # 5. Driver ranking (permutation importance, RF).
    top = drivers[:7]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.barh([d["feature"] for d in top][::-1], [d["importance"] for d in top][::-1], color=BLUE)
    ax.set_xlabel("AUC drop when shuffled (permutation importance)")
    ax.set_title("What predicts D30 retention — activation and cards linked dominate")
    fig.tight_layout()
    fig.savefig(CHARTS / "driver_importance.png"); plt.close(fig)

    # 6. Daily active share, days 1-60.
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(daily["day"], daily["share_all"] * 100, color=BLUE, label="All users")
    ax.plot(daily["day"], daily["share_activated"] * 100, color=TEAL, label="Activated")
    ax.plot(daily["day"], daily["share_not_activated"] * 100, color=RED, label="Not activated")
    ax.set_xlabel("Days since signup"); ax.set_ylabel("% active that day")
    ax.set_title("Daily activity: activated users stay active, the rest fade in two weeks")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(CHARTS / "daily_activity.png"); plt.close(fig)

    h = metrics["headline"]
    print(f"users={metrics['n_users']} activation={h['activation_rate']:.3f} "
          f"D7={h['d7_retention']:.3f} D30={h['d30_retention']:.3f} D60={h['d60_retention']:.3f}")
    for n in names:
        r = results[n]
        f1 = r.get("f1_at_best_threshold", r.get("f1"))
        print(f"{n}: AUC={r['auc']:.3f} acc={r['accuracy']:.3f} F1(best-thr)={f1:.3f} "
              f"lift@20%={r['lift']['lift']:.2f}x")
    print(f"top driver: {drivers[0]['feature']} ({drivers[0]['importance']:.3f})")
    print(f"experiment: {experiment['n_per_arm']} per arm, "
          f"{experiment['weeks_to_recruit']} weeks to recruit")
    print("metrics.json + 6 charts written")


if __name__ == "__main__":
    main()
