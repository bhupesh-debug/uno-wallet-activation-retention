"""Synthetic user-event data for Uno Wallet activation & retention analysis.

100% SYNTHETIC. No real users, no real Uno Wallet data. The generator encodes a
plausible product story with KNOWN drivers so the analysis can be checked
against ground truth:

  * Activation = first successful optimised payment within 7 days of signup.
  * Cards linked and activation are the strongest retention drivers (by design).
  * Channel acts mostly THROUGH cards linked / activation (referral users link
    more cards) — a confounder the analysis must not mistake for a channel
    effect. That is deliberate: it is the correlation-vs-causation lesson.
  * Reward earned in week 1 is a WEAK driver — it correlates with retention
    because active users earn more, not because earning causes retention.

Reproducible: fixed seed, numpy Generator. Run: python run_all.py
"""
import numpy as np
import pandas as pd

SEED = 20261125
N_USERS = 2400
N_WEEKS = 12
USERS_PER_WEEK = N_USERS // N_WEEKS
HORIZON_DAYS = 60

CHANNELS = ["organic", "referral", "paid", "campus", "waitlist"]
CHANNEL_P = np.array([0.30, 0.25, 0.25, 0.12, 0.08])

# P(cards_linked = 0,1,2,3) by channel. Referral/organic link more cards.
CARDS_P = {
    "organic":  [0.18, 0.34, 0.32, 0.16],
    "referral":  [0.10, 0.28, 0.38, 0.24],
    "paid":     [0.30, 0.38, 0.24, 0.08],
    "campus":   [0.26, 0.36, 0.28, 0.10],
    "waitlist":  [0.14, 0.30, 0.36, 0.20],
}
CHANNEL_ACTIVATION_SHIFT = {
    "organic": 0.10, "referral": 0.35, "paid": -0.25, "campus": 0.0, "waitlist": 0.20,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_users=N_USERS, seed=SEED):
    """Return (users_df, events_df).

    users_df: one row per user with signup week, channel, cards_linked,
    activation, first-week behaviour and D7/D30 retention flags.
    events_df: one row per (user, day) with an active flag, days 1..60.
    """
    rng = np.random.default_rng(seed)
    users = []
    events = []
    per_week = n_users // N_WEEKS

    for i in range(n_users):
        signup_week = i // per_week
        channel = str(rng.choice(CHANNELS, p=CHANNEL_P))
        cards_linked = int(rng.choice([0, 1, 2, 3], p=CARDS_P[channel]))

        # --- Activation: first successful optimised payment within 7 days ---
        act_logit = -0.9 + 0.75 * cards_linked + CHANNEL_ACTIVATION_SHIFT[channel]
        act_logit += rng.normal(0, 0.35)  # unobserved user heterogeneity
        activated = bool(rng.random() < _sigmoid(act_logit))
        days_to_activation = None
        if activated:
            # geometric-ish: most activate on day 1-2, tail to day 7
            days_to_activation = int(min(7, 1 + rng.geometric(0.45) - 1))

        # --- First-week behaviour ---
        lam_txns = 0.7 + 0.9 * cards_linked + 0.9 * int(activated)
        first_week_txns = int(rng.poisson(lam_txns))
        reward_per_txn = np.clip(rng.normal(0.85, 0.35, size=max(first_week_txns, 1)), 0.05, None)
        reward_earned_wk1 = float(round(reward_per_txn.sum(), 2)) if first_week_txns else 0.0

        # --- Daily activity, days 1..60 ---
        # Latent daily-engagement logit: activation + cards dominate; first-week
        # transactions add a little; reward earned adds almost nothing (by design).
        # Daily probability is LOW (a wallet is used a few days a week, not
        # every day) and decays, so retention is a real, discriminating outcome.
        eng_logit = (-3.6 + 1.25 * int(activated) + 0.55 * (cards_linked >= 2)
                     + 0.35 * (cards_linked == 1) + 0.12 * min(first_week_txns, 6)
                     + 0.02 * min(reward_earned_wk1, 10.0)
                     + rng.normal(0, 0.45))
        p_day = _sigmoid(eng_logit)
        active_days = []
        for d in range(1, HORIZON_DAYS + 1):
            p = p_day * (0.98 ** d)  # engagement decays over the horizon
            if rng.random() < p:
                active_days.append(d)
        active_set = set(active_days)

        # Retention = active in a 3-day window around the milestone day,
        # so D30 is strictly harder than D7 (a later, narrower window).
        retained_d7 = bool(active_set & set(range(5, 8)))
        retained_d30 = bool(active_set & set(range(28, 31)))
        retained_d60 = bool(active_set & set(range(58, 61)))

        users.append({
            "user_id": f"U{i:05d}",
            "signup_week": signup_week,
            "channel": channel,
            "cards_linked": cards_linked,
            "activated": activated,
            "days_to_activation": days_to_activation,
            "first_week_txns": first_week_txns,
            "reward_earned_wk1": reward_earned_wk1,
            "retained_d7": retained_d7,
            "retained_d30": retained_d30,
            "retained_d60": retained_d60,
            "n_active_days_60": len(active_days),
        })
        for d in active_days:
            events.append({"user_id": f"U{i:05d}", "day": d, "active": True})

    users_df = pd.DataFrame(users)
    events_df = pd.DataFrame(events)
    return users_df, events_df


def save_synthetic(users_df, events_df, users_path, events_path):
    users_df.to_csv(users_path, index=False)
    events_df.to_csv(events_path, index=False)


if __name__ == "__main__":
    u, e = generate()
    print(f"users={len(u)} events={len(e)}")
    print(f"activation={u['activated'].mean():.3f}  D7={u['retained_d7'].mean():.3f}  "
          f"D30={u['retained_d30'].mean():.3f}  D60={u['retained_d60'].mean():.3f}")
    print(u.groupby("channel")[["activated", "retained_d7", "retained_d30"]].mean().round(3))
