"""User-level feature frame for retention modelling.

Target: retained_d30 (1 = at least one transaction on days 28-30).
Features are all known at the end of week 1 — nothing peeks at the future.
"""
import pandas as pd

FEATURES = [
    "cards_linked",
    "activated",
    "first_week_txns",
    "reward_earned_wk1",
    "days_to_activation",
    "signup_week",
]
CHANNELS = ["organic", "referral", "paid", "campus", "waitlist"]
TARGET = "retained_d30"


def build_user_frame(users_df):
    """Return a modelling frame: features + target + channel dummies."""
    df = users_df.copy()
    df["activated"] = df["activated"].astype(int)
    # Days to activation: non-activated users get 8 (outside the 7-day window).
    df["days_to_activation"] = df["days_to_activation"].fillna(8).astype(int)
    df["retained_d7"] = df["retained_d7"].astype(int)
    df["retained_d30"] = df["retained_d30"].astype(int)
    df["retained_d60"] = df["retained_d60"].astype(int)
    for ch in CHANNELS:
        df[f"channel_{ch}"] = (df["channel"] == ch).astype(int)
    return df


def model_columns():
    return FEATURES + [f"channel_{c}" for c in CHANNELS]


def daily_active_share(events_df, users_df, horizon=60):
    """Share of users active on each day 1..horizon (overall + split by activation)."""
    n = len(users_df)
    rows = []
    act_users = set(users_df.loc[users_df["activated"], "user_id"])
    for d in range(1, horizon + 1):
        day_users = set(events_df.loc[events_df["day"] == d, "user_id"])
        rows.append({
            "day": d,
            "share_all": len(day_users) / n,
            "share_activated": len(day_users & act_users) / max(len(act_users), 1),
            "share_not_activated": len(day_users - act_users) / max(n - len(act_users), 1),
        })
    return pd.DataFrame(rows)
