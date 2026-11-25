"""Cohort retention analysis: by signup week and by acquisition channel."""
import pandas as pd


def cohort_table(users_df):
    """One row per signup week: cohort size, activation, D7, D30, D60."""
    g = users_df.groupby("signup_week").agg(
        cohort_size=("user_id", "count"),
        activation_rate=("activated", "mean"),
        d7=("retained_d7", "mean"),
        d30=("retained_d30", "mean"),
        d60=("retained_d60", "mean"),
    ).reset_index()
    return g


def channel_table(users_df):
    """One row per channel: size, activation, D7, D30, mean cards linked."""
    g = users_df.groupby("channel").agg(
        n=("user_id", "count"),
        activation_rate=("activated", "mean"),
        d7=("retained_d7", "mean"),
        d30=("retained_d30", "mean"),
        mean_cards_linked=("cards_linked", "mean"),
        mean_first_week_txns=("first_week_txns", "mean"),
        mean_reward_wk1=("reward_earned_wk1", "mean"),
    ).reset_index()
    return g


def retention_by_activation_cards(users_df):
    """D30 retention split by activated x cards_linked — the driver table."""
    g = users_df.groupby(["activated", "cards_linked"]).agg(
        n=("user_id", "count"),
        d7=("retained_d7", "mean"),
        d30=("retained_d30", "mean"),
    ).reset_index()
    return g
