"""A/B experiment design for the activation intervention.

Intervention ("48-hour activation nudge"): new signups get an in-app
checklist + one push notification within 48 hours prompting them to link a
second card and complete their first optimised payment. Hypothesis: pushing
activation in the first 48 hours raises D30 retention.

This module computes the sample size from the OBSERVED baseline (never a
guessed number) and states the decision rule BEFORE any data is collected —
the rule is the deliverable; running the test is a future step.
"""
import math

Z_ALPHA_2SIDED_05 = 1.96   # two-sided alpha = 0.05
Z_BETA_80 = 0.84           # power = 80%


def sample_size_per_arm(p0, mde, alpha_z=Z_ALPHA_2SIDED_05, beta_z=Z_BETA_80):
    """Per-arm n to detect an absolute uplift `mde` from baseline `p0`."""
    p1 = p0 + mde
    p_bar = (p0 + p1) / 2.0
    n = ((alpha_z * math.sqrt(2 * p_bar * (1 - p_bar))
          + beta_z * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2) / (mde ** 2)
    return int(math.ceil(n))


def design_experiment(baseline_d30, mde=0.03, signups_per_week=200,
                      alpha=0.05, power=0.80):
    """Return the full experiment spec, with the decision rule written first."""
    n_arm = sample_size_per_arm(baseline_d30, mde)
    total = 2 * n_arm
    weeks = math.ceil(total / signups_per_week)
    return {
        "intervention": ("48-hour activation nudge: in-app checklist + one push "
                         "notification prompting a second linked card and a first "
                         "optimised payment within 48h of signup."),
        "hypothesis": ("Pushing activation into the first 48 hours raises D30 "
                       "retention by at least 3 percentage points."),
        "randomisation": "At signup, 50/50, stratified by acquisition channel.",
        "primary_metric": "D30 retention (any transaction on days 28-30).",
        "guardrails": [
            "Activation rate must not fall by more than 1pp.",
            "No increase in support contacts per 1,000 signups.",
            "No increase in linked-card removals within 30 days.",
        ],
        "secondary_metrics": [
            "D7 retention", "cards linked by day 7", "reward earned in week 1",
            "days to activation (median)",
        ],
        "baseline_d30": round(float(baseline_d30), 4),
        "mde_pp": mde * 100,
        "alpha": alpha,
        "power": power,
        "n_per_arm": n_arm,
        "total_n": total,
        "signups_per_week": signups_per_week,
        "weeks_to_recruit": weeks,
        "srm_check": "Chi-square on allocation counts; stop and investigate if p < 0.001.",
        "decision_rule": (
            "SHIP if all of: (1) D30 uplift >= +3pp, (2) two-sided p < 0.05, "
            "(3) activation rate not worse by >1pp, (4) no guardrail breached, "
            "(5) no sample-ratio mismatch. KILL if D30 uplift <= 0pp or any "
            "guardrail breached. ITERATE (one more variant, same rule) if uplift "
            "is between 0 and +3pp. The rule is fixed before launch; it is not "
            "renegotiated after seeing results."
        ),
        "causation_note": (
            "The driver ranking in this project is OBSERVATIONAL: activated users "
            "retain better, but activation also marks users who were already "
            "engaged. Correlation is not causation — only this randomised test "
            "can establish a causal link between causing activation and retention."
        ),
    }
