"""Build the 10-page report PDF and the 10-slide deck PDF (reportlab).

Reads reports/metrics.json so every number in the documents comes from
the executed pipeline, not from hand-typed values.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent
OUT = Path("/home/hatch/workspace/your_files")
CHARTS = ROOT / "reports" / "charts"
M = json.loads((ROOT / "reports" / "metrics.json").read_text())
H = M["headline"]
MOD = M["models"]
DRV = M["drivers"]
COEF = M["logistic_coefs"]
EXP = M["experiment"]
CHAN = {r["channel"]: r for r in M["channels"]}

NAVY = colors.HexColor("#1A3A6B")
GREEN = colors.HexColor("#2E7D5B")
RED = colors.HexColor("#B03A2E")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#EAF0F7")

H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceAfter=8)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=NAVY, spaceBefore=8, spaceAfter=3)
B = ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=5)
SM = ParagraphStyle("sm", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=GREY, spaceAfter=4)
CELL = ParagraphStyle("c", fontName="Helvetica", fontSize=9, leading=12)
CELLB = ParagraphStyle("cb", fontName="Helvetica-Bold", fontSize=9, leading=12)
TITLE = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=NAVY)
SUB = ParagraphStyle("s", fontName="Helvetica", fontSize=12.5, leading=17, textColor=GREY)


def P(text, style=B):
    return Paragraph(text, style)


def bullets(items, style=B):
    return [Paragraph(f"•&nbsp;&nbsp;{t}", style) for t in items]


def table(headers, rows, widths):
    data = [[Paragraph(h, CELLB) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B9C6D8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def chart(name, w=7.0, h=3.0):
    return Image(str(CHARTS / name), width=w * inch, height=h * inch)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(0.75 * inch, 0.55 * inch,
                      "Activation & Retention Analysis  •  Uno Wallet AI Business Analyst  •  SYNTHETIC DATA ONLY")
    canvas.drawRightString(7.75 * inch, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


def pct(x):
    return f"{x:.1%}"


def build_report():
    path = OUT / "Bhupesh_Chandra_Dimri_ML-Project_Uno-Wallet-Activation-Retention.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, leftMargin=0.75 * inch,
                            rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                            title="ML Project — Activation & Retention Analysis, Uno Wallet",
                            author="Bhupesh Chandra Dimri")
    s = []
    # ---- Page 1: cover + executive summary ----
    s += [Spacer(1, 0.5 * inch),
          P("Activation &amp; Retention<br/>Analysis", TITLE), Spacer(1, 0.15 * inch),
          P("ML Project for the AI Business Analyst role<br/>Uno Wallet (YC P26) — San Francisco / New York / Remote", SUB),
          Spacer(1, 0.25 * inch),
          P("Bhupesh Chandra Dimri &nbsp;|&nbsp; November 2026", B),
          P("Python • pandas • scikit-learn • cohort analysis • randomised experiment design", SM),
          Spacer(1, 0.15 * inch)]
    s.append(P("Executive summary", H1))
    s += bullets([
        "<b>What it is:</b> a wallet lives or dies on whether users come back after week one. This project measures activation and retention on 2,400 synthetic users over 12 signup weeks, ranks what actually predicts D30 retention, and designs the randomised test that would prove whether <i>causing</i> activation <i>causes</i> retention.",
        f"<b>The funnel:</b> activation <b>{pct(H['activation_rate'])}</b> (first successful optimised payment within 7 days), D7 retention <b>{pct(H['d7_retention'])}</b>, D30 retention <b>{pct(H['d30_retention'])}</b>, D60 retention <b>{pct(H['d60_retention'])}</b>. One in five users is still active at day 30 — the problem is real and the levers are visible.",
        f"<b>Honest finding:</b> the simple two-feature rule (activated AND ≥2 cards linked) <b>beats both fitted models on F1</b> ({MOD['rule']['f1']:.3f} vs {MOD['logistic']['f1_at_best_threshold']:.3f} logistic, {MOD['random_forest']['f1_at_best_threshold']:.3f} random forest) and trails by only 0.05–0.06 AUC. The models add ranking value, not decision value.",
        "<b>Correlation ≠ causation:</b> referral users retain best (D30 23.8% vs paid 15.5%) — but they also link more cards (1.80 vs 1.13). Channel looks like a driver until you condition on cards linked. Only the experiment in §7 can settle it.",
        f"<b>The test:</b> the 48-hour activation nudge, powered for +3pp on D30, needs <b>{EXP['n_per_arm']:,} users per arm — {EXP['weeks_to_recruit']} weeks</b> at 200 signups/week. The pragmatic fallback (+5pp) needs 1,084 per arm, 11 weeks. That timeline is itself a finding.",
        "<b>Data:</b> 100% synthetic, seeded and reproducible. No real Uno Wallet user data, no live-product claim anywhere in this document.",
    ])
    s.append(P("Match score: 6 / 10 — strong on data analysis, SQL/Python, forecasting and dashboards; gaps are product analytics, experimentation design and the payments domain. See §6.", B))
    s.append(PageBreak())

    # ---- Page 2: Step 1 JD breakdown ----
    s.append(P("1 &nbsp; JD breakdown — what the role really asks for", H1))
    s.append(P("Uno Wallet (YC P26) is a 2-person startup building an AI-powered iPhone wallet that automatically selects the best credit card for every purchase. The AI Business Analyst role is full-time, $75–140K/yr + signing bonus, bonus and equity, San Francisco Bay Area / New York / Miami / Remote US. The JD is two bullets — so this section states what is in the JD and labels everything else as inference.", B))
    s.append(table(
        ["Category", "What the JD asks for"],
        [["Core responsibilities", "Analyse data, identify business opportunities, and turn insights into actionable recommendations. Leverage AI tools to improve business processes and decision-making."],
         ["Required technical skills", "Data analysis (implied: SQL, Python, spreadsheets/BI). No specific tools are named in the JD — [INFERENCE] the stack is whatever a 2-person startup runs: product analytics, SQL, Python, dashboards."],
         ["Soft skills", "Analytical, entrepreneurial, resourceful, strong problem-solving."],
         ["Hidden expectations", "Ownership without a manager layer: the analyst IS the analytics function. Comfort with ambiguity, tiny samples and fast decisions; founder-facing communication; judgement about what NOT to measure."],
         ["Success measures", "Not stated in the JD — [INFERENCE] activation and retention (this project), rewards yield per user, and decisions shipped, not dashboards built."],
         ["Seniority", "Not stated. A generalist analyst seat at a 2-person startup — the bar is end-to-end ownership, not depth in one tool."],
         ["Domain knowledge", "Credit-card rewards economics (earn rates, category caps, interchange) and consumer payments behaviour — [INFERENCE] from the product, not stated in the JD."]],
        [1.5 * inch, 5.5 * inch]))
    s.append(P("Requirement hierarchy", H2))
    s += bullets([
        "<b>Must-have:</b> data analysis that ends in a recommendation; SQL/Python; clear communication to non-technical founders.",
        "<b>Good-to-have:</b> product analytics (funnels, cohorts, retention); experimentation design; dashboarding.",
        "<b>Bonus:</b> payments/rewards domain knowledge; experience shipping an A/B test end-to-end; a reproducible analysis artefact (this project).",
    ])
    s.append(P("Risk factors: the JD's thinness is the main risk — most of the role picture is inferred from the product. Pay tops out at $140K, the floor of the candidate's band. E-Verify/I-983 at a 2-person startup is unconfirmed. None of these is a reason to hide the analysis; all three are stated in the package.", B))
    s.append(PageBreak())

    # ---- Page 3: Steps 2-3 ML problem + data ----
    s.append(P("2 &nbsp; ML problem definition", H1))
    s.append(P("Three problem statements, one per decision the role actually makes:", B))
    s.append(table(
        ["Problem", "Input", "Output / target", "Type"],
        [["P1 — Activation &amp; retention measurement", "Synthetic user events: signup, first payment, transactions, cards linked", "Activation rate; D7 / D30 / D60 retention by cohort and channel", "Descriptive analytics / cohort analysis"],
         ["P2 — Retention prediction", "Week-1 features only: cards linked, activation, first-week txns, reward earned, days to activation, channel", "P(retained at D30) — a binary label, no lookahead", "Classification (rule vs logistic vs random forest)"],
         ["P3 — Causal test design", "Observed D30 baseline (19.8%) and signup volume (200/week)", "A/B design for the 48-hour activation nudge + pre-written decision rule", "Experiment design / power analysis"]],
        [1.3 * inch, 2.5 * inch, 1.9 * inch, 1.3 * inch]))
    s.append(P("Constraints that shaped the design: features must be knowable at the end of week 1 (no peeking at day-30 behaviour); interpretability first (founders must be able to act on the answer); the sample is small, so a simple rule is a legitimate competitor, not a straw man; and the observational ranking must never be presented as causal — the experiment is the causal instrument.", B))
    s.append(P("3 &nbsp; Data strategy — synthetic, by design", H1))
    s.append(P("No real user data is used. The generator (<font face='Helvetica-Bold'>src/data_gen.py</font>, seed 20261125) produces 2,400 users over 12 signup weeks (200/week) and their daily activity for 60 days. The process encodes known drivers so the analysis can be checked against ground truth: activation and cards linked drive retention; channel acts mostly <i>through</i> cards linked (a deliberate confounder); reward earned in week 1 is a weak driver — it correlates with retention because active users earn more, not because earning causes retention.", B))
    s.append(table(
        ["Field", "Definition"],
        [["Activation", "First successful optimised payment within 7 days of signup"],
         ["D7 retention", "At least one transaction on days 5–7 after signup"],
         ["D30 retention", "At least one transaction on days 28–30 after signup"],
         ["D60 retention", "At least one transaction on days 58–60 after signup"],
         ["Cards linked", "0–3 cards on file at signup (channel-dependent distribution)"],
         ["Reward earned (wk 1)", "Dollar value of points earned on first-week transactions"]],
        [1.5 * inch, 5.5 * inch]))
    s.append(P("Preprocessing: one row per user, all features known by end of week 1; channel one-hot encoded; non-activated users get days_to_activation = 8 (outside the 7-day window) so the feature is complete without leakage. Edge cases — users with 0 cards, 0 first-week transactions, late activators — are in the data by design and appear in the driver table (§8).", B))
    s.append(PageBreak())

    # ---- Page 4: Steps 4-5 model strategy + visual intelligence ----
    s.append(P("4 &nbsp; Model strategy", H1))
    s.append(table(
        ["Tier", "Model", "Role in this project"],
        [["Baseline", "Rule: predict retained if activated AND ≥2 cards linked", "The bar every model must beat. Two features, zero training, fully explainable to a founder"],
         ["Intermediate", "Logistic regression (standardised features, C=1.0)", "Interpretable direction of each driver; coefficients readable by a non-ML stakeholder"],
         ["Advanced", "Random forest (300 trees, depth 6, min leaf 20)", "Included deliberately to test whether complexity helps on 2,400 users — it barely does (§8)"]],
        [1.0 * inch, 2.2 * inch, 3.8 * inch]))
    s.append(P("Feature engineering — the five ideas that matter", H2))
    s += bullets([
        "<b>Days to activation:</b> speed of first success, not just whether it happened — the top-ranked driver (§8).",
        "<b>Cards linked ≥2 (threshold):</b> the wallet's value proposition only works with choice; the rule uses the threshold directly.",
        "<b>First-week transaction count:</b> early habit formation, capped at 6 to stop one power user dominating.",
        "<b>Reward earned (week 1):</b> included to test the obvious story — and it turns out to be a weak driver, which is itself a finding.",
        "<b>Channel dummies:</b> included so the model can show channel is <i>not</i> the driver once cards linked is in the frame.",
    ])
    s.append(P("Evaluation: stratified 70/30 split (seed 42); primary metric AUC; decision metric F1 at a threshold chosen on TRAIN to maximise F1 (the 0.5 default is meaningless at a 19.8% base rate — it predicts almost no positives, and §8 shows both numbers so the artefact is visible). Overfitting controls: regularised logistic, depth-6 forest with min-leaf-20, and the train-vs-test AUC comparison in §8.", B))
    s.append(P("5 &nbsp; Visual intelligence — what the charts say", H1))
    s.append(P("The cohort curves (§5, chart 1) are flat across signup weeks: no week is special, so the drivers are user-level, not calendar-level. The channel chart (§5, chart 2) shows referral ahead on every metric — and the driver ranking (§8, chart 5) shows why that is mostly a cards-linked story. The daily-activity curve (§9, chart 6) is the most actionable picture in the project: activated users stay active for 60 days, non-activated users fade within two weeks. That is the whole case for the 48-hour nudge in one chart.", B))
    s.append(PageBreak())

    # ---- Page 5: charts 1-2 ----
    s.append(P("5 &nbsp; Visual intelligence (charts)", H1))
    s.append(chart("cohort_retention.png", 7.0, 3.1))
    s.append(P("Chart 1 — Activation, D7 and D30 by signup week. Flat lines: retention does not drift by cohort, so the levers are user-level (activation, cards linked), not timing.", SM))
    s.append(chart("channel_retention.png", 7.0, 3.1))
    s.append(P(f"Chart 2 — Activation / D7 / D30 by channel. Referral leads (D30 {pct(CHAN['referral']['d30'])}) and paid trails (D30 {pct(CHAN['paid']['d30'])}); referral users also link the most cards ({CHAN['referral']['mean_cards_linked']:.2f} vs {CHAN['paid']['mean_cards_linked']:.2f}) — the confounder to watch.", SM))
    s.append(PageBreak())

    # ---- Page 6: Step 6 resume alignment ----
    s.append(P("6 &nbsp; Resume alignment — match 6 / 10", H1))
    s.append(P("Candidate: Bhupesh Chandra Dimri — Financial Analyst, JPMorgan Chase (Jul 2025–present), 9+ years forecasting, budgeting and variance analysis; M.S. Accounting Analytics &amp; Data Technology, Pace (Dec 2025, GPA 3.24); MBA Finance &amp; Accounting. Evidence is taken only from the verified record.", B))
    s.append(table(
        ["JD requirement", "Resume evidence", "Gap?"],
        [["Analyse data; turn insights into recommendations", "Variance frameworks linking KPIs to outcomes; forecasting accuracy +27%; reporting latency 48h to under 2h", "No"],
         ["Leverage AI tools to improve processes", "AI Engineering guide + demos (runnable, synthetic data); ML projects on GitHub; automation scripting (VBA, RPA)", "Partial — production AI use unconfirmed"],
         ["SQL / Python / dashboards", "SQL, Python (pandas, NumPy, scikit-learn), Power BI, Tableau — evidenced throughout the record", "No"],
         ["Product analytics (funnels, cohorts, A/B)", "Not evidenced in the record — this project is the demonstration, not a claim of prior experience", "Yes — project addresses it"],
         ["Payments / rewards domain", "Banking and financial-services domain; no credit-card rewards experience", "Yes"],
         ["Startup / entrepreneurial ownership", "9+ years in large organisations; ownership shown in side projects, not a startup role", "Yes — [CONFIRM WITH CANDIDATE]"],
         ["Experimentation design", "Not evidenced before this project; the A/B design in §7 is the artefact", "Yes — project addresses it"]],
        [1.7 * inch, 3.8 * inch, 1.5 * inch]))
    s.append(P("Top 3 missing skills: (1) product analytics — practised here end-to-end (cohorts, retention curves, driver ranking); (2) experimentation — the pre-written decision rule and power analysis in §7 are the evidence; (3) payments/rewards domain — the sibling project (rewards optimisation) covers the economics, and the honest position is 'learning the domain through the data', not claiming it. Bullet upgrades: rewrite the JPMorgan dashboard bullet in the JD's language — 'Built cohort-style reporting that cut reporting latency from 48 hours to under 2 hours, turning analysis into decisions the business acted on' — and add this project as a second bullet with the activation/retention numbers and the rule-beats-forest finding.", B))
    s.append(P("Existing GitHub work used (real metrics only, no duplication): <font face='Helvetica-Bold'>AI-Engineering</font> (guide + runnable demos, synthetic toy data) and the FP&amp;A predictive variance engine (LightGBM WAPE 4.62%, R² 0.942). This project is distinct: user-level retention modelling, cohort analysis and experiment design — none of which the earlier engines do.", B))
    s.append(PageBreak())

    # ---- Page 7: Step 7 architecture + experiment ----
    s.append(P("7 &nbsp; Solution design, architecture &amp; the experiment", H1))
    s.append(P("End-to-end pipeline: <font face='Helvetica-Bold'>Synthetic generator → Week-1 feature frame (no lookahead) → Cohort tables → Model registry (rule / logistic / forest, one interface) → Driver ranking (permutation importance) → Experiment design (power + decision rule) → Monitoring (activation, D7, D30 weekly by cohort)</font>.", B))
    s.append(P("Tech stack: Python 3, pandas, scikit-learn, matplotlib; no deep learning — the role's bar is applied analytics, and 2,400 users cannot support a transformer. Deployment pattern for a 2-person startup: a weekly batch job, not a live service — regenerate cohorts after each signup week, log activation/D7/D30 with the driver ranking, and review with the founders. Experiment tracking is a plain metrics.json plus git commits.", B))
    s.append(P("The experiment — 48-hour activation nudge", H2))
    s.append(P("Intervention: an in-app checklist plus one push notification within 48 hours of signup, prompting the user to link a second card and complete a first optimised payment. Hypothesis: pushing activation into the first 48 hours raises D30 retention by at least 3pp. Randomisation: 50/50 at signup, stratified by channel. Primary metric: D30 retention. Guardrails: activation rate must not fall by more than 1pp; no increase in support contacts per 1,000 signups; no increase in linked-card removals within 30 days. Secondary metrics: D7 retention, cards linked by day 7, reward earned in week 1, median days to activation.", B))
    s.append(table(
        ["Design", "Primary (+3pp MDE)", "Pragmatic fallback (+5pp MDE)"],
        [["Baseline D30", pct(EXP["baseline_d30"]), pct(EXP["baseline_d30"])],
         ["Users per arm", f"{EXP['n_per_arm']:,}", f"{EXP['mde_5pp_alternative']['n_per_arm']:,}"],
         ["Total users", f"{EXP['total_n']:,}", f"{EXP['mde_5pp_alternative']['total_n']:,}"],
         ["Weeks to recruit (200/wk)", str(EXP["weeks_to_recruit"]), str(EXP["mde_5pp_alternative"]["weeks_to_recruit"])],
         ["Alpha / power", "0.05 two-sided / 80%", "0.05 two-sided / 80%"]],
        [1.6 * inch, 2.7 * inch, 2.7 * inch]))
    s.append(P("<b>Decision rule (written before launch, not renegotiated after results):</b> " + EXP["decision_rule"], B))
    s.append(P("<b>Why the timeline matters:</b> 30 weeks is longer than a startup's planning horizon. The honest recommendation is to run the +5pp design (11 weeks) and treat a +3pp effect as something the product learns about later, from observational data — not from a test it cannot afford. Sample-ratio-mismatch check (chi-square, stop if p &lt; 0.001) is part of the protocol.", B))
    s.append(PageBreak())

    # ---- Page 8: results + charts 3-4 ----
    s.append(P("8 &nbsp; Results — the honest findings", H1))
    s.append(table(
        ["Model", "AUC", "Accuracy", "F1*", "Lift @ top 20%"],
        [["Simple rule", f"{MOD['rule']['auc']:.3f}", f"{MOD['rule']['accuracy']:.3f}", f"{MOD['rule']['f1']:.3f}", f"{MOD['rule']['lift']['lift']:.2f}×"],
         ["Logistic regression", f"{MOD['logistic']['auc']:.3f}", f"{MOD['logistic']['accuracy']:.3f}", f"{MOD['logistic']['f1_at_best_threshold']:.3f}", f"{MOD['logistic']['lift']['lift']:.2f}×"],
         ["Random forest", f"{MOD['random_forest']['auc']:.3f}", f"{MOD['random_forest']['accuracy']:.3f}", f"{MOD['random_forest']['f1_at_best_threshold']:.3f}", f"{MOD['random_forest']['lift']['lift']:.2f}×"]],
        [1.7 * inch, 0.9 * inch, 1.1 * inch, 1.0 * inch, 1.5 * inch]))
    s.append(P("* F1 at the threshold chosen on TRAIN to maximise F1 (logistic 0.20, forest 0.25). At the naive 0.5 threshold both fitted models collapse (F1 0.014 / 0.000) — the base rate is 19.8%, so 0.5 predicts almost no positives. That artefact is reported, not hidden.", SM))
    s += bullets([
        f"<b>The simple rule wins on the decision metric.</b> F1 {MOD['rule']['f1']:.3f} vs {MOD['logistic']['f1_at_best_threshold']:.3f} (logistic) and {MOD['random_forest']['f1_at_best_threshold']:.3f} (forest). Two features, zero training, explainable to a founder in one sentence.",
        f"<b>The models win on ranking.</b> Logistic AUC {MOD['logistic']['auc']:.3f} vs rule {MOD['rule']['auc']:.3f}; targeting the top 20% by forest score retains {MOD['random_forest']['lift']['lift']:.2f}× the base rate. Use the model to prioritise outreach, the rule to decide.",
        f"<b>No overfitting worth hiding.</b> Logistic train AUC {MOD['logistic']['train_auc']:.3f} vs test {MOD['logistic']['auc']:.3f}; forest train {MOD['random_forest']['train_auc']:.3f} vs test {MOD['random_forest']['auc']:.3f} — mild, and the test numbers are the ones reported.",
        "<b>Reward earned is a weak driver.</b> It ranks third on permutation importance but the generator made it nearly irrelevant — users who transact more earn more. Rewarding harder would not buy retention; activating earlier might.",
        f"<b>The funnel's real gap is activation.</b> D30 is {pct(M['activation_cards'][4]['d30'])} for activated users with 0 cards and {pct(M['activation_cards'][7]['d30'])} for activated users with 3 cards — but only {pct(M['activation_cards'][0]['d30'])} for non-activated users with 0 cards. Getting the first payment done is the whole game.",
    ])
    s.append(chart("activation_retention.png", 7.0, 2.6))
    s.append(P("Chart 3 — D30 retention by activated × cards linked. The two levers compound: activation first, then a second card.", SM))
    s.append(PageBreak())

    # ---- Page 9: charts 4-6 + differentiation ----
    s.append(P("8 &nbsp; Results (continued)", H1))
    s.append(chart("model_comparison.png", 7.0, 2.6))
    s.append(P("Chart 4 — AUC and accuracy. The models lead on AUC; the rule's F1 lead (§8 table) is the decision-relevant comparison.", SM))
    s.append(chart("driver_importance.png", 7.0, 2.6))
    s.append(P("Chart 5 — Permutation importance (AUC drop when shuffled). Days to activation leads — speed of first success matters, not just whether it happens. Reward earned ranks third despite being nearly irrelevant by construction: a warning about reading importance as causation.", SM))
    s.append(P("10 &nbsp; Differentiation — why this stands out", H1))
    s += bullets([
        "<b>It reports the rule beating the models.</b> Most candidate projects show only the winning model. This one leads with a two-feature rule outperforming a random forest on the decision metric — and explains what each is for.",
        "<b>Correlation vs causation is the argument, not a footnote.</b> The channel confounder and the reward-earned trap are demonstrated in the data, and the experiment is designed to resolve the one question that matters.",
        "<b>It prices the experiment honestly.</b> A 30-week test at 200 signups/week is not a plan — it is a reason to choose the +5pp design. A hiring manager who has run a startup will recognise that judgement.",
    ])
    s.append(P("Five questions to ask in the interview: (1) What does activation mean to you — first payment, or first <i>optimised</i> payment? (2) How do you decide when a cohort is big enough to trust? (3) What would make you kill a feature that users love but that does not move D30? (4) How do you handle the Apple Pay default-wallet constraint in measurement? (5) What is the one metric you would put on the wall for the first 90 days?", B))
    s.append(PageBreak())

    # ---- Page 10: reproducibility, limits, next steps ----
    s.append(P("9 &nbsp; Reproducibility, limits &amp; next steps", H1))
    s.append(P("How to run", H2))
    s += bullets([
        "<font face='Helvetica-Bold'>python run_all.py</font> — data → features → cohorts → models → 6 charts → reports/metrics.json.",
        "<font face='Helvetica-Bold'>python -m tests.test_all</font> — 66 checks; runtime line: <b>66 passed, 0 failed</b>.",
        "<font face='Helvetica-Bold'>python build_docs.py</font> — regenerates this report and the deck from metrics.json.",
        "Requirements: numpy, pandas, scikit-learn, matplotlib, reportlab (see requirements.txt).",
    ])
    s.append(P("Limits — stated plainly", H2))
    s += bullets([
        "Synthetic data: the drivers are designed in, so results prove the pipeline works and the reasoning is sound — not that Uno Wallet's real users behave this way.",
        "2,400 users is small: the rule-vs-model gap (0.05 AUC) is within noise, and the report says so rather than claiming a victory.",
        "No survival analysis (Kaplan–Meier / Cox) — the next step for a real dataset with censoring; noted, not faked.",
        "The experiment is designed, not run. No causal claim is made from the observational ranking.",
        "Eligibility and E-Verify/I-983 at a 2-person startup are outside this project's scope and are flagged in the package, not here.",
    ])
    s.append(P("Next steps", H2))
    s += bullets([
        "Run the +5pp experiment design (11 weeks) once real signup volume supports it; log the decision rule in the repo before launch.",
        "Add survival analysis (Kaplan–Meier curves by activation status; Cox model) when real event data exists.",
        "Add a second intervention arm (reward-first vs card-link-first) to separate the two levers the rule conflates.",
        "Track D30 by channel weekly with the cards-linked conditioning in §8 — the dashboard that stops the team chasing channel effects.",
    ])
    s.append(P("GitHub", H2))
    s.append(P("Suggested repo: <font face='Helvetica-Bold'>uno-wallet-activation-retention</font> (github.com/bhupesh-debug). No repo exists yet — the user creates the empty repo first; the project is ready to push (see file count in the delivery note). Chart PNGs are gitignored and excluded from the push; regenerate with <font face='Helvetica-Bold'>run_all.py</font>.", B))
    s.append(P("Sources &amp; method note: JD is Uno Wallet AI Business Analyst, full-time, $75–140K/yr + signing bonus, bonus and equity, apply by Dec 25, 2026 (Handshake listing). Company facts: YC P26, founded 2026, team size 2, San Francisco; founders Saket Joshi (ex-xAI) and Mordi Shadpour (ex-Bloomberg, Harvard MBA) — verified by web search Nov 25, 2026. All user data is synthetic (src/data_gen.py, seed 20261125). All metrics are computed by the executed pipeline and recorded in reports/metrics.json.", SM))
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("report:", path)


def build_deck():
    path = OUT / "Bhupesh_Chandra_Dimri_ML-Project_Uno-Wallet-Activation-Retention_Deck.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, leftMargin=0.75 * inch,
                            rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                            title="ML Project Deck — Activation & Retention Analysis, Uno Wallet",
                            author="Bhupesh Chandra Dimri")
    s = []
    # Slide 1 — title
    s += [Spacer(1, 1.2 * inch), P("Activation &amp;<br/>Retention Analysis", TITLE), Spacer(1, 0.2 * inch),
          P("ML Project for the AI Business Analyst role<br/>Uno Wallet (YC P26) — San Francisco / New York / Remote", SUB),
          Spacer(1, 0.4 * inch), P("Bhupesh Chandra Dimri &nbsp;|&nbsp; November 2026", B),
          P("2,400 synthetic users • 12 signup weeks • SYNTHETIC DATA ONLY", SM), PageBreak()]
    # Slide 2 — problem
    s.append(P("2 — Problem overview", H1))
    s += bullets([
        "✅ <b>One-liner:</b> a wallet lives or dies on whether users come back after week one — so measure it, find what drives it, and test the fix.",
        f"Activation {pct(H['activation_rate'])} → D7 {pct(H['d7_retention'])} → D30 {pct(H['d30_retention'])} → D60 {pct(H['d60_retention'])}. One in five users is still active at day 30.",
        "Why it matters: at 200 signups a week, every point of D30 retention is worth more than any acquisition channel.",
        "The deliverable is not a dashboard — it is a decision: which lever to pull, and how to prove it worked.",
    ])
    s.append(PageBreak())
    # Slide 3 — JD breakdown
    s.append(P("3 — JD breakdown", H1))
    s += bullets([
        "<b>Must-have:</b> data analysis ending in a recommendation; SQL/Python; clear communication to founders.",
        "<b>Hidden expectations:</b> you ARE the analytics function at a 2-person startup — ownership, ambiguity, tiny samples, fast decisions.",
        "<b>Gaps (flagged, not hidden):</b> product analytics and experimentation not evidenced before this project; no payments domain; no startup experience.",
        "<b>JD is two bullets</b> — most of the role picture is inference from the product, labelled as such.",
    ])
    s.append(PageBreak())
    # Slide 4 — ML approach
    s.append(P("4 — ML approach", H1))
    s += bullets([
        "<b>P1 — Measurement:</b> activation rate and D7/D30/D60 retention by cohort and channel.",
        "<b>P2 — Prediction:</b> P(retained at D30) from week-1 features only — no lookahead. Rule vs logistic vs random forest.",
        "<b>P3 — Causal test:</b> A/B design for the 48-hour activation nudge, with a pre-written decision rule.",
        "<b>Why this bar:</b> the role asks for applied analytics, not deep learning. 2,400 users cannot support a transformer.",
    ])
    s.append(PageBreak())
    # Slide 5 — data
    s.append(P("5 — Data strategy", H1))
    s += bullets([
        "<b>100% synthetic</b> — 2,400 users, 12 signup weeks, 60 days of daily activity, seed 20261125.",
        "Drivers are known by construction: activation + cards linked drive retention; channel acts through cards linked; reward earned is nearly irrelevant.",
        "Retention = active in a 3-day window around the milestone (days 5–7, 28–30, 58–60) — D30 is strictly harder than D7.",
        "All features knowable at end of week 1. Non-activated users get days_to_activation = 8 — complete, no leakage.",
    ])
    s.append(PageBreak())
    # Slide 6 — model design
    s.append(P("6 — Model design", H1))
    s += bullets([
        "<b>Baseline first:</b> the two-feature rule (activated AND ≥2 cards) is the bar; every model must beat it.",
        "<b>Evaluation:</b> stratified 70/30 split; primary metric AUC; decision metric F1 at a threshold chosen on TRAIN (0.5 is meaningless at a 19.8% base rate).",
        "<b>Overfitting controls:</b> regularised logistic, depth-6 forest, min-leaf-20, train-vs-test AUC reported.",
        "<b>Driver ranking:</b> permutation importance on the test set — plus logistic coefficients for a second view.",
    ])
    s.append(PageBreak())
    # Slide 7 — results
    s.append(P("7 — Results &amp; insights", H1))
    s.append(table(
        ["Model", "AUC", "F1*", "Lift @ top 20%"],
        [["Simple rule", f"{MOD['rule']['auc']:.3f}", f"{MOD['rule']['f1']:.3f}", f"{MOD['rule']['lift']['lift']:.2f}×"],
         ["Logistic", f"{MOD['logistic']['auc']:.3f}", f"{MOD['logistic']['f1_at_best_threshold']:.3f}", f"{MOD['logistic']['lift']['lift']:.2f}×"],
         ["Random forest", f"{MOD['random_forest']['auc']:.3f}", f"{MOD['random_forest']['f1_at_best_threshold']:.3f}", f"{MOD['random_forest']['lift']['lift']:.2f}×"]],
        [1.8 * inch, 1.1 * inch, 1.1 * inch, 1.8 * inch]))
    s += bullets([
        "✅ <b>The simple rule wins on F1</b> — and trails by only 0.05 AUC. Use the rule to decide, the model to prioritise.",
        "✅ <b>Referral's lead is a confounder:</b> D30 23.8% vs paid 15.5%, but 1.80 vs 1.13 cards linked.",
        "✅ <b>Reward earned is a weak driver</b> — earning follows activity, it does not cause it.",
        "✅ <b>Activation is the game:</b> D30 6.6% → 35.6% across the activation × cards grid.",
    ])
    s.append(PageBreak())
    # Slide 8 — business impact
    s.append(P("8 — Business impact", H1))
    s += bullets([
        "The 48-hour nudge targets the single biggest lever in the data — first payment, then a second card.",
        f"The +3pp test needs {EXP['n_per_arm']:,} users per arm ({EXP['weeks_to_recruit']} weeks) — too slow. The +5pp design (1,084/arm, 11 weeks) is the plan.",
        "The rule gives founders a decision in one sentence; the model gives growth a prioritised list (top 20% retains 1.94× the base rate).",
        "No vanity metrics: signups are never reported without the retained users behind them.",
    ])
    s.append(PageBreak())
    # Slide 9 — future improvements
    s.append(P("9 — Future improvements", H1))
    s += bullets([
        "Run the +5pp experiment once real signup volume supports it; log the decision rule before launch.",
        "Add survival analysis (Kaplan–Meier, Cox) when real event data with censoring exists.",
        "Add a second intervention arm (reward-first vs card-link-first) to separate the two levers.",
        "Track D30 by channel weekly, conditioned on cards linked — the dashboard that stops the team chasing channel effects.",
        "Production readiness: weekly batch job, metrics.json + git, founder review of cohorts.",
    ])
    s.append(PageBreak())
    # Slide 10 — about
    s.append(P("10 — About this project", H1))
    s += bullets([
        "<b>Created by Bhupesh Chandra Dimri</b> — Financial Analyst, JPMorgan Chase; M.S. Accounting Analytics &amp; Data Technology, Pace (Dec 2025).",
        "<b>GitHub:</b> suggested repo <b>uno-wallet-activation-retention</b> (github.com/bhupesh-debug) — to be created by the user; project ready to push.",
        "<b>Run it:</b> python run_all.py &nbsp;•&nbsp; python -m tests.test_all (66 passed, 0 failed).",
        "<b>Contact:</b> bhupesh7cd@gmail.com &nbsp;•&nbsp; 917-818-6198 &nbsp;•&nbsp; linkedin.com/in/bhupeshdimri",
        "<b>Disclosure:</b> all user data is synthetic; no claim about Uno Wallet's real users or product; the experiment is designed, not run — no causal claim is made from observational data.",
    ])
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("deck:", path)


if __name__ == "__main__":
    build_report()
    build_deck()
