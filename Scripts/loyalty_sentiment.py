"""
Loyalty Program Sentiment Analysis — Caesars Rewards vs MGM Rewards (v2)
=========================================================================
Evaluates customer sentiment toward casino loyalty programs by filtering
reviews for loyalty-related keywords and scoring them with VADER + TextBlob.

v2 changes:
  - Expanded loyalty keyword list (Task 1)
  - n_reviews + confidence_flag in monthly output (Task 2)
  - Weighted trend regressions (Task 2)
  - TextBlob polarity cross-check, conflicting signal detection (Task 3)
  - Mar 2025 Caesars crash investigation (Task 4)
  - Updated chart with volume panel, greyed-out low-n, TextBlob lines (Task 5)

Inputs:
  Data/las_vegas_reviews.csv
  Data/las_vegas_strip_entities.md  (for parent_company mapping)

Outputs:
  Data/loyalty_sentiment_summary.csv
  Data/loyalty_sentiment_monthly.csv
  Data/loyalty_sentiment_trend.png
"""

import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy import stats
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

REVIEWS_CSV = DATA_DIR / "las_vegas_reviews.csv"
ENTITIES_MD = DATA_DIR / "las_vegas_strip_entities.md"

OUT_SUMMARY = DATA_DIR / "loyalty_sentiment_summary.csv"
OUT_MONTHLY = DATA_DIR / "loyalty_sentiment_monthly.csv"
OUT_CHART = DATA_DIR / "loyalty_sentiment_trend.png"

# ── Task 1: Expanded loyalty keyword list (case-insensitive) ──
LOYALTY_KEYWORDS = [
    "rewards", "loyalty", "points", "tier", "status",
    "diamond", "platinum", "caesars rewards", "mgm rewards",
    "mlife", "m life", "total rewards", "card", "comp",
    "comped", "redeem", "redemption", "earned", "benefit",
    "free play", "freeplay", "upgrade", "elite",
]

# Compile regex — match multi-word phrases first for accuracy
LOYALTY_PATTERN = re.compile(
    r"(?i)\b(" + "|".join(
        re.escape(kw) for kw in sorted(LOYALTY_KEYWORDS, key=len, reverse=True)
    ) + r")\b"
)

# Minimum total reviews for a brand's analysis to be considered reliable
MIN_BRAND_REVIEWS = 200

# NLTK stop words (inline to avoid download dependency)
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    "also", "would", "could", "get", "got", "one", "two", "go", "went",
    "like", "really", "even", "still", "much", "back", "us", "well",
    "hotel", "room", "stay", "stayed", "vegas", "las", "strip", "place",
    "night", "time", "good", "great", "nice", "casino", "resort",
}


# ══════════════════════════════════════════════
# 1. Parse Parent Company from Entities Markdown
# ══════════════════════════════════════════════
def parse_parent_companies(filepath: Path) -> pd.DataFrame:
    rows = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.split("|")]
            if len(cols) < 5:
                continue
            name = cols[1]
            if name in ("", "Entity Name") or set(name) <= {"-", " "}:
                continue
            parent = cols[3]
            rows.append({"location_name": name, "parent_company": parent})
    df = pd.DataFrame(rows)
    print(f"[META] Parsed {len(df)} entities with parent company info")
    print(f"       MGM Resorts:          {(df['parent_company'] == 'MGM Resorts').sum()}")
    print(f"       Caesars Entertainment: {(df['parent_company'] == 'Caesars Entertainment').sum()}")
    return df


# ══════════════════════════════════════════════
# 2. Load and Prepare Review Data
# ══════════════════════════════════════════════
def load_reviews(filepath: Path, parent_map: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"\n[LOAD] Raw reviews: {df.shape[0]:,} rows")
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_date"]).copy()
    df["review_text"] = df["review_text"].fillna("")
    df = df.merge(parent_map, on="location_name", how="left")
    df["parent_company"] = df["parent_company"].fillna("Other")
    print(f"[PREP] Reviews with parent company: {df.shape[0]:,}")
    return df


# ══════════════════════════════════════════════
# 3. Filter Loyalty-Related Reviews (Task 1)
# ══════════════════════════════════════════════
def filter_loyalty_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task 1: Filter to reviews containing loyalty keywords.
    Print per-brand counts and flag brands with < MIN_BRAND_REVIEWS.
    """
    mask = df["review_text"].str.contains(LOYALTY_PATTERN, na=False)
    loyalty_df = df[mask].copy()

    loyalty_df["matched_keywords"] = loyalty_df["review_text"].apply(
        lambda text: ", ".join(sorted(set(m.lower() for m in LOYALTY_PATTERN.findall(text))))
    )

    loyalty_df["month"] = loyalty_df["review_date"].dt.to_period("M")

    print(f"\n[FILTER] Loyalty keyword matches: {len(loyalty_df):,} / {len(df):,} "
          f"({len(loyalty_df)/len(df)*100:.1f}%)")

    # Per-brand breakdown (Task 1)
    for company in ["MGM Resorts", "Caesars Entertainment", "Other"]:
        total = (df["parent_company"] == company).sum()
        filtered = (loyalty_df["parent_company"] == company).sum()
        pct = filtered / total * 100 if total > 0 else 0
        flag = ""
        if filtered < MIN_BRAND_REVIEWS:
            flag = f"  ⚠ BELOW {MIN_BRAND_REVIEWS} — sentiment analysis may be unreliable"
        print(f"    {company}: {filtered:,} / {total:,} reviews "
              f"({pct:.1f}% mention loyalty){flag}")

    # Top keywords
    all_kws = loyalty_df["matched_keywords"].str.split(", ").explode()
    print(f"\n  Top loyalty keywords:")
    for kw, cnt in all_kws.value_counts().head(10).items():
        print(f"    {kw:<20} {cnt:>5} mentions")

    return loyalty_df


# ══════════════════════════════════════════════
# 4. Sentiment Analysis — VADER + TextBlob (Task 3)
# ══════════════════════════════════════════════
def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each review using VADER (baseline) and TextBlob (cross-check).
    """
    print(f"\n[SENTIMENT] Scoring {len(df):,} loyalty reviews...")

    sia = SentimentIntensityAnalyzer()

    # VADER compound score
    df["vader_sentiment"] = df["review_text"].apply(
        lambda text: sia.polarity_scores(text)["compound"]
    )

    # TextBlob polarity score (Task 3)
    df["textblob_sentiment"] = df["review_text"].apply(
        lambda text: TextBlob(text).sentiment.polarity
    )

    # Classify VADER into categories
    df["sentiment_label"] = pd.cut(
        df["vader_sentiment"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    )

    print(f"  VADER  — mean={df['vader_sentiment'].mean():.4f}, "
          f"positive={((df['sentiment_label'] == 'positive').mean()*100):.1f}%")
    print(f"  TextBlob — mean={df['textblob_sentiment'].mean():.4f}")

    return df


# ══════════════════════════════════════════════
# 5. Summary Table (with cross-model check — Task 3)
# ══════════════════════════════════════════════
def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "─" * 65)
    print("TABLE 1: Loyalty Sentiment Summary by Company")
    print("─" * 65)

    summary = (
        df.groupby("parent_company")
        .agg(
            vader_avg=("vader_sentiment", "mean"),
            textblob_avg=("textblob_sentiment", "mean"),
            loyalty_mentions=("vader_sentiment", "count"),
            vader_median=("vader_sentiment", "median"),
            textblob_median=("textblob_sentiment", "median"),
            pct_positive=("sentiment_label", lambda x: (x == "positive").mean() * 100),
            pct_negative=("sentiment_label", lambda x: (x == "negative").mean() * 100),
        )
        .reset_index()
        .rename(columns={"parent_company": "Company"})
        .sort_values("loyalty_mentions", ascending=False)
    )

    print(summary.to_string(index=False, float_format="%.4f"))

    # Cross-model comparison: Caesars vs MGM (Task 3)
    mgm = df[df["parent_company"] == "MGM Resorts"]
    caesars = df[df["parent_company"] == "Caesars Entertainment"]

    cross_model_results = {}
    if len(mgm) > 1 and len(caesars) > 1:
        print(f"\n  ── Cross-Model Comparison: Caesars vs MGM ──")
        for model, col in [("VADER", "vader_sentiment"), ("TextBlob", "textblob_sentiment")]:
            mgm_mean = mgm[col].mean()
            cae_mean = caesars[col].mean()
            t_stat, p_val = stats.ttest_ind(
                caesars[col].values, mgm[col].values, equal_var=False
            )
            caesars_higher = cae_mean > mgm_mean
            sig = p_val < 0.05
            cross_model_results[model] = {
                "caesars_higher": caesars_higher,
                "significant": sig,
                "caesars_mean": cae_mean,
                "mgm_mean": mgm_mean,
                "p_value": p_val,
            }
            direction = "Caesars > MGM" if caesars_higher else "MGM > Caesars"
            print(f"    {model}: Caesars={cae_mean:.4f}, MGM={mgm_mean:.4f} → "
                  f"{direction} (p={p_val:.4f}, "
                  f"{'Sig ✓' if sig else 'Not sig ✗'})")

        # Task 3: Check whether finding holds under BOTH models
        vader_higher = cross_model_results["VADER"]["caesars_higher"]
        tb_higher = cross_model_results["TextBlob"]["caesars_higher"]
        if vader_higher == tb_higher:
            winner = "Caesars" if vader_higher else "MGM"
            print(f"\n    BOTH models agree: {winner} has higher loyalty sentiment.")
        else:
            print(f"\n    ⚠ Models DISAGREE on direction — treat with caution.")

    return summary, cross_model_results


# ══════════════════════════════════════════════
# 6. Monthly Trend (with n, confidence, TextBlob — Tasks 2–3)
# ══════════════════════════════════════════════
def compute_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "─" * 65)
    print("TABLE 2: Monthly Loyalty Sentiment Trend (with n + confidence)")
    print("─" * 65)

    # Monthly aggregation for both brands
    monthly = (
        df[df["parent_company"].isin(["MGM Resorts", "Caesars Entertainment"])]
        .groupby(["month", "parent_company"])
        .agg(
            vader_avg=("vader_sentiment", "mean"),
            textblob_avg=("textblob_sentiment", "mean"),
            n_reviews=("vader_sentiment", "count"),
        )
        .reset_index()
    )

    # Confidence flag (Task 2)
    def _conf(n):
        if n >= 30:
            return "high"
        elif n >= 10:
            return "medium"
        else:
            return "low - treat with caution"

    monthly["confidence_flag"] = monthly["n_reviews"].apply(_conf)

    # Pivot to wide format
    pivot_vader = monthly.pivot_table(index="month", columns="parent_company",
                                      values="vader_avg").reset_index()
    pivot_tb = monthly.pivot_table(index="month", columns="parent_company",
                                   values="textblob_avg").reset_index()
    pivot_n = monthly.pivot_table(index="month", columns="parent_company",
                                  values="n_reviews").reset_index()
    pivot_conf = monthly.pivot_table(index="month", columns="parent_company",
                                     values="confidence_flag",
                                     aggfunc="first").reset_index()

    # Build combined output
    result = pd.DataFrame({"Month": pivot_vader["month"].astype(str)})

    for company, short in [("Caesars Entertainment", "Caesars"),
                            ("MGM Resorts", "MGM")]:
        if company in pivot_vader.columns:
            result[f"{short} VADER"] = pivot_vader[company].values
        if company in pivot_tb.columns:
            result[f"{short} TextBlob"] = pivot_tb[company].values
        if company in pivot_n.columns:
            result[f"{short} n_reviews"] = pivot_n[company].values
        if company in pivot_conf.columns:
            result[f"{short} confidence"] = pivot_conf[company].values

    # Conflicting signal detection (Task 3)
    for short in ["Caesars", "MGM"]:
        v_col = f"{short} VADER"
        t_col = f"{short} TextBlob"
        if v_col in result.columns and t_col in result.columns:
            conflict = (
                ((result[v_col] > 0) & (result[t_col] < 0)) |
                ((result[v_col] < 0) & (result[t_col] > 0))
            )
            result[f"{short} signal"] = np.where(
                conflict, "conflicting", "aligned"
            )

    result = result.sort_values("Month").reset_index(drop=True)
    print(result.to_string(index=False, float_format="%.4f"))

    return result


# ══════════════════════════════════════════════
# 7. Weighted Momentum Analysis (Task 2)
# ══════════════════════════════════════════════
def momentum_analysis(monthly_df: pd.DataFrame) -> dict:
    """
    Weighted trend (polyfit with w=sqrt(n)), using VADER as primary.
    """
    print("\n" + "─" * 65)
    print("MOMENTUM ANALYSIS: Weighted Sentiment Trend Slopes")
    print("─" * 65)

    results = {}

    for short, label in [("MGM", "MGM Resorts"), ("Caesars", "Caesars Entertainment")]:
        v_col = f"{short} VADER"
        n_col = f"{short} n_reviews"
        tb_col = f"{short} TextBlob"

        if v_col not in monthly_df.columns:
            continue

        valid = monthly_df[[v_col, n_col, tb_col]].dropna()
        if len(valid) < 2:
            print(f"\n  {label}: Insufficient data for trend")
            continue

        x = np.arange(len(valid)).astype(float)
        y_vader = valid[v_col].values
        y_tb = valid[tb_col].values
        weights = np.sqrt(valid[n_col].values)  # sqrt(n) weights

        # Weighted OLS (Task 2)
        vader_coefs = np.polyfit(x, y_vader, deg=1, w=weights)
        tb_coefs = np.polyfit(x, y_tb, deg=1, w=weights)

        # Unweighted for comparison
        uw_slope, _, uw_r, uw_p, _ = stats.linregress(x, y_vader)

        results[label] = {
            "vader_slope_weighted": vader_coefs[0],
            "vader_slope_unweighted": uw_slope,
            "textblob_slope_weighted": tb_coefs[0],
            "n_months": len(valid),
            "avg_n": valid[n_col].mean(),
        }

        print(f"\n  {label} (n={len(valid)} months, avg {valid[n_col].mean():.0f} "
              f"reviews/mo):")
        print(f"    VADER weighted slope:   {vader_coefs[0]:+.4f}/mo")
        print(f"    VADER unweighted slope: {uw_slope:+.4f}/mo")
        print(f"    TextBlob weighted slope:{tb_coefs[0]:+.4f}/mo")

    return results


# ══════════════════════════════════════════════
# 8. Mar 2025 Caesars Crash Investigation (Task 4)
# ══════════════════════════════════════════════
def investigate_caesars_crash(df: pd.DataFrame) -> None:
    """
    Task 4: Isolate Caesars loyalty reviews Feb–May 2025 and diagnose
    the Mar 2025 sentiment crash.
    """
    print("\n" + "─" * 65)
    print("INVESTIGATION: Mar 2025 Caesars Sentiment Crash")
    print("─" * 65)

    caesars = df[df["parent_company"] == "Caesars Entertainment"].copy()
    caesars["month_str"] = caesars["month"].astype(str)

    window_months = ["2025-02", "2025-03", "2025-04", "2025-05"]
    window = caesars[caesars["month_str"].isin(window_months)].copy()

    print(f"\n  Caesars loyalty reviews in Feb–May 2025 window:")
    for m in window_months:
        subset = window[window["month_str"] == m]
        n = len(subset)
        if n == 0:
            print(f"    {m}: n=0 (no data)")
            continue
        avg_vader = subset["vader_sentiment"].mean()
        avg_tb = subset["textblob_sentiment"].mean()
        pct_neg = (subset["sentiment_label"] == "negative").mean() * 100
        print(f"    {m}: n={n}, VADER={avg_vader:.4f}, "
              f"TextBlob={avg_tb:.4f}, {pct_neg:.0f}% negative")

    # Focus on Mar 2025
    mar = window[window["month_str"] == "2025-03"]
    if mar.empty:
        print("\n  No March 2025 data found.")
        return

    n_mar = len(mar)
    avg_mar = mar["vader_sentiment"].mean()

    # Determine if low-n or genuine negative content
    print(f"\n  ── Diagnosis ──")
    if n_mar < 10:
        print(f"    The Mar 2025 crash (VADER={avg_mar:.4f}) is based on "
              f"only n={n_mar} reviews.")
        print(f"    VERDICT: LOW-N artifact — too few reviews for reliable score.")
    else:
        print(f"    The Mar 2025 crash (VADER={avg_mar:.4f}) is based on "
              f"n={n_mar} reviews.")
        print(f"    This is above the n=10 threshold, suggesting GENUINE "
              f"negative content.")

    # Most negative reviews (verbatim — Task 4)
    neg = mar.sort_values("vader_sentiment").head(5)
    print(f"\n  5 Most Negative Caesars Loyalty Reviews (Mar 2025):")
    for i, (_, row) in enumerate(neg.iterrows(), 1):
        text = row["review_text"][:300].replace("\n", " ").strip()
        print(f"    [{i}] VADER={row['vader_sentiment']:.3f} | "
              f"{row['location_name']}")
        print(f"        \"{text}{'...' if len(row['review_text']) > 300 else ''}\"")

    # Word frequency in negative reviews (simple theme extraction)
    neg_reviews = mar[mar["vader_sentiment"] < -0.05]
    if not neg_reviews.empty:
        words = []
        for text in neg_reviews["review_text"].values:
            for w in re.findall(r"[a-z]+", text.lower()):
                if w not in STOP_WORDS and len(w) > 2:
                    words.append(w)
        top_words = Counter(words).most_common(15)
        print(f"\n  Top themes in negative reviews (word frequency):")
        for word, cnt in top_words:
            print(f"    {word:<20} {cnt:>3} mentions")


# ══════════════════════════════════════════════
# 9. Visualization — Updated Chart (Task 5)
# ══════════════════════════════════════════════
def plot_sentiment_trend(monthly_df: pd.DataFrame, momentum: dict,
                         output_path: Path) -> None:
    """
    Task 5: Updated chart with:
      a) Volume panel showing n_reviews per brand
      b) Greyed-out low-confidence points
      c) VADER solid + TextBlob dashed lines
      d) Mar 2025 Caesars dip annotation
    """
    print("\n" + "─" * 65)
    print("VISUALIZATION: Updated Loyalty Sentiment Trend")
    print("─" * 65)

    monthly_df = monthly_df.copy()
    monthly_df["date"] = pd.to_datetime(monthly_df["Month"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1]})

    # ─── Top Panel: Sentiment lines ───
    ax = axes[0]

    mgm_color = "#FFB300"
    caesars_color = "#7B1FA2"

    for short, label, color, marker in [
        ("MGM", "MGM Resorts", mgm_color, "o"),
        ("Caesars", "Caesars Entertainment", caesars_color, "s"),
    ]:
        v_col = f"{short} VADER"
        tb_col = f"{short} TextBlob"
        conf_col = f"{short} confidence"

        if v_col not in monthly_df.columns:
            continue

        valid = monthly_df[monthly_df[v_col].notna()].copy()
        if valid.empty:
            continue

        # Separate high/medium vs low confidence points (Task 5b)
        if conf_col in valid.columns:
            hi = valid[valid[conf_col] != "low - treat with caution"]
            lo = valid[valid[conf_col] == "low - treat with caution"]
        else:
            hi = valid
            lo = pd.DataFrame()

        # VADER solid line (Task 5c)
        ax.plot(hi["date"], hi[v_col], color=color, linewidth=2.2,
                marker=marker, markersize=6, label=f"{label} (VADER)",
                alpha=0.85, zorder=3)
        if not lo.empty:
            ax.scatter(lo["date"], lo[v_col], color=color, marker=marker,
                       s=50, alpha=0.20, zorder=3, linewidths=1.5,
                       edgecolors=color, facecolors=color)

        # TextBlob dashed line (Task 5c)
        if tb_col in valid.columns:
            ax.plot(valid["date"], valid[tb_col], color=color, linewidth=1.5,
                    linestyle="--", marker=marker, markersize=3,
                    label=f"{label} (TextBlob)", alpha=0.45, zorder=2)

    # Annotate Mar 2025 Caesars dip (Task 5d)
    mar_row = monthly_df[monthly_df["Month"] == "2025-03"]
    if not mar_row.empty and "Caesars VADER" in mar_row.columns:
        mar_val = mar_row["Caesars VADER"].values[0]
        mar_n = mar_row["Caesars n_reviews"].values[0]
        if pd.notna(mar_val):
            mar_date = pd.Timestamp("2025-03-01")
            ax.annotate(
                f"Mar 2025 crash\n"
                f"VADER={mar_val:.3f}\n"
                f"n={int(mar_n)}",
                xy=(mar_date, mar_val),
                xytext=(mar_date + pd.Timedelta(days=60), mar_val - 0.25),
                fontsize=9, fontweight="bold", color=caesars_color,
                arrowprops=dict(arrowstyle="->", color=caesars_color, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor=caesars_color, alpha=0.9),
            )

    # Neutral line
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="-", alpha=0.5, zorder=1)

    # Shade sentiment zones
    ax.axhspan(0.05, 1.0, alpha=0.04, color="green", zorder=0)
    ax.axhspan(-1.0, -0.05, alpha=0.04, color="red", zorder=0)

    ax.set_title(
        "Las Vegas Strip — Loyalty Program Sentiment Trend\n"
        "MGM Rewards vs Caesars Rewards (VADER solid / TextBlob dashed)",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_ylabel("Sentiment Score\n(−1 = Negative, +1 = Positive)", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ─── Bottom Panel: Volume bars (Task 5a) ───
    ax2 = axes[1]
    bar_width = 12  # days

    for short, label, color in [
        ("MGM", "MGM", mgm_color),
        ("Caesars", "Caesars", caesars_color),
    ]:
        n_col = f"{short} n_reviews"
        if n_col not in monthly_df.columns:
            continue
        valid = monthly_df[monthly_df[n_col].notna()]
        offset = -bar_width/2 if short == "MGM" else bar_width/2
        dates = valid["date"] + pd.Timedelta(days=offset)
        ax2.bar(dates, valid[n_col], width=bar_width, color=color,
                alpha=0.6, label=f"{label} n", zorder=2)

    # Low confidence threshold line
    ax2.axhline(10, color="red", linewidth=1, linestyle=":", alpha=0.7)
    ax2.text(monthly_df["date"].iloc[0], 10, " n=10 (low conf.)",
             fontsize=8, color="red", va="bottom")
    ax2.axhline(30, color="orange", linewidth=1, linestyle=":", alpha=0.5)
    ax2.text(monthly_df["date"].iloc[0], 30, " n=30 (high conf.)",
             fontsize=8, color="orange", va="bottom")

    ax2.set_title("Monthly Review Volume (Loyalty Keyword Reviews)",
                   fontsize=11, fontweight="bold")
    ax2.set_ylabel("n reviews", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.set_axisbelow(True)

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(h_pad=3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  → Chart saved to {output_path.name}")


# ══════════════════════════════════════════════
# 10. Executive Summary
# ══════════════════════════════════════════════
def print_executive_summary(monthly_df: pd.DataFrame,
                            cross_model: dict,
                            momentum: dict) -> None:
    print("\n" + "=" * 65)
    print("EXECUTIVE SUMMARY")
    print("=" * 65)

    # 1. Caesars vs MGM — which models agree?
    if cross_model:
        vader_res = cross_model.get("VADER", {})
        tb_res = cross_model.get("TextBlob", {})
        v_higher = vader_res.get("caesars_higher", None)
        t_higher = tb_res.get("caesars_higher", None)
        v_sig = vader_res.get("significant", False)
        t_sig = tb_res.get("significant", False)

        if v_higher is not None and t_higher is not None:
            if v_higher == t_higher:
                winner = "Caesars" if v_higher else "MGM"
                sig_note = ""
                if v_sig and t_sig:
                    sig_note = " (statistically significant under both)."
                elif v_sig or t_sig:
                    sig_note = " (significant under one model only — moderate confidence)."
                else:
                    sig_note = " (not statistically significant under either — low confidence)."
                print(f"  1. {winner} loyalty sentiment is higher than "
                      f"{'MGM' if winner == 'Caesars' else 'Caesars'}'s under "
                      f"BOTH VADER and TextBlob{sig_note}")
            else:
                print(f"  1. Models DISAGREE: VADER says "
                      f"{'Caesars > MGM' if v_higher else 'MGM > Caesars'}, "
                      f"TextBlob says "
                      f"{'Caesars > MGM' if t_higher else 'MGM > Caesars'}. "
                      f"Treat any directional claim with caution.")
    else:
        print("  1. Insufficient data for cross-model comparison.")

    # 2. Trend reliability based on avg monthly n
    all_ns = []
    for short in ["MGM", "Caesars"]:
        n_col = f"{short} n_reviews"
        if n_col in monthly_df.columns:
            all_ns.extend(monthly_df[n_col].dropna().values)
    avg_n = np.mean(all_ns) if all_ns else 0
    if avg_n >= 30:
        reliability = "HIGH"
    elif avg_n >= 10:
        reliability = "MEDIUM"
    else:
        reliability = "LOW"
    print(f"  2. Trend reliability: {reliability} (average monthly n = "
          f"{avg_n:.0f} loyalty-keyword reviews per brand).")

    # 3. Months to disregard
    disregard = []
    for short in ["MGM", "Caesars"]:
        conf_col = f"{short} confidence"
        if conf_col in monthly_df.columns:
            low = monthly_df[monthly_df[conf_col] == "low - treat with caution"]
            for _, row in low.iterrows():
                disregard.append(f"{row['Month']} ({short}, n="
                                 f"{int(row[f'{short} n_reviews'])})")
    if disregard:
        print(f"  3. Disregard these months when citing to clients:")
        for item in disregard:
            print(f"     • {item}")
    else:
        print(f"  3. No months flagged as low confidence.")


# ══════════════════════════════════════════════
# 11. Main Pipeline
# ══════════════════════════════════════════════
def main():
    print("=" * 65)
    print("Loyalty Program Sentiment Analysis (v2)")
    print("Caesars Rewards vs MGM Rewards")
    print("=" * 65)

    # Step 1: Load parent company mapping
    parent_map = parse_parent_companies(ENTITIES_MD)

    # Step 2: Load and prepare reviews
    reviews = load_reviews(REVIEWS_CSV, parent_map)

    # Step 3: Filter loyalty-related reviews (Task 1)
    loyalty = filter_loyalty_reviews(reviews)

    if loyalty.empty:
        print("\n⚠ No loyalty-related reviews found — exiting")
        return

    # Step 4: Sentiment scoring — VADER + TextBlob (Task 3)
    loyalty = compute_sentiment(loyalty)

    # Step 5: Summary table (Task 3 cross-model check)
    summary, cross_model = compute_summary_table(loyalty)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"\n  → Saved to {OUT_SUMMARY.name}")

    # Step 6: Monthly trend (Tasks 2–3)
    monthly = compute_monthly_trend(loyalty)
    monthly.to_csv(OUT_MONTHLY, index=False)
    print(f"  → Saved to {OUT_MONTHLY.name}")

    # Step 7: Weighted momentum analysis (Task 2)
    momentum = momentum_analysis(monthly)

    # Step 8: Mar 2025 crash investigation (Task 4)
    investigate_caesars_crash(loyalty)

    # Step 9: Updated visualization (Task 5)
    plot_sentiment_trend(monthly, momentum, OUT_CHART)

    # Step 10: Executive summary
    print_executive_summary(monthly, cross_model, momentum)

    print("\n" + "=" * 65)
    print("Analysis Complete")
    print("=" * 65)
    print(f"  Output files:")
    print(f"    • {OUT_SUMMARY}")
    print(f"    • {OUT_MONTHLY}")
    print(f"    • {OUT_CHART}")


if __name__ == "__main__":
    main()
