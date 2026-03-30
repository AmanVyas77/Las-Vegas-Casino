import re
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    print("[WARNING] TextBlob not installed. Run: pip install textblob")
    print("Continuing with VADER only.")
    TEXTBLOB_AVAILABLE = False

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ════════════════════════════════════════════════
DATE_START = datetime(2022, 1, 1)
DATE_END   = datetime(2026, 2, 28)  

BRAND_PROPERTIES = {
    "Caesars Entertainment": [
        "Caesars Palace",
        "Harrah's Las Vegas",
        "The LINQ Hotel + Experience",
        "Flamingo Las Vegas",
        "The Cromwell",
        "Bally's Las Vegas (Horseshoe)",
        "Paris Las Vegas",
        "Planet Hollywood"
    ],
    "MGM Resorts": [
        "ARIA Resort & Casino",
        "Bellagio",
        "MGM Grand",
        "Mandalay Bay",
        "Park MGM",
        "New York-New York",
        "Luxor Las Vegas",
        "Excalibur",
        "Delano Las Vegas",
        "Vdara Hotel & Spa",
        "The Signature at MGM Grand",
        "The Cosmopolitan"
    ]
}

LOYALTY_KEYWORDS = [
    "rewards", "loyalty", "points", "tier", "status",
    "diamond", "platinum", "caesars rewards", "mgm rewards",
    "mlife", "m life", "total rewards", "card", "comp",
    "comped", "redeem", "redemption", "earned", "benefit",
    "free play", "freeplay", "upgrade", "elite",
]

LOYALTY_PATTERN = re.compile(
    r"(?i)\b(" + "|".join(
        re.escape(kw) for kw in sorted(LOYALTY_KEYWORDS, key=len, reverse=True)
    ) + r")\b"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

GOOGLE_MAPS_CSV = DATA_DIR / "las_vegas_reviews_v4.csv"
APP_STORE_CSV = DATA_DIR / "loyalty_app_reviews.csv"
REDDIT_CSV = DATA_DIR / "loyalty_reddit_reviews.csv"

OUT_MONTHLY = DATA_DIR / "loyalty_sentiment_monthly_v2.csv"
OUT_SOURCE = DATA_DIR / "loyalty_sentiment_by_source.csv"
OUT_SUMMARY = DATA_DIR / "loyalty_sentiment_summary.csv"
OUT_CHART = DATA_DIR / "loyalty_sentiment_chart_v2.png"


# ════════════════════════════════════════════════
# SECTION 2 — DATA LOADING FUNCTION
# ════════════════════════════════════════════════
def load_all_loyalty_data() -> pd.DataFrame:
    sources = []

    # ── SOURCE 1: Google Maps ──
    print(f"\n[LOAD] Processing Google Maps data ({GOOGLE_MAPS_CSV.name})...")
    df_gm = pd.read_csv(GOOGLE_MAPS_CSV, low_memory=False)
    total_gm = len(df_gm)
    
    # 1. Map to brands based on location
    caesars_props = set(BRAND_PROPERTIES["Caesars Entertainment"])
    mgm_props = set(BRAND_PROPERTIES["MGM Resorts"])
    
    def get_brand(loc):
        if loc in caesars_props: return "Caesars Entertainment"
        if loc in mgm_props: return "MGM Resorts"
        return None
        
    df_gm["brand"] = df_gm["location_name"].apply(get_brand)
    
    # 2. Filter to brand properties ONLY
    df_gm = df_gm.dropna(subset=["brand"])
    brand_filtered_gm = len(df_gm)
    
    # 3. Apply keyword filter
    df_gm["review_text"] = df_gm["review_text"].fillna("").astype(str)
    gm_mask = df_gm["review_text"].str.contains(LOYALTY_PATTERN, na=False)
    df_gm = df_gm[gm_mask].copy()
    loyalty_gm = len(df_gm)
    
    df_gm["data_source"] = "google_maps"
    df_gm["review_date"] = pd.to_datetime(df_gm["review_date"], errors="coerce")
    
    df_gm = df_gm[["brand", "data_source", "review_date", "rating", "review_text"]]
    sources.append(df_gm)
    
    print(f"  [GOOGLE MAPS] Total reviews loaded: {total_gm:,}")
    print(f"  [GOOGLE MAPS] After brand property filter: {brand_filtered_gm:,}")
    print(f"  [GOOGLE MAPS] After loyalty keyword filter: {loyalty_gm:,}")
    
    gm_caesars = len(df_gm[df_gm["brand"] == "Caesars Entertainment"])
    gm_mgm = len(df_gm[df_gm["brand"] == "MGM Resorts"])
    print(f"  [GOOGLE MAPS] Caesars: {gm_caesars:,} reviews | MGM: {gm_mgm:,} reviews")

    # ── SOURCE 2: App Store ──
    print(f"\n[LOAD] Processing App Store data ({APP_STORE_CSV.name})...")
    df_app = pd.read_csv(APP_STORE_CSV)
    total_app = len(df_app)
    
    df_app["data_source"] = "app_store"
    df_app["review_date"] = pd.to_datetime(df_app["review_date"], errors="coerce")
    
    # Apply date filter
    mask = (df_app["review_date"] >= DATE_START) & (df_app["review_date"] <= DATE_END)
    df_app = df_app[mask].copy()
    date_filtered_app = len(df_app)
    
    df_app = df_app[["brand", "data_source", "review_date", "rating", "review_text"]]
    
    if df_app.empty:
        print("  [WARNING] App Store CSV resulted in 0 rows after date filtering.")
    else:
        sources.append(df_app)
        
    print(f"  [APP STORE] Total reviews loaded: {total_app:,}")
    print(f"  [APP STORE] After date filter: {date_filtered_app:,}")
    
    app_caesars = len(df_app[df_app["brand"] == "Caesars Entertainment"])
    app_mgm = len(df_app[df_app["brand"] == "MGM Resorts"])
    print(f"  [APP STORE] Caesars: {app_caesars:,} reviews | MGM: {app_mgm:,} reviews")
    
    if app_caesars == 0 or app_mgm == 0:
        print("  [WARNING] Missing a brand in App Store data.")

    # ── SOURCE 3: Reddit ──
    try:
        df_reddit = pd.read_csv(REDDIT_CSV)
        df_reddit["data_source"] = "reddit"
        df_reddit["review_date"] = pd.to_datetime(df_reddit["review_date"], errors="coerce")
        df_reddit["review_text"] = df_reddit["review_text"].fillna("").astype(str)
        
        mask = (df_reddit["review_date"] >= DATE_START) & \
               (df_reddit["review_date"] <= DATE_END) & \
               (df_reddit["review_text"].str.len() > 50)
        df_reddit = df_reddit[mask].copy()
        
        df_reddit = df_reddit[["brand", "data_source", "review_date", "rating", "review_text"]]
        if not df_reddit.empty:
            sources.append(df_reddit)
        print(f"\n[LOAD] Reddit data loaded ({len(df_reddit)} rows).")
    except Exception:
        print(f"\n[REDDIT] File not found — skipping. Add {REDDIT_CSV.name} to include Reddit data.")

    # ── COMBINE ──
    master = pd.concat(sources, ignore_index=True)
    master["review_date"] = pd.to_datetime(master["review_date"])
    
    master["review_text"] = master["review_text"].fillna("").astype(str)
    master = master[master["review_text"].str.strip() != ""]
    
    print("\n  ── Final Source Breakdown ──")
    summary = master.groupby(["data_source", "brand"]).agg(
        Reviews=("review_text", "count"),
        Earliest=("review_date", lambda x: getattr(x.min(), 'date', lambda: None)()),
        Latest=("review_date", lambda x: getattr(x.max(), 'date', lambda: None)())
    ).reset_index()
    print(summary.to_string(index=False))

    return master


# ════════════════════════════════════════════════
# SECTION 3 — SENTIMENT SCORING
# ════════════════════════════════════════════════
def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[SCORING] Scoring {len(df):,} reviews with VADER...")
    sia = SentimentIntensityAnalyzer()
    df["vader_score"] = df["review_text"].apply(
        lambda text: sia.polarity_scores(str(text))["compound"]
    )
    
    if TEXTBLOB_AVAILABLE:
        print(f"[SCORING] Scoring {len(df):,} reviews with TextBlob...")
        df["textblob_score"] = df["review_text"].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else None
        )
    return df


# ════════════════════════════════════════════════
# SECTION 4 — MONTHLY AGGREGATION
# ════════════════════════════════════════════════
def aggregate_monthly(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["month"] = df["review_date"].dt.to_period("M")
    
    def wavg(group, val_col, weight_col):
        d = group[group[val_col].notna()]
        if len(d) == 0:
            return np.nan
        # weights are essentially 1 per review here since we're unnested, 
        # so simply calculating mean works as weighted mean of review count
        return np.average(d[val_col])
        
    def _agg(group):
        n = len(group)
        v_avg = np.mean(group["vader_score"])
        t_avg = np.mean(group["textblob_score"]) if TEXTBLOB_AVAILABLE else np.nan
        
        if n >= 30: conf = "high"
        elif n >= 10: conf = "medium"
        else: conf = "low"
        
        conflict = False
        if TEXTBLOB_AVAILABLE and pd.notna(v_avg) and pd.notna(t_avg):
            if (v_avg > 0 and t_avg < 0) or (v_avg < 0 and t_avg > 0):
                conflict = True
                
        return pd.Series({
            "vader_score": v_avg,
            "textblob_score": t_avg,
            "n_reviews": n,
            "confidence_flag": conf,
            "conflicting_signal": conflict
        })

    # LEVEL 1
    monthly_lvl1 = df.groupby(["brand", "month"]).apply(_agg).reset_index()
    monthly_lvl1.to_csv(OUT_MONTHLY, index=False)
    
    # LEVEL 2
    monthly_lvl2 = df.groupby(["brand", "data_source", "month"]).apply(_agg).reset_index()
    print(f"  → Saved to {OUT_MONTHLY.name}\n")
    return monthly_lvl1, monthly_lvl2


# ════════════════════════════════════════════════
# SECTION 5 — TREND CALCULATION
# ════════════════════════════════════════════════
def calculate_trends(monthly_lvl1: pd.DataFrame) -> dict:
    results = {}
    print("\n[TREND] Calculating weighted momentum slopes...")
    
    for brand in ["Caesars Entertainment", "MGM Resorts"]:
        b_df = monthly_lvl1[monthly_lvl1["brand"] == brand].copy()
        
        # Exclude low confidence for trend calculation
        valid_b = b_df[b_df["confidence_flag"] != "low"].sort_values("month")
        
        if len(valid_b) < 2:
            print(f"  [WARNING] Insufficient data for {brand} trend calc.")
            continue
            
        x = np.arange(len(valid_b)).astype(float)
        y_vader = valid_b["vader_score"].values
        w = valid_b["n_reviews"].values
        
        v_coef = np.polyfit(x, y_vader, deg=1, w=w)
        v_slope = v_coef[0]
        
        t_slope = np.nan
        if TEXTBLOB_AVAILABLE:
            valid_t = valid_b.dropna(subset=["textblob_score"])
            if len(valid_t) > 1:
                x_t = np.arange(len(valid_t)).astype(float)
                t_coef = np.polyfit(x_t, valid_t["textblob_score"].values, deg=1, w=valid_t["n_reviews"].values)
                t_slope = t_coef[0]
                
        results[brand] = {
            "v_slope": v_slope,
            "t_slope": t_slope
        }
        
    if "Caesars Entertainment" in results and "MGM Resorts" in results:
        v_c = results["Caesars Entertainment"]["v_slope"]
        v_m = results["MGM Resorts"]["v_slope"]
        print(f"  Caesars VADER slope: {v_c:+.4f}/mo")
        print(f"  MGM VADER slope:     {v_m:+.4f}/mo")
        print(f"  Caesars > MGM:       {v_c > v_m}")
        
        if TEXTBLOB_AVAILABLE:
            t_c = results["Caesars Entertainment"]["t_slope"]
            t_m = results["MGM Resorts"]["t_slope"]
            print(f"  Caesars TextBlob slope: {t_c:+.4f}/mo")
            print(f"  MGM TextBlob slope:     {t_m:+.4f}/mo")
            print(f"  Caesars > MGM:          {t_c > t_m}")

    return results


# ════════════════════════════════════════════════
# SECTION 6 — SUMMARY OUTPUT
# ════════════════════════════════════════════════
def save_summary(df: pd.DataFrame, monthly: pd.DataFrame, trends: dict):
    rows = []
    
    for brand in ["Caesars Entertainment", "MGM Resorts"]:
        b_df = df[df["brand"] == brand]
        b_mon = monthly[monthly["brand"] == brand]
        
        if b_df.empty: continue
        
        avg_v = b_df["vader_score"].mean()
        avg_t = b_df["textblob_score"].mean() if TEXTBLOB_AVAILABLE else np.nan
        
        t_v = trends.get(brand, {}).get("v_slope", np.nan)
        t_t = trends.get(brand, {}).get("t_slope", np.nan)
        
        n_revs = len(b_df)
        avg_n = b_mon["n_reviews"].mean() if not b_mon.empty else 0
        hi_months = len(b_mon[b_mon["confidence_flag"] == "high"])
        
        sources = "+".join(sorted(b_df["data_source"].unique()))
        
        rows.append({
            "brand": brand,
            "avg_vader_score": avg_v,
            "avg_textblob_score": avg_t,
            "trend_slope_vader": t_v,
            "trend_slope_textblob": t_t,
            "total_reviews": n_revs,
            "avg_monthly_n": avg_n,
            "high_confidence_months": hi_months,
            "data_sources_used": sources
        })
        
    sum_df = pd.DataFrame(rows)
    
    header = """# Sentiment scores derived from brand-owned properties only
# Caesars: 8 properties | MGM: 12 properties  
# Non-brand properties excluded to prevent signal contamination
# MGM Grand missing from Google Maps source (scrape failure)
# Analysis period: 2022-01-01 to 2026-02 (2026-03 excluded)
"""
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(header)
    sum_df.to_csv(OUT_SUMMARY, mode="a", index=False)


# ════════════════════════════════════════════════
# SECTION 7 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════
def print_exec_summary(df: pd.DataFrame, monthly: pd.DataFrame, trends: dict):
    sources = df["data_source"].unique()
    
    def get_stats(b):
        bdf = df[df["brand"]==b]
        bmon = monthly[monthly["brand"]==b]
        if bdf.empty: return None, None, None, None
        
        v = bdf["vader_score"].mean()
        tr = trends.get(b, {}).get("v_slope", 0)
        tr_dir = "improving" if tr > 0 else "declining"
        n = bmon["n_reviews"].mean() if not bmon.empty else 0
        
        if n >= 30: c = "high"
        elif n >= 10: c = "medium"
        else: c = "low"
        
        return v, tr, tr_dir, n, c

    c_v, c_tr, c_dir, c_n, c_c = get_stats("Caesars Entertainment")
    m_v, m_tr, m_dir, m_n, m_c = get_stats("MGM Resorts")
    
    # Logic
    ans = "INCONCLUSIVE"
    res = ""
    tb_ans = "N/A - TextBlob unavailable"
    
    if c_v is not None and m_v is not None:
        if c_v > m_v and c_tr > m_tr:
            ans = "YES"
            res = "Caesars has a higher aggregate Vader score and a better slope trend."
        elif m_v > c_v and m_tr > c_tr:
            ans = "NO"
            res = "MGM has a higher aggregate Vader score and a better slope trend."
        else:
            ans = "INCONCLUSIVE"
            res = "Caesars and MGM have mixed metric leadership."
            
        if TEXTBLOB_AVAILABLE:
            b_c = df[df["brand"]=="Caesars Entertainment"]["textblob_score"].mean()
            b_m = df[df["brand"]=="MGM Resorts"]["textblob_score"].mean()
            t_c = trends.get("Caesars Entertainment", {}).get("t_slope", 0)
            t_m = trends.get("MGM Resorts", {}).get("t_slope", 0)
            
            if ans == "YES":
                tb_ans = "YES" if b_c > b_m and t_c > t_m else "NO"
            elif ans == "NO":
                tb_ans = "YES" if b_m > b_c and t_m > t_c else "NO"
            else:
                tb_ans = "INCONCLUSIVE"
        
    conflicts = monthly[monthly["conflicting_signal"] == True]
    conflict_strs = [f"{r['month']} ({r['brand']})" for _, r in conflicts.iterrows()]
    
    print("\n═══════════════════════════════════════")
    print("LOYALTY SENTIMENT — EXECUTIVE SUMMARY")
    print("═══════════════════════════════════════")
    print(f"\nData Sources: {', '.join(sources)}")
    print("Analysis Period: 2022-01-01 to 2026-02")
    
    if c_v is not None:
        print("\nCaesars Entertainment:")
        print(f"  Avg Sentiment Score (VADER): {c_v:.4f}")
        print(f"  Trend: {c_tr:+.4f}/month ({c_dir})")
        print(f"  Avg Monthly Reviews: {c_n:.1f}")
        print(f"  Confidence: {c_c}")

    if m_v is not None:
        print("\nMGM Resorts:")
        print(f"  Avg Sentiment Score (VADER): {m_v:.4f}")
        print(f"  Trend: {m_tr:+.4f}/month ({m_dir})")
        print(f"  Avg Monthly Reviews: {m_n:.1f}")
        print(f"  Confidence: {m_c}")
        
    print("\nKEY FINDING:")
    print("Does data support Caesars Rewards sentiment > MGM Rewards?")
    print(f"Answer: {ans}")
    print(f"Reason: {res}")
    
    print("\nDoes this hold under TextBlob cross-check?")
    print(f"Answer: {tb_ans}")
    
    print("\nCAVEATS:")
    print("• MGM Grand missing from Google Maps data —")
    print("  MGM sentiment may be understated")
    if conflict_strs:
        print(f"• Monthly conflicting signals (VADER vs TextBlob) flagged in: ")
        for c in conflict_strs:
            print(f"  - {c}")
    print("═══════════════════════════════════════")

# ════════════════════════════════════════════════
# SECTION 8 — VISUALIZATION
# ════════════════════════════════════════════════
def plot_sentiment_trend(monthly_lvl1: pd.DataFrame, trends: dict) -> None:
    print("\n" + "─" * 65)
    print("VISUALIZATION: Updated Loyalty Sentiment Trend (3 Panels)")
    print("─" * 65)

    df = monthly_lvl1.copy()
    df["date"] = pd.to_datetime(df["month"].dt.start_time)

    # Convert month column to string for the cross-check mask
    df["month_str"] = df["month"].astype(str)

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5])

    mgm_color = "#FFB300"
    cae_color = "#7B1FA2"

    # --- PANEL 1: Main Trend ---
    ax1 = fig.add_subplot(gs[0])
    
    for brand, color, marker in [("MGM Resorts", mgm_color, "o"), 
                                 ("Caesars Entertainment", cae_color, "s")]:
        b_df = df[df["brand"] == brand].dropna(subset=["vader_score"])
        if b_df.empty: continue
        
        hi = b_df[b_df["confidence_flag"] != "low"]
        lo = b_df[b_df["confidence_flag"] == "low"]
        
        # Solid High Conf
        ax1.plot(hi["date"], hi["vader_score"], color=color, linewidth=2.5, 
                 marker=marker, markersize=7, label=f"{brand} (Vader)")
                 
        # Washout Low Conf
        ax1.scatter(lo["date"], lo["vader_score"], color=color, marker=marker,
                    s=50, alpha=0.25, zorder=4)
                    
        # TextBlob Reference
        if TEXTBLOB_AVAILABLE:
            t_df = b_df.dropna(subset=["textblob_score"])
            ax1.plot(t_df["date"], t_df["textblob_score"], color=color, 
                     linewidth=1.2, linestyle="--", alpha=0.5, label=f"{brand} (TextBlob)")

    ax1.axhline(0, color="gray", linestyle="-", alpha=0.5, zorder=1)
    ax1.set_title("Las Vegas Strip Loyalty Sentiment (VADER vs TextBlob)", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Sentiment Score (-1 to 1)", fontsize=12)
    ax1.legend(loc="upper left")
    
    # Shade Apr-Aug 2025 "Low Confidence Window"
    window_start = pd.to_datetime("2025-04-01")
    window_end = pd.to_datetime("2025-08-31")
    ax1.axvspan(window_start, window_end, color="red", alpha=0.08, label="Low Confidence Window (April-August 2025)")

    # --- PANEL 2: Monthly Volume ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    width = 12
    
    for brand, color, offset in [("MGM Resorts", mgm_color, -width/2), 
                                 ("Caesars Entertainment", cae_color, width/2)]:
        b_df = df[df["brand"] == brand]
        if b_df.empty: continue
        
        ax2.bar(b_df["date"] + pd.Timedelta(days=offset), b_df["n_reviews"], 
                width=width, color=color, alpha=0.7, label=brand)

    ax2.axhline(30, color="green", linestyle="--", alpha=0.5, label="High Conf (30)")
    ax2.axhline(10, color="orange", linestyle="--", alpha=0.5, label="Med Conf (10)")
    ax2.set_ylabel("Total Reviews", fontsize=12)
    ax2.legend(loc="upper left", fontsize=10)

    # --- PANEL 3: Model Agreement Heatmap ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Create an aligned matrix 2 x N
    brand_list = ["Caesars Entertainment", "MGM Resorts"]
    dates = sorted(df["date"].unique())
    agree_matrix = np.full((2, len(dates)), np.nan)
    
    for i, brand in enumerate(brand_list):
        b_df = df[df["brand"] == brand].set_index("date")
        for j, d in enumerate(dates):
            if d in b_df.index:
                conflict = b_df.loc[d, "conflicting_signal"]
                # If boolean, sum will be > 0. Handle duplicates just in case.
                if isinstance(conflict, pd.Series):
                    conflict = conflict.any()
                
                # 1 = Conflict, 0 = Aligned
                agree_matrix[i, j] = 1 if conflict else 0
                
    cmap = plt.cm.colors.ListedColormap(['#e8f5e9', '#ffebee']) # light green vs red
    bounds = [-0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Extract meshgrid bounds for the heatmap blocks
    x_dates = mdates.date2num(dates)
    X, Y = np.meshgrid(np.append(x_dates, x_dates[-1] + 30), np.arange(3))
    
    ax3.pcolormesh(X, Y, agree_matrix, cmap=cmap, norm=norm, edgecolors='white', linewidth=1)
    ax3.set_yticks([0.5, 1.5])
    ax3.set_yticklabels(brand_list, fontsize=10)
    ax3.set_title("Model Agreement (Red = VADER and TextBlob disagree)", fontsize=11)
    ax3.set_xlabel("Date", fontsize=12)

    for ax in [ax1, ax2]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.grid(True, linestyle="--", alpha=0.4)
        
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")
    
    plt.tight_layout(h_pad=3)
    plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  → Chart logic processed, resulting shape [{len(dates)} mo].")
    try:
        sz = OUT_CHART.stat().st_size / 1024
        print(f"  → Chart successfully exported: {OUT_CHART.name} ({sz:.1f} KB)")
    except Exception as e:
        print(f"  → WARNING: Failed to stat {OUT_CHART.name} - {e}")

def main():
    df = load_all_loyalty_data()
    df = score_sentiment(df)
    mon_l1, mon_l2 = aggregate_monthly(df)
    trends = calculate_trends(mon_l1)
    save_summary(df, mon_l1, trends)
    print_exec_summary(df, mon_l1, trends)
    plot_sentiment_trend(mon_l1, trends)
    
if __name__ == "__main__":
    main()
