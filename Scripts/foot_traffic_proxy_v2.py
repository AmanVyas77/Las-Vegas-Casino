"""
Foot Traffic Proxy — Normalized Review-Based Traffic Index v2
=============================================================
Transforms ~50k raw Google Maps reviews into a monthly
"Reviews per 100 Rooms" metric, aggregated by location and
Strip region (North / South).

Handles missing properties, approximate dates, low coverage, 
and partial months.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

REVIEWS_CSV = DATA_DIR / "las_vegas_reviews_v4.csv"
ENTITIES_MD = DATA_DIR / "las_vegas_strip_entities.md"

OUT_LOCATION = DATA_DIR / "monthly_traffic_by_location.csv"
OUT_REGION = DATA_DIR / "monthly_traffic_by_region.csv"
OUT_CHART = DATA_DIR / "foot_traffic_north_vs_south.png"

DATA_HOLE_THRESHOLD = 0.30

# ── ISSUE 1 ──
MISSING_PROPERTIES = ["MGM Grand", "New York-New York"]

# ── ISSUE 2 ──
EXCLUDED_LOW_COVERAGE = [
    "Aria Patisserie",
    "SARDINIA Restaurant",
    "Aria Resort & Casino Parking",
    "Lupo by Wolfgang Puck (Mandalay Bay)"
]

# ── ISSUE 4 ──
PARTIAL_MONTHS = ["2026-03"]

# ── ISSUE 3 ──
DATE_PRECISION_WARNING = """# WARNING: All dates parsed from relative strings. 
# Monthly counts may have ±2 week error for recent reviews 
# and ±6 month error for reviews older than 1 year.
# Treat monthly trends as directionally correct, 
# not precise.
"""

# Geographic Boundary Definitions
STRIP_BOUNDARIES = {
    "North": {
        "description": "Sahara Ave to Spring Mountain Rd",
        "lat_range": (36.126, 36.150),
        "address_range": "~2800–3400 Las Vegas Blvd S",
    },
    "South": {
        "description": "Spring Mountain Rd to Russell Rd / Mandalay Bay",
        "lat_range": (36.080, 36.126),
        "address_range": "~3400–3950 Las Vegas Blvd S",
    },
}

BOUNDARY_NOTES = {
    "Caesars Palace": "North",
    "The Cromwell": "North",
    "Flamingo Las Vegas": "North",
    "The Cosmopolitan": "South",
    "Planet Hollywood": "South",
    "Paris Las Vegas": "South",
    "Bally's Las Vegas (Horseshoe)": "South",
    "Park MGM": "South",
    "MGM Grand": "South",
}

def save_csv_with_warning(df: pd.DataFrame, filepath: Path) -> None:
    """Save CSV with the DATE_PRECISION_WARNING header."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(DATE_PRECISION_WARNING)
    df.to_csv(filepath, mode='a', index=False)

def parse_entity_metadata(filepath: Path) -> pd.DataFrame:
    rows = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.split("|")]
            if len(cols) < 6:
                continue
            name = cols[1]
            if name in ("", "Entity Name") or set(name) <= {"-", " "}:
                continue

            strip_region = cols[4]
            room_raw = cols[5]

            room_count = None
            if room_raw != "N/A":
                cleaned = room_raw.replace(",", "").strip()
                try:
                    room_count = int(cleaned)
                except ValueError:
                    room_count = None

            rows.append({
                "location_name": name,
                "room_count": room_count,
                "strip_region": strip_region,
            })

    df = pd.DataFrame(rows)
    return df

def load_and_clean_reviews(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, low_memory=False)
    
    # Pre-merge exclusions
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_date"]).copy()
    df["month"] = df["review_date"].dt.to_period("M").astype(str)
    
    # ── ISSUE 4: Exclude Partial Months ──
    for p_month in PARTIAL_MONTHS:
        excluded = len(df[df["month"] == p_month])
        if excluded > 0:
            print(f"[EXCLUDED] {p_month} flagged as partial month (scrape date: 2026-03-10). Excluded from all calculations to prevent artifact. Dropped {excluded} reviews.")
    df = df[~df["month"].isin(PARTIAL_MONTHS)]

    # ── ISSUE 2: Low Coverage Non-Brand ──
    for prop in EXCLUDED_LOW_COVERAGE:
        n_revs = len(df[df["location_name"] == prop])
        if n_revs > 0:
            print(f"[EXCLUDED] {prop} — {n_revs} reviews, below minimum threshold for reliable index calculation")
    df = df[~df["location_name"].isin(EXCLUDED_LOW_COVERAGE)]

    return df

def merge_metadata(reviews: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta_subset = meta[["location_name", "room_count"]]
    merged = reviews.merge(meta_subset, on="location_name", how="left")
    merged = merged.dropna(subset=["strip_region"]).copy()
    return merged

def compute_monthly_by_location(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["location_name", "month", "strip_region", "room_count"])
        .agg(
            monthly_review_count=("review_date", "count"),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
    )

    grouped["normalized_traffic_index"] = grouped.apply(
        lambda row: (row["monthly_review_count"] / row["room_count"]) * 100
        if pd.notna(row["room_count"]) and row["room_count"] > 0
        else None,
        axis=1,
    )

    grouped = grouped.sort_values(["location_name", "month"]).reset_index(drop=True)

    # ── ISSUE 3: Smoothed Traffic Index ──
    grouped["smoothed_traffic_index"] = (
        grouped.groupby("location_name")["normalized_traffic_index"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    grouped["data_quality_flag"] = "ok"
    for name, idx in grouped.groupby("location_name").groups.items():
        sub = grouped.loc[idx].sort_values("month")
        counts = sub["monthly_review_count"].values
        months = sub["month"].values
        for pos in range(len(counts)):
            if pos < 3:
                continue
            trail_avg = np.mean(counts[pos - 3 : pos])
            if trail_avg > 0 and counts[pos] < DATA_HOLE_THRESHOLD * trail_avg:
                grouped.loc[sub.index[pos], "data_quality_flag"] = "suspected_hole"

    return grouped

def compute_monthly_by_region(location_df: pd.DataFrame) -> pd.DataFrame:
    # Use smoothed traffic index as primary
    hotels = location_df[location_df["smoothed_traffic_index"].notna()].copy()

    regional = (
        hotels.groupby(["strip_region", "month"])
        .agg(
            mean_traffic_index=("smoothed_traffic_index", "mean"),
            total_reviews=("monthly_review_count", "sum"),
            num_properties=("location_name", "nunique"),
        )
        .reset_index()
    )

    pivot_index = regional.pivot_table(
        index="month",
        columns="strip_region",
        values="mean_traffic_index",
    ).reset_index()
    pivot_index.columns.name = None

    pivot_reviews = regional.pivot_table(
        index="month",
        columns="strip_region",
        values="total_reviews",
    ).reset_index()
    pivot_reviews.columns.name = None

    pivot = pivot_index.copy()
    col_map = {"month": "Month"}
    if "North" in pivot.columns:
        col_map["North"] = "North Strip Index"
    if "South" in pivot.columns:
        col_map["South"] = "South Strip Index"
    pivot = pivot.rename(columns=col_map)

    if "North" in pivot_reviews.columns:
        pivot["north_total_reviews"] = pivot_reviews["North"].values
    else:
        pivot["north_total_reviews"] = np.nan
    if "South" in pivot_reviews.columns:
        pivot["south_total_reviews"] = pivot_reviews["South"].values
    else:
        pivot["south_total_reviews"] = np.nan

    pivot = pivot.sort_values("Month").reset_index(drop=True)

    # ── ISSUE 1: Footnote Flag ──
    pivot["mgm_grand_missing"] = True

    return pivot

def plot_regional_traffic(region_df: pd.DataFrame, output_path: Path) -> None:
    region_df = region_df.copy()
    region_df["date"] = pd.to_datetime(region_df["Month"])

    region_df["total_reviews"] = (
        region_df.get("north_total_reviews", pd.Series(0)).fillna(0)
        + region_df.get("south_total_reviews", pd.Series(0)).fillna(0)
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    ax2 = ax.twinx()
    bar_width = 20
    ax2.bar(
        region_df["date"],
        region_df["total_reviews"],
        width=bar_width,
        color="#CCCCCC",
        alpha=0.35,
        label="Total Reviews",
        zorder=1,
    )
    ax2.set_ylabel("Total Monthly Reviews (all locations)", fontsize=10, color="#888888")
    ax2.tick_params(axis="y", labelcolor="#888888")
    ax2.set_axisbelow(True)

    north_color = "#2196F3"
    south_color = "#FF5722"

    if "North Strip Index" in region_df.columns:
        ax.plot(
            region_df["date"], region_df["North Strip Index"],
            color=north_color, linewidth=2.2, marker="o", markersize=4,
            label="North Strip", alpha=0.9, zorder=3,
        )
    if "South Strip Index" in region_df.columns:
        ax.plot(
            region_df["date"], region_df["South Strip Index"],
            color=south_color, linewidth=2.2, marker="s", markersize=4,
            label="South Strip", alpha=0.9, zorder=3,
        )

    ax.set_title(
        "Las Vegas Strip — Monthly Foot Traffic Proxy\n(Reviews per 100 Rooms, 3-Mo Smoothed)",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Smoothed Traffic Index\n(Reviews / 100 Rooms)", fontsize=12)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=10, loc="upper left", framealpha=0.9)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    for label, col, color in [
        ("North", "North Strip Index", north_color),
        ("South", "South Strip Index", south_color),
    ]:
        if col in region_df.columns:
            avg = region_df[col].mean()
            ax.axhline(avg, color=color, linestyle=":", alpha=0.5)
            ax.text(
                region_df["date"].iloc[-1], avg,
                f"  avg = {avg:.1f}", color=color, fontsize=9,
                va="bottom", fontweight="bold",
            )

    # ── Annotate Caveats ──
    fig.text(0.01, 0.01, "CAUTION: MGM Grand / NY-NY missing. South Strip understated.\nDates approx (±2-wk to ±6-mo error).", 
             fontsize=9, color="red", style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    print("=" * 60)
    print("Foot Traffic Proxy — Analysis Pipeline v2")
    print("=" * 60)
    
    # ── ISSUE 1: WARNING ──
    print(f"[WARNING] The following properties have no data and will\n"
          f"be excluded from all calculations: {', '.join(MISSING_PROPERTIES)}.\n"
          f"MGM Grand is a key anchor property — MGM South Strip index will be understated.\n"
          f"Flag this in all outputs.\n")

    meta = parse_entity_metadata(ENTITIES_MD)
    reviews = load_and_clean_reviews(REVIEWS_CSV)
    merged = merge_metadata(reviews, meta)

    location_df = compute_monthly_by_location(merged)
    save_csv_with_warning(location_df, OUT_LOCATION)

    region_df = compute_monthly_by_region(location_df)
    save_csv_with_warning(region_df, OUT_REGION)

    plot_regional_traffic(region_df, OUT_CHART)

    print("\n--- OUTPUT CHECK ---")
    print(f"Data/monthly_traffic_by_region.csv -> Top 5 rows:")
    print(region_df.head(5).to_string())
    print(f"\nData/monthly_traffic_by_region.csv -> Last 5 rows:")
    print(region_df.tail(5).to_string())
    
    north_avg = region_df["North Strip Index"].mean()
    south_avg = region_df["South Strip Index"].mean()
    print(f"\nNorth Strip average index: {north_avg:.2f}")
    print(f"South Strip average index: {south_avg:.2f}")

    earliest_val_month = region_df["Month"].min()
    
    # ── STEP 3: ANALYST CAVEAT SUMMARY ──
    print("\n" + "-"*60)
    print("DATA CAVEATS (as of 2026-03-10):")
    print("• Foot traffic index based on Google Maps review counts")
    print("  normalized by room count (reviews per 100 rooms/month).")
    print("• All review dates parsed from relative timestamps —")
    print("  monthly figures carry ±2 week precision for recent")
    print("  data and ±6 month precision for historical data.")
    print("  Trends should be interpreted directionally.")
    print("• MGM Grand and New York-New York excluded from MGM South")
    print("  Strip index due to missing scrape data. MGM South Strip")
    print("  figures are therefore understated — treat MGM index as")
    print("  a floor estimate.")
    print(f"• Analysis period: {earliest_val_month} to February 2026.")
    print("  March 2026 excluded as partial month.")
    print("• 4 non-brand low-coverage properties excluded from index.")
    print("-" * 60)


if __name__ == "__main__":
    main()
