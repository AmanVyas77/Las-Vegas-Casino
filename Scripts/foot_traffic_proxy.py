"""
Foot Traffic Proxy — Normalized Review-Based Traffic Index
==========================================================
Transforms ~50k raw Google Maps reviews into a monthly
"Reviews per 100 Rooms" metric, aggregated by location and
Strip region (North / South).

Inputs:
  Data/las_vegas_reviews.csv        — raw review data
  Data/las_vegas_strip_entities.md  — entity metadata (room counts, regions)

Outputs:
  Data/monthly_traffic_by_location.csv  — Dataset 1 (with data_quality_flag)
  Data/monthly_traffic_by_region.csv    — Dataset 2 (with review count cols)
  Data/foot_traffic_north_vs_south.png  — Visualization (with review bars)
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

REVIEWS_CSV = DATA_DIR / "las_vegas_reviews.csv"
ENTITIES_MD = DATA_DIR / "las_vegas_strip_entities.md"

OUT_LOCATION = DATA_DIR / "monthly_traffic_by_location.csv"
OUT_REGION = DATA_DIR / "monthly_traffic_by_region.csv"
OUT_CHART = DATA_DIR / "foot_traffic_north_vs_south.png"

# Threshold: a month is flagged as "suspected_hole" if its review count
# falls below this fraction of the 3-month trailing average.
DATA_HOLE_THRESHOLD = 0.30


# ──────────────────────────────────────────────
# Geographic Boundary Definitions (Task 2)
# ──────────────────────────────────────────────
# The Las Vegas Strip is divided at approximately Spring Mountain Rd /
# Sands Ave (~36.1265°N latitude, ~3400 block of Las Vegas Blvd S).
#
# North Strip: Sahara Ave (~36.147°N) south to Spring Mountain Rd (~36.126°N)
#   Roughly Las Vegas Blvd addresses ~2800 – ~3400
#   Lat/long bounding box: 36.126°N – 36.150°N, 115.180°W – 115.155°W
#
# South Strip: Spring Mountain Rd (~36.126°N) south to Russell Rd (~36.084°N)
#   Roughly Las Vegas Blvd addresses ~3400 – ~3950
#   Lat/long bounding box: 36.080°N – 36.126°N, 115.185°W – 115.155°W
#
# The dividing line (Spring Mountain Rd) was chosen because it aligns with
# the traditional industry boundary used by the LVCVA and places the
# Wynn/Encore cluster firmly in the North and the Bellagio/CityCenter
# cluster firmly in the South.
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

# Annotations for properties near the geographic boundary.
# These explain why each ambiguous property was placed into its bucket.
BOUNDARY_NOTES = {
    # ── North Strip (at or near boundary) ──
    "Caesars Palace":
        "North — main tower at ~36.117°N, but front entrance and forum shops "
        "extend south past Flamingo Rd. Placed North because the hotel "
        "registration, rooms, and casino floor are north of Flamingo Rd.",
    "The Cromwell":
        "North — sits on the NE corner of Flamingo Rd & Las Vegas Blvd "
        "(~36.116°N). Could be 'center strip'; classified North because it "
        "shares the same block as Flamingo Las Vegas and LINQ, both North.",
    "Flamingo Las Vegas":
        "North — on the NW corner of Flamingo Rd (~36.116°N). Traditionally "
        "grouped with Harrah's / LINQ in the central-north cluster.",
    # ── South Strip (at or near boundary) ──
    "The Cosmopolitan":
        "South — main entrance at ~36.110°N, directly south of the Bellagio "
        "fountains. Clearly south of Flamingo Rd.",
    "Planet Hollywood":
        "South — at ~36.109°N between Paris and the Cosmopolitan. Squarely "
        "in the Caesars Entertainment south cluster (with Paris & Bally's).",
    "Paris Las Vegas":
        "South — at ~36.112°N, directly across Las Vegas Blvd from Bellagio. "
        "Grouped with Bally's/Horseshoe which it physically connects to.",
    "Bally's Las Vegas (Horseshoe)":
        "South — at ~36.113°N, directly across the intersection from Caesars "
        "Palace (classified North). The two properties are across the street "
        "but on opposite sides of the Flamingo Rd dividing line.",
    "Park MGM":
        "South — at ~36.098°N, between ARIA and New York-New York. No "
        "geographic ambiguity.",
    "MGM Grand":
        "South — at ~36.102°N on Tropicana Ave. No geographic ambiguity.",
}


# ══════════════════════════════════════════════
# 1. Parse Entity Metadata from Markdown
# ══════════════════════════════════════════════
def parse_entity_metadata(filepath: Path) -> pd.DataFrame:
    """
    Parse the markdown table to extract entity name,
    room count, and strip region for each row.
    Returns a DataFrame with columns:
        location_name, room_count, strip_region
    """
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
            # Skip header and separator rows
            if name in ("", "Entity Name") or set(name) <= {"-", " "}:
                continue

            strip_region = cols[4]  # "North" or "South"
            room_raw = cols[5]      # e.g. "3,500" or "N/A"

            # Parse room count — set to NaN for non-hotel entities
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
    print(f"[META] Parsed {len(df)} entities from {filepath.name}")
    print(f"       {df['room_count'].notna().sum()} with room counts, "
          f"{df['room_count'].isna().sum()} without (restaurants/non-hotels)")

    # ── Task 2: Print full property lists so analyst can verify ──
    _print_boundary_audit(df)

    return df


def _print_boundary_audit(meta: pd.DataFrame) -> None:
    """Print the North/South property classification for analyst review."""
    hotels = meta[meta["room_count"].notna()].copy()
    print(f"\n[BOUNDARY AUDIT] Strip region classification "
          f"(boundary defined in STRIP_BOUNDARIES dict):")
    print(f"  North: {STRIP_BOUNDARIES['North']['description']} "
          f"| Lat {STRIP_BOUNDARIES['North']['lat_range']}")
    print(f"  South: {STRIP_BOUNDARIES['South']['description']} "
          f"| Lat {STRIP_BOUNDARIES['South']['lat_range']}")

    for region in ("North", "South"):
        subset = hotels[hotels["strip_region"] == region].sort_values("location_name")
        print(f"\n  ── {region.upper()} STRIP ({len(subset)} properties) ──")
        for _, row in subset.iterrows():
            name = row["location_name"]
            rooms = int(row["room_count"])
            note = BOUNDARY_NOTES.get(name, "")
            note_str = f"  ⚠ {note}" if note else ""
            print(f"    • {name} ({rooms:,} rooms){note_str}")


# ══════════════════════════════════════════════
# 2. Load and Clean Review Data
# ══════════════════════════════════════════════
def load_and_clean_reviews(filepath: Path) -> pd.DataFrame:
    """
    Load the raw reviews CSV.
    - Convert review_date to datetime.
    - Drop rows with invalid or missing dates.
    """
    df = pd.read_csv(filepath)
    print(f"\n[LOAD] Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Convert review_date to datetime, coerce errors to NaT
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    # Count and drop invalid / missing dates
    invalid_dates = df["review_date"].isna().sum()
    df = df.dropna(subset=["review_date"]).copy()
    print(f"[CLEAN] Dropped {invalid_dates:,} rows with invalid/missing dates")
    print(f"        Remaining: {df.shape[0]:,} rows")

    # Create a month-period column for grouping (e.g. 2024-01)
    df["month"] = df["review_date"].dt.to_period("M")

    return df


# ══════════════════════════════════════════════
# 3. Merge Reviews with Entity Metadata
# ══════════════════════════════════════════════
def merge_metadata(reviews: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join reviews with entity metadata to add
    room_count and strip_region columns.
    """
    merged = reviews.merge(meta, on="location_name", how="left")

    matched = merged["strip_region"].notna().sum()
    unmatched = merged["strip_region"].isna().sum()
    print(f"\n[MERGE] Matched: {matched:,} reviews | Unmatched: {unmatched:,}")

    if unmatched > 0:
        missing = merged.loc[merged["strip_region"].isna(), "location_name"].unique()
        print(f"        Unmatched locations: {list(missing)}")

    # Drop rows without metadata (shouldn't happen if lists align)
    merged = merged.dropna(subset=["strip_region"]).copy()
    return merged


# ══════════════════════════════════════════════
# 4. Monthly Aggregation by Location
#    (with artifact detection — Task 1)
# ══════════════════════════════════════════════
def compute_monthly_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset 1: Monthly review counts and normalized traffic index by location.
    Normalized Traffic Index = (Monthly Review Count / Room Count) * 100
    For entities without room counts (restaurants), the raw monthly
    review count is kept; the normalized index is set to NaN.

    After aggregation, each location-month is flagged via data_quality_flag:
      - "suspected_hole" if count < 30% of 3-month trailing average
      - "ok" otherwise
    Rows with < 3 months of preceding history get "ok" (insufficient data
    to compute a trailing average).
    """
    # Group by location and month
    grouped = (
        df.groupby(["location_name", "month", "strip_region", "room_count"])
        .agg(
            monthly_review_count=("review_date", "count"),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
    )

    # Compute normalized index (reviews per 100 rooms)
    grouped["normalized_traffic_index"] = grouped.apply(
        lambda row: (row["monthly_review_count"] / row["room_count"]) * 100
        if pd.notna(row["room_count"]) and row["room_count"] > 0
        else None,
        axis=1,
    )

    # Sort by location then month
    grouped = grouped.sort_values(["location_name", "month"]).reset_index(drop=True)

    # Convert period to string for CSV export
    grouped["month"] = grouped["month"].astype(str)

    # ── Task 1: Artifact detection — 3-month trailing average ──
    grouped["data_quality_flag"] = "ok"
    for name, idx in grouped.groupby("location_name").groups.items():
        sub = grouped.loc[idx].sort_values("month")
        counts = sub["monthly_review_count"].values
        months = sub["month"].values
        for pos in range(len(counts)):
            if pos < 3:
                continue  # not enough history
            trail_avg = np.mean(counts[pos - 3 : pos])
            if trail_avg > 0 and counts[pos] < DATA_HOLE_THRESHOLD * trail_avg:
                grouped.loc[sub.index[pos], "data_quality_flag"] = "suspected_hole"

    # ── Print flagging summary ──
    flagged = grouped[grouped["data_quality_flag"] == "suspected_hole"]
    n_flagged = len(flagged)
    n_total = len(grouped)
    print(f"\n[AGG-1] Monthly by location: {n_total:,} rows")
    print(f"        Locations: {grouped['location_name'].nunique()}")
    print(f"        Date range: {grouped['month'].min()} → {grouped['month'].max()}")

    print(f"\n[DATA QUALITY] {n_flagged} location-months flagged as "
          f"'suspected_hole' (count < {DATA_HOLE_THRESHOLD:.0%} of "
          f"3-mo trailing avg)")
    if n_flagged > 0:
        flag_counts = (
            flagged.groupby("location_name").size()
            .sort_values(ascending=False)
        )
        print(f"  Locations with most flags:")
        for loc, cnt in flag_counts.items():
            months_str = ", ".join(
                flagged.loc[flagged["location_name"] == loc, "month"].values
            )
            print(f"    • {loc}: {cnt} month(s) — [{months_str}]")

    return grouped


# ══════════════════════════════════════════════
# 5. Regional Aggregation (North vs. South)
#    (with review count transparency — Task 3)
# ══════════════════════════════════════════════
def compute_monthly_by_region(location_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset 2: Monthly aggregated normalized traffic index by strip region.
    Only includes hotel entities (those with room counts) for a fair
    per-room comparison.

    Output columns:
        Month | North Strip Index | South Strip Index
             | north_total_reviews | south_total_reviews
    """
    # Filter to entities that have room counts (hotels/casinos)
    hotels = location_df[location_df["normalized_traffic_index"].notna()].copy()

    # Group by region and month, take the mean of normalized indices
    regional = (
        hotels.groupby(["strip_region", "month"])
        .agg(
            mean_traffic_index=("normalized_traffic_index", "mean"),
            total_reviews=("monthly_review_count", "sum"),
            num_properties=("location_name", "nunique"),
        )
        .reset_index()
    )

    # Pivot traffic index
    pivot_index = regional.pivot_table(
        index="month",
        columns="strip_region",
        values="mean_traffic_index",
    ).reset_index()
    pivot_index.columns.name = None

    # Pivot total reviews (Task 3)
    pivot_reviews = regional.pivot_table(
        index="month",
        columns="strip_region",
        values="total_reviews",
    ).reset_index()
    pivot_reviews.columns.name = None

    # Merge the two pivots
    pivot = pivot_index.copy()
    col_map = {"month": "Month"}
    if "North" in pivot.columns:
        col_map["North"] = "North Strip Index"
    if "South" in pivot.columns:
        col_map["South"] = "South Strip Index"
    pivot = pivot.rename(columns=col_map)

    # Add review count columns (Task 3)
    if "North" in pivot_reviews.columns:
        pivot["north_total_reviews"] = pivot_reviews["North"].values
    else:
        pivot["north_total_reviews"] = np.nan
    if "South" in pivot_reviews.columns:
        pivot["south_total_reviews"] = pivot_reviews["South"].values
    else:
        pivot["south_total_reviews"] = np.nan

    # Sort by month
    pivot = pivot.sort_values("Month").reset_index(drop=True)

    print(f"\n[AGG-2] Monthly by region: {pivot.shape[0]} months")
    print(pivot.to_string(index=False))

    return pivot


# ══════════════════════════════════════════════
# 6. Visualization — North vs. South Strip
#    (with review-count bars & low-confidence
#     shading — Task 4)
# ══════════════════════════════════════════════
def plot_regional_traffic(region_df: pd.DataFrame, output_path: Path) -> None:
    """
    Creates a dual-line chart showing monthly normalized traffic
    index for North Strip vs. South Strip over time.

    Task 4 additions:
      - Secondary y-axis with grey bar chart of total monthly review count.
      - Red shaded regions over months where aggregate review count
        drops below 30% of the prior 3-month average.
    """
    # Convert month strings to datetime for plotting
    region_df = region_df.copy()
    region_df["date"] = pd.to_datetime(region_df["Month"])

    # Compute total reviews across both regions for the bar chart
    region_df["total_reviews"] = (
        region_df.get("north_total_reviews", pd.Series(0)).fillna(0)
        + region_df.get("south_total_reviews", pd.Series(0)).fillna(0)
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    # ── Secondary y-axis: review count bars (Task 4) ──
    ax2 = ax.twinx()
    bar_width = 20  # days
    ax2.bar(
        region_df["date"],
        region_df["total_reviews"],
        width=bar_width,
        color="#CCCCCC",
        alpha=0.35,
        label="Total Reviews",
        zorder=1,
    )
    ax2.set_ylabel("Total Monthly Reviews (all locations)", fontsize=10,
                    color="#888888")
    ax2.tick_params(axis="y", labelcolor="#888888")
    ax2.set_axisbelow(True)

    # ── Low-confidence shading (Task 4) ──
    # Flag months where aggregate review count < 30% of 3-mo trailing avg
    total_vals = region_df["total_reviews"].values
    dates = region_df["date"].values
    shaded_label_added = False
    for i in range(3, len(total_vals)):
        trail_avg = np.mean(total_vals[i - 3 : i])
        if trail_avg > 0 and total_vals[i] < DATA_HOLE_THRESHOLD * trail_avg:
            dt = pd.Timestamp(dates[i])
            # Shade approximately one month around the flagged point
            left = dt - pd.Timedelta(days=15)
            right = dt + pd.Timedelta(days=15)
            label = "Data Gap / Low Confidence" if not shaded_label_added else None
            ax.axvspan(left, right, alpha=0.15, color="red",
                       zorder=0, label=label)
            shaded_label_added = True

    # ── Primary y-axis: traffic index lines ──
    north_color = "#2196F3"  # blue
    south_color = "#FF5722"  # orange-red

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

    # Styling
    ax.set_title(
        "Las Vegas Strip — Monthly Foot Traffic Proxy\n(Reviews per 100 Rooms)",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Normalized Traffic Index\n(Reviews / 100 Rooms)", fontsize=12)

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=10, loc="upper left", framealpha=0.9)

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Annotate overall averages
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Chart saved to {output_path.name}")


# ══════════════════════════════════════════════
# 7. Main Pipeline
# ══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Foot Traffic Proxy — Analysis Pipeline")
    print("=" * 60)

    # Step 1: Parse entity metadata (room counts, regions)
    meta = parse_entity_metadata(ENTITIES_MD)

    # Step 2: Load and clean reviews
    reviews = load_and_clean_reviews(REVIEWS_CSV)

    # Step 3: Merge reviews with metadata
    merged = merge_metadata(reviews, meta)

    # Step 4: Dataset 1 — Monthly normalized index by location
    location_df = compute_monthly_by_location(merged)
    location_df.to_csv(OUT_LOCATION, index=False)
    print(f"[SAVE] Dataset 1 → {OUT_LOCATION.name}")

    # Step 5: Dataset 2 — Monthly aggregated index by region
    region_df = compute_monthly_by_region(location_df)
    region_df.to_csv(OUT_REGION, index=False)
    print(f"[SAVE] Dataset 2 → {OUT_REGION.name}")

    # Step 6: Visualization
    plot_regional_traffic(region_df, OUT_CHART)

    # Summary stats
    print("\n" + "=" * 60)
    print("Pipeline Complete — Summary")
    print("=" * 60)
    print(f"  Total reviews processed: {len(merged):,}")
    print(f"  Unique locations:        {merged['location_name'].nunique()}")
    print(f"  Date range:              {merged['review_date'].min().date()} → "
          f"{merged['review_date'].max().date()}")
    print(f"  Output files:")
    print(f"    • {OUT_LOCATION}")
    print(f"    • {OUT_REGION}")
    print(f"    • {OUT_CHART}")


if __name__ == "__main__":
    main()
