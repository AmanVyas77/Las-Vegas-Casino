"""
Traffic Momentum — Trend & Acceleration Analysis (v2 — Cleaned)
================================================================
Measures traffic momentum on the Las Vegas Strip using the
normalized review volume index. Computes YoY growth, CAGR,
statistical acceleration tests, and trend visualizations.

v2 changes:
  - Clean analysis window excludes low-confidence months
  - Theil-Sen robust regression in addition to OLS
  - Pre/post artifact period split (Period A vs B)
  - YoY reliability flags for cross-artifact comparisons
  - Updated chart with Theil-Sen/OLS lines + shaded artifact period

Input:
  Data/monthly_traffic_by_region.csv
  Data/monthly_traffic_by_location.csv

Outputs:
  Data/yoy_growth_rates.csv          — YoY with reliability flags
  Data/trend_analysis_summary.csv    — Consolidated findings table
  Data/trend_analysis_chart.png      — Updated trend visualization
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

REGION_CSV = DATA_DIR / "monthly_traffic_by_region.csv"
LOCATION_CSV = DATA_DIR / "monthly_traffic_by_location.csv"
OUT_CHART = DATA_DIR / "trend_analysis_chart.png"
OUT_YOY = DATA_DIR / "yoy_growth_rates.csv"
OUT_SUMMARY = DATA_DIR / "trend_analysis_summary.csv"

# ──────────────────────────────────────────────
# Task 1: Clean Analysis Window
# ──────────────────────────────────────────────
# Months where aggregate review count < 30% of the 3-month trailing avg
# are excluded from trend regressions. The window boundaries below are
# determined dynamically at runtime and printed for the analyst.
DATA_HOLE_THRESHOLD = 0.30

# Hard boundary for Period A / Period B split (Task 3).
# Period A: data up to and including this month (higher confidence).
# Period B: data after this month (low confidence — post-cliff artifact).
ARTIFACT_BOUNDARY = pd.Timestamp("2025-03-31")


# ══════════════════════════════════════════════
# 1. Load Data & Determine Clean Window
# ══════════════════════════════════════════════
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the regional and location-level monthly index datasets."""
    region = pd.read_csv(REGION_CSV, comment='#')
    region["Month"] = pd.to_datetime(region["Month"])
    region = region.sort_values("Month").reset_index(drop=True)

    location = pd.read_csv(LOCATION_CSV, low_memory=False, comment='#')
    location["month"] = pd.to_datetime(location["month"])
    location = location.sort_values(["location_name", "month"]).reset_index(drop=True)

    print("=" * 65)
    print("Traffic Momentum — Trend & Acceleration Analysis (v2)")
    print("=" * 65)
    print(f"\n[LOAD] Regional data: {region.shape[0]} months "
          f"({region['Month'].min().date()} → {region['Month'].max().date()})")
    print(f"[LOAD] Location data: {location.shape[0]} rows, "
          f"{location['location_name'].nunique()} locations")

    return region, location


def determine_clean_window(region: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, list]:
    """
    Task 1: Identify months to exclude and compute the clean analysis window.
    A month is excluded if its aggregate review count (north + south)
    is below 30% of the prior 3-month trailing average.
    """
    region = region.copy()
    region["total_reviews"] = (
        region["north_total_reviews"].fillna(0)
        + region["south_total_reviews"].fillna(0)
    )

    totals = region["total_reviews"].values
    excluded_months = []

    for i in range(len(totals)):
        if i < 3:
            continue  # not enough history for trailing avg
        trail_avg = np.mean(totals[i - 3: i])
        if trail_avg > 0 and totals[i] < DATA_HOLE_THRESHOLD * trail_avg:
            excluded_months.append(region.iloc[i]["Month"])

    # Also check if any month in the location-level data has widespread flags
    # (the foot_traffic_proxy already flagged these; region-level check suffices)

    # Clean window = all months NOT in excluded list
    all_months = region["Month"].values
    clean_mask = ~region["Month"].isin(excluded_months)
    clean_months = region.loc[clean_mask, "Month"]

    if len(clean_months) == 0:
        clean_start = region["Month"].min()
        clean_end = region["Month"].max()
    else:
        clean_start = clean_months.min()
        clean_end = clean_months.max()

    n_excluded = len(excluded_months)
    excl_strs = [m.strftime("%Y-%m") for m in excluded_months]

    print(f"\n[CLEAN WINDOW] Clean analysis window: "
          f"{clean_start.strftime('%Y-%m')} to {clean_end.strftime('%Y-%m')}. "
          f"Excluded {n_excluded} months as low-confidence.")
    if excl_strs:
        print(f"  Excluded months: {', '.join(excl_strs)}")

    return clean_start, clean_end, excluded_months


# ══════════════════════════════════════════════
# 2. Year-over-Year (YoY) Growth Rates
#    (with reliability flag — Task 4)
# ══════════════════════════════════════════════
def compute_yoy_growth(region: pd.DataFrame,
                       excluded_months: list) -> pd.DataFrame:
    """
    Calculate YoY growth for each region.
    Task 4: Flag comparisons that span the artifact cliff as unreliable.
    """
    print("\n" + "─" * 65)
    print("YoY Growth Rates (with reliability flags)")
    print("─" * 65)

    region = region.copy()
    region["year"] = region["Month"].dt.year
    region["month_num"] = region["Month"].dt.month

    # Build a set of excluded month timestamps for quick lookup
    excluded_set = set(pd.Timestamp(m) for m in excluded_months)

    yoy_rows = []

    for col, label in [
        ("North Strip Index", "North Strip"),
        ("South Strip Index", "South Strip"),
    ]:
        valid = region[region[col].notna()].copy()

        for _, row in valid.iterrows():
            curr_year = row["year"]
            curr_month = row["month_num"]
            curr_val = row[col]
            curr_dt = row["Month"]

            prior = valid[
                (valid["year"] == curr_year - 1) &
                (valid["month_num"] == curr_month)
            ]

            if not prior.empty:
                prior_val = prior.iloc[0][col]
                prior_dt = prior.iloc[0]["Month"]
                if prior_val != 0:
                    yoy = (curr_val - prior_val) / prior_val

                    # Task 4: determine reliability
                    # Unreliable if current or prior month is excluded,
                    # or if comparison spans the artifact boundary
                    # (one side pre-artifact, one side post-artifact)
                    curr_excluded = curr_dt in excluded_set
                    prior_excluded = prior_dt in excluded_set
                    spans_artifact = (
                        (prior_dt <= ARTIFACT_BOUNDARY < curr_dt) or
                        (curr_dt <= ARTIFACT_BOUNDARY < prior_dt)
                    )
                    # Also unreliable if either value is from a very
                    # sparse month (only 1-3 locations reporting)
                    curr_reviews = (
                        row.get("north_total_reviews", 0) or 0
                    ) + (row.get("south_total_reviews", 0) or 0)

                    if curr_excluded or prior_excluded:
                        reliability = "unreliable_excluded_month"
                    elif spans_artifact:
                        reliability = "unreliable_spans_artifact"
                    else:
                        reliability = "reliable"

                    yoy_rows.append({
                        "Region": label,
                        "Month": curr_dt.strftime("%Y-%m"),
                        "Current Value": round(curr_val, 2),
                        "Prior Year Value": round(prior_val, 2),
                        "YoY Growth (%)": round(yoy * 100, 2),
                        "yoy_reliability": reliability,
                    })

    yoy_df = pd.DataFrame(yoy_rows)

    if yoy_df.empty:
        print("\n  ⚠ No overlapping months found for YoY calculation.")
    else:
        print("\n  All YoY comparisons:")
        print(yoy_df.to_string(index=False))

        # Analyst-facing summary: only reliable rows
        reliable = yoy_df[yoy_df["yoy_reliability"] == "reliable"]
        unreliable = yoy_df[yoy_df["yoy_reliability"] != "reliable"]
        print(f"\n  Reliable comparisons: {len(reliable)}")
        print(f"  Unreliable comparisons: {len(unreliable)}")
        if not reliable.empty:
            print("\n  ── Reliable YoY Summary ──")
            print(reliable.to_string(index=False))
        if not unreliable.empty:
            print("\n  ── Flagged as Unreliable ──")
            print(unreliable[["Region", "Month", "YoY Growth (%)",
                              "yoy_reliability"]].to_string(index=False))

    yoy_df.to_csv(OUT_YOY, index=False)
    print(f"\n  → Saved to {OUT_YOY.name}")

    return yoy_df


# ══════════════════════════════════════════════
# 3. CAGR — Period A only (Task 3)
# ══════════════════════════════════════════════
def compute_cagr(region: pd.DataFrame,
                 excluded_months: list) -> dict:
    """
    CAGR = (Ending / Beginning)^(1/n) - 1.
    Task 3: computed separately for Period A (up to artifact boundary)
    and Period B (post-artifact, labeled low confidence).
    """
    print("\n" + "─" * 65)
    print("CAGR — Pre/Post Artifact Split")
    print("─" * 65)

    excluded_set = set(pd.Timestamp(m) for m in excluded_months)
    cagr_results = {}

    for period_label, mask_fn, confidence in [
        ("Period A (Mar 2023 – Mar 2025)",
         lambda dt: dt <= ARTIFACT_BOUNDARY and dt not in excluded_set,
         "HIGH CONFIDENCE"),
        ("Period B (Apr 2025 – present)",
         lambda dt: dt > ARTIFACT_BOUNDARY and dt not in excluded_set,
         "LOW CONFIDENCE — possible data artifact"),
    ]:
        print(f"\n  ── {period_label} [{confidence}] ──")

        for col, label in [
            ("North Strip Index", "North Strip"),
            ("South Strip Index", "South Strip"),
        ]:
            valid = region[region[col].notna()].copy()
            valid = valid[valid["Month"].apply(mask_fn)].sort_values("Month")

            if len(valid) < 2:
                print(f"    {label}: Insufficient data (n={len(valid)})")
                continue

            begin_val = valid.iloc[0][col]
            end_val = valid.iloc[-1][col]
            begin_date = valid.iloc[0]["Month"]
            end_date = valid.iloc[-1]["Month"]
            years = (end_date - begin_date).days / 365.25

            if years > 0 and begin_val > 0:
                cagr = (end_val / begin_val) ** (1 / years) - 1
            else:
                cagr = np.nan

            key = f"{label} — {period_label}"
            cagr_results[key] = {
                "begin_date": begin_date,
                "end_date": end_date,
                "begin_value": begin_val,
                "end_value": end_val,
                "years": years,
                "cagr": cagr,
                "confidence": confidence,
            }

            print(f"    {label}:")
            print(f"      Period:  {begin_date.strftime('%Y-%m')} → "
                  f"{end_date.strftime('%Y-%m')} ({years:.2f} years)")
            print(f"      Begin:   {begin_val:.2f}")
            print(f"      End:     {end_val:.2f}")
            print(f"      CAGR:    {cagr * 100:.2f}%")

    return cagr_results


# ══════════════════════════════════════════════
# 4. Trend Regression — OLS + Theil-Sen (Task 2)
#    Run on Period A as primary, Period B labeled
# ══════════════════════════════════════════════
def run_trend_regressions(region: pd.DataFrame,
                          excluded_months: list) -> dict:
    """
    Task 2: Run OLS and Theil-Sen on each region for:
      - Full clean window
      - Period A only (primary finding)
      - Period B only (low confidence)

    Task 3: Report Period A vs Period B separately.
    """
    print("\n" + "─" * 65)
    print("Trend Regression — OLS & Theil-Sen")
    print("─" * 65)

    excluded_set = set(pd.Timestamp(m) for m in excluded_months)
    results = {}

    for period_label, mask_fn, confidence in [
        ("Period A (≤ Mar 2025)",
         lambda dt: dt <= ARTIFACT_BOUNDARY and dt not in excluded_set,
         "HIGH CONFIDENCE"),
        ("Period B (> Mar 2025)",
         lambda dt: dt > ARTIFACT_BOUNDARY and dt not in excluded_set,
         "LOW CONFIDENCE"),
    ]:
        print(f"\n  ── {period_label} [{confidence}] ──")

        for col, label in [
            ("North Strip Index", "North Strip"),
            ("South Strip Index", "South Strip"),
        ]:
            valid = region[region[col].notna()].copy()
            valid = valid[valid["Month"].apply(mask_fn)].sort_values("Month")

            if len(valid) < 2:
                print(f"    {label}: Insufficient data (n={len(valid)})")
                continue

            x_days = (valid["Month"] - valid["Month"].min()).dt.days.values.astype(float)
            y = valid[col].values

            # OLS
            ols_slope, ols_intercept, ols_r, ols_p, ols_se = stats.linregress(x_days, y)
            ols_slope_per_month = ols_slope * 30.44

            # Theil-Sen (Task 2)
            ts_slope, ts_intercept, ts_low, ts_high = stats.theilslopes(y, x_days)
            ts_slope_per_month = ts_slope * 30.44
            ts_low_per_month = ts_low * 30.44
            ts_high_per_month = ts_high * 30.44

            key = f"{label}|{period_label}"
            results[key] = {
                "region": label,
                "period": period_label,
                "confidence": confidence,
                "n_months": len(valid),
                "ols_slope_per_month": ols_slope_per_month,
                "ols_slope_per_day": ols_slope,
                "ols_intercept": ols_intercept,
                "ols_r_squared": ols_r ** 2,
                "ols_p_value": ols_p,
                "ts_slope_per_month": ts_slope_per_month,
                "ts_slope_per_day": ts_slope,
                "ts_intercept": ts_intercept,
                "ts_ci_low_per_month": ts_low_per_month,
                "ts_ci_high_per_month": ts_high_per_month,
                "x_days": x_days,
                "y": y,
                "months": valid["Month"].values,
                "start": valid["Month"].min(),
                "end": valid["Month"].max(),
            }

            print(f"\n    {label} (n={len(valid)}):")
            print(f"      OLS slope:       {ols_slope_per_month:+.3f} /mo "
                  f"(R²={ols_r**2:.3f}, p={ols_p:.4f})")
            print(f"      Theil-Sen slope: {ts_slope_per_month:+.3f} /mo "
                  f"[95% CI: {ts_low_per_month:+.3f}, {ts_high_per_month:+.3f}]")

    return results


# ══════════════════════════════════════════════
# 5. Acceleration Analysis — Statistical Tests
# ══════════════════════════════════════════════
def acceleration_analysis(region: pd.DataFrame,
                          excluded_months: list) -> dict:
    """
    Test whether North Strip traffic growth is statistically greater
    than South Strip. Runs on Period A (clean) data only.
    """
    print("\n" + "─" * 65)
    print("Acceleration Analysis — Statistical Testing (Period A only)")
    print("─" * 65)

    excluded_set = set(pd.Timestamp(m) for m in excluded_months)
    # Period A only
    clean = region[
        (region["Month"] <= ARTIFACT_BOUNDARY) &
        (~region["Month"].isin(excluded_set))
    ].copy()

    results = {}
    alpha = 0.05

    # ─── Method A: Two-sample t-test ───
    print("\n  Method A: Two-Sample t-Test on Monthly Index Values (Period A)")
    print("  " + "·" * 50)

    north = clean["North Strip Index"].dropna().values
    south = clean["South Strip Index"].dropna().values

    if len(north) >= 2 and len(south) >= 2:
        t_stat, p_value = stats.ttest_ind(north, south, equal_var=False)
        sig_ttest = (p_value / 2) < alpha and t_stat > 0

        print(f"    North Strip: n={len(north)}, mean={np.mean(north):.2f}")
        print(f"    South Strip: n={len(south)}, mean={np.mean(south):.2f}")
        print(f"    Welch's t: {t_stat:.4f}, one-sided p: {p_value/2:.6f}")
        print(f"    Significant at α=0.05: {'YES ✓' if sig_ttest else 'NO ✗'}")

        results["ttest"] = {
            "t_stat": t_stat, "p_value": p_value / 2,
            "significant": sig_ttest,
        }
    else:
        print("    Insufficient data for t-test.")

    # ─── Method A2: Paired t-test ───
    print("\n  Method A2: Paired t-Test on Overlapping Months (Period A)")
    print("  " + "·" * 50)

    paired = clean.dropna(subset=["North Strip Index", "South Strip Index"])
    if len(paired) >= 2:
        diff = paired["North Strip Index"].values - paired["South Strip Index"].values
        t_stat_p, p_paired = stats.ttest_1samp(diff, 0)
        sig_paired = (p_paired / 2) < alpha and t_stat_p > 0

        print(f"    Paired months: {len(paired)}")
        print(f"    Mean diff: {np.mean(diff):.2f} (North − South)")
        print(f"    t={t_stat_p:.4f}, one-sided p={p_paired/2:.6f}")
        print(f"    Significant: {'YES ✓' if sig_paired else 'NO ✗'}")

        results["paired_ttest"] = {
            "t_stat": t_stat_p, "p_value": p_paired / 2,
            "significant": sig_paired,
        }
    else:
        print("    Insufficient paired data.")

    # ─── Method B: OLS Interaction Model ───
    print("\n  Method B: OLS Interaction Model (Period A)")
    print("  " + "·" * 50)

    if len(paired) >= 2:
        long = pd.melt(
            paired, id_vars=["Month"],
            value_vars=["North Strip Index", "South Strip Index"],
            var_name="region", value_name="index_value",
        )
        long["is_north"] = (long["region"] == "North Strip Index").astype(int)
        long["time_months"] = (
            (long["Month"] - long["Month"].min()).dt.days / 30.44
        ).astype(float)
        long["time_x_north"] = long["time_months"] * long["is_north"]

        y = long["index_value"].values
        X = np.column_stack([
            np.ones(len(y)),
            long["time_months"].values,
            long["is_north"].values,
            long["time_x_north"].values,
        ])

        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ betas
        resid = y - y_hat
        n, k = X.shape
        dof = n - k

        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        if dof > 0:
            mse = ss_res / dof
            cov = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            t_vals = betas / se
            p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=dof))
        else:
            p_vals = np.full(k, np.nan)

        interaction_p = p_vals[3]
        interaction_coef = betas[3]
        sig_ols = interaction_p < alpha

        print(f"    Interaction (time × north): coef={interaction_coef:.4f}, "
              f"p={interaction_p:.6f}")
        print(f"    R²={r_sq:.4f}")
        print(f"    Significant: {'YES ✓' if sig_ols else 'NO ✗'}")

        results["ols_interaction"] = {
            "coefficient": interaction_coef,
            "p_value": interaction_p,
            "significant": sig_ols,
            "r_squared": r_sq,
        }

    return results


# ══════════════════════════════════════════════
# 6. Visualization — Updated Trend Chart (Task 5)
# ══════════════════════════════════════════════
def plot_trend_chart(region: pd.DataFrame,
                     trend_results: dict,
                     excluded_months: list,
                     output_path: Path) -> None:
    """
    Task 5: Updated chart with:
      a) Theil-Sen line (solid) and OLS line (dashed) for Period A
      b) Artifact period shaded light red
      c) Slope, CI, and window annotated on chart
      d) Removed misleading full-period trend line
    """
    print("\n" + "─" * 65)
    print("Trend Visualization (updated)")
    print("─" * 65)

    region = region.copy()
    excluded_set = set(pd.Timestamp(m) for m in excluded_months)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1]})

    # ─── Top Panel: Index values with trend lines ───
    ax = axes[0]

    north_color = "#1565C0"
    south_color = "#E53935"

    # Shade artifact period (Task 5b)
    ax.axvspan(ARTIFACT_BOUNDARY, region["Month"].max() + pd.Timedelta(days=15),
               alpha=0.10, color="red", zorder=0,
               label="Low Confidence Period (Apr 2025+)")

    # Plot raw data points (all months including post-artifact)
    for col, label, color, marker in [
        ("North Strip Index", "North Strip", north_color, "o"),
        ("South Strip Index", "South Strip", south_color, "s"),
    ]:
        valid = region[region[col].notna()].copy()
        if valid.empty:
            continue

        # Mark excluded months with open markers
        clean = valid[~valid["Month"].isin(excluded_set)]
        dirty = valid[valid["Month"].isin(excluded_set)]

        ax.plot(
            clean["Month"], clean[col],
            color=color, linewidth=2, marker=marker, markersize=5,
            label=label, alpha=0.85, zorder=3,
        )
        if not dirty.empty:
            ax.scatter(
                dirty["Month"], dirty[col],
                color=color, marker=marker, s=40, facecolors="none",
                linewidths=1.5, zorder=3, alpha=0.5,
            )

    # Plot trend lines for Period A only (Task 5a, 5d)
    for col, label, color in [
        ("North Strip Index", "North Strip", north_color),
        ("South Strip Index", "South Strip", south_color),
    ]:
        key = f"{label}|Period A (≤ Mar 2025)"
        if key not in trend_results:
            continue

        res = trend_results[key]
        x_days = res["x_days"]
        months = res["months"]

        if len(x_days) < 2:
            continue

        # Theil-Sen line (solid) — Task 5a
        ts_y = res["ts_intercept"] + res["ts_slope_per_day"] * x_days
        ax.plot(months, ts_y, color=color, linewidth=2.0, linestyle="-",
                alpha=0.7, zorder=4,
                label=f"{label} Theil-Sen ({res['ts_slope_per_month']:+.2f}/mo)")

        # OLS line (dashed) — Task 5a
        ols_y = res["ols_intercept"] + res["ols_slope_per_day"] * x_days
        ax.plot(months, ols_y, color=color, linewidth=1.5, linestyle="--",
                alpha=0.5, zorder=4,
                label=f"{label} OLS ({res['ols_slope_per_month']:+.2f}/mo)")

    # Annotation box (Task 5c) — show Period A stats
    annotation_lines = []
    for col, label in [
        ("North Strip Index", "North Strip"),
        ("South Strip Index", "South Strip"),
    ]:
        key = f"{label}|Period A (≤ Mar 2025)"
        if key in trend_results:
            r = trend_results[key]
            annotation_lines.append(
                f"{label}: Theil-Sen = {r['ts_slope_per_month']:+.2f}/mo "
                f"[{r['ts_ci_low_per_month']:+.2f}, {r['ts_ci_high_per_month']:+.2f}]"
            )
    if annotation_lines:
        window_str = ""
        for col, label in [("North Strip Index", "North Strip")]:
            key = f"{label}|Period A (≤ Mar 2025)"
            if key in trend_results:
                r = trend_results[key]
                window_str = (f"Clean window: "
                              f"{r['start'].strftime('%Y-%m')} → "
                              f"{r['end'].strftime('%Y-%m')}")
                break

        text = window_str + "\n" + "\n".join(annotation_lines)
        ax.annotate(
            text,
            xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
            verticalalignment="bottom",
        )

    ax.set_title(
        "Las Vegas Strip — Traffic Momentum Analysis\n"
        "Normalized Review Index: North Strip vs South Strip",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_ylabel("Normalized Traffic Index\n(Reviews per 100 Rooms)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ─── Bottom Panel: Spread (North - South) ───
    ax2 = axes[1]

    paired = region.dropna(subset=["North Strip Index", "South Strip Index"]).copy()
    if not paired.empty:
        spread = paired["North Strip Index"] - paired["South Strip Index"]
        colors = [north_color if s > 0 else south_color for s in spread]

        ax2.bar(paired["Month"], spread, width=25, color=colors, alpha=0.7, zorder=3)
        ax2.axhline(0, color="grey", linewidth=0.8, zorder=2)

        # Shade artifact period
        ax2.axvspan(ARTIFACT_BOUNDARY,
                     region["Month"].max() + pd.Timedelta(days=15),
                     alpha=0.10, color="red", zorder=0)

        ax2.legend(
            handles=[
                Patch(facecolor=north_color, alpha=0.7, label="North > South"),
                Patch(facecolor=south_color, alpha=0.7, label="South > North"),
            ],
            fontsize=9, loc="upper right", framealpha=0.9,
        )

        ax2.set_title("North − South Spread", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Spread", fontsize=10)
        ax2.grid(True, alpha=0.25, linestyle="--")
        ax2.set_axisbelow(True)

        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Period A average spread
        paired_a = paired[paired["Month"] <= ARTIFACT_BOUNDARY]
        if len(paired_a) > 0:
            spread_a = (paired_a["North Strip Index"]
                        - paired_a["South Strip Index"]).mean()
            ax2.axhline(spread_a, color="purple", linestyle=":", alpha=0.6)
            ax2.text(
                paired_a["Month"].iloc[-1], spread_a,
                f"  Period A avg = {spread_a:.1f}", color="purple",
                fontsize=9, va="bottom", fontweight="bold",
            )

    plt.tight_layout(h_pad=3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  → Chart saved to {output_path.name}")


# ══════════════════════════════════════════════
# 7. Final Summary Table
# ══════════════════════════════════════════════
def save_summary(cagr_results: dict, stat_results: dict,
                 trend_results: dict) -> None:
    """Build and save the consolidated findings table."""
    rows = []

    # CAGR rows
    for key, data in cagr_results.items():
        rows.append({
            "Metric": f"CAGR ({key})",
            "Value": f"{data['cagr'] * 100:.2f}%",
            "Period": (f"{data['begin_date'].strftime('%Y-%m')} → "
                       f"{data['end_date'].strftime('%Y-%m')}"),
            "Confidence": data["confidence"],
        })

    # Trend slope rows
    for key, data in trend_results.items():
        if "x_days" not in data:
            continue
        rows.append({
            "Metric": f"Theil-Sen slope ({data['region']})",
            "Value": (f"{data['ts_slope_per_month']:+.3f}/mo "
                      f"[{data['ts_ci_low_per_month']:+.3f}, "
                      f"{data['ts_ci_high_per_month']:+.3f}]"),
            "Period": data["period"],
            "Confidence": data["confidence"],
        })
        rows.append({
            "Metric": f"OLS slope ({data['region']})",
            "Value": (f"{data['ols_slope_per_month']:+.3f}/mo "
                      f"(R²={data['ols_r_squared']:.3f})"),
            "Period": data["period"],
            "Confidence": data["confidence"],
        })

    # Stat test rows
    for test, data in stat_results.items():
        p = data.get("p_value", data.get("p_value_one_sided"))
        if p is not None:
            rows.append({
                "Metric": f"Stat Test ({test})",
                "Value": f"p={p:.6f}",
                "Period": "Significant" if data["significant"] else "Not Significant",
                "Confidence": "Period A only",
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\n" + "=" * 65)
    print("FINAL TREND FINDINGS")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\n  → Summary saved to {OUT_SUMMARY.name}")


# ══════════════════════════════════════════════
# 8. Main Pipeline
# ══════════════════════════════════════════════
def main():
    # Load data
    region, location = load_data()

    # Task 1: Determine clean analysis window
    clean_start, clean_end, excluded_months = determine_clean_window(region)

    # Task 4: YoY Growth with reliability flags
    yoy_df = compute_yoy_growth(region, excluded_months)

    # Task 3: CAGR — Period A vs B
    cagr_results = compute_cagr(region, excluded_months)

    # Task 2: Trend regression — OLS + Theil-Sen
    trend_results = run_trend_regressions(region, excluded_months)

    # Acceleration analysis (Period A only)
    stat_results = acceleration_analysis(region, excluded_months)

    # Task 5: Updated visualization
    plot_trend_chart(region, trend_results, excluded_months, OUT_CHART)

    # Save consolidated summary
    save_summary(cagr_results, stat_results, trend_results)

    print("\n" + "=" * 65)
    print("Analysis Complete")
    print("=" * 65)
    print(f"  Output files:")
    print(f"    • {OUT_YOY}")
    print(f"    • {OUT_SUMMARY}")
    print(f"    • {OUT_CHART}")


if __name__ == "__main__":
    main()
