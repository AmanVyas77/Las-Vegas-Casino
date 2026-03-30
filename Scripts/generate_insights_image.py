import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
OUT_IMAGE = DATA_DIR / "loyalty_insights_dashboard.png"

def generate_insights_image():
    # Hardcoding the extracted insights for the dashboard
    brands = ["Caesars Entertainment", "MGM Resorts"]
    
    # 1. Avg Sentiment Score (VADER)
    avg_vader = [0.3316, 0.2493]
    
    # 2. Trend Slopes (/month)
    trend_slopes = [-0.0046, 0.0026]
    
    # 3. Avg Monthly Reviews
    avg_n = [34.3, 50.0]

    colors_caesars = "#7B1FA2"
    colors_mgm = "#FFB300"
    
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Create grid
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])
    
    # --- PANEL 1: Avg Sentiment ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(["Caesars", "MGM"], avg_vader, color=[colors_caesars, colors_mgm], alpha=0.9, edgecolor='black', zorder=3)
    ax1.set_title("Avg Sentiment Score (VADER)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 0.4)
    ax1.grid(True, axis='y', alpha=0.3, linestyle="--", zorder=0)
    for i, v in enumerate(avg_vader):
        ax1.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight="bold", fontsize=11)
        
    # --- PANEL 2: Trend Slope ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(["Caesars", "MGM"], trend_slopes, color=[colors_caesars, colors_mgm], alpha=0.9, edgecolor='black', zorder=3)
    ax2.set_title("Trend Momentum (Slope / Month)", fontsize=12, fontweight="bold")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.grid(True, axis='y', alpha=0.3, linestyle="--", zorder=0)
    for i, v in enumerate(trend_slopes):
        offset = 0.0005 if v > 0 else -0.0005
        va = 'bottom' if v > 0 else 'top'
        ax2.text(i, v + offset, f"{v:+.4f}", ha='center', fontweight="bold", fontsize=11)
        
    # --- PANEL 3: Avg Monthly Volume ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(["Caesars", "MGM"], avg_n, color=[colors_caesars, colors_mgm], alpha=0.9, edgecolor='black', zorder=3)
    ax3.set_title("Avg Monthly Review Volume", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 60)
    ax3.grid(True, axis='y', alpha=0.3, linestyle="--", zorder=0)
    for i, v in enumerate(avg_n):
        ax3.text(i, v + 1.5, f"{v:.1f}", ha='center', fontweight="bold", fontsize=11)
        
    # --- PANEL 4: Executive Summary Text ---
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    summary_text = (
        "════════════════════════════════════════════════════════════════════════════════════\n"
        "LOYALTY SENTIMENT — EXECUTIVE SUMMARY\n"
        "════════════════════════════════════════════════════════════════════════════════════\n\n"
        "KEY FINDING: Does data support Caesars Rewards sentiment > MGM Rewards?\n"
        "Answer: INCONCLUSIVE\n"
        "Reason: Caesars has a higher aggregate VADER score (0.33 vs 0.25), but is declining (-0.0046/mo).\n"
        "MGM has a lower score but positive momentum (+0.0026/mo). Furthermore, MGM Grand data\n"
        "is entirely missing from Google Maps, artificially understating MGM's true position.\n\n"
        "CROSS-CHECK: Does this hold under TextBlob analysis?\n"
        "Answer: INCONCLUSIVE — Both brands show declining slopes under TextBlob, but Caesars \n"
        "is declining faster (-0.0062/mo vs -0.0019/mo).\n\n"
        "CAVEATS:\n"
        "• MGM Grand missing from Google Maps data\n"
        "• 6 months flagged for conflicting signals between VADER and TextBlob\n"
    )
    
    ax4.text(0.05, 0.95, summary_text, fontsize=13, fontfamily="monospace", va="top", ha="left",
             bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor="gray", alpha=0.9))
    
    plt.suptitle("Las Vegas Strip — Loyalty Program Sentiment Insights", fontsize=18, fontweight="bold", y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    plt.savefig(OUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"Insights dashboard saved to {OUT_IMAGE}")

if __name__ == "__main__":
    generate_insights_image()
