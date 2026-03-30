"""
Build Las Vegas Strip Map
Generates an interactive HTML map from foot traffic data.
"""

import sys
import os
from pathlib import Path

# TASK 1 — INSTALL CHECK
try:
    import folium
    import branca
except ImportError:
    print("Run: pip install folium branca")
    sys.exit(1)

import pandas as pd
import numpy as np

def main():
    # Setup paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "Data"
    coords_csv = DATA_DIR / "entity_coordinates.csv"
    traffic_csv = DATA_DIR / "monthly_traffic_by_location.csv"
    output_html = DATA_DIR / "las_vegas_strip_map.html"

    # TASK 2 — PREPARE THE DATA
    df_coords = pd.read_csv(coords_csv)
    df_traffic = pd.read_csv(traffic_csv, comment='#')

    # Compute per-property summary
    def aggregate_traffic(group):
        total_reviews = group["monthly_review_count"].sum()
        avg_traffic_index = group["normalized_traffic_index"].mean()
        avg_rating = group["avg_rating"].mean()
        
        ok_data = group[group["data_quality_flag"] == "ok"]
        months_of_data = ok_data["month"].nunique()
        
        # highest index
        idx_max = group["normalized_traffic_index"].idxmax()
        peak_month = group.loc[idx_max, "month"] if pd.notna(idx_max) else None
        
        strip_region = group["strip_region"].iloc[0] if not group.empty else None
        
        return pd.Series({
            "total_reviews": total_reviews,
            "avg_traffic_index": avg_traffic_index,
            "avg_rating": round(avg_rating, 2) if pd.notna(avg_rating) else np.nan,
            "months_of_data": months_of_data,
            "peak_month": peak_month,
            "strip_region": strip_region
        })

    # Apply aggregations
    df_summary = df_traffic.groupby("location_name").apply(aggregate_traffic, include_groups=False).reset_index()

    # Merge coordinates with the summary
    merged = df_coords.merge(df_summary, on="location_name", how="left")

    # Drop any rows where latitude or longitude is null
    missing_coords = merged["latitude"].isna() | merged["longitude"].isna()
    dropped_props = merged[missing_coords]["location_name"].tolist()
    if dropped_props:
        print("WARNING: Dropping properties due to missing coordinates:")
        for p in dropped_props:
            print(f"  - {p}")
            
    merged = merged[~missing_coords].copy()

    # TASK 3 — BUILD THE MAP
    # Center the map on the Las Vegas Strip: [36.1147, -115.1728], zoom=14.
    m = folium.Map(location=[36.1147, -115.1728], zoom_start=14)

    # Calculate min/max traffic index for scaling
    min_idx = df_summary["avg_traffic_index"].min()
    max_idx = df_summary["avg_traffic_index"].max()

    def get_scaled_radius(val):
        """Radius scaled by avg_traffic_index (min=8, max=20)"""
        if pd.isna(val) or pd.isna(min_idx) or min_idx == max_idx:
            return 8
        scale = (val - min_idx) / (max_idx - min_idx)
        return max(8, min(8 + scale * 12, 20))

    # Add markers
    key_properties = {"Caesars Palace", "MGM Grand", "Bellagio", "Wynn Las Vegas"}
    no_data_props = []

    for _, row in merged.iterrows():
        name = row["location_name"]
        lat = row["latitude"]
        lon = row["longitude"]
        
        # Check if traffic data exists
        has_data = pd.notna(row.get("avg_traffic_index"))

        if has_data:
            region = row["strip_region"]
            # Color "#2196F3" (blue) for North Strip, "#FF5722" for South Strip
            color = "#2196F3" if region == "North" else "#FF5722"
            radius = get_scaled_radius(row["avg_traffic_index"])
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 220px;">
                <b style="font-size: 16px;">{name}</b><br>
                <b>Strip Region:</b> {region}<br>
                <b>Avg "Traffic" Index:</b> {row["avg_traffic_index"]:.2f}<br>
                <b>Total Reviews:</b> {int(row["total_reviews"])}<br>
                <b>Avg Rating:</b> {row["avg_rating"]:.2f} &#9733;<br>
                <b>Peak Month:</b> {row["peak_month"]}<br>
                <b>Months of Clean Data:</b> {int(row["months_of_data"])}
            </div>
            """
        else:
            # TASK 6 — HANDLE MISSING TRAFFIC DATA
            no_data_props.append(name)
            color = "#9E9E9E"  # grey circle
            radius = 6         # fixed radius
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <b style="font-size: 16px;">{name}</b><br>
                No traffic data available
            </div>
            """
            
        # Add the CircleMarker
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=name,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

        # TASK 5 — HIGHLIGHT KEY PROPERTIES
        if name in key_properties:
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="darkblue", icon="star", prefix="fa")
            ).add_to(m)

    # Print warnings for properties missing traffic data
    if no_data_props:
        print("\nWARNING: No traffic data available for these properties (plotted as grey pins):")
        for p in no_data_props:
            print(f"  - {p}")

    # TASK 4 — ADD A LEGEND AND TITLE
    legend_html = '''
    <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 220px; height: 130px; 
     background-color: rgba(255, 255, 255, 0.9); z-index:9999; font-size:14px;
     border:2px solid grey; border-radius: 5px; padding: 10px; font-family: Arial, sans-serif;">
     <b style="font-size: 16px;">Legend</b><br>
     <div style="margin-top: 5px;">
       <i style="background:#2196F3; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> North Strip<br>
       <i style="background:#FF5722; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> South Strip<br>
       <i style="background:#9E9E9E; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> No Data<br>
     </div>
     <small style="display:block; margin-top: 8px; color: #555;">Circle size = relative traffic index</small>
    </div>
    '''
    m.get_root().html.add_child(branca.element.Element(legend_html))

    title_html = '''
    <div style="position: fixed; 
     top: 10px; left: 50vw; transform: translateX(-50%); width: 450px; height: auto; 
     background-color: rgba(255, 255, 255, 0.9); z-index:9999; font-size:16px; font-family: Arial, sans-serif;
     border:2px solid grey; border-radius: 5px; padding: 10px; text-align:center;
     box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
     <b style="font-size: 18px;">Las Vegas Strip — Foot Traffic Proxy Map</b><br>
     <small style="color: #444;">Source: Google Maps Reviews (Normalized by Room Count)</small>
    </div>
    '''
    m.get_root().html.add_child(branca.element.Element(title_html))

    # Save output
    m.save(output_html)
    
    # Final output summary
    file_size_kb = os.path.getsize(output_html) / 1024
    num_full_data = len(merged) - len(no_data_props)
    
    print("\n" + "="*50)
    print("MAPPING SUMMARY")
    print("="*50)
    print(f"Total properties plotted: {len(merged)}")
    print(f"Properties with full traffic data: {num_full_data}")
    print(f"Properties with no data (grey pins): {len(no_data_props)}")
    print(f"Properties dropped due to missing coordinates: {len(dropped_props)}")
    print(f"\nMap saved successfully to:\n  {output_html} ({file_size_kb:.1f} KB)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
