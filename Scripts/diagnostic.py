import pandas as pd

def run_diagnostics():
    filepath = "Data/las_vegas_reviews_v4.csv"
    
    print("=" * 60)
    print("DATA QUALITY SANITY CHECK")
    print("=" * 60)
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # DIAGNOSTIC 1 — BASIC SHAPE
    print("\n--- DIAGNOSTIC 1 — BASIC SHAPE ---")
    print(f"Total number of rows: {len(df)}")
    unique_locations = df['location_name'].nunique()
    print(f"Total number of unique location_name values: {unique_locations}")
    
    if unique_locations < 60:
        print(f"FLAG: Unique locations ({unique_locations}) is significantly lower than expected (67).")
    else:
        print(f"Unique locations looks close to 67.")
        
    print(f"Columns present: {', '.join(df.columns)}")

    # DIAGNOSTIC 2 — DATE RANGE
    print("\n--- DIAGNOSTIC 2 — DATE RANGE ---")
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    earliest_date = df['review_date'].min()
    latest_date = df['review_date'].max()
    
    print(f"Earliest review_date: {earliest_date.strftime('%Y-%m-%d') if pd.notnull(earliest_date) else 'N/A'}")
    print(f"Latest review_date: {latest_date.strftime('%Y-%m-%d') if pd.notnull(latest_date) else 'N/A'}")
    
    threshold_start = pd.Timestamp('2022-06-01')
    if earliest_date > threshold_start:
        print(f"FLAG: Earliest date ({earliest_date.strftime('%Y-%m-%d')}) is NOT on or before 2022-06-01.")
    else:
        print("Earliest date check passed (on or before 2022-06-01).")
        
    threshold_end = pd.Timestamp('2026-03-01')
    if latest_date < threshold_end:
        print(f"FLAG: Latest date ({latest_date.strftime('%Y-%m-%d')}) is NOT 2026-03 or later.")
    else:
        print("Latest date check passed (2026-03 or later).")

    # DIAGNOSTIC 3 — REVIEWS PER PROPERTY
    print("\n--- DIAGNOSTIC 3 — REVIEWS PER PROPERTY ---")
    
    prop_stats = df.groupby('location_name').agg(
        review_count=('rating', 'count'),
        earliest_date=('review_date', 'min'),
        latest_date=('review_date', 'max')
    ).reset_index().sort_values('review_count')
    
    print(f"{'location_name':<40} | {'review_count':<12} | {'earliest_date':<13} | {'latest_date':<13} | {'flags'}")
    print("-" * 100)
    for _, row in prop_stats.iterrows():
        flags = []
        if row['review_count'] < 10:
            flags.append("CRITICAL — likely scraping failure")
        elif row['review_count'] < 50:
            flags.append("LOW COVERAGE — may need re-scrape")
            
        e_date = row['earliest_date'].strftime('%Y-%m-%d') if pd.notnull(row['earliest_date']) else 'N/A'
        l_date = row['latest_date'].strftime('%Y-%m-%d') if pd.notnull(row['latest_date']) else 'N/A'
        
        flag_str = " | ".join(flags)
        print(f"{row['location_name']:<40} | {row['review_count']:<12} | {e_date:<13} | {l_date:<13} | {flag_str}")

    # DIAGNOSTIC 4 — MONTHLY VOLUME CHECK
    print("\n--- DIAGNOSTIC 4 — MONTHLY VOLUME CHECK ---")
    df['month'] = df['review_date'].dt.to_period('M')
    
    monthly_stats = df.groupby('month').agg(
        total_reviews=('rating', 'count'),
        locations_reporting=('location_name', 'nunique')
    ).reset_index()
    
    # Sort chronologically
    monthly_stats = monthly_stats.sort_values('month')
    monthly_stats['prev_reviews'] = monthly_stats['total_reviews'].shift(1)
    
    print(f"{'month':<10} | {'total_reviews':<15} | {'locations_reporting':<20} | {'flags'}")
    print("-" * 75)
    for _, row in monthly_stats.iterrows():
        flags = []
        if pd.isna(row['month']):
             continue
             
        if row['total_reviews'] < 100:
            flags.append("sparse")
            
        if pd.notna(row['prev_reviews']) and row['prev_reviews'] > 0:
            drop_pct = (row['prev_reviews'] - row['total_reviews']) / row['prev_reviews']
            if drop_pct > 0.70:
                flags.append("possible artifact (drop > 70%)")
                
        flag_str = " | ".join(flags)
        print(f"{str(row['month']):<10} | {row['total_reviews']:<15} | {row['locations_reporting']:<20} | {flag_str}")

    # DIAGNOSTIC 5 — BRAND PROPERTY COVERAGE
    print("\n--- DIAGNOSTIC 5 — BRAND PROPERTY COVERAGE ---")
    caesars_props = [
        "Caesars Palace", "Harrah's Las Vegas", "The LINQ Hotel + Experience", 
        "Flamingo Las Vegas", "The Cromwell", "Bally's Las Vegas (Horseshoe)", 
        "Paris Las Vegas", "Planet Hollywood"
    ]
    mgm_props = [
        "ARIA Resort & Casino", "Bellagio", "MGM Grand", "Mandalay Bay", 
        "Park MGM", "New York-New York", "Luxor Las Vegas", "Excalibur", 
        "Delano Las Vegas", "Vdara Hotel & Spa", "The Signature at MGM Grand", 
        "The Cosmopolitan"
    ]
    
    print("\nCaesars Properties:")
    check_brand_props(prop_stats, caesars_props)
    
    print("\nMGM Properties:")
    check_brand_props(prop_stats, mgm_props)

    # DIAGNOSTIC 6 — DATA QUALITY FLAGS
    print("\n--- DIAGNOSTIC 6 — DATA QUALITY FLAGS ---")
    if 'data_quality_flag' in df.columns:
        print("data_quality_flag breakdown:")
        counts = df['data_quality_flag'].value_counts()
        total = len(df)
        for val, count in counts.items():
            print(f"  {val}: {count} ({count/total*100:.1f}%)")
    else:
        print("Column 'data_quality_flag' not found.")
        
    if 'date_precision' in df.columns:
         print("\ndate_precision breakdown:")
         counts = df['date_precision'].value_counts()
         total = len(df)
         
         approximate_pct = 0
         for val, count in counts.items():
             pct = count/total*100
             print(f"  {val}: {count} ({pct:.1f}%)")
             if val == 'approximate':
                 approximate_pct = pct
                 
         if approximate_pct > 30:
             print(f"\nFLAG: {approximate_pct:.1f}% of dates are 'approximate'. ISO timestamp extraction is not working well.")
    else:
        print("Column 'date_precision' not found.")
        
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    
    issues = []
    if unique_locations < 60:
        issues.append(f"Missing locations (only {unique_locations} found, expected 67)")
    
    cr_props = prop_stats[prop_stats['review_count'] < 10]['location_name'].tolist()
    if cr_props:
        issues.append(f"{len(cr_props)} properties have CRITICAL review counts (<10)")
        
    low_props = prop_stats[(prop_stats['review_count'] >= 10) & (prop_stats['review_count'] < 50)]['location_name'].tolist()
    if low_props:
        issues.append(f"{len(low_props)} properties have LOW COVERAGE (<50)")
        
    if 'date_precision' in df.columns and approximate_pct > 30:
        issues.append("ISO timestamp extraction failed for >30% of reviews (mostly 'approximate' dates)")
        
    if issues:
        print("NO-GO")
        print("Issues needing attention before proceeding:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("GO")
        print("Dataset looks clean, proceed to foot_traffic_proxy.py")


def check_brand_props(prop_stats, brand_list):
    for prop in brand_list:
        match = prop_stats[prop_stats['location_name'] == prop]
        if match.empty:
            print(f"{prop:<35} | 0            | N/A           | N/A           | CRITICAL — missing completely")
        else:
            row = match.iloc[0]
            count = row['review_count']
            e_date = row['earliest_date'].strftime('%Y-%m-%d') if pd.notnull(row['earliest_date']) else 'N/A'
            l_date = row['latest_date'].strftime('%Y-%m-%d') if pd.notnull(row['latest_date']) else 'N/A'
            
            status = "OK"
            if count < 10:
                status = "CRITICAL"
            elif count < 50:
                status = "LOW"
                
            print(f"{prop:<35} | {count:<12} | {e_date:<13} - {l_date:<13} | {status}")


if __name__ == "__main__":
    run_diagnostics()
