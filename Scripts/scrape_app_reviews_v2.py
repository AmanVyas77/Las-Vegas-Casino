import sys
import os
import time
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import json
    import requests
    from google_play_scraper import reviews as gp_reviews, Sort
    from app_store_scraper import AppStore
except ImportError:
    print(
        "ERROR: Required libraries not found.\n"
        "Please install them by running:\n"
        "  pip install google-play-scraper app-store-scraper requests\n"
    )
    sys.exit(1)

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
if not DATA_DIR.exists():
    DATA_DIR.mkdir()
OUTPUT_FILE = DATA_DIR / "loyalty_app_reviews.csv"

DATE_START = datetime(2022, 1, 1)
DATE_END = datetime(2026, 12, 31)
MAX_REVIEWS_PER_APP_PER_STORE = 5000

APPS = [
    {
        "brand": "Caesars Entertainment",
        "app_name": "Caesars Rewards",
        "google_play_id": "com.caesars.playbytr",
        "apple_app_id": "720839830",
        "apple_app_name": "caesars-rewards-resort-offers",
    },
    {
        "brand": "MGM Resorts",
        "app_name": "MGM Rewards",
        "google_play_id": "com.mgmresorts.mgmresorts",
        "apple_app_id": "366518979",
        "apple_app_name": "mgm-rewards",
    },
]

# NOTE FOR ANALYST:
# Caesars Google Play ID (com.caesars.playbytr) is their 
# casino/rewards hybrid app. Reviews will contain loyalty 
# sentiment mixed with gameplay feedback. The loyalty keyword 
# filter in loyalty_sentiment_v2.py will isolate relevant content.

def fetch_apple_reviews_rss(app_id: str, brand: str, app_name: str) -> list[dict]:
    """
    Fallback: fetch Apple App Store reviews via iTunes RSS feed.
    Returns up to 500 reviews (10 pages x 50 reviews).
    No library required — uses requests only.
    """
    reviews = []
    for page in range(1, 11):   # pages 1-10
        url = (
            f"https://itunes.apple.com/us/rss/customerreviews/"
            f"page={page}/id={app_id}/sortby=mostrecent/json"
        )
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
            entries = data.get("feed", {}).get("entry", [])
            if not entries:
                break
            for entry in entries:
                # Skip the first entry if it's the app metadata rather than a review
                if "im:rating" not in entry:
                    continue
                reviews.append({
                    "brand": brand,
                    "app_name": app_name,
                    "data_source": "apple_app_store",
                    "review_date": pd.to_datetime(
                        entry.get("updated", {}).get("label", None)
                    ),
                    "rating": float(
                        entry.get("im:rating", {}).get("label", 0)
                    ),
                    "review_text": entry.get("content", {})
                                       .get("label", ""),
                    "date_precision": "exact"
                })
            time.sleep(1)
        except Exception as e:
            print(f"[RSS FALLBACK] Page {page} failed: {e}")
            break
    return reviews

def scrape_google_play(app: dict) -> list[dict]:
    """Scrape reviews from Google Play Store for the given app."""
    logger.info(f"Fetching Google Play reviews for {app['app_name']}...")
    all_reviews = []
    
    try:
        result, _ = gp_reviews(
            app["google_play_id"],
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=MAX_REVIEWS_PER_APP_PER_STORE,
        )
        logger.info(f"  Fetched {len(result)} raw reviews from Google Play.")

        for r in result:
            review_date = r["at"]
            all_reviews.append({
                "brand": app["brand"],
                "app_name": app["app_name"],
                "data_source": "google_play",
                "review_date": review_date,
                "rating": r["score"],
                "review_text": r["content"],
                "date_precision": "exact",
            })
        
        logger.info(f"  Kept {len(all_reviews)} reviews.")
        
        if len(all_reviews) == 0:
            print(
                f"[WARNING] Google Play returned 0 reviews for "
                f"{app['brand']} ({app['google_play_id']}).\n"
                f"  Possible causes:\n"
                f"  1. App ID is correct but app has very few reviews\n"
                f"  2. google-play-scraper is being rate limited\n"
                f"  3. The app is a casino/gaming app and Google Play "
                f"     may restrict scraping for gambling category apps\n"
                f"  Action: Manually verify at:\n"
                f"  https://play.google.com/store/apps/details"
                f"?id={app['google_play_id']}"
            )
            
        return all_reviews

    except Exception as e:
        logger.error(f"  Error fetching Google Play reviews for {app['app_name']}: {e}")
        # Rate limit / error handling with 1 retry
        logger.info("  Sleeping 30s before retry...")
        time.sleep(30)
        try:
            result, _ = gp_reviews(
                app["google_play_id"],
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                count=MAX_REVIEWS_PER_APP_PER_STORE,
            )
            for r in result:
                review_date = r["at"]
                all_reviews.append({
                    "brand": app["brand"],
                    "app_name": app["app_name"],
                    "data_source": "google_play",
                    "review_date": review_date,
                    "rating": r["score"],
                    "review_text": r["content"],
                    "date_precision": "exact",
                })
            logger.info(f"  Retry successful. Kept {len(all_reviews)} reviews.")
            
            if len(all_reviews) == 0:
                print(
                    f"[WARNING] Google Play returned 0 reviews for "
                    f"{app['brand']} ({app['google_play_id']}).\n"
                    f"  Possible causes:\n"
                    f"  1. App ID is correct but app has very few reviews\n"
                    f"  2. google-play-scraper is being rate limited\n"
                    f"  3. The app is a casino/gaming app and Google Play "
                    f"     may restrict scraping for gambling category apps\n"
                    f"  Action: Manually verify at:\n"
                    f"  https://play.google.com/store/apps/details"
                    f"?id={app['google_play_id']}"
                )
            
            return all_reviews
        except Exception as retry_err:
            logger.error(f"  Retry failed for {app['app_name']} (Google Play): {retry_err}")
            return []

def scrape_apple_app_store(app: dict) -> list[dict]:
    """Scrape reviews from Apple App Store for the given app."""
    logger.info(f"Fetching Apple App Store reviews for {app['app_name']}...")
    all_reviews = []
    
    try:
        app_store = AppStore(
            country="us",
            app_name=app["apple_app_name"],
            app_id=app["apple_app_id"]
        )
        app_store.review(how_many=MAX_REVIEWS_PER_APP_PER_STORE)
        raw_reviews = app_store.reviews
        logger.info(f"  Fetched {len(raw_reviews)} raw reviews from Apple App Store.")

        if not raw_reviews:
            logger.warning(f"  App store returned 0 reviews. Forcing RSS fallback.")
            raise Exception("No reviews retrieved from AppStore library.")

        for r in raw_reviews:
            review_date = r["date"]
            all_reviews.append({
                "brand": app["brand"],
                "app_name": app["app_name"],
                "data_source": "apple_app_store",
                "review_date": review_date,
                "rating": r["rating"],
                "review_text": r["review"],
                "date_precision": "exact",
            })
                
        logger.info(f"  Kept {len(all_reviews)} reviews.")
        return all_reviews

    except Exception as e:
        logger.warning(f"  Library fetch failed for {app['app_name']}: {e}. Trying RSS fallback...")
        all_reviews = fetch_apple_reviews_rss(app["apple_app_id"], app["brand"], app["app_name"])
        
        if len(all_reviews) == 0:
             print(f"[WARNING] No Apple App Store reviews retrieved for {app['brand']}. "
                   f"Verify the app_id is correct at apps.apple.com")
        else:
             logger.info(f"  RSS fallback successful. Fetched {len(all_reviews)} reviews.")
             
        return all_reviews

def main():
    logger.info("Starting Loyalty App Review Scraper...")
    all_data = []

    for app in APPS:
        gp_data = scrape_google_play(app)
        all_data.extend(gp_data)
        
        apple_data = scrape_apple_app_store(app)
        all_data.extend(apple_data)

    if not all_data:
        logger.warning("No reviews collected. Exiting.")
        return

    # Combine into DataFrame
    df = pd.DataFrame(all_data)
    
    # Optional: Format review_date to standard ISO string if not already
    df['review_date'] = pd.to_datetime(df['review_date'], utc=True).dt.tz_localize(None)
    
    print(f"\n[DIAGNOSTIC] Raw reviews before date filter:")
    print(df.groupby(["brand","data_source"])["review_date"]
            .agg(["count","min","max"]))
            
    df = df[
      (df["review_date"] >= DATE_START) & 
      (df["review_date"] <= DATE_END)
    ]

    # Ensure output structure
    columns_order = [
        "brand", "app_name", "data_source", "review_date",
        "rating", "review_text", "date_precision"
    ]
    df = df[columns_order]

    # Sort
    df = df.sort_values(by=["brand", "review_date"], ascending=[True, False])

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(df)} total reviews to {OUTPUT_FILE}")

    # Generate and print summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY: LOYALTY APP REVIEWS")
    print("="*80)
    
    summary = df.groupby(["brand", "data_source"]).agg(
        Total_Reviews=("rating", "count"),
        Avg_Rating=("rating", "mean"),
        Min_Date=("review_date", "min"),
        Max_Date=("review_date", "max")
    ).reset_index()

    print(f"{'Brand':<25} | {'Store':<16} | {'Total':<6} | {'Date Range':<22} | {'Avg Rating':<10}")
    print("-" * 80)
    
    for _, row in summary.iterrows():
        date_range = f"{row['Min_Date'].strftime('%Y-%m')} to {row['Max_Date'].strftime('%Y-%m')}"
        store = "Google Play" if row["data_source"] == "google_play" else "App Store"
        print(f"{row['brand']:<25} | {store:<16} | {row['Total_Reviews']:<6} | {date_range:<22} | {row['Avg_Rating']:.2f}")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()
