"""
Las Vegas Strip — Review Scraping Pipeline (Google Maps)
========================================================
Scrapes Google Maps review data for every entity listed in
Data/las_vegas_strip_entities.md and writes results to
Data/las_vegas_reviews.csv.

Libraries: selenium, beautifulsoup4, pandas, requests
Run:  python Scripts/scrape_reviews_v2.py
"""

import os
import re
import csv
import json
import time
import random
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "Data"
INPUT_FILE = DATA_DIR / "las_vegas_strip_entities.md"
OUTPUT_FILE = DATA_DIR / "las_vegas_reviews.csv"
COORDS_FILE = DATA_DIR / "entity_coordinates.csv"

# Date window for reviews (inclusive)                    # FIX 5
DATE_START = datetime(2022, 1, 1)
DATE_END   = datetime(2026, 12, 31)

# Throttle settings (seconds)
PAGE_DELAY_MIN = 2
PAGE_DELAY_MAX = 5
SCROLL_DELAY_MIN = 2.5                                  # FIX 3
SCROLL_DELAY_MAX = 5.0                                  # FIX 3
REQUEST_TIMEOUT = 20

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF = 5

# Maximum scroll iterations per entity to load reviews   # FIX 1
MAX_SCROLL_ITERATIONS = 600
MAX_REVIEWS_PER_PROPERTY = 3000

# Checkpoint file for resume support                     # FIX 4
CHECKPOINT_FILE = DATA_DIR / "scrape_checkpoint.json"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reference date for relative date parsing
NOW = datetime(2026, 3, 6)


# ──────────────────────────────────────────────
# 1. Parse Entity List from Markdown
# ──────────────────────────────────────────────
def parse_entity_list(filepath: Path) -> list[str]:
    """
    Read the markdown table and return a deduplicated list of entity names.
    Skips header / separator rows.
    """
    entities = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.split("|")]
            if len(cols) < 3:
                continue
            name = cols[1]
            # Skip header and separator rows
            if name in ("", "Entity Name") or set(name) <= {"-", " "}:
                continue
            entities.append(name)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    logger.info(f"Loaded {len(unique)} entities from {filepath.name}")
    return unique


# ──────────────────────────────────────────────
# 2. Initialize Selenium WebDriver
# ──────────────────────────────────────────────
def init_driver() -> webdriver.Chrome:
    """
    Create a Chrome WebDriver configured for Google Maps scraping.
    Using headed mode for reliability with Google Maps JS rendering.
    """
    chrome_opts = Options()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--window-size=1920,1080")
    chrome_opts.add_argument("--disable-blink-features=AutomationControlled")
    chrome_opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    chrome_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=chrome_opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    logger.info("Chrome WebDriver initialised (headless)")
    return driver


# ──────────────────────────────────────────────
# 3. Parse Relative Dates from Google Maps
# ──────────────────────────────────────────────
def parse_relative_date(raw: str) -> datetime | None:
    """
    Google Maps displays dates like:
      'a month ago', '2 months ago', 'a year ago', '3 years ago',
      '2 weeks ago', '5 days ago', 'a day ago', etc.
    Convert to an approximate datetime relative to NOW.
    """
    if not raw:
        return None

    raw = raw.strip().lower()

    # Direct date patterns (sometimes Google shows actual dates)
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue

    # Relative patterns
    # "a day ago", "an hour ago" etc.
    if "just now" in raw or "a second ago" in raw or "seconds ago" in raw:
        return NOW
    if "a minute ago" in raw or "minutes ago" in raw:
        return NOW
    if "an hour ago" in raw or "hours ago" in raw:
        return NOW

    # Days
    m = re.search(r"(\d+)\s*days?\s*ago", raw)
    if m:
        return NOW - timedelta(days=int(m.group(1)))
    if "a day ago" in raw or "yesterday" in raw:
        return NOW - timedelta(days=1)

    # Weeks
    m = re.search(r"(\d+)\s*weeks?\s*ago", raw)
    if m:
        return NOW - timedelta(weeks=int(m.group(1)))
    if "a week ago" in raw:
        return NOW - timedelta(weeks=1)

    # Months
    m = re.search(r"(\d+)\s*months?\s*ago", raw)
    if m:
        return NOW - relativedelta(months=int(m.group(1)))
    if "a month ago" in raw:
        return NOW - relativedelta(months=1)

    # Years
    m = re.search(r"(\d+)\s*years?\s*ago", raw)
    if m:
        return NOW - relativedelta(years=int(m.group(1)))
    if "a year ago" in raw:
        return NOW - relativedelta(years=1)

    return None


# ──────────────────────────────────────────────
# 4. Search Google Maps for Entity
# ──────────────────────────────────────────────
def search_google_maps(driver: webdriver.Chrome, entity_name: str) -> bool:
    """
    Search for the entity on Google Maps and click the first result.
    Returns True if a place was found and selected.
    """
    query = f"{entity_name} Las Vegas Strip"
    search_url = f"https://www.google.com/maps/search/{requests.utils.quote(query)}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"  Searching Google Maps for '{entity_name}' (attempt {attempt})")
            driver.get(search_url)
            time.sleep(random.uniform(PAGE_DELAY_MIN, PAGE_DELAY_MAX))

            # Wait for either a single place result or a list of results
            try:
                WebDriverWait(driver, REQUEST_TIMEOUT).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR,
                        "div.fontHeadlineSmall, h1.DUwDvf, div[role='feed']"))
                )
            except TimeoutException:
                logger.warning(f"  Timeout waiting for search results (attempt {attempt})")
                continue

            # Check if we landed directly on a place page (single result)
            try:
                driver.find_element(By.CSS_SELECTOR, "h1.DUwDvf, div.tAiQdd h1")
                logger.info(f"  → Landed on place page directly")
                return True
            except NoSuchElementException:
                pass

            # Multiple results — click the first one
            try:
                results = driver.find_elements(By.CSS_SELECTOR, "div.fontHeadlineSmall, a.hfpxzc")
                if results:
                    # Some headless runs intercept clicks with floating divs.
                    # Remove potential overlaps first before clicking.
                    driver.execute_script(
                        "arguments[0].scrollIntoView(true);"
                        "if(document.querySelector('.bJzME')) document.querySelector('.bJzME').remove();"
                        "arguments[0].click();", 
                        results[0]
                    )
                    time.sleep(random.uniform(2, 4))
                    # Verify we're on a place page
                    WebDriverWait(driver, REQUEST_TIMEOUT).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "h1.DUwDvf, div.tAiQdd h1"))
                    )
                    logger.info(f"  → Selected first search result")
                    return True
            except (NoSuchElementException, TimeoutException):
                pass

            logger.warning(f"  Could not find a valid place result (attempt {attempt})")

        except WebDriverException as exc:
            logger.warning(f"  Search error (attempt {attempt}): {exc}")
            time.sleep(RETRY_BACKOFF * attempt)

    return False


# ──────────────────────────────────────────────
# 4b. Extract Place Coordinates
# ──────────────────────────────────────────────
def extract_place_coordinates(driver: webdriver.Chrome) -> dict:
    """
    After landing on a Google Maps place page, extract:
      - latitude and longitude from the URL (format: /@LAT,LNG,zoom)
      - place_id from the URL (format: /place/Name/data=!...!1s<PLACE_ID>!)
    Returns a dict: {"latitude": float, "longitude": float, "place_id": str}
    Returns empty dict if extraction fails.
    """
    url = driver.current_url
    
    # Extract lat/lng
    lat_lng_match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', url)
    if not lat_lng_match:
        logger.warning(f"  Could not extract coordinates from URL: {url}")
        return {}
        
    latitude = round(float(lat_lng_match.group(1)), 6)
    longitude = round(float(lat_lng_match.group(2)), 6)
    
    # Extract Place ID
    place_id = None
    place_id_match = re.search(r'!1s([\w:]+)!', url)
    if place_id_match:
        place_id = place_id_match.group(1)
        
    return {
        "latitude": latitude,
        "longitude": longitude,
        "place_id": place_id
    }


# ──────────────────────────────────────────────
# 5. Navigate to the Reviews Tab
# ──────────────────────────────────────────────
def open_reviews_tab(driver: webdriver.Chrome) -> bool:
    """
    Click the 'Reviews' tab on a Google Maps place page.
    Returns True if reviews section was opened successfully.
    """
    try:
        # Look for the reviews tab button
        tabs = driver.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            tab_text = tab.text.strip().lower()
            if "review" in tab_text:
                tab.click()
                time.sleep(random.uniform(2, 3))
                logger.info("  → Opened Reviews tab")
                return True

        # Fallback: click the review count link
        review_links = driver.find_elements(By.CSS_SELECTOR,
            "button[jsaction*='review'], span[jsan*='review']")
        for link in review_links:
            if "review" in link.text.lower():
                link.click()
                time.sleep(random.uniform(2, 3))
                logger.info("  → Opened Reviews via review count link")
                return True

        logger.warning("  Could not find Reviews tab")
        return False

    except (NoSuchElementException, ElementClickInterceptedException) as exc:
        logger.warning(f"  Error opening reviews tab: {exc}")
        return False


# ──────────────────────────────────────────────
# 6. Sort Reviews by Newest
# ──────────────────────────────────────────────
def sort_reviews_newest(driver: webdriver.Chrome) -> None:
    """
    Click the sort dropdown and select 'Newest' to get chronological reviews.
    """
    try:
        # Find the sort button (usually a dropdown menu)
        sort_btn = driver.find_elements(By.CSS_SELECTOR,
            "button[aria-label*='Sort'], button[data-value='sort'],"
            " button.g88MCb")
        if not sort_btn:
            # Fallback: look for "Most relevant" text
            sort_btn = driver.find_elements(By.XPATH,
                "//button[contains(., 'Most relevant') or contains(., 'Sort')]")

        if sort_btn:
            sort_btn[0].click()
            time.sleep(1.5)

            # Click "Newest" option in the dropdown
            menu_items = driver.find_elements(By.CSS_SELECTOR,
                "div[role='menuitemradio'], li[role='menuitemradio'],"
                " div.fxNQSd")
            for item in menu_items:
                if "newest" in item.text.lower():
                    item.click()
                    time.sleep(random.uniform(2, 3))
                    logger.info("  → Sorted reviews by Newest")
                    return

            logger.info("  Could not find 'Newest' option — using default sort")
        else:
            logger.info("  No sort button found — using default sort")

    except Exception as exc:
        logger.warning(f"  Error sorting reviews: {exc}")


# ──────────────────────────────────────────────
# 7. Scroll and Load Reviews
# ──────────────────────────────────────────────
def scroll_reviews(driver: webdriver.Chrome) -> int:
    """
    Scroll the reviews panel to load more reviews.
    Returns the number of review elements currently loaded.
    """
    # Find the scrollable reviews container
    scrollable = None
    try:
        # The reviews panel in Google Maps is a scrollable div
        scrollable_candidates = driver.find_elements(By.CSS_SELECTOR,
            "div.m6QErb.DxyBCb.kA9KIf.dS8AEf, div.m6QErb.DxyBCb")
        if scrollable_candidates:
            scrollable = scrollable_candidates[0]
    except Exception:
        pass

    if not scrollable:
        # Fallback: try to find any scrollable container with reviews
        try:
            scrollable = driver.find_element(By.CSS_SELECTOR,
                "div[role='feed'], div.section-scrollbox")
        except NoSuchElementException:
            return 0

    prev_count = 0
    stall_count = 0

    for i in range(MAX_SCROLL_ITERATIONS):
        # Scroll the container down
        driver.execute_script(
            "arguments[0].scrollTop = arguments[0].scrollHeight", scrollable
        )
        time.sleep(random.uniform(SCROLL_DELAY_MIN, SCROLL_DELAY_MAX))

        # Count currently loaded reviews
        reviews = driver.find_elements(By.CSS_SELECTOR,
            "div[data-review-id], div.jftiEf, div.WMbnJf")
        current_count = len(reviews)

        # FIX 1 — hard cap on reviews per property
        if current_count >= MAX_REVIEWS_PER_PROPERTY:
            logger.info(f"  Hit review cap ({MAX_REVIEWS_PER_PROPERTY}) — stopping scroll")
            break

        if current_count == prev_count:
            stall_count += 1
            if stall_count >= 10:                        # FIX 3 — raised from 5
                # FIX 3 — forced scroll fallback before giving up
                driver.execute_script(
                    "arguments[0].scrollTop = 9999999", scrollable
                )
                time.sleep(5)
                reviews_after = driver.find_elements(
                    By.CSS_SELECTOR, "div[data-review-id], div.jftiEf"
                )
                if len(reviews_after) > current_count:
                    stall_count = 0   # more loaded — reset and keep going
                    current_count = len(reviews_after)
                    prev_count = current_count
                    continue
                else:
                    break   # genuinely done
        else:
            stall_count = 0

        prev_count = current_count

        # Progress update every 10 scrolls
        if (i + 1) % 10 == 0:
            logger.info(f"  Scroll {i+1}: {current_count} reviews loaded so far")

    return prev_count


# ──────────────────────────────────────────────
# 8a. Extract ISO Dates from Google's Embedded JSON   (FIX 2)
# ──────────────────────────────────────────────
def extract_iso_dates(driver: webdriver.Chrome) -> list[str]:
    """
    Pull absolute ISO timestamps from Google Maps embedded JSON.
    Google embeds review publish dates in the page's script tags.
    Returns a list of ISO date strings like ['2024-03-15T10:22:00', ...]
    """
    try:
        script = """
        const scripts = document.querySelectorAll('script');
        for (let s of scripts) {
            if (s.text.includes('publishedAtDate') || 
                s.text.includes('reviewTime')) {
                return s.text;
            }
        }
        return null;
        """
        raw_json = driver.execute_script(script)
        if raw_json:
            dates = re.findall(
                r'"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', raw_json
            )
            return dates
    except Exception:
        pass
    return []


# ──────────────────────────────────────────────
# 8b. Extract Reviews from Page
# ──────────────────────────────────────────────
def extract_reviews(driver: webdriver.Chrome, entity_name: str) -> list[dict]:
    """
    Parse all loaded review elements and extract date, rating, and text.
    Filters to reviews within the DATE_START–DATE_END window.
    """
    # FIX 2 — attempt ISO date extraction before parsing HTML
    iso_dates = extract_iso_dates(driver)

    reviews = []
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Google Maps review containers (class .jftiEf or data-review-id)
    review_cards = soup.find_all("div", class_=re.compile(r"jftiEf"))
    if not review_cards:
        review_cards = soup.find_all("div", attrs={"data-review-id": True})
    if not review_cards:
        review_cards = soup.find_all("div", class_=re.compile(r"WMbnJf"))

    logger.info(f"  Found {len(review_cards)} review elements to parse")
    if iso_dates:
        logger.info(f"  Extracted {len(iso_dates)} ISO dates from embedded JSON")

    skipped_rating = 0
    skipped_date = 0
    skipped_range = 0

    for idx, card in enumerate(review_cards):
        # --- Rating ---
        rating = None

        # Pattern 1 (current 2026): Text like "3/5" or "5/5" in span.fzvQIb
        rating_el = card.find("span", class_=re.compile(r"fzvQIb"))
        if rating_el:
            rating_text = rating_el.get_text(strip=True)
            m = re.search(r"(\d)/\s*5", rating_text)
            if m:
                rating = float(m.group(1))

        # Pattern 2: aria-label like "5 stars" or "4 stars" or "Rated 4.0 out of 5"
        if rating is None:
            star_el = card.find(attrs={"aria-label": re.compile(r"\d.*star", re.I)})
            if star_el:
                m = re.search(r"(\d)", star_el["aria-label"])
                if m:
                    rating = float(m.group(1))

        # Pattern 3: role="img" with aria-label containing a digit
        if rating is None:
            star_el = card.find(attrs={"role": "img", "aria-label": re.compile(r"\d")})
            if star_el:
                m = re.search(r"(\d)", star_el.get("aria-label", ""))
                if m:
                    rating = float(m.group(1))

        # Pattern 4: Any element with class containing 'star' and a digit in text
        if rating is None:
            star_el = card.find(class_=re.compile(r"kvMYJc|DU9Pgb"))
            if star_el:
                m = re.search(r"(\d)\s*/\s*5", star_el.get_text(strip=True))
                if m:
                    rating = float(m.group(1))

        # Skip reviews missing rating
        if rating is None:
            skipped_rating += 1
            continue

        # --- Date ---                                   # FIX 2
        review_date = None
        date_precision = "approximate"

        # Try ISO date first (from embedded JSON)
        if idx < len(iso_dates):
            try:
                review_date = datetime.fromisoformat(iso_dates[idx])
                date_precision = "exact"
            except (ValueError, TypeError):
                pass

        # Fallback: parse relative date from the review card
        if review_date is None:
            raw_date = ""

            # Primary: span with class xRkPPb (verified 2026 selector)
            date_el = card.find("span", class_=re.compile(r"xRkPPb"))
            if date_el:
                raw_date = date_el.get_text(strip=True)

            # Fallback 1: span with class rsqaWe
            if not raw_date:
                date_el = card.find("span", class_=re.compile(r"rsqaWe"))
                if date_el:
                    raw_date = date_el.get_text(strip=True)

            # Fallback 2: any span containing "ago"
            if not raw_date:
                all_spans = card.find_all("span")
                for span in all_spans:
                    text = span.get_text(strip=True).lower()
                    if "ago" in text and any(w in text for w in
                        ["day", "week", "month", "year", "hour", "minute"]):
                        raw_date = span.get_text(strip=True)
                        break

            review_date = parse_relative_date(raw_date)
            date_precision = "approximate"

            if review_date is None:
                skipped_date += 1
                logger.debug(f"    Skipping review with unparseable date: '{raw_date}'")
                continue

        # Check date window
        if review_date < DATE_START or review_date > DATE_END:
            skipped_range += 1
            continue

        # --- Review Text ---
        review_text = ""

        # Primary: span.wiI7pd (verified 2026 selector)
        text_el = card.find("span", class_=re.compile(r"wiI7pd"))
        if text_el:
            review_text = text_el.get_text(strip=True)

        if not review_text:
            # Fallback: div.MyEned
            text_el = card.find("div", class_=re.compile(r"MyEned"))
            if text_el:
                review_text = text_el.get_text(strip=True)

        reviews.append({
            "location_name": entity_name,
            "review_date": review_date.strftime("%Y-%m-%d"),
            "rating": rating,
            "review_text": review_text,
            "date_precision": date_precision,
        })

    logger.info(f"  Parse stats: {len(reviews)} extracted, "
                f"{skipped_rating} skipped (no rating), "
                f"{skipped_date} skipped (bad date), "
                f"{skipped_range} skipped (out of range)")

    return reviews


# ──────────────────────────────────────────────
# 9. Expand Truncated Review Text
# ──────────────────────────────────────────────
def expand_reviews(driver: webdriver.Chrome) -> None:
    """
    Click all 'More' buttons to expand truncated review text.
    """
    try:
        more_buttons = driver.find_elements(By.CSS_SELECTOR,
            "button.w8nwRe.kyuRq, button[aria-label='See more'],"
            " button.M77dve")
        logger.info(f"  Expanding {len(more_buttons)} truncated reviews")
        for btn in more_buttons:
            try:
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.2)
            except (StaleElementReferenceException, ElementClickInterceptedException):
                pass
    except Exception:
        pass


# ──────────────────────────────────────────────
# 10. Scrape All Reviews for One Entity
# ──────────────────────────────────────────────
def scrape_entity_reviews(driver: webdriver.Chrome, entity_name: str) -> list[dict]:
    """
    Full pipeline for one entity: search → open reviews → sort → scroll → extract.
    """
    # Step 1: Search Google Maps
    if not search_google_maps(driver, entity_name):
        logger.warning(f"  Could not find '{entity_name}' on Google Maps — skipping")
        return [], {"latitude": None, "longitude": None, "place_id": None}

    # Extract coordinates after landing
    coords = extract_place_coordinates(driver)
    if coords:
        logger.info(f"  Coordinates: {coords['latitude']}, {coords['longitude']}")
    else:
        logger.warning(f"  Could not extract coordinates for '{entity_name}'")
        coords = {"latitude": None, "longitude": None, "place_id": None}

    # Step 2: Open Reviews tab
    if not open_reviews_tab(driver):
        logger.warning(f"  Could not open reviews for '{entity_name}' — skipping")
        return [], coords

    # Step 3: Sort by newest for chronological data
    sort_reviews_newest(driver)

    # Step 4: Scroll to load reviews
    total_loaded = scroll_reviews(driver)
    logger.info(f"  Total reviews loaded: {total_loaded}")

    # Step 5: Expand truncated text
    expand_reviews(driver)
    time.sleep(1)

    # Step 6: Extract and filter reviews
    reviews = extract_reviews(driver, entity_name)
    logger.info(f"  Extracted {len(reviews)} reviews within date range for '{entity_name}'")

    return reviews, coords


# ──────────────────────────────────────────────
# 11. Write Reviews to CSV (Incremental)
# ──────────────────────────────────────────────
FIELDNAMES = [
    "location_name", "strip_region", "latitude", "longitude", 
    "review_date", "date_precision", "rating", "review_text"
]


def init_csv(filepath: Path) -> None:                   # FIX 4 — only write header if new
    """Create the CSV file with header row if it does not already exist."""
    if not filepath.exists():
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()
        logger.info(f"Initialised output CSV: {filepath}")
    else:
        logger.info(f"Appending to existing CSV: {filepath}")


def append_reviews_to_csv(filepath: Path, reviews: list[dict], region_lookup: dict, coords: dict) -> None:
    """Append a batch of review dicts to the CSV file, enriching with region and coords."""
    if not reviews:
        return
        
    enriched_reviews = []
    for r in reviews:
        # Create a copy so we don't mutate the original dictionary in-place unexpectedly
        updated = dict(r)
        updated["strip_region"] = region_lookup.get(r["location_name"], "Unknown")
        updated["latitude"] = coords.get("latitude")
        updated["longitude"] = coords.get("longitude")
        enriched_reviews.append(updated)
        
    with open(filepath, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writerows(enriched_reviews)
    logger.info(f"  Wrote {len(enriched_reviews)} reviews to {filepath.name}")


# ──────────────────────────────────────────────
# 11b. Write Coordinates to CSV
# ──────────────────────────────────────────────
COORDS_FIELDNAMES = ["location_name", "latitude", "longitude", "place_id", "scrape_timestamp"]

def write_coordinates(entity_name: str, coords: dict) -> None:
    """Append extracted coordinates to the separate entity_coordinates.csv file."""
    if not COORDS_FILE.exists():
        with open(COORDS_FILE, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COORDS_FIELDNAMES)
            writer.writeheader()
            
    with open(COORDS_FILE, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COORDS_FIELDNAMES)
        writer.writerow({
            "location_name": entity_name,
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
            "place_id": coords.get("place_id"),
            "scrape_timestamp": datetime.now().isoformat()
        })


# ──────────────────────────────────────────────
# 11b. Checkpoint Load / Save                    (FIX 4)
# ──────────────────────────────────────────────
def load_checkpoint() -> set:
    """Return set of already-completed entity names."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_checkpoint(completed: set) -> None:
    """Write completed entity names to the checkpoint file."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(
            {
                "completed": list(completed),
                "last_updated": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

def load_coords_checkpoint() -> set:
    """Return set of entity names that already have coordinates saved."""
    coords_completed = set()
    if COORDS_FILE.exists():
        with open(COORDS_FILE, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                coords_completed.add(row.get("location_name"))
    return coords_completed


# ──────────────────────────────────────────────
# 12. Main Orchestrator
# ──────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("Las Vegas Strip — Review Scraping Pipeline (Google Maps)")
    logger.info("=" * 60)

    # Step 1: Load entity list
    entities = parse_entity_list(INPUT_FILE)
    if not entities:
        logger.error("No entities found — exiting")
        return

    # Build region lookup from the markdown file
    region_lookup = {}
    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.split("|")]
            if len(cols) < 5:
                continue
            name = cols[1]
            region = cols[4]
            # Skip header and separator rows
            if name in ("", "Entity Name") or set(name) <= {"-", " "}:
                continue
            region_lookup[name] = region

    # Step 2: Prepare output CSV
    init_csv(OUTPUT_FILE)

    # Step 3: Load checkpoint (FIX 4)
    completed = load_checkpoint()
    coords_completed = load_coords_checkpoint()
    
    if completed:
        logger.info(f"Checkpoint loaded: {len(completed)} entities have reviews scraped")
    if coords_completed:
        logger.info(f"Coords checkpoint loaded: {len(coords_completed)} entities have coordinates")

    # Step 4: Launch browser
    driver = init_driver()

    total_reviews = 0

    try:
        for idx, entity_name in enumerate(entities, start=1):
            
            needs_reviews = entity_name not in completed
            needs_coords = entity_name not in coords_completed

            # Skip entirely if we have BOTH reviews and coordinates
            if not needs_reviews and not needs_coords:
                logger.info(f"[{idx}/{len(entities)}] Skipping '{entity_name}' "
                            f"(checkpoint: reviews AND coordinates already saved)")
                continue

            logger.info(f"\n{'='*50}")
            logger.info(f"[{idx}/{len(entities)}] Processing: {entity_name}")
            logger.info(f"{'='*50}")

            try:
                # Fast coordinate fetch bypassing reviews if only coords are needed
                if not needs_reviews and needs_coords:
                    logger.info(f"  Reviews exist in checkpoint. Fetching ONLY coordinates...")
                    if not search_google_maps(driver, entity_name):
                         logger.warning(f"  Could not find '{entity_name}' on Google Maps — skipping")
                    else:
                         coords = extract_place_coordinates(driver)
                         if coords and (coords.get("latitude") is not None or coords.get("place_id") is not None):
                             write_coordinates(entity_name, coords)
                             logger.info(f"  ✓ Saved ONLY coordinates for {entity_name}")
                             
                    # Throttle short skips
                    sleep_sec = random.uniform(PAGE_DELAY_MIN, PAGE_DELAY_MAX)
                    logger.info(f"  Sleeping {sleep_sec:.1f}s before next entity...")
                    time.sleep(sleep_sec)
                    continue

                # Standard Scrape for both reviews and coordinates
                reviews, coords = scrape_entity_reviews(driver, entity_name)
            except Exception as e:
                logger.error(f"  Fatal error during scrape for '{entity_name}': {e}")
                logger.info("  Attempting to restart browser session...")
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = init_driver()
                logger.info(f"  Browser restarted. Skipping '{entity_name}' for now so the run can continue.")
                continue

            # Save coordinates
            if coords and (coords.get("latitude") is not None or coords.get("place_id") is not None):
                write_coordinates(entity_name, coords)

            # Save incrementally
            append_reviews_to_csv(OUTPUT_FILE, reviews, region_lookup, coords)
            total_reviews += len(reviews)

            # FIX 4 — checkpoint after each entity
            completed.add(entity_name)
            save_checkpoint(completed)

            logger.info(f"  ✓ {entity_name}: {len(reviews)} reviews | Running total: {total_reviews}")

            # Throttle between entities
            sleep_sec = random.uniform(PAGE_DELAY_MIN, PAGE_DELAY_MAX)
            logger.info(f"  Sleeping {sleep_sec:.1f}s before next entity...")
            time.sleep(sleep_sec)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user — saving progress and exiting")

    finally:
        driver.quit()
        logger.info("WebDriver closed")

    # Summary
    logger.info("=" * 60)
    logger.info(f"Pipeline complete. Total reviews collected: {total_reviews}")
    logger.info(f"Output saved to: {OUTPUT_FILE}")
    logger.info("=" * 60)

    # Quick preview
    if OUTPUT_FILE.exists() and total_reviews > 0:
        df = pd.read_csv(OUTPUT_FILE)
        logger.info(f"\nDataset shape: {df.shape}")
        logger.info(f"Date range: {df['review_date'].min()} → {df['review_date'].max()}")
        logger.info(f"Locations represented: {df['location_name'].nunique()}")
        logger.info(f"\nRating distribution:\n{df['rating'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
