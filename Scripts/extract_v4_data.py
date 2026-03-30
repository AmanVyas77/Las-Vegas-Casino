import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "Data" / "las_vegas_reviews.csv"
OUTPUT_FILE = BASE_DIR / "Data" / "las_vegas_reviews_v4.csv"

FIELDNAMES = [
    "location_name", "strip_region", "latitude", "longitude", 
    "review_date", "date_precision", "rating", "review_text"
]

v4_rows = []

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip the old 4-col header
        
        for row in reader:
            if len(row) == 8:
                v4_rows.append(dict(zip(FIELDNAMES, row)))
            elif len(row) != 4:
                print(f"Strange row length: {len(row)} -> {row[:3]}")

    if v4_rows:
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(v4_rows)
        print(f"Successfully recovered {len(v4_rows)} rows formatted for v4 and saved to {OUTPUT_FILE.name}")
    else:
        print("No 8-column rows found!")
except Exception as e:
    print(f"Error: {e}")
