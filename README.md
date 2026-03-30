# Las Vegas Casino — Alternative Data

Sentiment and foot traffic analysis for major Las Vegas Strip casino operators, using app store reviews and geospatial proxies as alternative data signals.

## Research Questions

- How is loyalty app sentiment trending across major casino operators?
- Which properties are gaining or losing foot traffic relative to peers?
- What are guests saying about food, rooms, gaming, and service?

## Data Sources

- **App Store Reviews** — Google Play loyalty app reviews scraped for major operators
- **Foot Traffic Proxy** — Geospatial and review-volume signals used to approximate visitor counts
- **Entities** — Strip casino operators with coordinates mapped to north/south corridor segments

## Folder Structure

```
Las Vegas Casino/
├── Scripts/
│   ├── scrape_app_reviews_v2.py      # Scrapes Google Play loyalty app reviews
│   ├── scrape_reviews_v3.py          # Review scraper (v3)
│   ├── scrape_reviews_v4.py          # Review scraper (v4, current)
│   ├── extract_v4_data.py            # Parses and structures raw review data
│   ├── loyalty_sentiment.py          # VADER sentiment scoring on reviews
│   ├── loyalty_sentiment_v2.py       # Sentiment scoring (v2, current)
│   ├── foot_traffic_proxy.py         # Foot traffic proxy model
│   ├── foot_traffic_proxy_v2.py      # Foot traffic proxy (v2, current)
│   ├── trend_analysis.py             # YoY trend and growth rate analysis
│   ├── build_map.py                  # Interactive map of Strip entities
│   ├── generate_insights_image.py    # Dashboard image generation
│   ├── diagnostic.py                 # Data quality diagnostic tool
│   └── Scrape.ipynb                  # Exploratory scraping notebook
├── Data/
│   ├── las_vegas_reviews_v4.csv          # Raw app reviews
│   ├── loyalty_app_reviews.csv           # Processed loyalty reviews
│   ├── loyalty_sentiment_summary.csv     # Sentiment scores by operator
│   ├── loyalty_sentiment_by_source.csv   # Sentiment broken out by app source
│   ├── loyalty_sentiment_monthly.csv     # Monthly sentiment time series
│   ├── loyalty_sentiment_monthly_v2.csv  # Monthly sentiment (v2)
│   ├── monthly_traffic_by_location.csv   # Traffic proxy by property
│   ├── monthly_traffic_by_region.csv     # Traffic proxy by north/south corridor
│   ├── trend_analysis_summary.csv        # YoY growth rates summary
│   ├── yoy_growth_rates.csv              # Year-over-year growth by operator
│   ├── entity_coordinates.csv            # Strip entity lat/long coordinates
│   ├── las_vegas_strip_entities.md       # Entity definitions and metadata
│   ├── las_vegas_strip_map.html          # Interactive Strip map
│   ├── loyalty_sentiment_chart_v2.png    # Sentiment trend chart
│   ├── loyalty_sentiment_trend.png       # Sentiment trend visualization
│   ├── loyalty_insights_dashboard.png    # Summary dashboard
│   ├── foot_traffic_north_vs_south.png   # North vs. south corridor traffic
│   └── trend_analysis_chart.png          # YoY trend chart
```

## How to Run

```bash
# 1. Scrape reviews
python Scripts/scrape_reviews_v4.py

# 2. Extract and clean data
python Scripts/extract_v4_data.py

# 3. Score sentiment
python Scripts/loyalty_sentiment_v2.py

# 4. Analyze foot traffic
python Scripts/foot_traffic_proxy_v2.py

# 5. Run trend analysis
python Scripts/trend_analysis.py

# 6. Build map and visuals
python Scripts/build_map.py
python Scripts/generate_insights_image.py
```

## Key Outputs

- `loyalty_sentiment_monthly_v2.csv` — Monthly sentiment scores by operator (use for time series)
- `yoy_growth_rates.csv` — Year-over-year growth signals
- `loyalty_insights_dashboard.png` — Summary dashboard image
- `las_vegas_strip_map.html` — Interactive map (open in browser)
