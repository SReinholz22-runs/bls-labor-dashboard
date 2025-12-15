import os
import json
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
CONFIG = BASE / "config"
RAW_DIR = BASE / "data" / "raw"
OUT_DIR = BASE / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SERIES_PATH = CONFIG / "series.json"
OUT_CSV = OUT_DIR / "bls_timeseries.csv"


def load_series_ids():
    with SERIES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_bls(series_ids, startyear, endyear):
    api_key = os.environ.get("BLS_API_KEY")
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": series_ids,
        "startyear": str(startyear),
        "endyear": str(endyear),
    }
    if api_key:
        payload["registrationKey"] = api_key

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def raw_to_tidy(raw_json):
    results = raw_json.get("Results", raw_json.get("results", {}))
    series_list = results.get("series", [])

    rows = []
    for s in series_list:
        sid = s["seriesID"]
        for obs in s.get("data", []):
            period = obs.get("period", "")
            if not period.startswith("M"):
                continue
            year = int(obs["year"])
            month = int(period[1:])
            date = pd.Timestamp(year=year, month=month, day=1)
            rows.append({"series_id": sid, "date": date, "value": float(obs["value"])})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def load_existing():
    if OUT_CSV.exists():
        return pd.read_csv(OUT_CSV, parse_dates=["date"])
    return pd.DataFrame(columns=["series_id", "date", "value"])


def main():
    series_ids = load_series_ids()
    existing = load_existing()

    today = datetime.now(UTC)
    startyear = today.year - 2
    endyear = today.year

    raw_json = fetch_bls(series_ids, startyear, endyear)

    ts = today.strftime("%Y%m%dT%H%M%SZ")
    raw_path = RAW_DIR / f"bls_raw_{ts}.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(raw_json, f, ensure_ascii=False, indent=2)

    new_df = raw_to_tidy(raw_json)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["series_id", "date"])
    combined = combined.sort_values(["series_id", "date"])

    combined.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(combined):,} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()