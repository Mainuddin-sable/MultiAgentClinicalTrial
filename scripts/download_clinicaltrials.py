import json
import time
from pathlib import Path

import requests


API_URL = "https://clinicaltrials.gov/api/v2/studies"

OUTPUT_DIR = Path("../data/raw/clinicaltrials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "hypertension_trials.json"


params = {
    "query.cond": "Hypertension",

    "filter.overallStatus": (
        "COMPLETED,"
        "RECRUITING,"
        "NOT_YET_RECRUITING,"
        "ACTIVE_NOT_RECRUITING"
    ),

    "pageSize": 1000,
    "countTotal": "true",
    "format": "json",
}


all_studies = []
page_token = None
page_number = 1


while True:

    request_params = params.copy()

    if page_token:
        request_params["pageToken"] = page_token

    print()
    print("=" * 60)
    print(f"Downloading page {page_number}...")
    print("=" * 60)

    try:
        response = requests.get(
            API_URL,
            params=request_params,
            timeout=60
        )

        response.raise_for_status()

    except requests.RequestException as error:
        print()
        print("ERROR: Could not download data.")
        print(error)
        break

    data = response.json()

    studies = data.get("studies", [])

    all_studies.extend(studies)

    print(f"Studies in this page : {len(studies)}")
    print(f"Studies downloaded   : {len(all_studies)}")

    page_token = data.get("nextPageToken")

    if not page_token:
        print()
        print("No more pages.")
        break

    page_number += 1

    time.sleep(1)


output_data = {
    "source": "ClinicalTrials.gov",
    "api_endpoint": API_URL,
    "query": "Hypertension",
    "total_downloaded": len(all_studies),
    "studies": all_studies
}


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output_data,
        file,
        indent=2,
        ensure_ascii=False
    )


print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(f"Total studies saved : {len(all_studies)}")
print(f"Output file         : {OUTPUT_FILE}")