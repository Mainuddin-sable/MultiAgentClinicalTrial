from pathlib import Path

import requests


# Folder where the eCFR files will be saved
ECFR_DIR = Path("../data/raw/ecfr")
ECFR_DIR.mkdir(parents=True, exist_ok=True)


# Official eCFR pages
DOCUMENTS = {
    "21_CFR_Part_50.html":
        "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-50",

    "21_CFR_Part_56.html":
        "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-56",

    "21_CFR_Part_312.html":
        "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-D/part-312",
}


HEADERS = {
    "User-Agent": "MultiAgentClinicalTrial-AcademicProject/1.0"
}


for filename, url in DOCUMENTS.items():

    print()
    print("=" * 60)
    print(f"Downloading {filename}")
    print("=" * 60)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=60
        )

        response.raise_for_status()

        output_file = ECFR_DIR / filename

        output_file.write_text(
            response.text,
            encoding="utf-8"
        )

        print(f"Saved successfully: {output_file}")

    except requests.RequestException as error:
        print(f"ERROR downloading {filename}")
        print(error)


print()
print("=" * 60)
print("eCFR DOWNLOAD COMPLETE")
print("=" * 60)