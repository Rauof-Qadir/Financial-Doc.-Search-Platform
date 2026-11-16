import json
from pathlib import Path

data = json.loads(
    Path("data/processed/apple_annual_report_chunks.json").read_text(encoding="utf-8")
)

target_pages = [23, 24, 25, 26, 31, 32, 33, 34, 35, 50, 51]

for chunk in data["chunks"]:
    if chunk["page_number"] in target_pages:
        print("\n" + "=" * 90)
        print("CHUNK ID:", chunk["chunk_id"])
        print("PDF PAGE:", chunk["page_number"])
        print("=" * 90)
        print(chunk["text"][:1100])
