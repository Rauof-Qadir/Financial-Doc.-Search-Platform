import json
from pathlib import Path

path = Path("evaluation/retrieval_eval_dataset.json")

questions = json.loads(
    path.read_text(encoding="utf-8")
)

updates = {
    "Q004": [
        "apple_annual_report-p031-c001",
        "apple_annual_report-p023-c002",
        "apple_annual_report-p038-c001",
        "apple_annual_report-p051-c001"
    ],
    "Q014": [
        "apple_annual_report-p033-c001",
        "apple_annual_report-p039-c001"
    ],
    "Q022": [
        "apple_annual_report-p050-c002"
    ],
    "Q023": [
        "apple_annual_report-p033-c001"
    ]
}

for item in questions:
    question_id = item["question_id"]

    if question_id in updates:
        item["relevant_chunk_ids"] = updates[question_id]
        print(f"Updated {question_id}: {len(updates[question_id])} labels")

path.write_text(
    json.dumps(questions, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8"
)

print("\nEvaluation dataset updated successfully.")
print("Total questions:", len(questions))
