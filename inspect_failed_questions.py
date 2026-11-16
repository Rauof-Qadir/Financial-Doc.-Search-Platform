import json
from pathlib import Path

questions = json.loads(
    Path("evaluation/retrieval_eval_dataset.json").read_text(encoding="utf-8")
)

for question_id in ["Q014", "Q022", "Q023"]:
    item = next(q for q in questions if q["question_id"] == question_id)

    print("\n" + "=" * 70)
    print("ID:", item["question_id"])
    print("QUESTION:", item["question"])
    print("CURRENT LABELS:")
    for chunk_id in item["relevant_chunk_ids"]:
        print(" -", chunk_id)
