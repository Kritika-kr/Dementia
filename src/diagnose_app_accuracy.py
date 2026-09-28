"""
Diagnostic: runs the EXACT same code path app.py uses (predict_single.predict)
against every transcript in your dataset, and compares to true labels.

This answers one question: does the deployed app's real-world accuracy match
what evaluate.py reported? If yes, misclassifications you're seeing live are
just the expected ~25-30% error rate showing up in small samples, not a bug.
If the number here is much WORSE than evaluate.py's ~72-74%, something is
actually broken in the deployment path specifically (wrong artifacts loaded,
feature mismatch, etc.) and needs fixing before you trust the app at all.

Usage (run from your project root, with the venv active, AFTER the app's
model artifacts exist in outputs/models and outputs/vectorizers):
    python -m src.diagnose_app_accuracy
"""
import pandas as pd
from . import config
from .predict_single import predict

LABEL_TO_INT = {v: k for k, v in config.LABEL_MAP.items()}


def main():
    df = pd.read_csv(config.RAW_CSV)
    correct = 0
    wrong_examples = []

    for _, row in df.iterrows():
        result = predict(row["text"])
        predicted_int = LABEL_TO_INT[result["label"]]
        is_correct = predicted_int == row["label"]
        correct += is_correct
        if not is_correct:
            wrong_examples.append({
                "speaker_id": row.get("speaker_id", "?"),
                "true_label": config.LABEL_MAP[row["label"]],
                "predicted": result["label"],
                "confidence": result["confidence"],
            })

    accuracy = correct / len(df)
    print(f"Deployed app accuracy on FULL dataset ({len(df)} transcripts): {accuracy:.3f}")
    print(f"(Note: this includes transcripts the model was trained on, so it will")
    print(f" look higher than your true held-out test accuracy — that's expected and")
    print(f" fine. The point here is just to confirm the pipeline runs without error")
    print(f" across every real transcript, not to re-measure generalization.)")

    control_to_dementia = sum(1 for e in wrong_examples if e["true_label"] == "Control")
    dementia_to_control = sum(1 for e in wrong_examples if e["true_label"] == "Dementia")
    print(f"\n{len(wrong_examples)} misclassified total:")
    print(f"  Control predicted as Dementia (false positive): {control_to_dementia}")
    print(f"  Dementia predicted as Control (false negative):  {dementia_to_control}")

    borderline = [e for e in wrong_examples
                  if 0.45 < max(e["confidence"].values()) < 0.60]
    print(f"\n{len(borderline)} of the {len(wrong_examples)} errors were near-50/50 "
          f"(model was uncertain, not confidently wrong) — these are less concerning "
          f"than a confident wrong answer would be.")

    print(f"\nFirst 10 misclassified examples:")
    for ex in wrong_examples[:10]:
        print(f"  speaker {ex['speaker_id']}: true={ex['true_label']}, "
              f"predicted={ex['predicted']}, confidence={ex['confidence']}")


if __name__ == "__main__":
    main()