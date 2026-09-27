"""
Builds data/transcripts.csv from the raw DementiaBank folder tree.

Expected layout (standard Pitt corpus layout):
    <root>/
        Control/
            cookie/   *.cha
            fluency/  *.cha
            recall/   *.cha
            sentence/ *.cha
        Dementia/
            cookie/   *.cha
            fluency/  *.cha
            recall/   *.cha
            sentence/ *.cha

Usage:
    python -m src.build_dataset --control /path/to/Control --dementia /path/to/Dementia
"""
import argparse
import csv
from pathlib import Path

from .cha_parser import parse_cha_file, parse_filename


def collect_group(group_dir: Path, label: int):
    rows = []
    if not group_dir.exists():
        print(f"WARNING: {group_dir} does not exist, skipping.")
        return rows

    for task_dir in sorted(group_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name.lower()
        for cha_file in sorted(task_dir.glob("*.cha")):
            speaker_id, visit = parse_filename(cha_file)
            text = parse_cha_file(cha_file)
            if not text.strip():
                continue
            rows.append({
                "speaker_id": speaker_id,
                "visit": visit,
                "task": task,
                "text": text,
                "label": label,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, help="Path to the Control group root")
    parser.add_argument("--dementia", required=True, help="Path to the Dementia group root")
    parser.add_argument("--out", default="data/transcripts.csv")
    args = parser.parse_args()

    rows = []
    rows += collect_group(Path(args.control), label=0)
    rows += collect_group(Path(args.dementia), label=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker_id", "visit", "task", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    n_control = sum(1 for r in rows if r["label"] == 0)
    n_dementia = sum(1 for r in rows if r["label"] == 1)
    n_speakers = len({r["speaker_id"] for r in rows})
    print(f"Wrote {len(rows)} transcripts ({n_control} control, {n_dementia} dementia) "
          f"from {n_speakers} unique speakers to {out_path}")

    task_counts = {}
    for r in rows:
        task_counts.setdefault((r["task"], r["label"]), 0)
        task_counts[(r["task"], r["label"])] += 1
    print("\nTask x Label breakdown (check this for the confound we found earlier):")
    for (task, label), count in sorted(task_counts.items()):
        print(f"  task={task:<10} label={label}  n={count}")


if __name__ == "__main__":
    main()
