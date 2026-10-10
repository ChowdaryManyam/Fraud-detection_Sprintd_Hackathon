
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from jiwer import wer

ROOT = Path.home() / "stt-project"
AUDIO_DIR = ROOT / "data" / "Audio"
REFERENCE_DIR = ROOT / "parsed_dialogues"
TRANSCRIPT_DIR = ROOT / "transcripts"
REPORT_DIR = ROOT / "stt_evaluation"

REPORT_DIR.mkdir(exist_ok=True)


def normalize(text):
    """Lowercase and tokenize words while preserving umlauts and accents."""
    text = text.lower().replace("’", "'")
    return re.findall(r"\w+(?:['']\w+)*", text, flags=re.UNICODE)


def normalized_text(text):
    return " ".join(normalize(text))


# Build a reference lookup keyed by filename stem.
references = {}
for path in sorted(REFERENCE_DIR.glob("*.json")):
    if path.name == "all_dialogues.json":
        continue

    data = json.loads(path.read_text(encoding="utf-8"))
    turns = data.get("turns", [])
    references[path.stem] = " ".join(
        turn.get("text", "").strip()
        for turn in turns
        if turn.get("text", "").strip()
    )

rows = []
missing_references = []
missing_transcripts = []

for audio_path in sorted(AUDIO_DIR.glob("*.wav")):
    # Example: Stufe1_D02-K1 -> Stufe1_D02
    reference_stem = audio_path.stem.rsplit("-", 1)[0]
    reference = references.get(reference_stem)

    transcript_path = TRANSCRIPT_DIR / f"{audio_path.stem}.txt"

    if reference is None:
        missing_references.append(audio_path.name)
        continue

    if not transcript_path.exists():
        missing_transcripts.append(audio_path.name)
        continue

    hypothesis = transcript_path.read_text(encoding="utf-8").strip()
    reference_norm = normalized_text(reference)
    hypothesis_norm = normalized_text(hypothesis)

    if not reference_norm:
        missing_references.append(audio_path.name)
        continue

    score = wer(reference_norm, hypothesis_norm)

    rows.append({
        "audio_file": audio_path.name,
        "reference_file": f"{reference_stem}.json",
        "reference_words": len(reference_norm.split()),
        "hypothesis_words": len(hypothesis_norm.split()),
        "reference_wer": round(score, 4),
        "reference_wer_percent": round(score * 100, 2),
    })

# Write per-recording results.
csv_path = REPORT_DIR / "recording_results.csv"
fields = [
    "audio_file",
    "reference_file",
    "reference_words",
    "hypothesis_words",
    "reference_wer",
    "reference_wer_percent",
]

with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

# Summarize by level. These are descriptive averages, not ground-truth metrics.
by_level = defaultdict(list)
for row in rows:
    level = row["audio_file"].split("_", 1)[0]
    by_level[level].append(row["reference_wer"])

summary = {
    "evaluation_type": "Reference-text comparison; scripts are not verified ground truth",
    "normalization": "Lowercase; compare Unicode word tokens; punctuation excluded",
    "recordings_evaluated": len(rows),
    "missing_references": missing_references,
    "missing_transcripts": missing_transcripts,
    "mean_recording_wer_percent": (
        round(sum(r["reference_wer"] for r in rows) / len(rows) * 100, 2)
        if rows else None
    ),
    "by_level_mean_recording_wer_percent": {
        level: round(sum(scores) / len(scores) * 100, 2)
        for level, scores in sorted(by_level.items())
    },
    "results_csv": str(csv_path),
}

summary_path = REPORT_DIR / "summary.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("STT reference-text evaluation complete")
print("Recordings evaluated:", len(rows))
print("Missing references:", len(missing_references))
print("Missing transcripts:", len(missing_transcripts))
print("Mean recording WER (%):", summary["mean_recording_wer_percent"])

print("\nMean WER by level (%):")
for level, score in summary["by_level_mean_recording_wer_percent"].items():
    print(f"  {level}: {score}")

print("\nResults:", csv_path)
print("Summary:", summary_path)
print("\nIMPORTANT: These are reference-text comparison scores,")
print("not verified measures of true transcription accuracy.")
