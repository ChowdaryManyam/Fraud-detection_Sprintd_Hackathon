
import csv
import json
from pathlib import Path

PROJECT_DIR = Path.home() / "stt-project"
TRANSCRIPT_DIR = PROJECT_DIR / "transcripts"
OUTPUT_FILE = PROJECT_DIR / "segment_index.csv"

rows = []
json_files = sorted(
    p for p in TRANSCRIPT_DIR.glob("*.json")
    if p.name != "failed_files.json"
)
errors = []

for path in json_files:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))

        audio_file = data.get("audio_file", "")
        model = data.get("model", "")
        language = data.get("language_detected", "")
        duration = data.get("duration_seconds")

        segments = data.get("segments", [])
        if not isinstance(segments, list):
            raise ValueError("'segments' is not a list")

        for index, segment in enumerate(segments, start=1):
            start = segment.get("start")
            end = segment.get("end")
            text = str(segment.get("text", "")).strip()

            if start is None or end is None:
                errors.append(f"{path.name}: segment {index} missing timestamp")
                continue

            if float(end) < float(start):
                errors.append(f"{path.name}: segment {index} ends before it starts")
                continue

            rows.append({
                "audio_file": audio_file,
                "json_file": path.name,
                "segment_id": index,
                "start_seconds": round(float(start), 3),
                "end_seconds": round(float(end), 3),
                "segment_duration_seconds": round(float(end) - float(start), 3),
                "text": text,
                "language_detected": language,
                "model": model,
                "audio_duration_seconds": duration,
            })

    except Exception as exc:
        errors.append(f"{path.name}: {exc}")

fieldnames = [
    "audio_file",
    "json_file",
    "segment_id",
    "start_seconds",
    "end_seconds",
    "segment_duration_seconds",
    "text",
    "language_detected",
    "model",
    "audio_duration_seconds",
]

with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"JSON files found: {len(json_files)}")
print(f"Timestamped segments indexed: {len(rows)}")
print(f"Errors: {len(errors)}")
print(f"CSV created: {OUTPUT_FILE}")

if errors:
    print("\nFirst errors:")
    for error in errors[:20]:
        print("-", error)

print("\nAudio files represented:")
print(len({row['audio_file'] for row in rows if row['audio_file']}))
