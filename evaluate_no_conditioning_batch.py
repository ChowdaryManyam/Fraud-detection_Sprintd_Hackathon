
import csv
import json
import re
import time
from pathlib import Path

from faster_whisper import WhisperModel
from jiwer import wer

ROOT = Path(".")
AUDIO_DIR = ROOT / "data" / "Audio"
ORIGINAL_DIR = ROOT / "transcripts"
REFERENCE_DIR = ROOT / "parsed_dialogues"
OUTPUT_DIR = ROOT / "stt_evaluation" / "no_conditioning_batch"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize(text):
    return re.findall(
        r"[a-zäöüß]+|\d+(?:[.,'’]\d+)*",
        text.lower(),
    )

def read_reference(stem):
    # Recording variants such as -K1/-K2 share a dialogue reference.
    ref_stem = stem.rsplit("-K", 1)[0]
    path = REFERENCE_DIR / f"{ref_stem}.json"

    data = json.loads(path.read_text(encoding="utf-8"))
    return " ".join(
        turn["text"]
        for turn in data["turns"]
        if turn.get("text")
    )

def read_original(stem):
    path = ORIGINAL_DIR / f"{stem}.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()

audio_files = sorted(AUDIO_DIR.glob("*.wav"))

if not audio_files:
    raise SystemExit(f"No WAV files found in {AUDIO_DIR}")

print(f"Found {len(audio_files)} WAV recordings.")
print("Loading large-v3 on CPU (int8)...")

model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8",
    cpu_threads=8,
)

results = []
start_all = time.time()

for index, audio_path in enumerate(audio_files, start=1):
    stem = audio_path.stem
    output_path = OUTPUT_DIR / f"{stem}.json"

    print(f"\n[{index}/{len(audio_files)}] {stem}")

    # Reuse an already completed output if the script is restarted.
    if output_path.exists():
        try:
            result_data = json.loads(
                output_path.read_text(encoding="utf-8")
            )
            if result_data.get("transcript") is not None:
                print("Using previously saved alternative transcript.")
            else:
                result_data = None
        except (json.JSONDecodeError, OSError):
            result_data = None
    else:
        result_data = None

    if result_data is None:
        start = time.time()

        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        segment_data = []
        transcript_parts = []

        for segment in segments:
            segment_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            })
            transcript_parts.append(segment.text)

        transcript = "".join(transcript_parts).strip()
        elapsed = time.time() - start

        result_data = {
            "audio_file": audio_path.name,
            "model": "large-v3",
            "settings": {
                "beam_size": 5,
                "vad_filter": True,
                "condition_on_previous_text": False,
            },
            "language_detected": info.language,
            "language_probability": info.language_probability,
            "duration_seconds": info.duration,
            "inference_seconds": round(elapsed, 2),
            "transcript": transcript,
            "segments": segment_data,
        }

        output_path.write_text(
            json.dumps(result_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"Transcribed in {elapsed:.1f}s; "
            f"{len(normalize(transcript))} words."
        )

    reference = read_reference(stem)
    reference_tokens = normalize(reference)
    alternative = result_data["transcript"]
    alternative_tokens = normalize(alternative)
    original = read_original(stem)

    alternative_wer = (
        100 * wer(
            " ".join(reference_tokens),
            " ".join(alternative_tokens),
        )
        if reference_tokens else None
    )

    original_wer = None
    if original is not None and reference_tokens:
        original_wer = 100 * wer(
            " ".join(reference_tokens),
            " ".join(normalize(original)),
        )

    results.append({
        "recording": stem,
        "level": stem.split("_", 1)[0],
        "reference_words": len(reference_tokens),
        "original_words": len(normalize(original)) if original is not None else "",
        "no_conditioning_words": len(alternative_tokens),
        "original_wer_percent": round(original_wer, 2) if original_wer is not None else "",
        "no_conditioning_wer_percent": round(alternative_wer, 2) if alternative_wer is not None else "",
        "inference_seconds": result_data.get("inference_seconds", ""),
    })

csv_path = OUTPUT_DIR / "recording_comparison.csv"

fieldnames = [
    "recording",
    "level",
    "reference_words",
    "original_words",
    "no_conditioning_words",
    "original_wer_percent",
    "no_conditioning_wer_percent",
    "inference_seconds",
]

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

valid = [
    row for row in results
    if isinstance(row["no_conditioning_wer_percent"], (int, float))
]
original_valid = [
    row for row in results
    if isinstance(row["original_wer_percent"], (int, float))
]

print("\n=== BATCH SUMMARY ===")
print(f"Recordings found: {len(audio_files)}")
print(f"Recordings evaluated: {len(valid)}")
print(f"Total elapsed time: {(time.time() - start_all) / 60:.1f} minutes")

if valid:
    mean_alt = sum(
        row["no_conditioning_wer_percent"] for row in valid
    ) / len(valid)
    print(f"Mean no-conditioning WER: {mean_alt:.2f}%")

if original_valid:
    mean_orig = sum(
        row["original_wer_percent"] for row in original_valid
    ) / len(original_valid)
    print(f"Mean original WER: {mean_orig:.2f}%")

print("\nMean WER by level:")
for level in sorted({row["level"] for row in valid}):
    level_rows = [
        row for row in valid if row["level"] == level
    ]
    mean_level = sum(
        row["no_conditioning_wer_percent"] for row in level_rows
    ) / len(level_rows)
    print(f"  {level}: {mean_level:.2f}% ({len(level_rows)} recordings)")

print(f"\nCSV comparison: {csv_path}")
print(f"Individual alternative transcripts: {OUTPUT_DIR}")
print(
    "\nNOTE: These are reference-script comparison scores, "
    "not verified true transcription accuracy."
)
