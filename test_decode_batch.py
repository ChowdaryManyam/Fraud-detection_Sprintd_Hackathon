
import json
import re
from pathlib import Path

from faster_whisper import WhisperModel
from jiwer import wer

ROOT = Path.home() / "stt-project"
AUDIO_DIR = ROOT / "data" / "Audio"
TRANSCRIPT_DIR = ROOT / "transcripts"
REFERENCE_DIR = ROOT / "parsed_dialogues"
OUTPUT_DIR = ROOT / "stt_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = [
    "Stufe1_D02-K1.wav",
    "Stufe2_C10-K4.wav",
    "Stufe3_C20-K4.wav",
]


def normalize(text):
    text = text.lower().replace("’", "'")
    return " ".join(re.findall(r"\w+(?:'\w+)*", text))


model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8",
    cpu_threads=8,
)

results = []

for audio_name in FILES:
    audio_path = AUDIO_DIR / audio_name
    stem = audio_path.stem
    reference_stem = stem.rsplit("-", 1)[0]

    reference_path = REFERENCE_DIR / f"{reference_stem}.json"
    original_path = TRANSCRIPT_DIR / f"{stem}.txt"

    if not audio_path.exists():
        print("SKIP: missing audio:", audio_name)
        continue
    if not reference_path.exists() or not original_path.exists():
        print("SKIP: missing reference or original transcript:", audio_name)
        continue

    ref_data = json.loads(reference_path.read_text(encoding="utf-8"))
    reference = " ".join(
        turn["text"].strip()
        for turn in ref_data.get("turns", [])
        if turn.get("text", "").strip()
    )

    original = original_path.read_text(encoding="utf-8")

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    segment_rows = [
        {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        }
        for seg in segments
    ]
    test_text = " ".join(seg["text"] for seg in segment_rows)

    original_wer = wer(normalize(reference), normalize(original))
    test_wer = wer(normalize(reference), normalize(test_text))

    output_path = OUTPUT_DIR / f"{stem}_no_conditioning.json"
    output_path.write_text(
        json.dumps({
            "audio_file": audio_name,
            "model": "large-v3",
            "language_detected": info.language,
            "language_probability": info.language_probability,
            "settings": {
                "beam_size": 5,
                "vad_filter": True,
                "condition_on_previous_text": False,
            },
            "transcript": test_text,
            "segments": segment_rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results.append({
        "audio_file": audio_name,
        "original_wer_percent": round(original_wer * 100, 2),
        "test_wer_percent": round(test_wer * 100, 2),
        "original_words": len(normalize(original).split()),
        "test_words": len(normalize(test_text).split()),
        "test_segments": len(segment_rows),
    })

    print(f"\n{audio_name}")
    print(f"  Original WER: {original_wer * 100:.2f}%")
    print(f"  Test WER:     {test_wer * 100:.2f}%")
    print(f"  Segments:     {len(segment_rows)}")

report_path = OUTPUT_DIR / "decode_batch_comparison.json"
report_path.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nSaved comparison:", report_path)
