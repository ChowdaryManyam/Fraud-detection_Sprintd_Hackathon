
import json
from pathlib import Path
from faster_whisper import WhisperModel

ROOT = Path.home() / "stt-project"
AUDIO = ROOT / "data" / "Audio" / "Stufe2_C15-K3.wav"
OUTPUT = ROOT / "stt_evaluation" / "Stufe2_C15-K3_no_conditioning.json"

model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8",
    cpu_threads=8,
)

segments, info = model.transcribe(
    str(AUDIO),
    beam_size=5,
    vad_filter=True,
    condition_on_previous_text=False,
)

segment_rows = []
for seg in segments:
    segment_rows.append({
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "text": seg.text.strip(),
    })

result = {
    "audio_file": AUDIO.name,
    "model": "large-v3",
    "language_detected": info.language,
    "language_probability": info.language_probability,
    "duration_seconds": info.duration,
    "settings": {
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
    },
    "transcript": " ".join(s["text"] for s in segment_rows),
    "segments": segment_rows,
}

OUTPUT.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Saved:", OUTPUT)
print("Detected language:", info.language)
print("Segments:", len(segment_rows))
print("\nTimestamped transcript:")
for seg in segment_rows:
    print(f"[{seg['start']:7.2f}-{seg['end']:7.2f}s] {seg['text']}")
