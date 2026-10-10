
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal
import torch
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)

MODEL_ID = "Flix-AI/flix-swissgerman-full"

ROOT = Path(".")
AUDIO_DIR = ROOT / "data" / "Audio"
REF_DIR = ROOT / "parsed_dialogues"
OUT_DIR = ROOT / "stt_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "Stufe1_D02-K1",
    "Stufe2_C10-K4",
    "Stufe3_C20-K4",
]

def normalize(text):
    return re.findall(r"[a-zäöüß]+|\d+(?:[.,'’]\d+)*", text.lower())

def read_reference(stem):
    ref_stem = stem.rsplit("-K", 1)[0]
    path = REF_DIR / f"{ref_stem}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return " ".join(
        turn["text"] for turn in data["turns"] if turn.get("text")
    )

def read_transcript(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)["transcript"]
    return text.strip()

def load_audio(path):
    audio, rate = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = np.gcd(rate, 16000)
        audio = scipy.signal.resample_poly(
            audio, 16000 // divisor, rate // divisor
        )
        rate = 16000
    return audio.astype(np.float32), rate

print("Loading Flix-AI processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading Flix-AI model on CPU...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, dtype=torch.float32
)
model.to("cpu")
model.eval()

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=-1,
)

results = []

for stem in FILES:
    audio_path = AUDIO_DIR / f"{stem}.wav"
    flix_path = OUT_DIR / f"{stem}_flix.txt"
    original_path = ROOT / "transcripts" / f"{stem}.txt"
    alt_path = OUT_DIR / f"{stem}_no_conditioning.json"

    print(f"\n--- Processing {stem} ---")

    if not audio_path.exists():
        print(f"Missing audio: {audio_path}")
        continue

    audio, rate = load_audio(audio_path)
    start = time.time()

    output = asr(
        {"raw": audio, "sampling_rate": rate},
        chunk_length_s=30,
        stride_length_s=5,
        generate_kwargs={"language": "de", "task": "transcribe"},
    )

    elapsed = time.time() - start
    flix_text = output["text"].strip()
    flix_path.write_text(flix_text + "\n", encoding="utf-8")

    reference = read_reference(stem)
    ref_tokens = normalize(reference)

    row = {
        "recording": stem,
        "reference_words": len(ref_tokens),
        "flix_words": len(normalize(flix_text)),
        "flix_wer_percent": round(
            100 * __import__("jiwer").wer(
                " ".join(ref_tokens),
                " ".join(normalize(flix_text)),
            ), 2
        ),
        "flix_seconds": round(elapsed, 1),
    }

    for label, path in [
        ("original", original_path),
        ("no_conditioning", alt_path),
    ]:
        if path.exists():
            hypothesis = read_transcript(path)
            row[f"{label}_words"] = len(normalize(hypothesis))
            row[f"{label}_wer_percent"] = round(
                100 * __import__("jiwer").wer(
                    " ".join(ref_tokens),
                    " ".join(normalize(hypothesis)),
                ), 2
            )
        else:
            row[f"{label}_words"] = ""
            row[f"{label}_wer_percent"] = ""

    results.append(row)

    print(f"Flix-AI WER: {row['flix_wer_percent']:.2f}%")
    print(f"Words: {row['flix_words']}")
    print(f"Inference time: {elapsed:.1f} seconds")
    print(f"Saved: {flix_path}")

csv_path = OUT_DIR / "flix_batch_comparison.csv"

if results:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nComparison saved to: {csv_path}")
    print("\n--- RESULTS ---")
    for row in results:
        print(
            f"{row['recording']}: "
            f"Original={row['original_wer_percent']}%, "
            f"No-conditioning={row['no_conditioning_wer_percent']}%, "
            f"Flix-AI={row['flix_wer_percent']}%, "
            f"Flix time={row['flix_seconds']}s"
        )
else:
    print("No recordings were processed.")
