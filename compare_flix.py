
import json
import re
from pathlib import Path
from jiwer import wer

ROOT = Path(".")
STEM = "Stufe2_C15-K3"
REFERENCE_JSON = ROOT / "parsed_dialogues" / "Stufe2_C15.json"

TRANSCRIPTS = {
    "Original large-v3": ROOT / "transcripts" / f"{STEM}.txt",
    "large-v3 no conditioning": (
        ROOT / "stt_evaluation" / f"{STEM}_no_conditioning.json"
    ),
    "Flix-AI": ROOT / "stt_evaluation" / f"{STEM}_flix.txt",
}

def normalize(text):
    # Lowercase and tokenize words/numbers, preserving German characters.
    return re.findall(r"[a-zäöüß]+|\d+(?:[.,'’]\d+)*", text.lower())

def read_reference(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = data["turns"]
    return " ".join(
        turn["text"] for turn in turns if turn.get("text")
    )

def read_transcript(path):
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        data = json.loads(text)
        return data["transcript"]

    return text.strip()

reference = read_reference(REFERENCE_JSON)
ref_tokens = normalize(reference)

if not ref_tokens:
    raise ValueError("Reference contains no words.")

print(f"Reference file: {REFERENCE_JSON}")
print(f"Reference word count: {len(ref_tokens)}")
print("\n--- WER comparison ---")

for name, path in TRANSCRIPTS.items():
    if not path.exists():
        print(f"{name}: FILE NOT FOUND: {path}")
        continue

    hypothesis = read_transcript(path)
    hyp_tokens = normalize(hypothesis)

    score = wer(
        " ".join(ref_tokens),
        " ".join(hyp_tokens),
    )

    print(
        f"{name:28s} "
        f"WER: {score * 100:7.2f}% | "
        f"words: {len(hyp_tokens):4d}"
    )

print(
    "\nNote: These are reference-script comparison scores, "
    "not verified true transcription accuracy."
)
