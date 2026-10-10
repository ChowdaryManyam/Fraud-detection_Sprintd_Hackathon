
from pathlib import Path
from jiwer import wer
import re

BASE = Path.home() / "stt-project"
REF = BASE / "data/Transkript/Stufe1_D02.txt"

HYPOTHESES = [
    BASE / "transcripts/Stufe1_D02-K1.txt",
    BASE / "transcripts/Stufe1_D02-K2.txt",
]

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())

reference = normalize(REF.read_text(encoding="utf-8"))

for path in HYPOTHESES:
    hypothesis = normalize(path.read_text(encoding="utf-8"))
    score = wer(reference, hypothesis)

    print(f"\nFile: {path.name}")
    print(f"WER: {score:.2%}")
    print(f"Word accuracy estimate: {max(0, 1-score):.2%}")
