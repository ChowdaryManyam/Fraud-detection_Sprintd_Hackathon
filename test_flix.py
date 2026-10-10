
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
AUDIO_PATH = Path("data/Audio/Stufe2_C15-K3.wav")
OUTPUT_PATH = Path("stt_evaluation/Stufe2_C15-K3_flix.txt")

def main():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Flix-AI processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading Flix-AI model on CPU...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()

    print(f"Loading audio: {AUDIO_PATH}")
    audio, sample_rate = sf.read(AUDIO_PATH)

    # Convert stereo audio to mono.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample to Whisper's expected 16 kHz if necessary.
    if sample_rate != 16000:
        divisor = np.gcd(sample_rate, 16000)
        audio = scipy.signal.resample_poly(
            audio,
            16000 // divisor,
            sample_rate // divisor,
        )
        sample_rate = 16000

    audio = audio.astype(np.float32)

    print("Setting up chunked ASR pipeline...")
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=-1,
    )

    print("Transcribing full recording in overlapping chunks...")
    start = time.time()

    result = asr(
        {
            "raw": audio,
            "sampling_rate": sample_rate,
        },
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
        generate_kwargs={
            "language": "de",
            "task": "transcribe",
        },
    )

    elapsed = time.time() - start
    transcript = result["text"].strip()

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        file.write(transcript + "\n")

    print("\n--- FLIX-AI TRANSCRIPT ---\n")
    print(transcript)

    print("\n--- VERIFICATION ---")
    print(f"Audio duration: {len(audio) / sample_rate:.1f} seconds")
    print(f"Transcript words: {len(transcript.split())}")
    print(f"Inference time: {elapsed:.1f} seconds")
    print(f"Saved transcript: {OUTPUT_PATH}")

    chunks = result.get("chunks", [])
    if chunks:
        print(f"Timestamped chunks: {len(chunks)}")
        print(f"Last chunk timestamp: {chunks[-1].get('timestamp')}")

if __name__ == "__main__":
    main()
