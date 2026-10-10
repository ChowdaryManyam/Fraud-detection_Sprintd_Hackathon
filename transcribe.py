
from pathlib import Path
import json
import logging

from faster_whisper import WhisperModel

BASE_DIR = Path.home() / "stt-project"
AUDIO_DIR = BASE_DIR / "data" / "Audio"
OUTPUT_DIR = BASE_DIR / "transcripts"

MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "int8"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    audio_files = sorted(
        p for p in AUDIO_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}
    )

    if not audio_files:
        raise SystemExit(f"No audio files found in {AUDIO_DIR}")

    logging.info("Loading Whisper %s on CPU...", MODEL_SIZE)
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type=COMPUTE_TYPE,
        cpu_threads=8,
    )

    succeeded = 0
    failed = []

    for index, audio_path in enumerate(audio_files, 1):
        txt_path = OUTPUT_DIR / f"{audio_path.stem}.txt"
        json_path = OUTPUT_DIR / f"{audio_path.stem}.json"

        if txt_path.exists() and json_path.exists():
            logging.info("[%d/%d] Already done: %s",
                         index, len(audio_files), audio_path.name)
            succeeded += 1
            continue

        try:
            logging.info("[%d/%d] Transcribing %s",
                         index, len(audio_files), audio_path.name)

            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                vad_filter=True,
            )

            segment_data = []
            text_parts = []

            for segment in segments:
                text_parts.append(segment.text.strip())
                segment_data.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                })

            transcript = " ".join(part for part in text_parts if part).strip()

            txt_path.write_text(transcript + "\n", encoding="utf-8")
            json_path.write_text(
                json.dumps({
                    "audio_file": audio_path.name,
                    "model": MODEL_SIZE,
                    "language_detected": info.language,
                    "language_probability": info.language_probability,
                    "duration_seconds": info.duration,
                    "transcript": transcript,
                    "segments": segment_data,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            succeeded += 1
            logging.info("Saved %s", txt_path.name)

        except Exception as exc:
            failed.append({"file": audio_path.name, "error": str(exc)})
            logging.exception("Failed: %s", audio_path.name)

    (OUTPUT_DIR / "failed_files.json").write_text(
        json.dumps(failed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nFinished: {succeeded}/{len(audio_files)} files processed or skipped.")
    print(f"Failed files: {len(failed)}")
    print(f"Transcripts: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
