# Swiss German STT — CallGuard

Speech-to-text pipeline and evaluation experiments for Swiss German banking-call recordings.

## Scope

This component focuses on transcription, transcript processing, timestamp indexing, and transcription-quality evaluation.

Fraud detection and compliance classification are outside my component's scope.

## Features

- Audio transcription using Faster-Whisper / Whisper large-v3
- Reference dialogue parsing and validation
- Timestamped segment indexing
- Word Error Rate (WER) evaluation
- Alternative decoding experiments
- Experimental evaluation of the Flix-AI Swiss German Whisper model

## Main scripts

| Script | Purpose |
|---|---|
| transcribe.py | Main transcription pipeline |
| parse_dialogues.py | Parse reference dialogues |
| validate_dialogues.py | Validate parsed dialogues |
| build_segment_index.py | Build timestamped segment index |
| evaluate_stt.py | Evaluate transcription results |
| evaluate_wer.py | WER evaluation |
| evaluate_no_conditioning_batch.py | Batch decoding experiment |
| test_decode_options.py | Test decoding settings |
| test_decode_batch.py | Batch decoding experiments |
| test_flix.py | Test Flix-AI |
| test_flix_batch.py | Batch Flix-AI experiments |
| compare_flix.py | Compare model outputs |
| test_wer.py | WER tests |

## Setup

Create a virtual environment and install dependencies:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Evaluation

Word Error Rate (WER) is used to compare generated transcripts with the provided reference scripts.

These scores are preliminary reference-text comparisons, not verified measurements of true transcription accuracy.

## Status

Draft submission — active development.
