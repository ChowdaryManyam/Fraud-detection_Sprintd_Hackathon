
from pathlib import Path
import json

BASE = Path.home() / "stt-project"
REF_DIR = BASE / "data" / "Transkript"
HYP_DIR = BASE / "transcripts"
REPORT = BASE / "wer_report.json"


def main():
    if not REF_DIR.is_dir():
        raise SystemExit(f"Reference directory not found: {REF_DIR}")

    if not HYP_DIR.is_dir():
        raise SystemExit(f"Transcript directory not found: {HYP_DIR}")

    references = sorted(REF_DIR.glob("*.txt"))
    hypotheses = sorted(HYP_DIR.glob("*.txt"))

    if not references:
        raise SystemExit("No reference TXT files found.")

    if not hypotheses:
        raise SystemExit("No generated TXT files found.")

    hypothesis_stems = {p.stem: p for p in hypotheses}
    matches = []
    unmatched = []

    for ref in references:
        exact = hypothesis_stems.get(ref.stem)
        segmented = sorted(HYP_DIR.glob(f"{ref.stem}-K*.txt"))

        if exact:
            candidates = [exact]
            match_type = "exact filename"
        elif segmented:
            candidates = segmented
            match_type = "segmented filename candidates"
        else:
            unmatched.append(ref.name)
            continue

        matches.append({
            "reference": ref.name,
            "hypotheses": [p.name for p in candidates],
            "match_type": match_type,
            "wer": None,
            "status": "alignment must be verified before scoring"
        })

    report = {
        "reference_count": len(references),
        "generated_transcript_count": len(hypotheses),
        "candidate_match_count": len(matches),
        "unmatched_references": unmatched,
        "matches": matches,
        "note": (
            "WER has not been calculated. Confirm that each generated "
            "transcript contains the same speech as its reference first."
        )
    }

    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Reference files:", len(references))
    print("Generated transcript files:", len(hypotheses))
    print("Candidate matches:", len(matches))
    print("Unmatched references:", unmatched)

    print("\nCandidate mappings:")
    for item in matches:
        print(
            f"{item['reference']} -> "
            f"{', '.join(item['hypotheses'])} "
            f"({item['match_type']})"
        )

    print("\nWER not calculated until alignment is verified.")
    print("Report saved to:", REPORT)


if __name__ == "__main__":
    main()
