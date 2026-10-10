
from pathlib import Path
import json
import re

BASE_DIR = Path.home() / "stt-project"
INPUT_DIR = BASE_DIR / "data" / "Transkript"
OUTPUT_DIR = BASE_DIR / "parsed_dialogues"

# Standard format: T001 · Beratung
STANDARD_HEADER = re.compile(
    r"^\s*(T\d+)\s*[·•]\s*(Beratung|Kundin)\s*$",
    re.IGNORECASE,
)

# M-dialogue format: T001 B: utterance / T002 K: utterance
M_HEADER = re.compile(
    r"^\s*(T\d+)\s+([BK])\s*:\s*(.*)$",
    re.IGNORECASE,
)

WORD_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)

ROLE_MAP = {
    "beratung": ("Beratung", "employee"),
    "kundin": ("Kundin", "customer"),
    "b": ("Beratung", "employee"),
    "k": ("Kundin", "customer"),
}


def parse_file(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    turns = []
    errors = []
    current = None

    def finish_turn():
        nonlocal current
        if current is None:
            return

        utterance = " ".join(current["text_parts"]).strip()
        speaker, role = ROLE_MAP[current["speaker_key"]]
        tokens = WORD_RE.findall(utterance)

        turns.append({
            "turn_id": current["turn_id"],
            "speaker": speaker,
            "speaker_role": role,
            "text": utterance,
            "tokens": tokens,
            "token_count": len(tokens),
        })
        current = None

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        standard = STANDARD_HEADER.match(line)
        m_header = M_HEADER.match(line)

        if standard:
            finish_turn()
            current = {
                "turn_id": standard.group(1),
                "speaker_key": standard.group(2).lower(),
                "text_parts": [],
            }
        elif m_header:
            finish_turn()
            current = {
                "turn_id": m_header.group(1),
                "speaker_key": m_header.group(2).lower(),
                "text_parts": [],
            }
            if m_header.group(3).strip():
                current["text_parts"].append(m_header.group(3).strip())
        elif current is not None:
            # Keep wrapped lines within the current utterance.
            current["text_parts"].append(line)
        else:
            # Introductory metadata in M files is intentionally ignored.
            # Unexpected text in a standard transcript is reported.
            if not path.stem.endswith(("M05", "M08", "M01", "M02", "M03", "M04")):
                errors.append({
                    "line": line_number,
                    "text": line,
                    "reason": "Text before first recognised turn header",
                })

    finish_turn()

    return {
        "source_file": path.name,
        "turn_count": len(turns),
        "token_count": sum(t["token_count"] for t in turns),
        "turns": turns,
        "parse_errors": errors,
    }


def main():
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Input directory not found: {INPUT_DIR}")

    files = sorted(INPUT_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dialogues = []
    total_errors = 0

    for path in files:
        dialogue = parse_file(path)
        all_dialogues.append(dialogue)
        total_errors += len(dialogue["parse_errors"])

        output_path = OUTPUT_DIR / f"{path.stem}.json"
        output_path.write_text(
            json.dumps(dialogue, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"{path.name}: {dialogue['turn_count']} turns, "
            f"{dialogue['token_count']} tokens, "
            f"{len(dialogue['parse_errors'])} parse errors"
        )

    combined_path = OUTPUT_DIR / "all_dialogues.json"
    combined_path.write_text(
        json.dumps(all_dialogues, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nDialogues processed: {len(all_dialogues)}")
    print(f"Total parse errors: {total_errors}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Combined JSON: {combined_path}")


if __name__ == "__main__":
    main()
