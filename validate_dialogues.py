
from pathlib import Path
import json

BASE_DIR = Path.home() / "stt-project"
INPUT_DIR = BASE_DIR / "parsed_dialogues"

VALID_ROLES = {
    "Beratung": "employee",
    "Kundin": "customer",
}

def validate(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    turns = data.get("turns", [])
    ids = [t.get("turn_id") for t in turns]

    if len(ids) != len(set(ids)):
        errors.append("Duplicate turn IDs")

    if data.get("turn_count") != len(turns):
        errors.append("Incorrect turn_count")

    for index, turn in enumerate(turns, start=1):
        speaker = turn.get("speaker")
        expected_role = VALID_ROLES.get(speaker)

        if expected_role is None:
            errors.append(f"{turn.get('turn_id')}: invalid speaker")

        elif turn.get("speaker_role") != expected_role:
            errors.append(f"{turn.get('turn_id')}: incorrect speaker role")

        if not turn.get("text", "").strip():
            errors.append(f"{turn.get('turn_id')}: empty text")

        tokens = turn.get("tokens", [])
        if not tokens:
            errors.append(f"{turn.get('turn_id')}: empty tokens")

        if turn.get("token_count") != len(tokens):
            errors.append(f"{turn.get('turn_id')}: incorrect token_count")

        expected_id = f"T{index:03d}"
        if turn.get("turn_id") != expected_id:
            errors.append(
                f"Expected {expected_id}, got {turn.get('turn_id')}"
            )

    if data.get("parse_errors"):
        errors.append("Parser reported errors")

    return errors, len(turns)


def main():
    files = sorted(
        p for p in INPUT_DIR.glob("*.json")
        if p.name != "all_dialogues.json"
    )

    total_turns = 0
    failed = 0

    for path in files:
        errors, count = validate(path)
        total_turns += count

        if errors:
            failed += 1
            print(f"\nFAIL: {path.name}")
            for error in errors:
                print(" -", error)
        else:
            print(f"PASS: {path.name} ({count} turns)")

    print("\nValidation summary")
    print("Dialogue files:", len(files))
    print("Total turns:", total_turns)
    print("Failed dialogue files:", failed)
    print("Result:", "PASS" if files and failed == 0 else "CHECK ERRORS")


if __name__ == "__main__":
    main()
