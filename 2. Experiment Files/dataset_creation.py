import os
import csv
import re

# Resolve paths relative to this script (so it works no matter where you run it from)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "super_final_pics")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "brain_study_metadata.csv")

HEADERS = ["filename", "source", "gender", "emotion"]

# Split filename into tokens by any non-alphanumeric separator
TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def tokens_from_filename(filename: str) -> set[str]:
    base = os.path.splitext(filename)[0].lower()
    return {t for t in TOKEN_SPLIT_RE.split(base) if t}


def classify_source(toks: set[str]) -> str:
    # AI images contain token "ai" (works for AI/ai)
    return "ai" if "ai" in toks else "non-ai"


def classify_gender(toks: set[str]) -> str:
    # AI set uses exact tokens: man / woman
    if "man" in toks:
        return "male"
    if "woman" in toks:
        return "female"

    # Real set often uses tokens like man10 / woman7 (because numbers stick to the token)
    # Detect those too:
    if any(t.startswith("man") for t in toks):
        return "male"
    if any(t.startswith("woman") for t in toks):
        return "female"

    return "unknown"


def classify_emotion(toks: set[str]) -> str:
    # Happy: happy OR smiling
    if "happy" in toks or "smiling" in toks:
        return "happy"
    # Sad: sad
    if "sad" in toks:
        return "sad"
    return "unknown"


def main() -> None:
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(
            f"Image folder not found: {IMAGE_FOLDER}\n"
            "Make sure 'super_final_pictures' is in the same folder as this script."
        )

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)

        for filename in sorted(os.listdir(IMAGE_FOLDER)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            toks = tokens_from_filename(filename)
            source = classify_source(toks)
            gender = classify_gender(toks)
            emotion = classify_emotion(toks)

            writer.writerow([filename, source, gender, emotion])

    print(f"Created: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()