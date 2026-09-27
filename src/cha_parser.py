"""
Parses raw CHAT (.cha) transcript files from DementiaBank/TalkBank.

Extracts only the PARTICIPANT's speech (skips *INV: investigator lines),
strips CHAT annotation codes, and returns plain lowercase text in the
same style as original_data.csv (space-separated words, hesitation
fillers like "uh"/"um" preserved).

Filename convention: "001-0.cha" -> speaker_id="001", visit=0
"""
import re
from pathlib import Path

# Lines that carry participant speech (PAR is standard; some corpora use PAR1/PAR2)
PARTICIPANT_TIERS = ("*PAR:", "*PAR1:", "*PAR2:")


def parse_filename(path: Path):
    """'001-0.cha' -> ('001', 0). Falls back gracefully if the pattern differs."""
    stem = path.stem
    match = re.match(r"(\w+)-(\d+)", stem)
    if match:
        return match.group(1), int(match.group(2))
    return stem, 0


def _clean_utterance(line: str) -> str:
    # Drop the speaker tier code itself, e.g. "*PAR:\t"
    line = re.sub(r"^\*\w+:\s*", "", line)

    # Timing marks: bullet \x15start_end\x15
    line = re.sub(r"\x15\d+_\d+\x15", "", line)

    # Fillers/hesitations: &-uh -> uh   (KEEP the word — this is the hesitation signal)
    line = re.sub(r"&-(\w+)", r"\1", line)
    # Word fragments/false starts: &+word -> drop
    line = re.sub(r"&\+\w+", "", line)

    # Retracing / repetition markers [/], [//], [///]
    line = re.sub(r"\[/+\]", "", line)
    # Any other bracketed annotation: [: replacement], [* error-code], [+ ...], etc.
    line = re.sub(r"\[[^\]]*\]", "", line)

    # Angle-bracket scope markers around repeated/corrected material: keep the words, drop < >
    line = line.replace("<", "").replace(">", "")

    # Pause markers (.) (..) (...)
    line = re.sub(r"\(\.{1,3}\)", "", line)

    # Non-verbal/paralinguistic tags like &=laughs, (coughs)
    line = re.sub(r"&=\w+", "", line)

    # Postcodes / terminators / special CHAT symbols
    line = re.sub(r"\+[.,\"<]+", "", line)
    line = re.sub(r"[\"]", "", line)

    # Standard sentence punctuation used as utterance terminators in CHAT
    line = re.sub(r"[.!?]+(\s|$)", " ", line)
    line = line.replace(",", " ")

    # Collapse whitespace
    line = re.sub(r"\s+", " ", line).strip()
    return line.lower()


def parse_cha_file(path: Path) -> str:
    """Return the cleaned, concatenated participant speech for one .cha file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    utterances = []
    capturing = False
    buffer = ""

    for line in lines:
        if line.startswith(PARTICIPANT_TIERS):
            if buffer:
                utterances.append(buffer)
            buffer = line
            capturing = True
        elif line.startswith("*") or line.startswith("%"):
            # a different speaker tier or a dependent tier (%mor, %gra) -> stop capturing
            if buffer:
                utterances.append(buffer)
            buffer = ""
            capturing = False
        elif capturing and line.startswith("\t"):
            # continuation line of the same participant utterance
            buffer += " " + line.strip()

    if buffer:
        utterances.append(buffer)

    cleaned = [_clean_utterance(u) for u in utterances]
    cleaned = [c for c in cleaned if c]
    return " ".join(cleaned)


if __name__ == "__main__":
    import sys
    sample = Path(sys.argv[1])
    speaker_id, visit = parse_filename(sample)
    print(f"speaker_id={speaker_id}  visit={visit}")
    print(parse_cha_file(sample))
