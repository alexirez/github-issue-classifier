"""Builds the model input text for each issue: raw and cleaned versions.

python src/cleaning/clean_issues.py

Reads   data/processed/split_issues.jsonl
Writes  data/processed/cleaned_issues.jsonl   (every field of the input row, plus
                                               text_raw and text_clean)

text_raw   = title, blank line, body (only line endings and outer whitespace tidied)
text_clean = same, after: leading title prefixes removed ("[BUG] ", "docs(router): ",
             "Security: "), fenced code blocks -> [CODE], URLs -> [URL]
An empty body gives just the title. Row order is the same as the input file.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"

FENCE_RE = re.compile(r"(```|~~~).*?(?:\1|\Z)", re.DOTALL)   # an unclosed fence runs to the end
URL_RE = re.compile(r"https?://[^\s<>)\]\"']+")
TITLE_TAG_RE = re.compile(r"^\s*(?:\[[^\]\n]{1,40}\]\s*[:\-–—]?\s*)+")
# "type: ", "type(scope): ", "type/scope: " at the start of a title. The colon is required.
PREFIX_RE = re.compile(
    r"^\s*(?:feature[ _-]request|feat|feature|bugfix|bug|fix|documentation|docs?|"
    r"style|refactor|performance|perf|tests?|testing|build|ci|chore|revert|security|"
    r"question|enhancement|proposal|idea|deps|dependencies)"
    r"(?:\s*\([^)\n]{1,40}\)|/[\w.\-]{1,30})?!?\s*:\s*",
    re.IGNORECASE,
)
BLANKS_RE = re.compile(r"\n{3,}")


def normalize(s):
    """None -> '', Windows line endings -> '\n', outer whitespace trimmed."""
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def join(title, body):
    return f"{title}\n\n{body}" if body else title


def clean_title(title):
    """Drop leading [TAG] and type(scope): prefixes. A title that is only a prefix is left as it was."""
    stripped = TITLE_TAG_RE.sub("", title, count=1)
    stripped = PREFIX_RE.sub("", stripped, count=1).strip()
    return stripped or title


def clean_body(body):
    body = FENCE_RE.sub("[CODE]", body)    # before URLs, so links inside code vanish too
    body = URL_RE.sub("[URL]", body)
    return BLANKS_RE.sub("\n\n", body).strip()


def clean_issue(title, body):
    """Return (text_raw, text_clean) for one issue."""
    title, body = normalize(title), normalize(body)
    clean = join(URL_RE.sub("[URL]", clean_title(title)), clean_body(body))
    return join(title, body), clean


def main():
    src = PROC_DIR / "split_issues.jsonl"
    if not src.exists():
        print("Run src/training/split_data.py first.")
        return

    stats = Counter()
    with open(src, encoding="utf-8") as fin, \
         open(PROC_DIR / "cleaned_issues.jsonl", "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            title, body = normalize(r["title"]), normalize(r["body"])
            raw, clean = clean_issue(r["title"], r["body"])

            stats["issues"] += 1
            stats["empty body"] += not body
            stats["title prefix removed"] += clean_title(title) != title
            stats["code block replaced"] += bool(FENCE_RE.search(body))
            stats["URL replaced"] += bool(URL_RE.search(title + "\n" + body))
            stats["text changed by cleaning"] += raw != clean

            fout.write(json.dumps({**r, "text_raw": raw, "text_clean": clean}) + "\n")

    print("Cleaning summary:")
    for k, v in stats.items():
        pct = "" if k == "issues" else f"  ({100 * v / stats['issues']:.1f}%)"
        print(f"  {v:>7}  {k}{pct}")
    print(f"\nWrote {PROC_DIR / 'cleaned_issues.jsonl'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()