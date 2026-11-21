"""Builds the model input text for each issue: raw, cleaned, and template-free versions.

python src/cleaning/clean_issues.py

Reads   data/processed/split_issues.jsonl
Writes  data/processed/cleaned_issues.jsonl   (every field of the input row, plus
                                               text_raw, text_clean, text_notemplate)
        data/analysis/boilerplate_lines.csv   (every line treated as template text, per repo)

text_raw        = title, blank line, body (only line endings and outer whitespace tidied)
text_clean      = after: leading title prefixes removed ("[BUG] ", "docs(router): ",
                  "Security: "), fenced code blocks -> [CODE], URLs -> [URL]
text_notemplate = text_clean minus boilerplate lines. A line is boilerplate when it
                  (normalized: lowercase, whitespace collapsed) appears in at least
                  MIN_REPEATS issues AND at least MIN_SHARE of the issues of the same repo.
                  Templates are found from the data itself: no labels, no hand-written list.
An empty body gives just the title. Row order is the same as the input file.
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "analysis"

# --- template detection settings (tune these after reading boilerplate_lines.csv) ---
MIN_LINE_CHARS = 8    # shorter lines ("no", "yes", "---") are never treated as boilerplate
MIN_REPEATS = 5       # a line must appear in at least this many issues of its repo...
MIN_SHARE = 0.01      # ...and in at least this share of that repo's issues

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


# ---------------------------------------------------------------------------
# Rule-based cleaning
# ---------------------------------------------------------------------------
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


def clean_parts(title, body):
    """Return (cleaned title, cleaned body) for one issue."""
    title, body = normalize(title), normalize(body)
    return URL_RE.sub("[URL]", clean_title(title)), clean_body(body)


# ---------------------------------------------------------------------------
# Template (boilerplate) detection: lines that repeat across a repo's issues
# ---------------------------------------------------------------------------
def norm_line(line):
    return " ".join(line.lower().split())


def candidate_lines(body):
    """Set of normalized lines of a cleaned body that could be boilerplate."""
    out = set()
    for line in body.split("\n"):
        n = norm_line(line)
        if len(n) >= MIN_LINE_CHARS:
            out.add(n)
    return out


def find_boilerplate(bodies_by_repo):
    """repo -> list of cleaned bodies. Returns repo -> {boilerplate line: issues containing it}."""
    result = {}
    for repo, bodies in bodies_by_repo.items():
        counts = Counter()
        for b in bodies:
            counts.update(candidate_lines(b))      # a set, so each issue counts a line once
        need = max(MIN_REPEATS, MIN_SHARE * len(bodies))
        result[repo] = {line: c for line, c in counts.items() if c >= need}
    return result


def strip_boilerplate(body, boiler):
    kept = [ln for ln in body.split("\n") if norm_line(ln) not in boiler]
    return BLANKS_RE.sub("\n\n", "\n".join(kept)).strip()


def write_boilerplate_report(boiler, n_issues):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "boilerplate_lines.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repo", "repo_issues", "issues_with_line", "line"])
        for repo in sorted(boiler):
            for line, c in sorted(boiler[repo].items(), key=lambda kv: -kv[1]):
                w.writerow([repo, n_issues[repo], c, line])
    return out


# ---------------------------------------------------------------------------
def main():
    src = PROC_DIR / "split_issues.jsonl"
    if not src.exists():
        print("Run src/training/split_data.py first.")
        return

    with open(src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    # Pass 1: rule-based cleaning; collect each repo's cleaned bodies.
    cleaned = []
    bodies_by_repo = defaultdict(list)
    for r in rows:
        title, body = clean_parts(r["title"], r["body"])
        cleaned.append((title, body))
        bodies_by_repo[r["repo"]].append(body)

    # Pass 2: find each repo's template lines, then write all three text versions.
    boiler = find_boilerplate(bodies_by_repo)
    n_issues = {repo: len(b) for repo, b in bodies_by_repo.items()}

    stats = Counter()
    with open(PROC_DIR / "cleaned_issues.jsonl", "w", encoding="utf-8") as fout:
        for r, (title, body) in zip(rows, cleaned):
            title_n, body_n = normalize(r["title"]), normalize(r["body"])
            raw = join(title_n, body_n)
            clean = join(title, body)
            notemplate = join(title, strip_boilerplate(body, boiler[r["repo"]]))

            stats["issues"] += 1
            stats["empty body"] += not body_n
            stats["title prefix removed"] += clean_title(title_n) != title_n
            stats["code block replaced"] += bool(FENCE_RE.search(body_n))
            stats["URL replaced"] += bool(URL_RE.search(title_n + "\n" + body_n))
            stats["text changed by cleaning"] += raw != clean
            stats["template lines removed"] += clean != notemplate

            fout.write(json.dumps({**r, "text_raw": raw, "text_clean": clean,
                                   "text_notemplate": notemplate}) + "\n")

    report = write_boilerplate_report(boiler, n_issues)

    print("Cleaning summary:")
    for k, v in stats.items():
        pct = "" if k == "issues" else f"  ({100 * v / stats['issues']:.1f}%)"
        print(f"  {v:>7}  {k}{pct}")
    n_lines = sum(len(b) for b in boiler.values())
    n_repos = sum(1 for b in boiler.values() if b)
    print(f"\nTemplate lines found: {n_lines} distinct lines in {n_repos} of {len(boiler)} repos")
    print(f"Wrote {PROC_DIR / 'cleaned_issues.jsonl'}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()