"""Rough, LLM-assisted leakage count over a sample of cleaned issues. A learning tool, not a result.

Build the paste file:
    python src/cleaning/rough_audit.py -n 100
Paste the WHOLE file into an LLM. Save its short reply as data/analysis/rough_count.txt, then:
    python src/cleaning/rough_audit.py --review [FILE]

Reads   data/processed/cleaned_issues.jsonl
Writes  data/analysis/rough_audit.txt   (instructions + one issue per line)

Sampling is class-balanced and spreads each class across repos (round-robin), so rare
classes and small repos appear instead of 58% bug and a few big repos. Seeded.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "analysis"
PASTE_FILE = OUT_DIR / "rough_audit.txt"
COUNT_FILE = OUT_DIR / "rough_count.txt"

CLASSES = "bug, feature, documentation, question, maintenance, security, performance, other, dependency"

# Suggested categories. The LLM may add more; the review marks those "(new)".
CATEGORIES = ["template_heading", "template_phrase", "title_prefix",
              "explicit_type_statement", "environment_block", "label_mismatch"]
STARTERS = set(CATEGORIES)

PROMPT = f"""\
Below are GitHub issues, one JSON object per line: labels, maintainer_class, and the cleaned text
(removed_template_lines were already stripped; ignore them).
Count the issues whose text still gives away their type, by category. One issue can count in several.
Categories: {", ".join(CATEGORIES)} (label_mismatch = maintainer_class does not fit the content).
Add categories of your own if these miss something.
Reply with ONLY lines in this format, nothing else:
category1: 6
category2: 10
...etc, for easy copy and paste into a .txt file.
"""

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def repo_round_robin(rows, rng):
    """Yield rows one repo at a time, cycling through repos in random order."""
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)
    queues = list(by_repo.values())
    rng.shuffle(queues)
    for q in queues:
        rng.shuffle(q)
    while any(queues):
        for q in queues:
            if q:
                yield q.pop()


def pick_issues(rows, n, seed):
    """Class-balanced pick, each class spread across repos. Returned in random order."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)
    pickers = {c: repo_round_robin(rs, rng) for c, rs in by_class.items()}
    active = sorted(pickers)
    rng.shuffle(active)
    picked = []
    while len(picked) < n and active:
        for c in list(active):
            if len(picked) >= n:
                break
            nxt = next(pickers[c], None)
            if nxt is None:
                active.remove(c)
            else:
                picked.append(nxt)
    rng.shuffle(picked)
    return picked


# ---------------------------------------------------------------------------
# Building the paste file
# ---------------------------------------------------------------------------
def removed_lines(clean, notemplate, cap=8, width=100):
    """Lines that the template step removed from text_clean (deduplicated, capped)."""
    kept = set(notemplate.split("\n"))
    out, seen = [], set()
    for ln in clean.split("\n"):
        s = ln.strip()
        if s and ln not in kept and s not in seen:
            seen.add(s)
            out.append(s[:width])
    return out[:cap]


def build_record(r, max_chars):
    text = r["text_notemplate"]
    if len(text) > max_chars:
        text = text[:max_chars] + f" [truncated {len(r['text_notemplate']) - max_chars} chars]"
    rec = {"id": f"{r['repo']}#{r['number']}", "labels": r["labels"],
           "maintainer_class": r["class"]}
    if r["title"].strip() != r["text_notemplate"].split("\n", 1)[0]:
        rec["title_before_cleaning"] = r["title"].strip()
    rec["text"] = text
    removed = removed_lines(r["text_clean"], r["text_notemplate"])
    if removed:
        rec["removed_template_lines"] = removed
    return rec


def build(args):
    src = PROC_DIR / "cleaned_issues.jsonl"
    if not src.exists():
        print("Run src/cleaning/clean_issues.py first.")
        return
    with open(src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    picked = pick_issues(rows, min(args.num, len(rows)), args.seed)
    lines = [json.dumps(build_record(r, args.max_chars), ensure_ascii=False, separators=(",", ":"))
             for r in picked]

    out = Path(args.out) if args.out else PASTE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PROMPT + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

    chars = out.stat().st_size
    print(f"Wrote {len(picked)} issues from {len({r['repo'] for r in picked})} repos to {out}")
    print(f"Size: about {chars // 4:,} tokens. If too big to paste, use a smaller -n or --max-chars.")
    print(f"\nPaste the whole file into an LLM, save its reply as {COUNT_FILE}, then run:")
    print("  python src/cleaning/rough_audit.py --review")


# ---------------------------------------------------------------------------
# Reviewing the LLM's counts
# ---------------------------------------------------------------------------
LINE_RE = re.compile(r"^([A-Za-z_]\w*)\s*:\s*(\d+)\s*(?:\|\s*(.*))?$")


def parse_counts(text):
    """'name: 12' or 'name: 12 | definition' lines. Anything else (fences, chatter) is ignored."""
    counts, defs = {}, {}
    for line in text.splitlines():
        m = LINE_RE.match(line.strip().lstrip("-*• ").strip("`"))
        if m:
            counts[m.group(1)] = int(m.group(2))
            if m.group(3):
                defs[m.group(1)] = m.group(3).strip()
    return counts, defs


def review(path):
    if not path.exists():
        print(f"File not found: {path}")
        return
    counts, defs = parse_counts(path.read_text(encoding="utf-8"))
    if not counts:
        print(f"No 'name: number' lines found in {path}")
        return

    n = counts.get("total_audited")
    if not n and PASTE_FILE.exists():
        n = sum(1 for ln in PASTE_FILE.read_text(encoding="utf-8").splitlines() if ln.startswith("{"))
    n = n or 0

    print(f"Issues audited: {n or 'unknown'}\n")
    rows = sorted(((k, v) for k, v in counts.items() if k != "total_audited"), key=lambda kv: -kv[1])
    for name, v in rows:
        pct = f"{100 * v / n:5.1f}%" if n else "   n/a"
        tag = "" if name in STARTERS else "  (new)"
        extra = f"  {defs[name]}" if name in defs else ""
        warn = "  <- more than the issues audited, check the reply" if n and v > n else ""
        print(f"  {name:<26}{v:>4}  {pct}{tag}{extra}{warn}")
    print("\nThese are the LLM's rough suspicions, not measurements.")


def main():
    ap = argparse.ArgumentParser(description="Rough LLM-assisted leakage count over cleaned issues.")
    ap.add_argument("-n", "--num", type=int, default=100, help="issues to audit (default 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=800,
                    help="longest text shown per issue (default 800)")
    ap.add_argument("--out", help="paste file to write (default data/analysis/rough_audit.txt)")
    ap.add_argument("--review", nargs="?", const=str(COUNT_FILE), metavar="FILE",
                    help="tally the LLM's counts (default data/analysis/rough_count.txt)")
    args = ap.parse_args()

    if args.review:
        review(Path(args.review))
    else:
        build(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()