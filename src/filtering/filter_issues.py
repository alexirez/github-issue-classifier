"""Filters fetched issues, assigns one class each, reports class counts.

python src/filtering/filter_issues.py

Reads   data/raw/*.jsonl
Writes  data/processed/filtered_issues.jsonl   (kept issues + class + all matched classes)
        data/analysis/class_counts_filtered.csv (per-class issues, share, repos)
        data/analysis/filter_attrition.csv      (how many issues survive each step)
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from label_variants import match_classes, should_drop

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "analysis"
REQUIRED = ("repo", "number", "labels", "state", "created_at", "title", "body")


# ---------------------------------------------------------------------------
# Load + filter
# ---------------------------------------------------------------------------
def load_rows():
    rows, broken = [], 0
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    broken += 1
                    continue
                if isinstance(r, dict) and all(k in r for k in REQUIRED):
                    rows.append(r)
                else:
                    broken += 1
    return rows, broken, len(list(RAW_DIR.glob("*.jsonl")))


def filter_rows(rows):
    """Return (kept, attrition) where attrition is a list of (step, remaining)."""
    attrition = [("fetched (valid rows)", len(rows))]

    seen, uniq = set(), []
    for r in rows:
        key = (r["repo"], r["number"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    attrition.append(("after removing repeated (repo, number)", len(uniq)))

    uniq = [r for r in uniq if (r["title"] or "").strip()]
    attrition.append(("after removing empty titles", len(uniq)))

    undup = [r for r in uniq if not should_drop(r["labels"])]
    attrition.append(("after removing 'duplicate' issues", len(undup)))

    kept = []
    for r in undup:
        matched = match_classes(r["labels"])
        if not matched:
            continue
        kept.append({**r, "class": matched[0], "matched_classes": matched})
    attrition.append(("after removing issues with no type label", len(kept)))
    return kept, attrition


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def write_class_table(kept):
    by_class = defaultdict(list)
    for r in kept:
        by_class[r["class"]].append(r)
    total = len(kept)
    rows = []
    for cls in sorted(by_class, key=lambda c: -len(by_class[c])):
        items = by_class[cls]
        rows.append([cls, len(items), f"{100 * len(items) / total:.1f}%",
                     len({r["repo"] for r in items})])
    rows.append(["TOTAL", total, "100.0%", len({r["repo"] for r in kept})])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "class_counts_filtered.csv"
    header = ["class", "issues", "share", "repos"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print("\n" + "".join(f"{h:>13}" for h in header))
    for row in rows:
        print("".join(f"{str(v):>13}" for v in row))
    print(f"\nWrote {out}")


def write_attrition(attrition):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "filter_attrition.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "issues_remaining"])
        w.writerows(attrition)
    print("\nAttrition:")
    for step, n in attrition:
        print(f"  {n:>7}  {step}")
    print(f"Wrote {out}")


def report_overlap(kept):
    multi = [r for r in kept if len(r["matched_classes"]) > 1]
    print(f"\nIssues matching more than one class: {len(multi)} "
          f"({100 * len(multi) / len(kept):.1f}% of kept)")
    pairs = Counter()
    for r in multi:
        pairs.update(combinations(sorted(r["matched_classes"]), 2))
    for (a, b), n in pairs.most_common(10):
        print(f"  {n:>6}  {a} + {b}")


# ---------------------------------------------------------------------------
def main():
    rows, broken, n_files = load_rows()
    if not rows:
        print("No data in data/raw. Run fetch_data.py first.")
        return
    print(f"Loaded {len(rows)} rows from {n_files} files ({broken} broken/incomplete lines skipped)")

    kept, attrition = filter_rows(rows)
    if not kept:
        print("No issues survived filtering.")
        return

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROC_DIR / "filtered_issues.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    write_attrition(attrition)
    write_class_table(kept)
    report_overlap(kept)
    print(f"\nWrote {PROC_DIR / 'filtered_issues.jsonl'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()