"""Filters fetched issues, assigns one class each, splits by repository, reports class counts.

python src/filtering/filter_issues.py [--seed 42] [--trials 5000] [--val-frac 0.15] [--test-frac 0.15]

Reads   data/raw/*.jsonl
Writes  data/processed/filtered_issues.jsonl   (kept issues + class + all matched classes + split)
        data/processed/repo_splits.json        (repo -> train/val/test)
        data/analysis/class_counts_filtered.csv (per-class issues, repos, per-split counts)
        data/analysis/filter_attrition.csv      (how many issues survive each step)
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from label_variants import PRECEDENCE, match_classes, should_drop

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "analysis"
REQUIRED = ("repo", "number", "labels", "state", "created_at", "title", "body")
SPLITS = ("train", "val", "test")
MIN_PER_CLASS = 3      # preferred minimum issues of each class in val and test


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

    no_text = [r for r in uniq if not (r["title"] or "").strip()]
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
# Split by repository
# ---------------------------------------------------------------------------
def score_split(assign, repo_counts, totals, fracs):
    """Lower is better: class shares close to targets, no near-empty val/test classes."""
    per = {s: Counter() for s in SPLITS}
    for repo, split in assign.items():
        per[split].update(repo_counts[repo])
    score = 0.0
    for cls, total in totals.items():
        for s in SPLITS:
            n = per[s][cls]
            score += abs(n / total - fracs[s])
            if s != "train":
                score += 10 * max(0, min(MIN_PER_CLASS, total) - n) / MIN_PER_CLASS
    return score


def split_repos(kept, seed, trials, val_frac, test_frac):
    repo_counts = defaultdict(Counter)
    for r in kept:
        repo_counts[r["repo"]][r["class"]] += 1
    repos = sorted(repo_counts)
    totals = Counter(r["class"] for r in kept)
    fracs = {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    n_val, n_test = round(len(repos) * val_frac), round(len(repos) * test_frac)

    best, best_score = None, float("inf")
    for t in range(trials):
        rng = random.Random(seed * 1_000_003 + t)
        order = repos[:]
        rng.shuffle(order)
        assign = {}
        for i, repo in enumerate(order):
            assign[repo] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        sc = score_split(assign, repo_counts, totals, fracs)
        if sc < best_score:
            best, best_score = assign, sc
    print(f"Best repo split of {trials} trials (score {best_score:.3f}): "
          f"{Counter(best.values())['train']} train / {Counter(best.values())['val']} val / "
          f"{Counter(best.values())['test']} test repos")
    return best


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def write_class_table(kept):
    by_class = defaultdict(list)
    for r in kept:
        by_class[r["class"]].append(r)
    header = ["class", "issues", "share", "repos"]
    header += [f"{s}_issues" for s in SPLITS] + [f"{s}_repos" for s in SPLITS]
    total = len(kept)
    rows = []
    for cls in sorted(by_class, key=lambda c: -len(by_class[c])):
        items = by_class[cls]
        row = [cls, len(items), f"{100 * len(items) / total:.1f}%", len({r["repo"] for r in items})]
        row += [sum(1 for r in items if r["split"] == s) for s in SPLITS]
        row += [len({r["repo"] for r in items if r["split"] == s}) for s in SPLITS]
        rows.append(row)
    all_row = ["TOTAL", total, "100.0%", len({r["repo"] for r in kept})]
    all_row += [sum(1 for r in kept if r["split"] == s) for s in SPLITS]
    all_row += [len({r["repo"] for r in kept if r["split"] == s}) for s in SPLITS]
    rows.append(all_row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "class_counts_filtered.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print("\n" + "".join(f"{h:>13}" for h in header))
    for row in rows:
        print("".join(f"{str(v):>13}" for v in row))
    print(f"\nWrote {out}")


def write_attrition(attrition):
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


def report_small_classes(kept):
    for s in ("val", "test"):
        counts = Counter(r["class"] for r in kept if r["split"] == s)
        thin = [c for c in PRECEDENCE if counts[c] < MIN_PER_CLASS]
        if thin:
            print(f"WARNING: {s} has fewer than {MIN_PER_CLASS} issues for: {', '.join(thin)}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Filter issues, assign classes, split by repo.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    rows, broken, n_files = load_rows()
    if not rows:
        print("No data in data/raw. Run fetch_data.py first.")
        return
    print(f"Loaded {len(rows)} rows from {n_files} files ({broken} broken/incomplete lines skipped)")

    kept, attrition = filter_rows(rows)
    if not kept:
        print("No issues survived filtering.")
        return

    assign = split_repos(kept, args.seed, args.trials, args.val_frac, args.test_frac)
    for r in kept:
        r["split"] = assign[r["repo"]]

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROC_DIR / "filtered_issues.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    with open(PROC_DIR / "repo_splits.json", "w", encoding="utf-8") as f:
        json.dump(assign, f, indent=2, sort_keys=True)

    write_attrition(attrition)
    write_class_table(kept)
    report_overlap(kept)
    report_small_classes(kept)
    print(f"\nWrote {PROC_DIR / 'filtered_issues.jsonl'} and repo_splits.json "
          f"(seed={args.seed}, trials={args.trials})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()