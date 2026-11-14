"""Splits the filtered issues into train/val/test BY REPOSITORY.

python src/testing/split_data.py [--seed 42] [--trials 5000] [--val-frac 0.15] [--test-frac 0.15]

Reads   data/processed/filtered_issues.jsonl
Writes  data/processed/split_issues.jsonl      (each issue + its split)
        data/processed/repo_splits.json        (repo -> train/val/test)
        data/analysis/class_counts_split.csv   (per-class issues and repos in each split)

The only random step is choosing which repos go where, so it is seeded. Same seed +
same input file = same split. Report the seed in the paper.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "analysis"
SPLITS = ("train", "val", "test")
MIN_PER_CLASS = 3      # preferred minimum issues of each class in val and test


def load_issues():
    with open(PROC_DIR / "filtered_issues.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def split_repos(issues, seed, trials, val_frac, test_frac):
    repo_counts = defaultdict(Counter)
    for r in issues:
        repo_counts[r["repo"]][r["class"]] += 1
    repos = sorted(repo_counts)
    totals = Counter(r["class"] for r in issues)
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
    c = Counter(best.values())
    print(f"Best of {trials} trials (score {best_score:.3f}): "
          f"{c['train']} train / {c['val']} val / {c['test']} test repos")
    return best


def write_table(issues):
    by_class = defaultdict(list)
    for r in issues:
        by_class[r["class"]].append(r)
    header = ["class", "issues"] + [f"{s}_issues" for s in SPLITS] + [f"{s}_repos" for s in SPLITS]
    rows = []
    for cls in sorted(by_class, key=lambda c: -len(by_class[c])):
        items = by_class[cls]
        row = [cls, len(items)]
        row += [sum(1 for r in items if r["split"] == s) for s in SPLITS]
        row += [len({r["repo"] for r in items if r["split"] == s}) for s in SPLITS]
        rows.append(row)
    total = ["TOTAL", len(issues)]
    total += [sum(1 for r in issues if r["split"] == s) for s in SPLITS]
    total += [len({r["repo"] for r in issues if r["split"] == s}) for s in SPLITS]
    rows.append(total)

    out = OUT_DIR / "class_counts_split.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("\n" + "".join(f"{h:>13}" for h in header))
    for row in rows:
        print("".join(f"{str(v):>13}" for v in row))
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Split filtered issues into train/val/test by repo.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    path = PROC_DIR / "filtered_issues.jsonl"
    if not path.exists():
        print("Run src/filtering/filter_issues.py first.")
        return
    issues = load_issues()

    assign = split_repos(issues, args.seed, args.trials, args.val_frac, args.test_frac)
    for r in issues:
        r["split"] = assign[r["repo"]]

    with open(PROC_DIR / "split_issues.jsonl", "w", encoding="utf-8") as f:
        for r in issues:
            f.write(json.dumps(r) + "\n")
    with open(PROC_DIR / "repo_splits.json", "w", encoding="utf-8") as f:
        json.dump(assign, f, indent=2, sort_keys=True)

    write_table(issues)
    for s in ("val", "test"):
        counts = Counter(r["class"] for r in issues if r["split"] == s)
        thin = [c for c in sorted({r["class"] for r in issues}) if counts[c] < MIN_PER_CLASS]
        if thin:
            print(f"WARNING: {s} has fewer than {MIN_PER_CLASS} issues for: {', '.join(thin)}")
    print(f"\nseed={args.seed}, trials={args.trials}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()