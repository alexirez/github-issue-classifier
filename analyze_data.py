import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/analysis")
REQUIRED = ("repo", "number", "labels", "state", "created_at", "title", "body")
TOP_LABELS = 40


def resolve_files(args):
    """No args -> every file in data/raw. Args may be owner/repo or a file path."""
    if not args:
        return sorted(RAW_DIR.glob("*.jsonl"))
    files = []
    for a in args:
        p = Path(a)
        if not p.suffix:
            p = RAW_DIR / f"{a.replace('/', '__')}.jsonl"
        if p.exists():
            files.append(p)
        else:
            print(f"Not found: {a}")
    return files


def load_file(path):
    """Return (valid_rows, broken_lines, rows_missing_fields)."""
    rows, broken, bad_fields = [], 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                broken += 1
                continue
            if not isinstance(row, dict) or any(k not in row for k in REQUIRED):
                bad_fields += 1
                continue
            rows.append(row)
    return rows, broken, bad_fields


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def main():
    ap = argparse.ArgumentParser(description="Summarize fetched issue files.")
    ap.add_argument("repos", nargs="*", help="owner/repo or file path (default: all in data/raw)")
    args = ap.parse_args()

    files = resolve_files(args.repos)
    if not files:
        print("No files to analyze.")
        return

    label_counts, label_repos = Counter(), defaultdict(set)
    total = total_broken = total_bad = 0
    print(f"{'repo':38} {'valid':>6} {'broken':>6} {'badfld':>6} {'nolabel':>8} "
          f"{'avg_lbl':>7} {'nobody':>7}  date range")
    for path in files:
        rows, broken, bad = load_file(path)
        name = path.stem.replace("__", "/")
        n = len(rows)
        unlabeled = sum(1 for r in rows if not r["labels"])
        empty_body = sum(1 for r in rows if not r["body"].strip())
        avg = sum(len(r["labels"]) for r in rows) / n if n else 0
        dates = sorted(r["created_at"] for r in rows)
        span = f"{dates[0][:10]} to {dates[-1][:10]}" if dates else "n/a"
        print(f"{name:38} {n:>6} {broken:>6} {bad:>6} {pct(unlabeled, n):>8} "
              f"{avg:>7.1f} {pct(empty_body, n):>7}  {span}")
        for r in rows:
            for lab in r["labels"]:
                label_counts[lab] += 1
                label_repos[lab].add(name)
        total += n; total_broken += broken; total_bad += bad

    print(f"\nTOTAL: {total} valid issues in {len(files)} files "
          f"({total_broken} broken lines, {total_bad} rows missing fields)")

    print(f"\nTop {TOP_LABELS} labels (of {len(label_counts)} distinct):")
    print(f"{'label':40} {'count':>6} {'repos':>6}")
    for lab, c in label_counts.most_common(TOP_LABELS):
        print(f"{lab[:40]:40} {c:>6} {len(label_repos[lab]):>6}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "label_counts.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "count", "n_repos"])
        for lab, c in label_counts.most_common():
            w.writerow([lab, c, len(label_repos[lab])])
    print(f"\nFull label table written to {out}")

    repos_txt = Path("repos.txt")
    if repos_txt.exists() and not args.repos:
        wanted = [l.split("#")[0].strip() for l in repos_txt.read_text().splitlines()]
        have = {p.stem.replace("__", "/") for p in files}
        missing = [r for r in wanted if r and r not in have]
        if missing:
            print(f"\nIn repos.txt but no file (failed or not fetched yet): {', '.join(missing)}")


if __name__ == "__main__":
    main()