import json, os, time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ["GITHUB_TOKEN"]
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
MAX_PER_REPO = 600          # cap per repo so no single project dominates
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

skipped_run = []            # consecutive skipped repos, printed on one line


def flush_skipped():
    if skipped_run:
        print("skipped: " + ", ".join(skipped_run))
        skipped_run.clear()


def format_duration(seconds):
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)

    def unit(n, name):
        return f"{n} {name}" + ("" if n == 1 else "s")

    parts = []
    if hours:
        parts.append(unit(hours, "hour"))
    if hours or minutes:
        parts.append(unit(minutes, "minute"))
    parts.append(unit(secs, "second"))
    if not hours and not minutes:
        parts.append(unit(ms, "millisecond"))
    return ", ".join(parts)


def get(url, params):
    while True:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code in (403, 429) and r.headers.get("X-RateLimit-Remaining") == "0":
            wait = int(r.headers["X-RateLimit-Reset"]) - int(time.time()) + 5
            print(f"Rate limited, sleeping {wait}s")
            time.sleep(max(wait, 5))
            continue
        r.raise_for_status()
        return r


def scrape_repo(repo):
    out_path = OUT_DIR / f"{repo.replace('/', '__')}.jsonl"
    if out_path.exists():                       # resumable: skip finished repos
        skipped_run.append(repo)
        return
    flush_skipped()
    kept, page = [], 1
    while len(kept) < MAX_PER_REPO:
        r = get(f"https://api.github.com/repos/{repo}/issues",
                {"state": "all", "per_page": 100, "page": page,
                 "sort": "created", "direction": "desc"})
        items = r.json()
        if not items:
            break
        for it in items:
            if "pull_request" in it:            # endpoint also returns PRs
                continue
            if (it.get("user") or {}).get("type") == "Bot":
                continue
            kept.append({
                "repo": repo,
                "number": it["number"],
                "labels": [l["name"] for l in it["labels"]],
                "state": it["state"],
                "created_at": it["created_at"],
                "author_association": it.get("author_association"),
                "url": it["html_url"],
                "title": it["title"],
                "body": it.get("body") or "",
            })
        page += 1
    with open(out_path, "w") as f:
        for row in kept[:MAX_PER_REPO]:
            f.write(json.dumps(row) + "\n")
    print(f"{repo}: {len(kept[:MAX_PER_REPO])} issues")


if __name__ == "__main__":
    start = time.perf_counter()
    for line in Path("repos.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        try:
            scrape_repo(line)
        except requests.RequestException as e:
            status = e.response.status_code if e.response is not None else "network error"
            detail = e.response.text[:200] if e.response is not None else str(e)
            print(f"{status}: Failed to read from {line} ({detail})")
    flush_skipped()
    print(f"Finished in {format_duration(time.perf_counter() - start)}")