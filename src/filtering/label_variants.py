"""Maps maintainers' GitHub labels onto our issue categories.

Import this module for its constants and helpers; run it directly for a report:

    python src/filtering/label_variants.py [path/to/label_counts.csv]

How matching works
------------------
Labels are NORMALIZED first, then compared by exact equality (never substring):
  * lowercased
  * emoji and :shortcodes: removed   ("bug :beetle:" -> "bug", "🔩 p2-edge-case" -> "p2-edge-case")
  * spaces around ':' '/' '-' removed ("Type: Bug" -> "type:bug", "00 - Bug" -> "00-bug")
So variants below are written lowercase without emoji; no wildcard is needed.
Variants are normalized the same way at import, so writing "type: bug" also works.

PROVISIONAL: the precedence order, the Misc definition, and the class merges are
open decisions (see docs/METHODS.md). Fix them before final testing.
"""
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
_SHORTCODE = re.compile(r":[a-z0-9_+\-]+:")
_SEPARATORS = re.compile(r"\s*([:/\-])\s*")


def normalize_label(label):
    s = label.lower()
    s = _SHORTCODE.sub(" ", s)
    s = "".join(
        " " if (unicodedata.category(c) in ("So", "Sk", "Cs") or c in "️‍") else c
        for c in s
    )
    s = " ".join(s.split())
    s = _SEPARATORS.sub(r"\1", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Variants per class (lowercase, no emoji)
# ---------------------------------------------------------------------------
VARIANTS = {
    "bug": {
        "bug", "bugs", "c-bug", "type:bug", "00-bug", "p3-minor-bug", "bug/regression",
        "bug-regression", "bug-crash", "bug-vim", "defect", "kind/bug", "type: bug",
        "regression", "06-regression", "module:regression", "regression-untriaged",
        "regression-from-stable-to-stable", "regression-from-stable-to-beta",
        "regression-from-stable-to-nightly", "crash", "module:crash", "segfault", "panic",
        "i-crash", "i-ice", "i-miscompile", "i-unsound", "i-hang", "i-memleak", "memory leak",
        "module:correctness (silent)", "module:dependency bug",
    },
    "enhancement": {
        "enhancement", "c-enhancement", "type:enhancement", "01-enhancement", "improvement",
        "kind/enhancement",
    },
    "feature": {
        "feature", "feature-request", "feature request", "c-feature-request", "new feature",
        "type:feature", "type:feature request", "kind/feature", "function request",
        "new model", "suggestion",
    },
    "documentation": {
        "documentation", "docs", "04-documentation", "type:docs", "type:docs-bug",
        "module:docs", "a-docs", "kind/documentation", "kind/docs",
    },
    "question": {
        "question", "questions", "question/not a bug", "33-question", "closed:question",
        "usage question", "usage", "support", "type:support", "type:question",
        "kind/support", "kind/question", "c-question",
    },
    "performance": {
        "performance", "perf", "module:performance", "type:performance", "i-slow",
        "i-compiletime", "i-compilemem", "c-optimization", "area/performance", "slow",
    },
    "security": {
        "security", "a-security", "type:security", "kind/security", "area/security",
        "vulnerability", "cve", "security vulnerability",
    },
    "dependency": {
        "dependencies", "dependency", "dependency issue", "dependency update",
        "dependency-update", "deps", "a-dependency-resolution", "a-crate-dependencies",
    },
    "maintenance": {
        "maintenance", "03-maintenance", "chore", "refactor", "cleanup", "c-cleanup",
        "tech debt", "tech-debt", "technical debt", "a-technical-debt", "debt",
        "ci", "build", "build/ci", "ci/build", "module:ci", "module:build", "component:ci",
        "flaky", "flaky test", "i-flaky-test", "module:flaky-tests", "module:tests",
        "05-testing", "17-task", "task", "deprecate", "07-deprecation", "module:deprecation",
        "better-engineering",
    },
    # PROVISIONAL (open decision): explicit "not a real issue" labels only.
    # Unmapped issues are NOT Misc; they are dropped.
    "misc": {
        "spam", "invalid", "closed:invalid", "code agent slop", "off-topic", "offtopic",
        "out of scope",
    },
}

# Issues carrying any of these labels are dropped entirely (not useful training examples).
DROP_LABELS = {
    "duplicate", "*duplicate", "duplicate report", "closed:duplicate", "50-duplicate",
}

# Applied AFTER mapping to reduce the class set, e.g. {"enhancement": "feature",
# "performance": "bug"}. Decide before final testing; empty keeps all classes separate.
MERGES = {}

# PROVISIONAL precedence (first match wins). Rarer classes first so they keep examples.
PRECEDENCE = [
    "misc", "security", "dependency", "performance", "documentation", "question",
    "bug", "enhancement", "feature", "maintenance",
]

# ---------------------------------------------------------------------------
# Lookup tables built at import (with sanity checks)
# ---------------------------------------------------------------------------
def _build_lookup(variants):
    lookup = {}
    for cls, names in variants.items():
        for name in names:
            key = normalize_label(name)
            if key in lookup and lookup[key] != cls:
                raise ValueError(f"label {key!r} is in both {lookup[key]!r} and {cls!r}")
            lookup[key] = cls
    return lookup


LOOKUP = _build_lookup(VARIANTS)
DROP = {normalize_label(x) for x in DROP_LABELS}

_missing = set(VARIANTS) - set(PRECEDENCE)
if _missing:
    raise ValueError(f"classes missing from PRECEDENCE: {_missing}")


# ---------------------------------------------------------------------------
# Helpers used by filter_issues.py
# ---------------------------------------------------------------------------
def should_drop(labels):
    return any(normalize_label(l) in DROP for l in labels)


def match_classes(labels):
    """All classes matched by an issue's labels, ordered by precedence (after merges)."""
    found = set()
    for label in labels:
        key = normalize_label(label)
        cls = LOOKUP.get(key)
        if cls:
            found.add(MERGES.get(cls, cls))
    return sorted(found, key=lambda c: PRECEDENCE.index(c) if c in PRECEDENCE
                  else len(PRECEDENCE))


def assign_class(labels):
    """Single class by precedence, or None if no label maps (issue is then dropped)."""
    matched = match_classes(labels)
    return matched[0] if matched else None


# ---------------------------------------------------------------------------
# Report: how well do the variants cover the labels we actually fetched?
# ---------------------------------------------------------------------------
def _report(csv_path):
    per_class, unmapped = Counter(), Counter()
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key, count = normalize_label(row["label"]), int(row["count"])
            if key in DROP:
                per_class["(dropped: duplicate)"] += count
            elif key in LOOKUP:
                per_class[MERGES.get(LOOKUP[key], LOOKUP[key])] += count
            else:
                unmapped[row["label"]] += count
    print("Label occurrences mapped per class:")
    for cls, n in per_class.most_common():
        print(f"  {cls:24} {n:>6}")
    print(f"\nTop 40 UNMAPPED labels (extend VARIANTS if any are type labels):")
    for label, n in unmapped.most_common(40):
        print(f"  {n:>6}  {label}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "analysis" / "label_counts.csv"
    _report(path)