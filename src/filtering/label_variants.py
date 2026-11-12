"""Maps maintainers' GitHub labels onto issue categories (whole-word matching).

python src/filtering/label_variants.py [label_counts.csv] [--audit]

How a label is classified
-------------------------
1. Lowercase, remove emoji and :shortcodes:, split into words on every non-letter/digit
   ("Type: Bug" -> [type, bug]; "regression-from-stable-to-beta" -> [regression, from, ...]).
2. Skip the label if it is about WORKFLOW or LOCATION, not issue type:
   any word in WORKFLOW_WORDS ("needs testing", "triage review") or a first word in
   AREA_PREFIXES ("topic:audio", "platform/windows").
3. Remove negated words ("Question/Not a bug" -> question).
4. Drop filler: single letters, numbers, p0-p9, pronouns/articles, and type-namespace words
   ("type", "kind", "module", ...). What is left is the label's "core".
5. Match, in two tiers:
     STRONG - the keyword is distinctive; ANY core word that equals it counts
              ("bug-vim", "p3-minor-bug", "regression-from-stable-to-beta" -> bug).
     PHRASES - the keyword is short or ambiguous; the WHOLE core must equal the phrase
              ("ci", "build", "docs"). This keeps "a-build-scripts", "needs decision"
              (decision != ci) and "rustdoc" out.
Words are compared whole, never as substrings ("debug" is not "bug", "mcve" is not "cve").
"""
import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Word lists (all lowercase)
# ---------------------------------------------------------------------------
STRONG = {
    "bug": {
        "bug", "bugs", "bugfix", "bugfixes", "defect", "defects", "regression", "regressions",
        "crash", "crashes", "crashing", "segfault", "segfaults", "panic", "panics",
        "deadlock", "hang", "hangs", "broken",
    },
    "security": {
        "security", "vulnerability", "vulnerabilities", "vuln", "vulns", "cve", "cves",
        "xss", "csrf", "ssrf", "rce",
    },
    "question": {"question", "questions", "faq"},
    "performance": {
        "performance", "perf", "slow", "slowness", "slowdown", "optimization",
        "optimisation", "optimizations",
    },
    "documentation": {"documentation", "docstring", "docstrings", "readme"},
    "dependency": {"dependabot", "renovate"},
    "maintenance": {
        "maintenance", "chore", "chores", "cleanup", "debt",
        "deprecate", "deprecated", "deprecation", "flaky", "flakiness",
    },
    "enhancement": {"enhancement", "enhancements", "improvement", "improvements"},
    "feature": {"wishlist"},
    "misc": {"spam"},
}

PHRASES = {
    "bug": set(),
    "enhancement": set(),
    "feature": {
        "feature", "feature request", "feature requests", "new feature", "feature idea",
        "feature proposal", "suggestion", "proposal", "idea", "ideas",
    },
    "documentation": {"docs", "doc", "documenting", "tutorial", "typo", "typos"},
    "question": {"support", "support request", "help", "usage", "usage question", "how to"},
    "performance": {"speed"},
    "security": set(),
    "dependency": {
        "dependency", "dependencies", "dependency issue", "dependency issues", "deps", "dep",
        "dependency update", "dependency updates", "dependency management",
    },
    "maintenance": {
        "ci", "build", "build ci", "ci build", "infra", "infrastructure", "test", "tests",
        "testing", "task", "tasks", "tech debt", "technical debt", "refactor", "refactoring",
    },
    "misc": {"invalid", "off topic", "offtopic", "out of scope"},
}

# Issues with any of these words are dropped entirely (duplicates add nothing).
DROP_WORDS = {"duplicate"}

# Skip a label containing any of these: it describes workflow state, not issue type.
WORKFLOW_WORDS = {
    "needs", "need", "needed", "awaiting", "waiting", "blocked", "triage", "triaged",
    "pending",
}

# Skip a label whose FIRST word is one of these: it names a location/area, not a type.
AREA_PREFIXES = {
    "topic", "platform", "component", "comp", "scope", "os", "command", "ext", "oncall",
}

# Words that carry no meaning for matching.
IGNORED = {
    # namespace words that prefix type labels ("type: bug", "kind/bug", "module: docs")
    "type", "kind", "category", "module", "area", "issue", "issues", "label", "closed",
    # pronouns, articles, prepositions, conjunctions
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their", "the", "an", "of", "to", "for", "in", "on",
    "with", "and", "or", "is", "be", "as", "at", "by", "from", "this", "that",
}

NEGATIONS = {"not", "non", "no", "without"}
ARTICLES = {"a", "an", "the"}

# Applied AFTER mapping to reduce the class set, e.g. {"enhancement": "feature"}.
MERGES = {}

# first match wins, rarer classes first so they keep examples.
PRECEDENCE = [
    "misc", "security", "dependency", "performance", "documentation", "question",
    "bug", "enhancement", "feature", "maintenance",
]

# ---------------------------------------------------------------------------
# Tokenizing
# ---------------------------------------------------------------------------
_SHORTCODE = re.compile(r":[a-z0-9_+\-]+:")
_PRIORITY = re.compile(r"p\d+")


def _words(label):
    s = _SHORTCODE.sub(" ", label.lower())
    s = "".join(
        " " if (unicodedata.category(c) in ("So", "Sk", "Cs") or c in "️‍") else c
        for c in s
    )
    return [w for w in re.split(r"[\W_]+", s) if w]


def _strip_negated(words):
    out, i = [], 0
    while i < len(words):
        if words[i] in NEGATIONS:
            i += 1
            if i < len(words) and words[i] in ARTICLES:
                i += 1
            i += 1                                  # the negated word itself
        else:
            out.append(words[i])
            i += 1
    return out


def core_words(label):
    """The meaningful words of a label, or [] if it is workflow/location only."""
    words = _words(label)
    if not words or words[0] in AREA_PREFIXES or any(w in WORKFLOW_WORDS for w in words):
        return []
    return [w for w in _strip_negated(words)
            if len(w) > 1 and not w.isdigit() and not _PRIORITY.fullmatch(w)
            and w not in IGNORED]


# ---------------------------------------------------------------------------
# Lookup tables with validation checks
# ---------------------------------------------------------------------------
def _phrase_key(phrase):
    return " ".join(w for w in _words(phrase)
                    if len(w) > 1 and w not in IGNORED)


_STRONG_OWNER, _PHRASE_OWNER = {}, {}
for _cls, _words_set in STRONG.items():
    for _w in _words_set:
        if _STRONG_OWNER.setdefault(_w, _cls) != _cls:
            raise ValueError(f"word {_w!r} is STRONG for two classes")
for _cls, _phrases in PHRASES.items():
    for _p in _phrases:
        key = _phrase_key(_p)
        if _PHRASE_OWNER.setdefault(key, _cls) != _cls:
            raise ValueError(f"phrase {key!r} is in two classes")
_missing = (set(STRONG) | set(PHRASES)) - set(PRECEDENCE)
if _missing:
    raise ValueError(f"classes missing from PRECEDENCE: {_missing}")


# ---------------------------------------------------------------------------
# Public helpers (used by filter_issues.py)
# ---------------------------------------------------------------------------
def classify_label(label):
    """Return (set_of_classes, tier) for one label; tier is 'strong' or 'phrase'."""
    if any(w in DROP_WORDS for w in _words(label)):
        return {"drop"}, "strong"
    core = core_words(label)
    if not core:
        return set(), None
    strong = {_STRONG_OWNER[w] for w in core if w in _STRONG_OWNER}
    if strong:
        return strong, "strong"
    owner = _PHRASE_OWNER.get(" ".join(core))
    return ({owner}, "phrase") if owner else (set(), None)


def should_drop(labels):
    return any("drop" in classify_label(l)[0] for l in labels)


def match_classes(labels):
    """All classes matched by an issue's labels, ordered by precedence (after merges)."""
    found = set()
    for label in labels:
        classes, _ = classify_label(label)
        found |= {MERGES.get(c, c) for c in classes if c != "drop"}
    return sorted(found, key=lambda c: PRECEDENCE.index(c) if c in PRECEDENCE
                  else len(PRECEDENCE))


def assign_class(labels):
    """Single class by precedence, or None if no label maps (issue is then dropped)."""
    matched = match_classes(labels)
    return matched[0] if matched else None


# ---------------------------------------------------------------------------
# Report: how well do the lists cover the labels we actually fetched?
# ---------------------------------------------------------------------------
def _report(csv_path, audit):
    per_class, unmapped = Counter(), Counter()
    matched = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label, count = row["label"], int(row["count"])
            classes, tier = classify_label(label)
            if not classes:
                unmapped[label] += count
                continue
            for c in classes:
                c = "(dropped: duplicate)" if c == "drop" else MERGES.get(c, c)
                per_class[c] += count
                matched[c].append((count, label, tier))
    print("Label occurrences mapped per class:")
    for cls, n in per_class.most_common():
        print(f"  {cls:24} {n:>6}")
    if audit:
        for cls, items in matched.items():
            print(f"\n[{cls}] top matches (check for mistakes):")
            for count, label, tier in sorted(items, reverse=True)[:15]:
                print(f"  {count:>5}  {label}   ({tier})")
    print("\nTop 40 UNMAPPED labels (extend the lists if any are type labels):")
    for label, n in unmapped.most_common(40):
        print(f"  {n:>6}  {label}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else ROOT / "data" / "analysis" / "label_counts.csv"
    _report(path, audit="--audit" in sys.argv)