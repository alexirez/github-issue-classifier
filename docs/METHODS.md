# Methods

A classifier that reads the title and body of a GitHub issue and predicts one category (bug, feature, documentation, question, and others).

Last updated: 2026-11-12

## Goal

- One label per issue. It does not predict severity, owner, or duplicates, and it ignores pull requests.
- Main metric: **macro-F1**, the average of per-class F1 scores.
- Models: TF-IDF + logistic regression as the baseline, then sentence embeddings + a classifier as the main method. Fine-tuning DistilBERT is optional.
- Target: macro-F1 of 0.70 and +5 points over the baseline.

## Data

- Issues from 100 public repositories (repos used are specified in `configs/repos.txt`), collected via the GitHub REST API. Up to 600 of the newest issues per repo, so no single project dominates the data.
- Pull requests and bot-authored issues are removed. Usernames are not stored.
- Issue text is not redistributed. `data/` is git-ignored; the repo list, rules, and label tables are committed.
- The raw labels are messy. Labels are cleaned up programmatically before being fed into training.

## Labeling

Maintainers' labels are mapped to categories with fixed rules. There is no LLM labeling, so results are free and fully reproducible.

1. Lowercase each label, strip emojis, and split labels into whole words.
2. Skip labels about workflow or location, not type (`needs-triage`, `platform/windows`).
3. Handle negation ("Question/Not a bug" becomes question).
4. Match distinctive words anywhere in the label (`bug`, `regression`, `cve`). Match ambiguous short ones (`ci`, `build`, `docs`) only when they make the whole label.
5. If several categories match, a fixed precedence order picks just one.
6. Issues with no type label, and issues labeled duplicate, are dropped.
7. One rule set for every repo, with no per-repo overrides.

Maintainer labels are noisy, so they're checked against a hand-labeled gold set (below).

## Evaluation

- **Split by repository.** Training, validation, and test repos never overlap, so the score measures how the model handles projects it has never seen.
- Validation is used for tuning. The test set is scored once.
- **Gold set:** about 300 to 400 issues from roughly 20 new repos that never appear in training. Labelled by hand from the title and body only, without seeing maintainer labels.
- Reported: macro-F1, per-class precision and recall, and a confusion matrix. One ablation compares keeping vs. stripping code blocks and URLs.

## Status

| Stage | State |
|---|---|
| Fetch issues from GitHub | Done |
| Analyze raw data and label counts | Done |
| Label-mapping rules | Done, still being audited on the full fetch |
| Filter, split by repo, and report how many issues survive each step | Next |
| TF-IDF baseline | Next |
| Embedding classifier | Planned |
| Gold set | Planned |

## Known limitations

- **Rare classes.** In the first batch, security and dependency issues numbered about 29 and 9, against about 4,200 bugs. Public trackers rarely hold security reports. Even with more repositories added, these groups are harder to train for.
- Labels measure what maintainers chose to tag, not issue content, and keeping only labeled issues favors well-triaged repos.
- Fetching newest-first gives a narrow time window for busy repos.
- A single label cannot capture an issue with several intents.
- Predicting a category says nothing about priority.

## Reproducibility

Configurations are stored in `configs/`. Since labels are cleaned programmatically, it's fully deterministic. The data however, may not be. As a result, results may vary during repeated trials.