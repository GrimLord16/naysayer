"""Cleans raw GitHub issues and filters them down to a labeled training set."""

import json
import os
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(os.environ.get("NAYSAYER_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
TOP_N_LABELS = int(os.environ.get("TOP_N_LABELS", "25"))
MIN_LABEL_COUNT = int(os.environ.get("MIN_LABEL_COUNT", "5"))

# Workflow/process labels (triage-bot state, not content) — applied to most issues
# regardless of topic, so they dominate predictions and crowd out useful labels like
# `module: cuda`. Not something we want to suggest anyway (priority is explicitly
# out of scope: no reliable ground truth).
DEFAULT_EXCLUDE_LABELS = "triaged,bot-triaged,triage review,skipped,needs reproduction,high priority"
EXCLUDE_LABELS = {
    label.strip()
    for label in os.environ.get("EXCLUDE_LABELS", DEFAULT_EXCLUDE_LABELS).split(",")
    if label.strip()
}

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_URL_RE = re.compile(r"https?://\S+")
_MD_RE = re.compile(r"[#*_>-]")
_WHITESPACE_RE = re.compile(r"\s+")

BOT_SUFFIXES = ("[bot]", "-bot", "_bot")


def is_pull_request(issue: dict) -> bool:
    """The GitHub /issues endpoint mixes in pull requests; PRs carry a `pull_request` key."""
    return "pull_request" in issue


def is_bot(issue: dict) -> bool:
    user = issue.get("user") or {}
    if user.get("type") == "Bot":
        return True
    login = (user.get("login") or "").lower()
    return any(login.endswith(suffix) for suffix in BOT_SUFFIXES)


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = _CODE_BLOCK_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MD_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _issue_label_names(issue: dict) -> list[str]:
    return [label["name"] for label in issue.get("labels", []) if label["name"] not in EXCLUDE_LABELS]


def top_labels(issues: list[dict], n: int, min_count: int) -> list[str]:
    counts = Counter(name for issue in issues for name in _issue_label_names(issue))
    frequent = [label for label, count in counts.items() if count >= min_count]
    frequent.sort(key=lambda label: (-counts[label], label))
    return frequent[:n]


def preprocess(issues: list[dict], top_n: int, min_count: int) -> list[dict]:
    real_issues = [i for i in issues if not is_pull_request(i) and not is_bot(i)]
    label_set = set(top_labels(real_issues, top_n, min_count))

    records = []
    for issue in real_issues:
        labels = [name for name in _issue_label_names(issue) if name in label_set]
        if not labels:
            continue
        text = f"{issue['title']} {clean_text(issue.get('body'))}".strip()
        records.append(
            {
                "number": issue["number"],
                "text": text,
                "labels": labels,
                "created_at": issue["created_at"],
            }
        )
    return records


def load_raw(filename: str = "raw_issues.json") -> list[dict]:
    return json.loads((DATA_DIR / filename).read_text())


def save_processed(records: list[dict], filename: str = "processed_issues.json") -> Path:
    path = DATA_DIR / filename
    path.write_text(json.dumps(records))
    print(f"Saved {len(records)} processed issues to {path}")
    return path


if __name__ == "__main__":
    issues = load_raw()
    records = preprocess(issues, TOP_N_LABELS, MIN_LABEL_COUNT)
    save_processed(records)
