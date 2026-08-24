"""Fetches GitHub issues for a repo and saves them as raw JSON under data/."""

import json
import os
import time
from pathlib import Path

import requests

GITHUB_API = "https://api.github.com"
REPO_OWNER = os.environ.get("REPO_OWNER", "pytorch")
REPO_NAME = os.environ.get("REPO_NAME", "pytorch")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
MAX_ISSUES = int(os.environ.get("MAX_ISSUES", "20000"))

DATA_DIR = Path(os.environ.get("NAYSAYER_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
RATE_LIMIT_SAFETY_MARGIN = 50  # stop and sleep before actually hitting 0


def _headers() -> dict:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _respect_rate_limit(response: requests.Response) -> None:
    remaining = int(response.headers.get("x-ratelimit-remaining", "1"))
    if remaining <= RATE_LIMIT_SAFETY_MARGIN:
        reset_epoch = int(response.headers.get("x-ratelimit-reset", time.time()))
        wait = max(0, reset_epoch - int(time.time())) + 5
        print(f"Rate limit low ({remaining} left), sleeping {wait}s until reset")
        time.sleep(wait)


def fetch_issues(owner: str, repo: str, max_issues: int) -> list[dict]:
    """Fetches the most-recently-created `max_issues` issues (PRs included — the GitHub
    /issues endpoint mixes them in; preprocessing filters them out). Sorted by created
    date, not updated date: a capped pull needs a coherent, representative time slice,
    and `updated_at` order can land you on old issues bumped by bots/relabeling."""
    session = requests.Session()
    session.headers.update(_headers())

    issues: list[dict] = []
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues"
    params = {"state": "all", "per_page": 100, "sort": "created", "direction": "desc"}

    while url and len(issues) < max_issues:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        _respect_rate_limit(response)

        issues.extend(response.json())
        print(f"Fetched {len(issues)} issues so far")

        params = None  # only needed on the first request; the `next` link encodes it
        url = response.links.get("next", {}).get("url")

    return issues[:max_issues]


def save_raw(issues: list[dict], filename: str = "raw_issues.json") -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    path.write_text(json.dumps(issues))
    print(f"Saved {len(issues)} issues to {path}")
    return path


if __name__ == "__main__":
    issues = fetch_issues(REPO_OWNER, REPO_NAME, MAX_ISSUES)
    save_raw(issues)
