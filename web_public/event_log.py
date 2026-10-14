"""
Stores user events (visit, analysis, feedback, survey) persistently.

- Each event is appended as one JSON line to the `DATA_DIR/events_<server>.jsonl` file.
- If the HF_TOKEN and DATA_REPO environment variables are set, the folder is uploaded
  automatically every few minutes to a private Hugging Face Dataset repository (the Space disk is ephemeral).
- No personal data is kept: the CV link, IP address, name, etc. are NOT STORED. The visitor ID
  is a random UUID stored in a browser cookie.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_timer = None          # huggingface_hub.CommitScheduler
_data_dir: Path = None
_file: Path = None
_previous_dir: Path = None   # Events downloaded from previous runs (read-only)

SERVER_ID = os.environ.get("SERVER_ID") or uuid.uuid4().hex[:8]

EVENT_TYPES = {
    "visit",
    "analysis_started",
    "analysis_completed",
    "analysis_error",
    "results_viewed",
    "project_feedback",
    "survey",
}


def launch(project_root: Path, logger=None):
    """Prepares the logging system. Called once when the application starts."""
    global _timer, _data_dir, _file, _previous_dir

    _data_dir = Path(os.environ.get("DATA_DIR", project_root / "outputs" / "user_data"))
    new = _data_dir / "new"
    new.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    _file = new / f"events_{date}_{SERVER_ID}.jsonl"

    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("DATA_REPO")  # e.g. username/fairpass-user-data
    if not (token and repo):
        if logger:
            logger.info(f"📝 Events are written to a local file: {_file} (HF backup disabled)")
        return

    try:
        from huggingface_hub import CommitScheduler, snapshot_download

        # Download the data of previous runs for the stats (read-only)
        _previous_dir = _data_dir / "previous"
        try:
            snapshot_download(
                repo_id=repo, repo_type="dataset", token=token,
                local_dir=str(_previous_dir), allow_patterns=["data/*.jsonl"],
            )
        except Exception as e:  # The repository may still be empty
            if logger:
                logger.info(f"Could not download previous data (may be the first run): {e}")

        _timer = CommitScheduler(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(new),
            path_in_repo="data",
            every=int(os.environ.get("BACKUP_MINUTES", "5")),
            private=True,
            token=token,
        )
        if logger:
            logger.info(f"☁️  Events are backed up automatically to the {repo} dataset repository")
    except Exception as e:
        _timer = None
        if logger:
            logger.error(f"Could not start the HF backup, only local logging will be used: {e}")


def save(kind: str, **fields):
    """Records an event. Does not stop the application on error."""
    if _file is None:
        return
    if kind not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {kind}")
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "server": SERVER_ID,
        **fields,
    }
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    try:
        outer_lock = _timer.lock if _timer is not None else _lock
        with outer_lock:
            with open(_file, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


def all_events():
    """Returns all events on this server and (if any) from previous runs."""
    files = []
    if _data_dir is not None:
        files += sorted((_data_dir / "new").glob("*.jsonl"))
    if _previous_dir is not None and _previous_dir.exists():
        local_names = {d.name for d in files}
        files += [d for d in sorted(_previous_dir.rglob("*.jsonl")) if d.name not in local_names]
    return read_jsonl(files)


def read_jsonl(files):
    events = []
    for d in files:
        try:
            with open(d, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            continue
    return events


def backup_now():
    """Uploads the pending events to HF immediately (for the admin panel)."""
    if _timer is not None:
        _timer.trigger()
        return True
    return False
