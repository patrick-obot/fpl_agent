#!/usr/bin/env python3
"""
FPL Projections Updater (VPS-side)

Commits and pushes projected_points.csv that was SCP'd from the Raspberry Pi.
Checks CSV freshness before committing.

Usage:
    python update_projections.py              # Commit + push
    python update_projections.py --skip-commit  # Just show file info
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta


PROJECT_DIR = Path(__file__).parent
DATA_FILE = PROJECT_DIR / "data" / "projected_points.csv"


def check_freshness() -> bool:
    """Check if the CSV is reasonably fresh. Warns if stale."""
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found!")
        print("The Raspberry Pi has not uploaded the CSV yet.")
        return False

    age = datetime.now() - datetime.fromtimestamp(DATA_FILE.stat().st_mtime)

    if age > timedelta(hours=48):
        print(f"WARNING: CSV is {age.total_seconds() / 3600:.0f}h old — check if Pi cron is running")
    elif age > timedelta(hours=24):
        print(f"INFO: CSV is {age.total_seconds() / 3600:.0f}h old")

    return True


def show_file_info():
    """Display information about the CSV file."""
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found!")
        return False

    file_size = DATA_FILE.stat().st_size
    file_modified = datetime.fromtimestamp(DATA_FILE.stat().st_mtime)
    print(f"File: {DATA_FILE.name}")
    print(f"Size: {file_size:,} bytes")
    print(f"Modified: {file_modified.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    return True


def git_commit_and_push() -> bool:
    """Add, commit, and push the CSV file to git."""
    # Git add
    print("Adding to git...")
    result = subprocess.run(
        ["git", "add", "data/projected_points.csv"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Git add failed: {result.stderr}")
        return False

    # Check if there are changes to commit
    result = subprocess.run(
        ["git", "status", "--porcelain", "data/projected_points.csv"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )

    if not result.stdout.strip():
        print("No changes to commit - file is already up to date.")
        return True

    # Git commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("Committing...")
    result = subprocess.run(
        ["git", "commit", "-m", f"Update projected_points.csv ({timestamp})"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0 and "nothing to commit" not in result.stdout:
        print(f"Git commit failed: {result.stderr}")
        return False
    print("Committed!")

    # Git push
    print("Pushing to origin...")
    result = subprocess.run(
        ["git", "push"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Git push failed: {result.stderr}")
        return False
    print("Pushed!")

    return True


def main():
    skip_commit = "--skip-commit" in sys.argv

    print("=" * 60)
    print("FPL PROJECTIONS UPDATER")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Check freshness
    if not check_freshness():
        return 1

    # Step 2: Show file info
    if not show_file_info():
        return 1

    # Step 3: Git commit and push (unless skipped)
    if not skip_commit:
        if not git_commit_and_push():
            return 1

    print()
    print("=" * 60)
    print("SUCCESS! Projections updated.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
