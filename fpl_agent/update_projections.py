#!/usr/bin/env python3
"""
FPL Projections Updater

After manually downloading projected_points.csv from FPL Review,
run this script to commit and deploy to Railway.

Usage:
    1. Download CSV from https://fplreview.com/massive-data-planner23/
    2. Save as: fpl_agent/data/projected_points.csv
    3. Run: python update_projections.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


PROJECT_DIR = Path(__file__).parent
DATA_FILE = PROJECT_DIR / "data" / "projected_points.csv"


def main():
    print("=" * 60)
    print("FPL PROJECTIONS UPDATER")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check if file exists
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found!")
        print()
        print("Please download the CSV from FPL Review first:")
        print("  1. Go to https://fplreview.com/massive-data-planner23/")
        print("  2. Log in with Patreon")
        print("  3. Click 'Download CSV'")
        print(f"  4. Save as: {DATA_FILE}")
        return 1

    # Show file info
    file_size = DATA_FILE.stat().st_size
    file_modified = datetime.fromtimestamp(DATA_FILE.stat().st_mtime)
    print(f"File: {DATA_FILE.name}")
    print(f"Size: {file_size:,} bytes")
    print(f"Modified: {file_modified.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

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
        return 1

    # Check if there are changes to commit
    result = subprocess.run(
        ["git", "status", "--porcelain", "data/projected_points.csv"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )

    if not result.stdout.strip():
        print("No changes to commit - file is already up to date.")
        print()
        print("Deploying to Railway anyway...")
    else:
        # Git commit
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"Committing...")
        result = subprocess.run(
            ["git", "commit", "-m", f"Update projected_points.csv ({timestamp})"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True
        )
        if result.returncode != 0 and "nothing to commit" not in result.stdout:
            print(f"Git commit failed: {result.stderr}")
            return 1
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
            return 1
        print("Pushed!")

    # Deploy to Railway
    print()
    print("Deploying to Railway...")
    result = subprocess.run(
        ["railway", "up", "--service", "fpl-agent"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=300
    )
    if result.returncode != 0:
        print(f"Railway deploy failed: {result.stderr}")
        return 1

    print()
    print("=" * 60)
    print("SUCCESS! Projections updated and deployed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
