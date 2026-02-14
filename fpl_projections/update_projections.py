#!/usr/bin/env python3
"""
FPL Projections Updater

Automatically downloads projected_points.csv from FPL Review via Patreon,
then commits and pushes to git.

Usage:
    python update_projections.py [--skip-download] [--skip-commit]

Options:
    --skip-download  Skip the automatic download (use existing file)
    --skip-commit    Skip git commit and push (just download)
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import aiohttp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from fplreview_client import download_fplreview_projections


PROJECT_DIR = Path(__file__).parent


async def send_telegram_notification(config: Config, message: str) -> bool:
    """Send a Telegram notification."""
    if not config.telegram_bot_token or not config.telegram_chat_id:
        print("Telegram not configured, skipping notification")
        return False

    url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": config.telegram_chat_id,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status == 200:
                    print("Telegram notification sent")
                    return True
                else:
                    print(f"Telegram API error: {response.status}")
                    return False
    except Exception as e:
        print(f"Telegram notification failed: {e}")
        return False
DATA_FILE = PROJECT_DIR / "data" / "projected_points.csv"


async def download_projections(config: Config) -> bool:
    """Download projections CSV from FPL Review."""
    print("Downloading projections from FPL Review...")
    print()

    email = (
        getattr(config, "fpl_review_email", None)
        or os.getenv("FPL_REVIEW_EMAIL")
        or os.getenv("PATREON_EMAIL")
        or ""
    )
    password = (
        getattr(config, "fpl_review_password", None)
        or os.getenv("FPL_REVIEW_PASSWORD")
        or os.getenv("FPL_REVIEW_PASS")
        or os.getenv("PATREON_PASSWORD")
        or ""
    )
    headless_env = (os.getenv("FPL_REVIEW_HEADLESS") or "true").strip().lower()
    headless = headless_env not in {"0", "false", "no", "off"}

    if not email or not password:
        print("ERROR: FPL_REVIEW_EMAIL and FPL_REVIEW_PASSWORD must be set in .env")
        print()
        print("Add the following to your .env file:")
        print("  FPL_REVIEW_EMAIL=your_patreon_email")
        print("  FPL_REVIEW_PASSWORD=your_patreon_password")
        print("  (or use FPL_REVIEW_PASS)")
        await send_telegram_notification(
            config,
            "❌ <b>FPL Review Download Failed</b>\n\nCredentials not configured."
        )
        return False

    csv_path = await download_fplreview_projections(
        email=email,
        password=password,
        download_dir=config.data_dir,
        headless=headless,
        logger=config.logger,
        team_id=str(config.fpl_team_id),
    )

    if csv_path and csv_path.exists():
        file_size = csv_path.stat().st_size
        print(f"Download successful: {csv_path}")

        # Send success notification
        await send_telegram_notification(
            config,
            f"✅ <b>FPL Review Projections Downloaded</b>\n\n"
            f"📁 File: projected_points.csv\n"
            f"📊 Size: {file_size:,} bytes\n"
            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return True
    else:
        print("ERROR: Download failed!")
        print("Check the screenshots in data/ folder for debugging.")

        # Send failure notification
        await send_telegram_notification(
            config,
            f"❌ <b>FPL Review Download Failed</b>\n\n"
            f"Check screenshots in data/ folder.\n"
            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return False


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


async def main():
    # Parse command line arguments
    skip_download = "--skip-download" in sys.argv
    skip_commit = "--skip-commit" in sys.argv

    print("=" * 60)
    print("FPL PROJECTIONS UPDATER")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load config
    config = Config.from_env()

    # Step 1: Download projections (unless skipped)
    if not skip_download:
        success = await download_projections(config)
        if not success:
            return 1
        print()

    # Step 2: Verify file exists and show info
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
    sys.exit(asyncio.run(main()))
