#!/usr/bin/env python3
"""
FPL Review CSV Downloader & Uploader (Raspberry Pi)

Downloads projected_points.csv from FPL Review via Patreon OAuth,
then SCPs it to the VPS. Designed to run via cron.

Usage:
    python download_and_upload.py              # Headless (cron)
    python download_and_upload.py --no-headless  # Interactive (first run, Patreon login)
"""

import asyncio
import fcntl
import logging
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

# Import FPLReviewClient from same directory
sys.path.insert(0, str(Path(__file__).parent))
from fplreview_client import FPLReviewClient


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load configuration from .env file."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print(f"ERROR: {env_file} not found. Copy .env.example to .env and fill in values.")
        sys.exit(1)

    load_dotenv(env_file)

    required = ["FPL_REVIEW_EMAIL", "FPL_REVIEW_PASSWORD", "VPS_HOST", "VPS_USER", "VPS_TARGET_PATH"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing required env vars: {', '.join(missing)}")
        sys.exit(1)

    return {
        "fpl_review_email": os.getenv("FPL_REVIEW_EMAIL"),
        "fpl_review_password": os.getenv("FPL_REVIEW_PASSWORD"),
        "fpl_team_id": os.getenv("FPL_TEAM_ID", ""),
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
        "vps_host": os.getenv("VPS_HOST"),
        "vps_user": os.getenv("VPS_USER"),
        "vps_ssh_key": os.getenv("VPS_SSH_KEY", "~/.ssh/fpl_pi_to_vps"),
        "vps_target_path": os.getenv("VPS_TARGET_PATH"),
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("fpl_pi")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

async def send_telegram(config: dict, message: str) -> None:
    """Send Telegram notification. Silently fails if not configured."""
    token = config.get("telegram_bot_token")
    chat_id = config.get("telegram_chat_id")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)):
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SCP Upload
# ---------------------------------------------------------------------------

def scp_to_vps(local_path: Path, config: dict, logger: logging.Logger) -> bool:
    """SCP the CSV file to VPS."""
    ssh_key = os.path.expanduser(config["vps_ssh_key"])
    target = f"{config['vps_user']}@{config['vps_host']}:{config['vps_target_path']}"

    cmd = [
        "scp",
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        str(local_path),
        target,
    ]

    logger.info(f"SCP: {local_path.name} -> {target}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode == 0:
        logger.info("SCP upload successful")
        return True
    else:
        logger.error(f"SCP failed: {result.stderr.strip()}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(headless: bool = True) -> bool:
    """Download CSV and upload to VPS. Returns True on success."""
    logger = setup_logging()
    config = load_config()

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"Starting FPL Review download ({timestamp})")

    # Step 1: Download CSV
    client = FPLReviewClient(
        email=config["fpl_review_email"],
        password=config["fpl_review_password"],
        download_dir=data_dir,
        logger=logger,
        team_id=config["fpl_team_id"],
    )

    csv_path = await client.download_projections_csv(headless=headless)

    if not csv_path or not csv_path.exists():
        logger.error("CSV download failed")
        await send_telegram(config,
            f"❌ <b>FPL Review Download Failed</b>\n\n"
            f"📍 Source: Raspberry Pi\n"
            f"⏰ {timestamp}\n"
            f"Check screenshots in {data_dir}")
        return False

    file_size = csv_path.stat().st_size
    logger.info(f"CSV downloaded: {csv_path} ({file_size:,} bytes)")

    # Step 2: SCP to VPS
    ok = scp_to_vps(csv_path, config, logger)

    if ok:
        await send_telegram(config,
            f"✅ <b>FPL Review Projections Updated</b>\n\n"
            f"📍 Source: Raspberry Pi\n"
            f"📁 {csv_path.name} ({file_size:,} bytes)\n"
            f"🖥️ Uploaded to VPS\n"
            f"⏰ {timestamp}")
    else:
        await send_telegram(config,
            f"⚠️ <b>FPL Review CSV Downloaded but SCP Failed</b>\n\n"
            f"📍 Source: Raspberry Pi\n"
            f"📁 Local file OK ({file_size:,} bytes)\n"
            f"❌ SCP to VPS failed\n"
            f"⏰ {timestamp}")

    return ok


def main():
    headless = "--no-headless" not in sys.argv

    # File lock to prevent overlapping cron runs
    lock_path = "/tmp/fpl_download.lock"
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another download is already running, exiting.")
        sys.exit(0)

    try:
        success = asyncio.run(run(headless=headless))
        sys.exit(0 if success else 1)
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


if __name__ == "__main__":
    main()
