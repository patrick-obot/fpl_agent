"""
Test FPL Review data collection.

Usage:
    python test_fplreview.py
    python test_fplreview.py --visible  # Show browser window
"""

import asyncio
import sys
from pathlib import Path

from src.config import Config
from src.fplreview_client import download_fplreview_projections


async def main():
    # Parse args
    headless = "--visible" not in sys.argv

    # Load config
    config = Config.from_env()

    if not config.fpl_review_email or not config.fpl_review_password:
        print("ERROR: FPL Review credentials not configured!")
        print()
        print("Add to your .env file:")
        print("  FPL_REVIEW_EMAIL=your_patreon_email")
        print("  FPL_REVIEW_PASSWORD=your_patreon_password")
        return 1

    print("=" * 60)
    print("FPL Review Projection Downloader")
    print("=" * 60)
    print(f"Email: {config.fpl_review_email}")
    print(f"Team ID: {config.fpl_team_id}")
    print(f"Headless: {headless}")
    print()

    # Download projections
    print("Downloading projections...")
    csv_path = await download_fplreview_projections(
        email=config.fpl_review_email,
        password=config.fpl_review_password,
        download_dir=config.data_dir,
        headless=headless,
        logger=config.logger,
        team_id=str(config.fpl_team_id),
    )

    if csv_path:
        print()
        print("=" * 60)
        print(f"SUCCESS! Downloaded to: {csv_path}")
        print("=" * 60)

        # Show file info
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)[:10]}...")
        return 0
    else:
        print()
        print("FAILED to download projections.")
        print("Check data/fplreview_debug.png for screenshot.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
