#!/usr/bin/env python3
"""
Test script for the enhanced DataCollector.
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.fpl_client import FPLClient
from src.data_collector import DataCollector


async def main():
    """Test the data collector."""
    config = Config.from_env()
    config.dry_run = False  # Use real API

    print("=" * 70)
    print("DATA COLLECTOR TEST")
    print("=" * 70)
    print()

    async with FPLClient(config) as client:
        collector = DataCollector(config, client)

        # Clear cache to test fresh collection
        print("[0] Clearing cache for fresh test...")
        collector.clear_cache()
        print()

        # Collect all data
        print("[1] Collecting all data (fixtures, news, projections)...")
        print("-" * 70)
        df = await collector.collect_all(gameweeks_ahead=5)
        print()

        # Show DataFrame info
        print("[2] DataFrame Summary:")
        print("-" * 70)
        print(f"    Total players: {len(df)}")
        print(f"    Columns: {len(df.columns)}")
        print(f"    Columns: {', '.join(df.columns[:15])}...")
        print()

        # Show top players by composite score
        print("[3] Top 15 Players by Composite Score:")
        print("-" * 70)
        top_cols = ["name", "team", "position", "price", "form", "fixture_difficulty", "composite_score"]
        print(df[top_cols].head(15).to_string(index=False))
        print()

        # Show top players by position
        print("[4] Top 5 Players by Position:")
        print("-" * 70)
        for pos in ["GOALKEEPER", "DEFENDER", "MIDFIELDER", "FORWARD"]:
            print(f"\n  {pos}S:")
            pos_df = collector.get_top_players(position=pos, limit=5)
            if not pos_df.empty:
                for _, row in pos_df.iterrows():
                    print(f"    {row['name']:15} {row['team']:4} {row['price']:5.1f}m  Form: {row['form']:4.1f}  Score: {row['composite_score']:.1f}")
        print()

        # Show fixture difficulty ticker
        print("[5] Fixture Difficulty Ticker (Best to Worst):")
        print("-" * 70)
        ticker = collector.get_fixture_ticker(gameweeks=5)
        if not ticker.empty:
            # Find fixture columns
            gw_cols = [c for c in ticker.columns if c.startswith("gw") and not c.endswith("_diff")]
            display_cols = ["team", "avg_difficulty", "rating"] + gw_cols[:5]
            available_cols = [c for c in display_cols if c in ticker.columns]
            print(ticker[available_cols].head(20).to_string(index=False))
        print()

        # Show flagged players
        print("[6] Flagged Players (Injury/Availability Concerns):")
        print("-" * 70)
        flagged = collector.get_flagged_players()
        if not flagged.empty:
            flagged_display = flagged[["name", "team", "status", "news", "chance_of_playing", "availability"]].head(15)
            for _, row in flagged_display.iterrows():
                availability = f"{row['availability']*100:.0f}%" if row['availability'] < 1 else "OK"
                news_preview = row['news'][:40] + "..." if len(str(row['news'])) > 40 else row['news']
                print(f"    {row['name']:15} {row['team']:4} [{row['status']}] {availability:>4}  {news_preview}")
        else:
            print("    No flagged players!")
        print()

        # Show projected points
        print("[7] Players with Projected Points:")
        print("-" * 70)
        proj_cols = ["name", "team", "position", "price", "projected_total", "gw1_projected", "gw2_projected", "gw3_projected"]
        available_proj_cols = [c for c in proj_cols if c in df.columns]
        proj_df = df[df["projected_total"] > 0][available_proj_cols].head(15)
        if not proj_df.empty:
            print(proj_df.to_string(index=False))
        else:
            print("    No projected points loaded.")
        print()

        # Export data
        print("[8] Exporting data to CSV...")
        print("-" * 70)
        try:
            csv_path = collector.export_to_csv()
            print(f"    Player analysis: {csv_path}")
            fixtures_path = collector.export_fixtures_to_csv()
            print(f"    Fixture ticker: {fixtures_path}")
        except Exception as e:
            print(f"    Export error: {e}")
        print()

        # Show cache status
        print("[9] Cache Status:")
        print("-" * 70)
        print(f"    Cache file: {collector._cache_path}")
        print(f"    Cache entries: {len(collector._cache)}")
        print(f"    Cache TTL: {collector.CACHE_TTL / 3600:.1f} hours")
        print()

        print("=" * 70)
        print("DATA COLLECTOR TEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
