#!/usr/bin/env python3
"""
Test script for the enhanced Optimizer with sample data.
Demonstrates transfer optimization, captain selection, and chip strategy.
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
from src.optimizer import Optimizer


async def main():
    """Test the optimizer with real data."""
    config = Config.from_env()
    config.dry_run = False  # Use real API

    print("=" * 70)
    print("OPTIMIZER TEST")
    print("=" * 70)
    print()

    async with FPLClient(config) as client:
        collector = DataCollector(config, client)

        print("[1] Collecting data (may use cache)...")
        print("-" * 70)
        player_df = await collector.collect_all(gameweeks_ahead=5)
        print(f"    Players analyzed: {len(player_df)}")
        print()

        # Create optimizer
        optimizer = Optimizer(config, client, collector)

        print("[2] Running optimization...")
        print("-" * 70)

        # Run offline demo since we can't authenticate
        # In production, you would use: result = await optimizer.optimize(...)
        print("    Running offline demonstration with sample data...")
        print()
        await run_offline_demo(optimizer, player_df, collector)

        print()
        print("=" * 70)
        print("OPTIMIZER TEST COMPLETE")
        print("=" * 70)


async def run_offline_demo(optimizer: Optimizer, player_df, collector):
    """Run offline demonstration when API auth fails."""
    from src.fpl_client import Player
    from src.optimizer import (
        TransferOptimizer,
        CaptainSelector,
        ChipStrategyAdvisor,
        OptimizationResult,
    )

    print("[OFFLINE MODE] Using sample squad for demonstration")
    print("-" * 70)
    print()

    # Create sample squad from top players
    sample_squad = []
    positions = {1: 2, 2: 5, 3: 5, 4: 3}  # GK, DEF, MID, FWD

    for pos_id, count in positions.items():
        pos_df = player_df[player_df["position_id"] == pos_id].head(count)
        for _, row in pos_df.iterrows():
            player = Player(
                id=int(row["player_id"]),
                web_name=str(row["name"]),
                first_name=str(row.get("full_name", row["name"])).split()[0] if row.get("full_name") else str(row["name"]),
                second_name=str(row.get("full_name", row["name"])).split()[-1] if row.get("full_name") else str(row["name"]),
                team=int(row["team_id"]),
                team_name=str(row.get("team_name", "")),
                element_type=pos_id,
                now_cost=float(row["price"]),
                total_points=int(row.get("total_points", 0)),
                form=float(row.get("form", 0)),
                points_per_game=float(row.get("points_per_game", 0)),
                selected_by_percent=float(row.get("selected_by", 0)),
                minutes=int(row.get("minutes", 0)),
                goals_scored=int(row.get("goals", 0)),
                assists=int(row.get("assists", 0)),
                clean_sheets=int(row.get("clean_sheets", 0)),
                goals_conceded=0,
                bonus=int(row.get("bonus", 0)),
                bps=0,
                expected_goals=float(row.get("xG", 0)),
                expected_assists=float(row.get("xA", 0)),
                expected_goal_involvements=float(row.get("xGI", 0)),
                expected_goals_conceded=0.0,
                status="a",
                news="",
                news_added=None,
                chance_of_playing_this_round=None,
                chance_of_playing_next_round=None,
                transfers_in_event=0,
                transfers_out_event=0,
                cost_change_event=0,
                cost_change_start=0,
            )
            sample_squad.append(player)

    print(f"[SAMPLE SQUAD] {len(sample_squad)} players:")
    for i, p in enumerate(sample_squad):
        pos_name = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}[p.element_type]
        print(f"    {i+1:2}. {p.web_name:15} {pos_name:3} {p.now_cost:5.1f}m")
    print()

    # Run transfer optimization
    print("[3] Transfer Optimization (2 free transfers, 5.0m budget):")
    print("-" * 70)
    transfer_optimizer = TransferOptimizer(
        current_squad=sample_squad,
        player_df=player_df,
        budget=5.0,
        free_transfers=2,
        max_transfers=2,
    )
    transfers = transfer_optimizer.optimize()

    if transfers:
        for i, t in enumerate(transfers, 1):
            print(f"    {i}. {t}")
    else:
        print("    No transfers recommended - squad is optimal")
    print()

    # Run captain selection
    print("[4] Captain Selection:")
    print("-" * 70)
    starting_xi = sample_squad[:11]
    captain_selector = CaptainSelector(starting_xi, player_df)
    captains = captain_selector.select()

    if captains:
        print(f"    Captain:      {captains[0]}")
        if len(captains) > 1:
            print(f"    Vice-Captain: {captains[1]}")
        differentials = [c for c in captains if c.is_differential]
        if differentials:
            print(f"    Differential: {differentials[0]}")
    print()

    # Run chip analysis
    print("[5] Chip Strategy Analysis:")
    print("-" * 70)
    fixtures_df = collector.get_fixture_ticker()
    chip_advisor = ChipStrategyAdvisor(
        squad=sample_squad,
        player_df=player_df,
        fixtures_df=fixtures_df,
        available_chips=["wildcard", "bench_boost", "free_hit", "triple_captain"],
        free_transfers=2,
    )
    chips = chip_advisor.analyze()

    for chip in chips:
        status = ">>> PLAY" if chip.is_recommended else "   Hold"
        print(f"    {status} {chip.chip_name.upper()}: Score {chip.score:.0f}/100")
        for reason in chip.reasons[:2]:
            print(f"             - {reason}")
    print()

    # Validate squad
    print("[6] Squad Validation:")
    print("-" * 70)
    errors = optimizer.validate_squad(sample_squad)
    if errors:
        print("    Errors:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("    Squad is valid!")

    # Validate formation
    is_valid, formation = optimizer.validate_formation(starting_xi)
    print(f"    Formation: {formation} ({'Valid' if is_valid else 'Invalid'})")


if __name__ == "__main__":
    asyncio.run(main())
