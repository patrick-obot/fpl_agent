#!/usr/bin/env python3
"""
Test script to fetch current team data from FPL API.
Uses public endpoints where possible, authenticated endpoints where needed.
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.fpl_client import FPLClient, AuthenticationError, FPLAPIError


async def fetch_team_via_public_api(client: FPLClient, team_id: int, gameweek: int) -> dict:
    """Fetch team picks via public entry history endpoint."""
    # This endpoint is public and doesn't require authentication
    data = await client._request("GET", f"entry/{team_id}/event/{gameweek}/picks/")
    return data


async def main():
    """Fetch and display team data."""
    # Load config
    config = Config.from_env()
    config.dry_run = False  # Actually connect to API

    print("=" * 60)
    print("FPL CLIENT TEST - Fetching Your Team Data")
    print("=" * 60)
    print(f"Team ID: {config.fpl_team_id}")
    print()

    try:
        async with FPLClient(config) as client:
            # Get manager info (public endpoint)
            print("[1] Fetching manager info...")
            manager = await client.get_manager_info()
            print(f"    Team Name: {manager.name}")
            print(f"    Manager: {manager.full_name}")
            print(f"    Overall Points: {manager.summary_overall_points:,}")
            if manager.summary_overall_rank:
                print(f"    Overall Rank: {manager.summary_overall_rank:,}")
            print(f"    GW{manager.current_event} Points: {manager.summary_event_points}")
            print()

            # Get current gameweek
            print("[2] Fetching gameweek info...")
            gw = await client.get_current_gameweek()
            print(f"    Current Gameweek: {gw.id} ({gw.name})")
            print(f"    Deadline: {gw.deadline_time}")
            print(f"    Finished: {gw.finished}")
            if gw.average_entry_score:
                print(f"    Average Score: {gw.average_entry_score}")
            print()

            # Get all players for reference
            print("[3] Loading player database...")
            players = await client.get_players()
            player_map = {p.id: p for p in players}
            teams = await client.get_teams()
            team_map = {t.id: t for t in teams}
            print(f"    Loaded {len(players)} players from {len(teams)} teams")
            print()

            # Get team picks via public endpoint
            print("[4] Fetching your team picks...")
            picks_data = await fetch_team_via_public_api(client, config.fpl_team_id, gw.id)

            # Parse picks
            picks = picks_data.get("picks", [])
            entry_history = picks_data.get("entry_history", {})

            bank = entry_history.get("bank", 0) / 10
            value = entry_history.get("value", 0) / 10
            points = entry_history.get("points", 0)
            total_points = entry_history.get("total_points", 0)
            event_transfers = entry_history.get("event_transfers", 0)
            event_transfers_cost = entry_history.get("event_transfers_cost", 0)

            print(f"    Bank: {bank}m")
            print(f"    Squad Value: {value}m")
            print(f"    Total Value: {bank + value}m")
            print(f"    GW Points: {points} (Total: {total_points})")
            print(f"    Transfers this GW: {event_transfers} (Cost: {event_transfers_cost} pts)")
            print()

            # Active chip
            active_chip = picks_data.get("active_chip")
            if active_chip:
                print(f"    Active Chip: {active_chip}")
                print()

            # Starting XI
            print("[5] Starting XI:")
            print("-" * 60)
            print(f"    {'Pos':<4} {'Player':<18} {'Team':<5} {'Position':<10} {'Price':>6} {'Form':>5} {'Pts':>4}")
            print("-" * 60)

            captain_id = None
            vice_captain_id = None
            for pick in picks:
                if pick.get("is_captain"):
                    captain_id = pick["element"]
                if pick.get("is_vice_captain"):
                    vice_captain_id = pick["element"]

            for pick in picks[:11]:  # First 11 are starters
                player_id = pick["element"]
                player = player_map.get(player_id)
                multiplier = pick.get("multiplier", 1)

                if player:
                    captain_mark = ""
                    if pick.get("is_captain"):
                        captain_mark = " (C)"
                    elif pick.get("is_vice_captain"):
                        captain_mark = " (VC)"

                    status = ""
                    if not player.is_available:
                        status = f" [{player.status.upper()}]"
                    elif player.news:
                        status = " [!]"

                    gw_points = pick.get("points", 0) * multiplier if pick.get("points") else "-"

                    print(f"    {pick['position']:<4} {player.web_name:<18} {player.team_name:<5} {player.position.name:<10} {player.now_cost:>5.1f}m {player.form:>5.1f} {gw_points:>4}{captain_mark}{status}")
                else:
                    print(f"    {pick['position']:<4} Player ID {player_id} (not found)")
            print()

            # Bench
            print("[6] Bench:")
            print("-" * 60)
            for pick in picks[11:]:  # Rest are bench
                player_id = pick["element"]
                player = player_map.get(player_id)

                if player:
                    status = ""
                    if not player.is_available:
                        status = f" [{player.status.upper()}]"

                    print(f"    {pick['position']:<4} {player.web_name:<18} {player.team_name:<5} {player.position.name:<10} {player.now_cost:>5.1f}m {player.form:>5.1f}{status}")
                else:
                    print(f"    {pick['position']:<4} Player ID {player_id} (not found)")
            print()

            # Players with news/flags
            print("[7] Players with News/Flags:")
            print("-" * 60)
            squad_ids = {p["element"] for p in picks}
            flagged_players = [
                player_map[pid] for pid in squad_ids
                if pid in player_map and (player_map[pid].news or not player_map[pid].is_available)
            ]
            if flagged_players:
                for player in flagged_players:
                    availability = player.availability_percent
                    print(f"    {player.web_name} ({player.team_name})")
                    if player.news:
                        print(f"        News: {player.news}")
                    print(f"        Status: {player.status} | Chance: {availability}%")
            else:
                print("    No flagged players in your squad!")
            print()

            # Captain info
            captain = player_map.get(captain_id)
            vice = player_map.get(vice_captain_id)
            print("[8] Captaincy:")
            print("-" * 60)
            if captain:
                print(f"    Captain: {captain.web_name} ({captain.team_name}) - {captain.form} form")
            if vice:
                print(f"    Vice Captain: {vice.web_name} ({vice.team_name}) - {vice.form} form")
            print()

            # Get next gameweek fixtures
            print("[9] Next Fixtures for Your Teams:")
            print("-" * 60)
            next_gw = gw.id + 1 if gw.finished else gw.id
            fixtures = await client.get_fixtures(next_gw)

            team_fixtures = {}
            for fixture in fixtures:
                home_team = team_map.get(fixture.home_team)
                away_team = team_map.get(fixture.away_team)
                if home_team and away_team:
                    team_fixtures[fixture.home_team] = (
                        f"vs {away_team.short_name} (H)",
                        fixture.home_team_difficulty
                    )
                    team_fixtures[fixture.away_team] = (
                        f"vs {home_team.short_name} (A)",
                        fixture.away_team_difficulty
                    )

            shown_teams = set()
            for pick in picks:
                player = player_map.get(pick["element"])
                if player and player.team not in shown_teams:
                    team_obj = team_map.get(player.team)
                    fixture_info = team_fixtures.get(player.team)
                    if team_obj and fixture_info:
                        diff_bar = "*" * fixture_info[1]
                        print(f"    {team_obj.short_name:<4}: {fixture_info[0]:<15} Difficulty: {diff_bar} ({fixture_info[1]})")
                    shown_teams.add(player.team)

            print()
            print("=" * 60)
            print("Test completed successfully!")
            print("=" * 60)

    except AuthenticationError as e:
        print(f"\n[ERROR] Authentication failed: {e}")
        return 1

    except FPLAPIError as e:
        print(f"\n[ERROR] API error: {e}")
        return 1

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
