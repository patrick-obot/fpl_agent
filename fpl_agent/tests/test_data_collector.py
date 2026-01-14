"""Tests for data collection module."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.fpl_client import FPLClient, Player, Team
from src.data_collector import DataCollector, FixtureDifficulty, PlayerNews


class TestFixtureDifficulty:
    """Test suite for FixtureDifficulty dataclass."""

    def test_adjusted_difficulty_home(self):
        """Test adjusted difficulty for home games."""
        fixture = FixtureDifficulty(
            team_id=1,
            team_name="Arsenal",
            team_short="ARS",
            gameweek=1,
            opponent_id=2,
            opponent_name="Chelsea",
            opponent_short="CHE",
            is_home=True,
            difficulty=3,
        )

        # Home advantage reduces difficulty by 0.5
        assert fixture.adjusted_difficulty == 2.5

    def test_adjusted_difficulty_away(self):
        """Test adjusted difficulty for away games."""
        fixture = FixtureDifficulty(
            team_id=1,
            team_name="Arsenal",
            team_short="ARS",
            gameweek=1,
            opponent_id=2,
            opponent_name="Chelsea",
            opponent_short="CHE",
            is_home=False,
            difficulty=3,
        )

        # Away increases difficulty by 0.5
        assert fixture.adjusted_difficulty == 3.5


class TestPlayerNews:
    """Test suite for PlayerNews dataclass."""

    def test_is_available_status_a(self):
        """Test availability check for status 'a'."""
        news = PlayerNews(
            player_id=1,
            player_name="Salah",
            team_id=14,
            team_name="Liverpool",
            news="",
            chance_of_playing=None,
            status='a',
            source='fpl',
        )

        assert news.is_available is True

    def test_is_available_high_chance(self):
        """Test availability check with high chance of playing."""
        news = PlayerNews(
            player_id=1,
            player_name="Salah",
            team_id=14,
            team_name="Liverpool",
            news="Minor knock",
            chance_of_playing=75,
            status='d',
            source='fpl',
        )

        assert news.is_available is True

    def test_is_available_low_chance(self):
        """Test availability check with low chance of playing."""
        news = PlayerNews(
            player_id=1,
            player_name="Salah",
            team_id=14,
            team_name="Liverpool",
            news="Injured",
            chance_of_playing=25,
            status='i',
            source='fpl',
        )

        assert news.is_available is False


class TestDataCollector:
    """Test suite for DataCollector class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(dry_run=True)

    @pytest.fixture
    def mock_client(self, config):
        """Create mock FPL client."""
        client = MagicMock(spec=FPLClient)
        client.config = config

        # Mock current gameweek
        client.get_current_gameweek = AsyncMock(return_value=10)

        # Mock teams
        client.get_teams = AsyncMock(return_value=[
            Team(id=1, name="Arsenal", short_name="ARS", code=3,
                 strength=5, strength_overall_home=1300,
                 strength_overall_away=1280, strength_attack_home=1320,
                 strength_attack_away=1300, strength_defence_home=1280,
                 strength_defence_away=1260),
            Team(id=2, name="Chelsea", short_name="CHE", code=8,
                 strength=4, strength_overall_home=1250,
                 strength_overall_away=1230, strength_attack_home=1270,
                 strength_attack_away=1250, strength_defence_home=1230,
                 strength_defence_away=1210),
        ])

        # Mock fixtures
        client.get_fixtures = AsyncMock(return_value=[
            {
                "team_h": 1, "team_a": 2,
                "team_h_difficulty": 3, "team_a_difficulty": 4,
            }
        ])

        # Mock players
        client.get_players = AsyncMock(return_value=[
            Player(
                id=1, web_name="Saka", first_name="Bukayo", second_name="Saka",
                team=1, team_name="Arsenal", element_type=3,
                now_cost=9.0, total_points=100, form=7.0,
                points_per_game=6.5, selected_by_percent=30.0, minutes=800,
                goals_scored=5, assists=8, clean_sheets=3, goals_conceded=10,
                bonus=15, bps=300, expected_goals=4.5, expected_assists=6.0,
                expected_goal_involvements=10.5, expected_goals_conceded=8.0,
                status='a', news='', news_added=None,
                chance_of_playing_this_round=None, chance_of_playing_next_round=None,
                transfers_in_event=50000, transfers_out_event=30000,
                cost_change_event=0, cost_change_start=0
            ),
        ])

        # Mock bootstrap static
        client.get_bootstrap_static = AsyncMock(return_value={
            "elements": [
                {
                    "id": 1,
                    "news": "",
                    "chance_of_playing_next_round": None,
                    "status": "a",
                }
            ]
        })

        return client

    @pytest.fixture
    def collector(self, config, mock_client):
        """Create data collector instance."""
        return DataCollector(config, mock_client)

    @pytest.mark.asyncio
    async def test_collect_fixture_difficulties(self, collector):
        """Test fixture difficulty collection."""
        # API now takes a list of gameweeks
        difficulties = await collector.collect_fixture_difficulties(gameweeks=[10, 11])

        assert isinstance(difficulties, dict)
        # Should have team fixtures for the teams in the mock data
        assert 1 in difficulties or 2 in difficulties

    def test_get_fixture_difficulty_score_default(self, collector):
        """Test fixture score with no data returns default."""
        score = collector.get_fixture_difficulty_score(team_id=999)

        assert score == 3.0  # Default medium difficulty

    def test_player_news_availability_score(self, collector):
        """Test PlayerNews availability_score property."""
        news = PlayerNews(
            player_id=1,
            player_name="Test",
            team_id=1,
            team_name="Arsenal",
            news="Doubtful",
            chance_of_playing=50,
            status='d',
            source='fpl',
        )

        # availability_score should return 0.5 for 50% chance
        assert news.availability_score == 0.5

    def test_player_news_availability_from_status(self, collector):
        """Test PlayerNews availability_score from status when no chance specified."""
        news = PlayerNews(
            player_id=1,
            player_name="Test",
            team_id=1,
            team_name="Arsenal",
            news="",
            chance_of_playing=None,
            status='a',
            source='fpl',
        )

        # Status 'a' (available) should return 1.0
        assert news.availability_score == 1.0

    def test_get_projected_points_no_data(self, collector):
        """Test projected points with no loaded data returns 0."""
        points = collector.get_projected_points(player_id=1)

        # Returns 0.0 when no data available
        assert points == 0.0

    def test_get_team_fixtures(self, collector):
        """Test getting team fixtures returns None when no data."""
        fixtures = collector.get_team_fixtures(team_id=999)

        assert fixtures is None

    def test_export_fixtures_to_csv(self, collector):
        """Test exporting fixture analysis to CSV."""
        from src.data_collector import TeamFixtures

        # Add some fixture data using TeamFixtures
        collector._team_fixtures = {
            1: TeamFixtures(
                team_id=1,
                team_name="Arsenal",
                team_short="ARS",
                fixtures=[
                    FixtureDifficulty(
                        team_id=1, team_name="Arsenal", team_short="ARS",
                        gameweek=10, opponent_id=2, opponent_name="Chelsea",
                        opponent_short="CHE", is_home=True, difficulty=3
                    )
                ]
            )
        }

        output_path = collector.export_fixtures_to_csv()

        assert output_path.exists()
        assert output_path.suffix == ".csv"

        # Cleanup
        output_path.unlink()
