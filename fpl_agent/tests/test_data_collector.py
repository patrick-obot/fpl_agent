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
            gameweek=1,
            opponent_id=2,
            opponent_name="Chelsea",
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
            gameweek=1,
            opponent_id=2,
            opponent_name="Chelsea",
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
            news="",
            chance_of_playing=None,
            status='a',
        )

        assert news.is_available is True

    def test_is_available_high_chance(self):
        """Test availability check with high chance of playing."""
        news = PlayerNews(
            player_id=1,
            player_name="Salah",
            team_id=14,
            news="Minor knock",
            chance_of_playing=75,
            status='d',
        )

        assert news.is_available is True

    def test_is_available_low_chance(self):
        """Test availability check with low chance of playing."""
        news = PlayerNews(
            player_id=1,
            player_name="Salah",
            team_id=14,
            news="Injured",
            chance_of_playing=25,
            status='i',
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
            Team(id=1, name="Arsenal", short_name="ARS",
                 strength=5, strength_overall_home=1300,
                 strength_overall_away=1280, strength_attack_home=1320,
                 strength_attack_away=1300, strength_defence_home=1280,
                 strength_defence_away=1260),
            Team(id=2, name="Chelsea", short_name="CHE",
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
            Player(id=1, web_name="Saka", team=1, element_type=3,
                   now_cost=9.0, total_points=100, form=7.0,
                   selected_by_percent=30.0, minutes=800,
                   goals_scored=5, assists=8, clean_sheets=3,
                   expected_goals=4.5, expected_assists=6.0,
                   expected_goal_involvements=10.5),
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
        difficulties = await collector.collect_fixture_difficulties(gameweeks_ahead=1)

        assert len(difficulties) == 2  # Two teams
        assert 1 in difficulties
        assert 2 in difficulties

    def test_get_team_fixture_score_default(self, collector):
        """Test fixture score with no data."""
        score = collector.get_team_fixture_score(team_id=999)

        assert score == 3.0  # Default medium difficulty

    def test_get_player_availability_no_news(self, collector):
        """Test availability for player with no news."""
        availability = collector.get_player_availability(player_id=999)

        assert availability == 1.0  # Assume available

    def test_get_player_availability_with_news(self, collector):
        """Test availability for player with news."""
        collector._player_news[1] = PlayerNews(
            player_id=1,
            player_name="Test",
            team_id=1,
            news="Injured",
            chance_of_playing=50,
            status='d',
        )

        availability = collector.get_player_availability(player_id=1)

        assert availability == 0.5

    def test_get_projected_points_no_data(self, collector):
        """Test projected points with no loaded data."""
        points = collector.get_projected_points(player_id=1)

        assert points is None

    @pytest.mark.asyncio
    async def test_create_player_dataframe(self, collector, mock_client):
        """Test creating player analysis DataFrame."""
        players = await mock_client.get_players()
        df = collector.create_player_dataframe(players)

        assert len(df) == 1
        assert "player_id" in df.columns
        assert "name" in df.columns
        assert "fixture_score" in df.columns
        assert "availability" in df.columns
        assert "value_score" in df.columns

    def test_export_analysis(self, collector):
        """Test exporting analysis to CSV."""
        # Add some fixture data
        collector._fixture_difficulties = {
            1: [
                FixtureDifficulty(
                    team_id=1, team_name="Arsenal", gameweek=10,
                    opponent_id=2, opponent_name="Chelsea",
                    is_home=True, difficulty=3
                )
            ]
        }

        output_path = collector.export_analysis()

        assert output_path.exists()
        assert output_path.suffix == ".csv"

        # Cleanup
        output_path.unlink()
