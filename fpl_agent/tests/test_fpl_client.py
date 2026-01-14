"""Tests for FPL API client."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.fpl_client import FPLClient, Player, Team, FPLAPIError, AuthenticationError


class TestPlayer:
    """Test suite for Player dataclass."""

    def test_from_api(self):
        """Test creating Player from API response."""
        api_data = {
            "id": 1,
            "web_name": "Salah",
            "team": 14,
            "element_type": 3,
            "now_cost": 130,
            "total_points": 150,
            "form": "8.5",
            "selected_by_percent": "45.2",
            "minutes": 900,
            "goals_scored": 10,
            "assists": 5,
            "clean_sheets": 3,
            "expected_goals": "8.5",
            "expected_assists": "4.2",
            "expected_goal_involvements": "12.7",
        }

        player = Player.from_api(api_data)

        assert player.id == 1
        assert player.web_name == "Salah"
        assert player.team == 14
        assert player.element_type == 3
        assert player.now_cost == 13.0  # Converted from 130 to 13.0m
        assert player.total_points == 150
        assert player.form == 8.5
        assert player.expected_goal_involvements == 12.7

    def test_from_api_with_missing_optional_fields(self):
        """Test creating Player with missing optional fields."""
        api_data = {
            "id": 1,
            "web_name": "Test",
            "team": 1,
            "element_type": 1,
            "now_cost": 50,
            "total_points": 0,
        }

        player = Player.from_api(api_data)

        assert player.id == 1
        assert player.form == 0
        assert player.expected_goals == 0


class TestTeam:
    """Test suite for Team dataclass."""

    def test_from_api(self):
        """Test creating Team from API response."""
        api_data = {
            "id": 14,
            "name": "Liverpool",
            "short_name": "LIV",
            "strength": 5,
            "strength_overall_home": 1340,
            "strength_overall_away": 1320,
            "strength_attack_home": 1350,
            "strength_attack_away": 1330,
            "strength_defence_home": 1280,
            "strength_defence_away": 1260,
        }

        team = Team.from_api(api_data)

        assert team.id == 14
        assert team.name == "Liverpool"
        assert team.short_name == "LIV"
        assert team.strength == 5


class TestFPLClient:
    """Test suite for FPLClient class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(dry_run=True, fpl_team_id=12345)

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return FPLClient(config)

    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test async context manager."""
        async with client:
            assert client._session is not None

        assert client._session is None

    @pytest.mark.asyncio
    async def test_authenticate_dry_run(self, client):
        """Test authentication in dry-run mode."""
        result = await client.authenticate()
        assert result is True

    @pytest.mark.asyncio
    async def test_get_my_team_dry_run(self, client):
        """Test getting team data in dry-run mode."""
        async with client:
            team = await client.get_my_team()

            assert "picks" in team
            assert "transfers" in team
            assert len(team["picks"]) == 15

    @pytest.mark.asyncio
    async def test_make_transfers_dry_run(self, client):
        """Test making transfers in dry-run mode."""
        transfers = [
            {"element_in": 100, "element_out": 200},
        ]

        async with client:
            result = await client.make_transfers(transfers)

            assert result["status"] == "dry_run"
            assert result["transfers"] == transfers

    @pytest.mark.asyncio
    async def test_set_captain_dry_run(self, client):
        """Test setting captain in dry-run mode."""
        async with client:
            result = await client.set_captain(player_id=1, vice_captain_id=2)

            assert result["status"] == "dry_run"
            assert result["captain"] == 1
            assert result["vice_captain"] == 2

    @pytest.mark.asyncio
    async def test_authentication_required_live_mode(self):
        """Test that authentication is required in live mode."""
        config = Config(dry_run=False)
        client = FPLClient(config)

        with pytest.raises(AuthenticationError):
            await client.authenticate()

    def test_mock_team_data(self, client):
        """Test mock team data structure."""
        mock_data = client._get_mock_team_data()

        assert len(mock_data["picks"]) == 15
        assert mock_data["transfers"]["bank"] == 50

        # Check captain/vice-captain are set
        captains = [p for p in mock_data["picks"] if p["is_captain"]]
        vice_captains = [p for p in mock_data["picks"] if p["is_vice_captain"]]

        assert len(captains) == 1
        assert len(vice_captains) == 1


class TestFPLClientRateLimiting:
    """Test rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limiter_exists(self):
        """Test that rate limiter is configured."""
        assert FPLClient._rate_limiter is not None
        assert FPLClient._rate_limiter.max_rate == 100
