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
    async def test_login_dry_run(self, client):
        """Test login in dry-run mode."""
        async with client:
            result = await client.login()
            assert result is True
            assert client._authenticated is True

    @pytest.mark.asyncio
    async def test_get_my_team_dry_run(self, client):
        """Test getting team data in dry-run mode returns MyTeam dataclass."""
        async with client:
            team = await client.get_my_team()

            # MyTeam is a dataclass, not a dict
            assert hasattr(team, 'picks')
            assert hasattr(team, 'bank')
            assert hasattr(team, 'free_transfers')
            assert len(team.picks) == 15

    @pytest.mark.asyncio
    async def test_make_transfers_dry_run(self, client):
        """Test making transfers in dry-run mode."""
        from src.fpl_client import Transfer

        transfers = [
            Transfer(element_in=100, element_out=200),
        ]

        async with client:
            result = await client.make_transfers(transfers)

            assert result.success is True
            assert "dry_run" in result.message.lower() or "dry run" in result.message.lower()

    @pytest.mark.asyncio
    async def test_set_captain_dry_run(self, client):
        """Test setting captain in dry-run mode."""
        async with client:
            # Without confirm, returns preview
            result = await client.set_captain(captain_id=1, vice_captain_id=2, confirm=False)

            assert result["success"] is True
            assert result["captain"] == 1
            assert result["vice_captain"] == 2

    @pytest.mark.asyncio
    async def test_login_live_mode_without_credentials(self):
        """Test that login fails in live mode without credentials."""
        config = Config(dry_run=False, fpl_email="", fpl_password="")
        client = FPLClient(config)

        async with client:
            with pytest.raises(AuthenticationError):
                await client.login()

    def test_mock_team_data(self, client):
        """Test mock team data structure via _get_mock_team."""
        mock_team = client._get_mock_team()

        # MyTeam dataclass attributes
        assert len(mock_team.picks) == 15
        assert mock_team.bank == 5.0
        assert mock_team.free_transfers == 2

        # Check captain/vice-captain via properties
        assert mock_team.captain_id is not None
        assert mock_team.vice_captain_id is not None


class TestFPLClientRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_limiter_exists(self):
        """Test that rate limiter is configured."""
        assert FPLClient._rate_limiter is not None
        # Rate limiter is set to 1 request per second for safety
        assert FPLClient._rate_limiter.max_rate == 1


class TestFPLClientDataFetching:
    """Test data fetching methods."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(dry_run=True, fpl_team_id=12345)

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return FPLClient(config)

    @pytest.mark.asyncio
    async def test_get_players(self, client):
        """Test fetching all players."""
        async with client:
            players = await client.get_players()

            assert isinstance(players, list)
            assert len(players) > 0
            assert all(isinstance(p, Player) for p in players)

    @pytest.mark.asyncio
    async def test_get_teams(self, client):
        """Test fetching all teams."""
        async with client:
            teams = await client.get_teams()

            assert isinstance(teams, list)
            assert len(teams) == 20  # Premier League has 20 teams
            assert all(isinstance(t, Team) for t in teams)

    @pytest.mark.asyncio
    async def test_get_current_gameweek(self, client):
        """Test fetching current gameweek."""
        async with client:
            gw = await client.get_current_gameweek()

            assert gw is not None
            assert hasattr(gw, 'id')
            assert hasattr(gw, 'name')
            assert 1 <= gw.id <= 38
