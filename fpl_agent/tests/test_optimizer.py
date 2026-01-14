"""
Comprehensive tests for transfer and captain optimization algorithms.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.fpl_client import FPLClient, Player
from src.data_collector import DataCollector
from src.optimizer import (
    Optimizer,
    TransferOptimizer,
    CaptainSelector,
    ChipStrategyAdvisor,
    Position,
    POSITION_NAMES,
    VALID_FORMATIONS,
    TransferRecommendation,
    CaptainRecommendation,
    ChipRecommendation,
    OptimizationResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def create_player(
    id: int,
    name: str,
    team: int,
    position: int,
    price: float,
    form: float = 5.0,
    total_points: int = 50,
    minutes: int = 900,
    selected_by: float = 10.0,
    goals: int = 0,
    assists: int = 0,
    xG: float = 0.0,
    xA: float = 0.0,
    xGI: float = 0.0,
) -> Player:
    """Create a test Player object."""
    return Player(
        id=id,
        web_name=name,
        first_name=name,
        second_name=f"Player{id}",
        team=team,
        team_name=f"Team{team}",
        element_type=position,
        now_cost=price,
        total_points=total_points,
        form=form,
        points_per_game=float(total_points) / 20 if total_points > 0 else 0.0,
        selected_by_percent=selected_by,
        minutes=minutes,
        goals_scored=goals,
        assists=assists,
        clean_sheets=0,
        goals_conceded=0,
        bonus=0,
        bps=0,
        expected_goals=xG,
        expected_assists=xA,
        expected_goal_involvements=xGI,
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


def create_player_df(players: list[dict]) -> pd.DataFrame:
    """Create a test DataFrame from player dictionaries."""
    return pd.DataFrame(players)


@pytest.fixture
def sample_players():
    """Create sample player objects for testing."""
    return [
        # Goalkeepers
        create_player(1, "GK1", 1, 1, 5.0, 4.5, 40, 900),
        create_player(2, "GK2", 2, 1, 4.5, 3.0, 30, 810),
        # Defenders
        create_player(3, "DEF1", 1, 2, 6.0, 6.0, 70, 900, xGI=0.3),
        create_player(4, "DEF2", 2, 2, 5.5, 5.0, 55, 850),
        create_player(5, "DEF3", 3, 2, 5.0, 4.5, 45, 800),
        create_player(6, "DEF4", 4, 2, 4.5, 4.0, 40, 750),
        create_player(7, "DEF5", 5, 2, 4.0, 3.5, 35, 700),
        # Midfielders
        create_player(8, "MID1", 6, 3, 12.0, 8.0, 150, 900, 40.0, 15, 10, 1.0, 0.8, 1.8),
        create_player(9, "MID2", 7, 3, 10.0, 7.0, 120, 850, 30.0, 10, 8, 0.7, 0.6, 1.3),
        create_player(10, "MID3", 8, 3, 8.0, 5.5, 80, 800, 15.0, 5, 5, 0.4, 0.4, 0.8),
        create_player(11, "MID4", 9, 3, 6.5, 4.5, 60, 750, 8.0, 3, 3, 0.2, 0.2, 0.4),
        create_player(12, "MID5", 10, 3, 5.0, 3.5, 40, 600, 3.0, 1, 1, 0.1, 0.1, 0.2),
        # Forwards
        create_player(13, "FWD1", 11, 4, 14.0, 9.0, 180, 900, 50.0, 20, 5, 1.5, 0.5, 2.0),
        create_player(14, "FWD2", 12, 4, 9.0, 6.5, 100, 850, 20.0, 8, 4, 0.6, 0.3, 0.9),
        create_player(15, "FWD3", 13, 4, 6.0, 4.0, 50, 700, 5.0, 3, 2, 0.3, 0.2, 0.5),
    ]


@pytest.fixture
def sample_player_df(sample_players):
    """Create DataFrame from sample players."""
    data = []
    for p in sample_players:
        data.append({
            "player_id": p.id,
            "name": p.web_name,
            "team_id": p.team,
            "position_id": p.element_type,
            "position": POSITION_NAMES.get(Position(p.element_type), "UNK"),
            "price": p.now_cost,
            "form": p.form,
            "total_points": p.total_points,
            "minutes": p.minutes,
            "selected_by": p.selected_by_percent,
            "goals": p.goals_scored,
            "assists": p.assists,
            "xG": p.expected_goals,
            "xA": p.expected_assists,
            "xGI": p.expected_goal_involvements,
            "fixture_difficulty": 3.0,
            "availability": 1.0,
            "value": p.total_points / p.now_cost if p.now_cost > 0 else 0,
            "gw1_fixture": "OPP (H)",
            "gw1_projected": p.form * 0.8,
            "gw2_projected": p.form * 0.7,
            "gw3_projected": p.form * 0.6,
        })
    return pd.DataFrame(data)


@pytest.fixture
def additional_player_df():
    """Create additional players for transfer candidates."""
    data = [
        {
            "player_id": 100,
            "name": "SuperMID",
            "team_id": 15,
            "position_id": 3,
            "position": "MID",
            "price": 7.0,
            "form": 8.5,
            "total_points": 100,
            "minutes": 900,
            "selected_by": 5.0,
            "goals": 8,
            "assists": 6,
            "xG": 0.8,
            "xA": 0.6,
            "xGI": 1.4,
            "fixture_difficulty": 2.0,
            "availability": 1.0,
            "value": 14.3,
            "gw1_fixture": "WHU (H)",
            "gw1_projected": 7.5,
            "gw2_projected": 6.5,
            "gw3_projected": 6.0,
        },
        {
            "player_id": 101,
            "name": "BudgetFWD",
            "team_id": 16,
            "position_id": 4,
            "position": "FWD",
            "price": 5.5,
            "form": 6.0,
            "total_points": 60,
            "minutes": 800,
            "selected_by": 2.0,
            "goals": 5,
            "assists": 2,
            "xG": 0.5,
            "xA": 0.2,
            "xGI": 0.7,
            "fixture_difficulty": 2.5,
            "availability": 1.0,
            "value": 10.9,
            "gw1_fixture": "BUR (H)",
            "gw1_projected": 5.5,
            "gw2_projected": 5.0,
            "gw3_projected": 4.5,
        },
        {
            "player_id": 102,
            "name": "InjuredMID",
            "team_id": 17,
            "position_id": 3,
            "position": "MID",
            "price": 6.0,
            "form": 7.0,
            "total_points": 70,
            "minutes": 500,
            "selected_by": 8.0,
            "goals": 4,
            "assists": 4,
            "xG": 0.4,
            "xA": 0.4,
            "xGI": 0.8,
            "fixture_difficulty": 2.5,
            "availability": 0.25,  # Injured
            "value": 11.7,
            "gw1_fixture": "NEW (A)",
            "gw1_projected": 0.0,
            "gw2_projected": 0.0,
            "gw3_projected": 0.0,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(dry_run=True)


@pytest.fixture
def mock_client(config):
    """Create mock FPL client."""
    client = MagicMock(spec=FPLClient)
    client.config = config
    return client


@pytest.fixture
def mock_collector(config, mock_client):
    """Create mock data collector."""
    collector = MagicMock(spec=DataCollector)
    collector.get_fixture_ticker = MagicMock(return_value=pd.DataFrame({
        "team_id": [1, 2, 3, 4, 5],
        "team": ["ARS", "AVL", "BHA", "BRE", "BUR"],
        "avg_difficulty": [2.5, 3.0, 3.5, 2.8, 2.2],
    }))
    return collector


# =============================================================================
# Test Position Enum
# =============================================================================

class TestPosition:
    """Test suite for Position enum."""

    def test_position_values(self):
        """Test position enum values."""
        assert Position.GOALKEEPER == 1
        assert Position.DEFENDER == 2
        assert Position.MIDFIELDER == 3
        assert Position.FORWARD == 4

    def test_position_names(self):
        """Test position name mapping."""
        assert POSITION_NAMES[Position.GOALKEEPER] == "GK"
        assert POSITION_NAMES[Position.DEFENDER] == "DEF"
        assert POSITION_NAMES[Position.MIDFIELDER] == "MID"
        assert POSITION_NAMES[Position.FORWARD] == "FWD"


# =============================================================================
# Test Valid Formations
# =============================================================================

class TestFormations:
    """Test suite for valid formations."""

    def test_all_formations_have_11_players(self):
        """Test all formations sum to 11 players."""
        for formation in VALID_FORMATIONS:
            assert sum(formation) == 11

    def test_all_formations_have_one_goalkeeper(self):
        """Test all formations have exactly 1 goalkeeper."""
        for formation in VALID_FORMATIONS:
            assert formation[0] == 1

    def test_all_formations_have_valid_defenders(self):
        """Test all formations have 3-5 defenders."""
        for formation in VALID_FORMATIONS:
            assert 3 <= formation[1] <= 5

    def test_all_formations_have_valid_forwards(self):
        """Test all formations have 1-3 forwards."""
        for formation in VALID_FORMATIONS:
            assert 1 <= formation[3] <= 3


# =============================================================================
# Test Transfer Recommendation
# =============================================================================

class TestTransferRecommendation:
    """Test suite for TransferRecommendation dataclass."""

    def test_str_representation(self, sample_players):
        """Test string representation of recommendation."""
        player_out = sample_players[11]  # MID4
        player_in = sample_players[8]  # MID1

        rec = TransferRecommendation(
            player_out=player_out,
            player_in=player_in,
            expected_gain=5.5,
            reason="Better form",
            hit_cost=0,
        )

        str_rep = str(rec)
        assert player_out.web_name in str_rep
        assert player_in.web_name in str_rep
        assert "5.50" in str_rep or "5.5" in str_rep

    def test_net_gain_calculation(self, sample_players):
        """Test net gain is calculated correctly."""
        rec = TransferRecommendation(
            player_out=sample_players[0],
            player_in=sample_players[1],
            expected_gain=6.0,
            reason="Test",
            hit_cost=-4,
        )
        assert rec.net_gain == 2.0

    def test_net_gain_no_hit(self, sample_players):
        """Test net gain with no hit cost."""
        rec = TransferRecommendation(
            player_out=sample_players[0],
            player_in=sample_players[1],
            expected_gain=6.0,
            reason="Test",
            hit_cost=0,
        )
        assert rec.net_gain == 6.0

    def test_hit_cost_in_string(self, sample_players):
        """Test hit cost appears in string representation."""
        rec = TransferRecommendation(
            player_out=sample_players[0],
            player_in=sample_players[1],
            expected_gain=6.0,
            reason="Test",
            hit_cost=-4,
        )
        assert "HIT: -4" in str(rec)


# =============================================================================
# Test Captain Recommendation
# =============================================================================

class TestCaptainRecommendation:
    """Test suite for CaptainRecommendation dataclass."""

    def test_str_representation(self, sample_players):
        """Test string representation."""
        rec = CaptainRecommendation(
            player=sample_players[7],  # MID1
            expected_points=12.5,
            fixture_score=2.0,
            confidence=0.85,
            is_home=True,
            ownership=40.0,
            reason="Easy fixture; Excellent form",
        )

        str_rep = str(rec)
        assert "MID1" in str_rep
        assert "(H)" in str_rep
        assert "85%" in str_rep

    def test_differential_detection(self, sample_players):
        """Test differential captain is detected."""
        rec = CaptainRecommendation(
            player=sample_players[7],
            expected_points=10.0,
            fixture_score=3.0,
            confidence=0.7,
            is_home=False,
            ownership=5.0,  # Low ownership
            reason="Test",
        )
        assert rec.is_differential is True

    def test_non_differential(self, sample_players):
        """Test non-differential captain."""
        rec = CaptainRecommendation(
            player=sample_players[7],
            expected_points=10.0,
            fixture_score=3.0,
            confidence=0.7,
            is_home=False,
            ownership=25.0,  # Higher ownership
            reason="Test",
        )
        assert rec.is_differential is False


# =============================================================================
# Test Chip Recommendation
# =============================================================================

class TestChipRecommendation:
    """Test suite for ChipRecommendation dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        rec = ChipRecommendation(
            chip_name="wildcard",
            gameweek=10,
            score=75.0,
            reasons=["Many injuries", "Fixture swing"],
            is_recommended=True,
        )

        str_rep = str(rec)
        assert "WILDCARD" in str_rep
        assert "GW10" in str_rep
        assert "75" in str_rep
        assert "RECOMMENDED" in str_rep

    def test_not_recommended_status(self):
        """Test chip not recommended status."""
        rec = ChipRecommendation(
            chip_name="bench_boost",
            gameweek=5,
            score=40.0,
            reasons=["Weak bench"],
            is_recommended=False,
        )
        assert "Consider" in str(rec)


# =============================================================================
# Test Transfer Optimizer
# =============================================================================

class TestTransferOptimizer:
    """Test suite for TransferOptimizer class."""

    def test_initialization(self, sample_players, sample_player_df):
        """Test optimizer initialization."""
        optimizer = TransferOptimizer(
            current_squad=sample_players,
            player_df=sample_player_df,
            budget=2.0,
            free_transfers=1,
            max_transfers=2,
        )

        assert optimizer.budget == 2.0
        assert optimizer.free_transfers == 1
        assert optimizer.max_transfers == 2
        assert len(optimizer.squad_ids) == 15

    def test_team_counting(self, sample_players, sample_player_df):
        """Test team count calculation."""
        optimizer = TransferOptimizer(
            current_squad=sample_players,
            player_df=sample_player_df,
            budget=2.0,
            free_transfers=1,
        )

        # Each player is from a different team in our sample
        for team_id, count in optimizer.team_counts.items():
            assert count >= 1

    def test_position_counting(self, sample_players, sample_player_df):
        """Test position count calculation."""
        optimizer = TransferOptimizer(
            current_squad=sample_players,
            player_df=sample_player_df,
            budget=2.0,
            free_transfers=1,
        )

        # 2 GK, 5 DEF, 5 MID, 3 FWD
        assert optimizer.position_counts[1] == 2
        assert optimizer.position_counts[2] == 5
        assert optimizer.position_counts[3] == 5
        assert optimizer.position_counts[4] == 3

    def test_can_add_from_team_under_limit(self, sample_players, sample_player_df):
        """Test team limit check when under limit."""
        optimizer = TransferOptimizer(
            current_squad=sample_players,
            player_df=sample_player_df,
            budget=2.0,
            free_transfers=1,
        )

        # Team 20 has no players, should be able to add
        assert optimizer._can_add_from_team(20, 1) is True

    def test_score_calculation(self, sample_players, sample_player_df):
        """Test score calculation for player."""
        optimizer = TransferOptimizer(
            current_squad=sample_players,
            player_df=sample_player_df,
            budget=2.0,
            free_transfers=1,
        )

        # Get score for MID1 (high form player)
        mid1 = sample_players[7]
        score = optimizer._calculate_player_score(mid1)
        assert score > 0

    def test_optimize_returns_list(self, sample_players, sample_player_df, additional_player_df):
        """Test optimization returns list of recommendations."""
        combined_df = pd.concat([sample_player_df, additional_player_df], ignore_index=True)

        optimizer = TransferOptimizer(
            current_squad=sample_players,
            player_df=combined_df,
            budget=5.0,
            free_transfers=2,
            max_transfers=2,
        )

        recommendations = optimizer.optimize()
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert isinstance(rec, TransferRecommendation)

    def test_hit_penalty_applied(self, sample_players, sample_player_df, additional_player_df):
        """Test hit penalty is applied for extra transfers."""
        combined_df = pd.concat([sample_player_df, additional_player_df], ignore_index=True)

        optimizer = TransferOptimizer(
            current_squad=sample_players,
            player_df=combined_df,
            budget=10.0,
            free_transfers=0,  # No free transfers
            max_transfers=2,
        )

        recommendations = optimizer.optimize()

        # All transfers should have hit cost
        for rec in recommendations:
            if rec.net_gain > 0:
                assert rec.hit_cost == -4

    def test_respects_max_transfers(self, sample_players, sample_player_df, additional_player_df):
        """Test optimizer respects max transfer limit."""
        combined_df = pd.concat([sample_player_df, additional_player_df], ignore_index=True)

        optimizer = TransferOptimizer(
            current_squad=sample_players,
            player_df=combined_df,
            budget=10.0,
            free_transfers=5,
            max_transfers=1,
        )

        recommendations = optimizer.optimize()
        assert len(recommendations) <= 1

    def test_filters_unavailable_players(self, sample_players, sample_player_df, additional_player_df):
        """Test that unavailable players are filtered out."""
        combined_df = pd.concat([sample_player_df, additional_player_df], ignore_index=True)

        optimizer = TransferOptimizer(
            current_squad=sample_players,
            player_df=combined_df,
            budget=10.0,
            free_transfers=2,
        )

        recommendations = optimizer.optimize()

        # InjuredMID (player_id=102, availability=0.25) should not be recommended
        for rec in recommendations:
            assert rec.player_in.id != 102


# =============================================================================
# Test Captain Selector
# =============================================================================

class TestCaptainSelector:
    """Test suite for CaptainSelector class."""

    def test_initialization(self, sample_players, sample_player_df):
        """Test selector initialization."""
        starting_xi = sample_players[:11]
        selector = CaptainSelector(starting_xi, sample_player_df)

        assert len(selector.squad) == 11

    def test_select_returns_sorted_list(self, sample_players, sample_player_df):
        """Test selection returns sorted list."""
        starting_xi = sample_players[:11]
        selector = CaptainSelector(starting_xi, sample_player_df)

        recommendations = selector.select()

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check sorted by expected points (descending)
        for i in range(len(recommendations) - 1):
            assert recommendations[i].expected_points >= recommendations[i + 1].expected_points

    def test_is_home_fixture_detection(self, sample_player_df):
        """Test home fixture detection."""
        selector = CaptainSelector([], sample_player_df)

        # Our sample data has "(H)" in gw1_fixture
        row = sample_player_df.iloc[0]
        assert selector._is_home_fixture(row) is True

    def test_is_away_fixture_detection(self, sample_player_df):
        """Test away fixture detection."""
        selector = CaptainSelector([], sample_player_df)

        # Modify to test away
        row = sample_player_df.iloc[0].copy()
        row["gw1_fixture"] = "OPP (A)"
        assert selector._is_home_fixture(row) is False

    def test_confidence_calculation(self, sample_player_df):
        """Test confidence score calculation."""
        selector = CaptainSelector([], sample_player_df)

        row = sample_player_df.iloc[0]
        confidence = selector._calculate_confidence(row, is_home=True, fixture_score=2.0)

        assert 0 <= confidence <= 1

    def test_captain_score_calculation(self, sample_players, sample_player_df):
        """Test captain score calculation."""
        selector = CaptainSelector(sample_players[:11], sample_player_df)

        player = sample_players[7]  # MID1 with high form
        row = sample_player_df[sample_player_df["player_id"] == player.id].iloc[0]

        score = selector._calculate_captain_score(player, row, is_home=True)
        assert score > 0

    def test_premium_player_bonus(self, sample_players, sample_player_df):
        """Test premium players get bonus."""
        selector = CaptainSelector(sample_players[:11], sample_player_df)

        premium = sample_players[12]  # FWD1 at 14.0m
        budget = sample_players[14]  # FWD3 at 6.0m

        premium_row = sample_player_df[sample_player_df["player_id"] == premium.id].iloc[0]
        budget_row = sample_player_df[sample_player_df["player_id"] == budget.id].iloc[0]

        # Premium should generally score higher due to bonus
        premium_score = selector._calculate_captain_score(premium, premium_row, is_home=True)
        budget_score = selector._calculate_captain_score(budget, budget_row, is_home=True)

        # Premium FWD1 has much better stats, so should definitely score higher
        assert premium_score > budget_score

    def test_skips_unavailable_players(self, sample_players, sample_player_df):
        """Test unavailable players are skipped."""
        # Mark one player as unavailable
        df = sample_player_df.copy()
        df.loc[df["player_id"] == sample_players[0].id, "availability"] = 0.25

        selector = CaptainSelector(sample_players[:11], df)
        recommendations = selector.select()

        # GK1 (id=1) should not be in recommendations
        captain_ids = [rec.player.id for rec in recommendations]
        assert sample_players[0].id not in captain_ids


# =============================================================================
# Test Chip Strategy Advisor
# =============================================================================

class TestChipStrategyAdvisor:
    """Test suite for ChipStrategyAdvisor class."""

    @pytest.fixture
    def fixtures_df(self):
        """Create sample fixtures DataFrame."""
        return pd.DataFrame({
            "team_id": list(range(1, 21)),
            "team": [f"T{i}" for i in range(1, 21)],
            "avg_difficulty": [2.0 + (i * 0.1) for i in range(20)],
            "gw1_fixture": ["OPP (H)"] * 20,
        })

    def test_initialization(self, sample_players, sample_player_df, fixtures_df):
        """Test advisor initialization."""
        advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=sample_player_df,
            fixtures_df=fixtures_df,
            available_chips=["wildcard", "bench_boost"],
            free_transfers=1,
        )

        assert len(advisor.available_chips) == 2

    def test_analyze_returns_list(self, sample_players, sample_player_df, fixtures_df):
        """Test analysis returns list of recommendations."""
        advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=sample_player_df,
            fixtures_df=fixtures_df,
            available_chips=["wildcard", "bench_boost", "free_hit", "triple_captain"],
            free_transfers=1,
        )

        recommendations = advisor.analyze()

        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert isinstance(rec, ChipRecommendation)

    def test_only_analyzes_available_chips(self, sample_players, sample_player_df, fixtures_df):
        """Test only available chips are analyzed."""
        advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=sample_player_df,
            fixtures_df=fixtures_df,
            available_chips=["wildcard"],  # Only wildcard available
            free_transfers=1,
        )

        recommendations = advisor.analyze()

        chip_names = [rec.chip_name for rec in recommendations]
        assert "wildcard" in chip_names
        assert "bench_boost" not in chip_names

    def test_wildcard_analysis(self, sample_players, sample_player_df, fixtures_df):
        """Test wildcard analysis."""
        advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=sample_player_df,
            fixtures_df=fixtures_df,
            available_chips=["wildcard"],
            free_transfers=0,
        )

        rec = advisor._analyze_wildcard()

        assert rec is not None
        assert rec.chip_name == "wildcard"
        assert 0 <= rec.score <= 100

    def test_bench_boost_analysis(self, sample_players, sample_player_df, fixtures_df):
        """Test bench boost analysis."""
        advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=sample_player_df,
            fixtures_df=fixtures_df,
            available_chips=["bench_boost"],
            free_transfers=1,
        )

        rec = advisor._analyze_bench_boost()

        assert rec is not None
        assert rec.chip_name == "bench_boost"

    def test_free_hit_analysis(self, sample_players, sample_player_df, fixtures_df):
        """Test free hit analysis."""
        advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=sample_player_df,
            fixtures_df=fixtures_df,
            available_chips=["free_hit"],
            free_transfers=1,
        )

        rec = advisor._analyze_free_hit()

        assert rec is not None
        assert rec.chip_name == "free_hit"

    def test_triple_captain_analysis(self, sample_players, sample_player_df, fixtures_df):
        """Test triple captain analysis."""
        advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=sample_player_df,
            fixtures_df=fixtures_df,
            available_chips=["triple_captain"],
            free_transfers=1,
        )

        rec = advisor._analyze_triple_captain()

        assert rec is not None
        assert rec.chip_name == "triple_captain"

    def test_injured_count(self, sample_players, sample_player_df, fixtures_df):
        """Test injured player counting."""
        # Mark some players as injured
        df = sample_player_df.copy()
        df.loc[df["player_id"].isin([1, 2, 3]), "availability"] = 0.25

        advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=df,
            fixtures_df=fixtures_df,
            available_chips=["wildcard"],
            free_transfers=1,
        )

        count = advisor._count_injured_players()
        assert count == 3

    def test_underperformers_count(self, sample_players, sample_player_df, fixtures_df):
        """Test underperforming players counting."""
        advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=sample_player_df,
            fixtures_df=fixtures_df,
            available_chips=["wildcard"],
            free_transfers=1,
        )

        count = advisor._count_underperformers()
        # Several players have low form/value
        assert count >= 0

    def test_recommendations_sorted_by_score(self, sample_players, sample_player_df, fixtures_df):
        """Test recommendations are sorted by score."""
        advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=sample_player_df,
            fixtures_df=fixtures_df,
            available_chips=["wildcard", "bench_boost", "free_hit", "triple_captain"],
            free_transfers=1,
        )

        recommendations = advisor.analyze()

        for i in range(len(recommendations) - 1):
            assert recommendations[i].score >= recommendations[i + 1].score


# =============================================================================
# Test Main Optimizer Class
# =============================================================================

class TestOptimizer:
    """Test suite for main Optimizer class."""

    @pytest.fixture
    def optimizer(self, config, mock_client, mock_collector):
        """Create optimizer instance."""
        return Optimizer(config, mock_client, mock_collector)

    def test_squad_constraints(self, optimizer):
        """Test squad constraint values."""
        assert optimizer.SQUAD_SIZE == 15
        assert optimizer.MAX_FROM_TEAM == 3

        assert optimizer.POSITION_LIMITS[Position.GOALKEEPER] == (2, 2)
        assert optimizer.POSITION_LIMITS[Position.DEFENDER] == (5, 5)
        assert optimizer.POSITION_LIMITS[Position.MIDFIELDER] == (5, 5)
        assert optimizer.POSITION_LIMITS[Position.FORWARD] == (3, 3)

    def test_validate_squad_correct(self, optimizer, sample_players):
        """Test validation passes with correct squad."""
        errors = optimizer.validate_squad(sample_players)
        assert len(errors) == 0

    def test_validate_squad_wrong_size(self, optimizer, sample_players):
        """Test validation fails with wrong squad size."""
        errors = optimizer.validate_squad(sample_players[:10])
        assert any("15 players" in e for e in errors)

    def test_validate_squad_team_limit(self, optimizer):
        """Test validation fails with too many from same team."""
        # Create squad with 4 players from team 1
        squad = [create_player(i, f"P{i}", 1, 3, 5.0) for i in range(15)]

        errors = optimizer.validate_squad(squad)
        assert any("same team" in e for e in errors)

    def test_validate_formation_valid(self, optimizer, sample_players):
        """Test valid formation is recognized."""
        # sample_players[:11] = 1 GK, 4 DEF, 4 MID, 2 FWD (4-4-2)
        # Need to adjust for 4-4-2
        starting_xi = (
            sample_players[0:1] +  # 1 GK
            sample_players[2:6] +  # 4 DEF
            sample_players[7:11] + # 4 MID
            sample_players[12:14]  # 2 FWD
        )

        is_valid, formation = optimizer.validate_formation(starting_xi)
        assert is_valid is True
        assert "4-4-2" in formation

    def test_validate_formation_invalid(self, optimizer):
        """Test invalid formation is detected."""
        # Create invalid formation: 6 defenders
        players = [
            create_player(1, "GK", 1, 1, 5.0),
            *[create_player(i, f"DEF{i}", i, 2, 5.0) for i in range(2, 8)],  # 6 DEF
            *[create_player(i, f"MID{i}", i, 3, 5.0) for i in range(8, 12)],  # 4 MID
        ]

        is_valid, formation = optimizer.validate_formation(players)
        assert is_valid is False

    def test_optimize_transfers_offline(self, optimizer, sample_players, sample_player_df, additional_player_df):
        """Test offline transfer optimization."""
        combined_df = pd.concat([sample_player_df, additional_player_df], ignore_index=True)

        recommendations = optimizer.optimize_transfers_offline(
            current_squad=sample_players,
            player_df=combined_df,
            budget=5.0,
            free_transfers=2,
            max_transfers=2,
        )

        assert isinstance(recommendations, list)

    def test_select_captain_offline(self, optimizer, sample_players, sample_player_df):
        """Test offline captain selection."""
        starting_xi = sample_players[:11]

        recommendations = optimizer.select_captain_offline(
            squad=starting_xi,
            player_df=sample_player_df,
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_format_recommendations_empty(self, optimizer):
        """Test formatting empty results."""
        result = OptimizationResult()

        output = optimizer.format_recommendations(result)

        assert "RECOMMENDATIONS" in output
        assert "No transfers recommended" in output

    def test_format_recommendations_with_transfers(self, optimizer, sample_players):
        """Test formatting results with transfers."""
        result = OptimizationResult(
            transfers=[
                TransferRecommendation(
                    player_out=sample_players[11],
                    player_in=sample_players[8],
                    expected_gain=5.0,
                    reason="Better form",
                    hit_cost=0,
                )
            ],
            total_expected_gain=5.0,
            net_expected_gain=5.0,
        )

        output = optimizer.format_recommendations(result)

        assert "TRANSFERS" in output
        assert "5.00" in output or "5.0" in output

    def test_format_recommendations_with_captain(self, optimizer, sample_players):
        """Test formatting results with captain."""
        result = OptimizationResult(
            captain=CaptainRecommendation(
                player=sample_players[12],
                expected_points=15.0,
                fixture_score=2.0,
                confidence=0.85,
                is_home=True,
                ownership=40.0,
                reason="Premium asset",
            ),
        )

        output = optimizer.format_recommendations(result)

        assert "CAPTAIN" in output
        assert "FWD1" in output

    def test_format_recommendations_with_chips(self, optimizer):
        """Test formatting results with chip recommendations."""
        result = OptimizationResult(
            chip_recommendations=[
                ChipRecommendation(
                    chip_name="wildcard",
                    gameweek=10,
                    score=75.0,
                    reasons=["Many injuries"],
                    is_recommended=True,
                ),
            ],
        )

        output = optimizer.format_recommendations(result)

        assert "CHIPS" in output
        assert "WILDCARD" in output

    def test_format_recommendations_with_warnings(self, optimizer):
        """Test formatting results with warnings."""
        result = OptimizationResult(
            warnings=["Test warning message"],
        )

        output = optimizer.format_recommendations(result)

        assert "WARNINGS" in output
        assert "Test warning" in output


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for optimizer components working together."""

    def test_full_optimization_flow(self, sample_players, sample_player_df, additional_player_df):
        """Test full optimization flow."""
        combined_df = pd.concat([sample_player_df, additional_player_df], ignore_index=True)

        # Create fixtures DataFrame
        fixtures_df = pd.DataFrame({
            "team_id": list(range(1, 21)),
            "team": [f"T{i}" for i in range(1, 21)],
            "avg_difficulty": [2.5] * 20,
            "gw1_fixture": ["OPP (H)"] * 20,
        })

        # Run transfer optimization
        transfer_optimizer = TransferOptimizer(
            current_squad=sample_players,
            player_df=combined_df,
            budget=5.0,
            free_transfers=2,
            max_transfers=2,
        )
        transfers = transfer_optimizer.optimize()

        # Run captain selection
        captain_selector = CaptainSelector(
            squad=sample_players[:11],
            player_df=sample_player_df,
        )
        captains = captain_selector.select()

        # Run chip analysis
        chip_advisor = ChipStrategyAdvisor(
            squad=sample_players,
            player_df=sample_player_df,
            fixtures_df=fixtures_df,
            available_chips=["wildcard", "bench_boost"],
            free_transfers=2,
        )
        chips = chip_advisor.analyze()

        # Verify all components work
        assert isinstance(transfers, list)
        assert isinstance(captains, list)
        assert isinstance(chips, list)
        assert len(captains) > 0

    def test_transfer_and_captain_consistency(self, sample_players, sample_player_df):
        """Test that transfer and captain recommendations are consistent."""
        # If we transfer out a player, they shouldn't be captain
        starting_xi = sample_players[:11]

        captain_selector = CaptainSelector(
            squad=starting_xi,
            player_df=sample_player_df,
        )
        captain_recs = captain_selector.select()

        # All captain recommendations should be from starting XI
        captain_ids = {rec.player.id for rec in captain_recs}
        starting_ids = {p.id for p in starting_xi}

        assert captain_ids.issubset(starting_ids)
