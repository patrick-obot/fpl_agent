#!/usr/bin/env python3
"""
Unit tests for the Executor module.

Tests cover:
- Data classes (TeamState, Alert, TransferDecision, etc.)
- AuditLogger for decision audit trail
- StateManager for rollback capability
- NotificationService for approval notifications
- Executor class with all safety features
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.fpl_client import Player
from src.executor import (
    ExecutionStatus,
    AlertLevel,
    TeamState,
    Alert,
    TransferDecision,
    CaptainDecision,
    ExecutionPlan,
    ExecutionResult,
    AuditEntry,
    AuditLogger,
    StateManager,
    NotificationService,
    Executor,
    DEFAULT_MIN_EXPECTED_GAIN,
    DEFAULT_MIN_CONFIDENCE,
    safe_execute,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_data_dir):
    """Create a test configuration."""
    cfg = Config.from_env()
    cfg.data_dir = temp_data_dir
    cfg.dry_run = True  # Default to dry run for tests
    return cfg


class MockSquadPick:
    """Mock SquadPick for testing."""
    def __init__(self, element, position, is_captain=False, is_vice_captain=False, multiplier=1):
        self.element = element
        self.position = position
        self.is_captain = is_captain
        self.is_vice_captain = is_vice_captain
        self.multiplier = multiplier


class MockMyTeam:
    """Mock MyTeam dataclass for testing."""
    def __init__(self):
        self.picks = [
            MockSquadPick(100, 1, is_captain=False, is_vice_captain=False),
            MockSquadPick(200, 2, is_captain=True, is_vice_captain=False),
            MockSquadPick(300, 3, is_captain=False, is_vice_captain=True),
        ]
        self.bank = 5.0
        self.free_transfers = 2
        self.transfers_made = 0
        self.total_value = 100.0
        self.transfers = {"bank": 50, "limit": 2, "made": 0}

    @property
    def captain_id(self):
        for pick in self.picks:
            if pick.is_captain:
                return pick.element
        return None

    @property
    def vice_captain_id(self):
        for pick in self.picks:
            if pick.is_vice_captain:
                return pick.element
        return None


@pytest.fixture
def mock_client():
    """Create a mock FPL client."""
    client = AsyncMock()

    # Return MockMyTeam instead of dict
    client.get_my_team.return_value = MockMyTeam()

    mock_gameweek = MagicMock()
    mock_gameweek.id = 21
    client.get_current_gameweek.return_value = mock_gameweek

    # Mock transfer result
    mock_transfer_result = MagicMock()
    mock_transfer_result.success = True
    mock_transfer_result.message = "Transfer successful"
    client.make_transfers.return_value = mock_transfer_result

    # Mock captain setting
    client.set_captain.return_value = {"success": True}

    return client


def create_player(
    player_id: int,
    name: str,
    team_id: int = 1,
    position: int = 3,
    price: float = 10.0,
    form: float = 5.0,
    total_points: int = 100,
    minutes: int = 1000,
) -> Player:
    """Create a Player object for testing."""
    return Player(
        id=player_id,
        web_name=name,
        first_name=name.split()[0] if " " in name else name,
        second_name=name.split()[-1] if " " in name else name,
        team=team_id,
        team_name="TestTeam",
        element_type=position,
        now_cost=price,
        total_points=total_points,
        form=form,
        points_per_game=5.0,
        selected_by_percent=10.0,
        minutes=minutes,
        goals_scored=5,
        assists=5,
        clean_sheets=5,
        goals_conceded=10,
        bonus=10,
        bps=200,
        expected_goals=5.0,
        expected_assists=3.0,
        expected_goal_involvements=8.0,
        expected_goals_conceded=10.0,
        status="a",
        news="",
        news_added=None,
        chance_of_playing_this_round=None,
        chance_of_playing_next_round=None,
        transfers_in_event=1000,
        transfers_out_event=500,
        cost_change_event=0,
        cost_change_start=0,
    )


class MockTransfer:
    """Mock transfer recommendation."""
    def __init__(
        self,
        player_out: Player,
        player_in: Player,
        expected_gain: float = 5.0,
        hit_cost: int = 0,
    ):
        self.player_out = player_out
        self.player_in = player_in
        self.expected_gain = expected_gain
        self.hit_cost = hit_cost
        self.net_gain = expected_gain + hit_cost
        self.reason = "Test transfer reason"


class MockCaptainChoice:
    """Mock captain choice."""
    def __init__(
        self,
        player: Player,
        expected_points: float = 8.0,
        confidence: float = 0.85,
        is_differential: bool = False,
    ):
        self.player = player
        self.expected_points = expected_points
        self.confidence = confidence
        self.is_differential = is_differential
        self.reason = "Test captain reason"


class MockBenchOrder:
    """Mock bench order."""
    def __init__(self, player_ids: list = None):
        self._ids = player_ids or []

    def to_list(self) -> list[int]:
        return self._ids


class MockOptimizationResult:
    """Mock optimization result."""
    def __init__(
        self,
        transfers: list = None,
        captain: MockCaptainChoice = None,
        vice_captain: MockCaptainChoice = None,
        starting_xi: list = None,
        bench_order: MockBenchOrder = None,
    ):
        self.transfers = transfers or []
        self.captain = captain
        self.vice_captain = vice_captain
        self.starting_xi = starting_xi or []
        self.bench_order = bench_order


# =============================================================================
# Test Enums
# =============================================================================

class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses are defined."""
        statuses = [
            "pending", "dry_run", "awaiting_approval", "approved",
            "rejected", "confirmed", "executing", "success",
            "failed", "cancelled", "rolled_back", "timeout"
        ]
        for status in statuses:
            assert hasattr(ExecutionStatus, status.upper())

    def test_status_values(self):
        """Test status values are correct."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.DRY_RUN.value == "dry_run"
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.FAILED.value == "failed"


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_all_levels_exist(self):
        """Test all alert levels are defined."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"


# =============================================================================
# Test Data Classes
# =============================================================================

class TestTeamState:
    """Tests for TeamState data class."""

    def test_creation(self):
        """Test TeamState creation."""
        state = TeamState(
            timestamp=datetime.now(),
            picks=[{"element": 1, "position": 1}],
            captain_id=100,
            vice_captain_id=200,
            bank=5.0,
            free_transfers=2,
            total_points=500,
            gameweek=21,
        )
        assert state.captain_id == 100
        assert state.bank == 5.0

    def test_to_dict(self):
        """Test serialization to dict."""
        now = datetime.now()
        state = TeamState(
            timestamp=now,
            picks=[{"element": 1}],
            captain_id=100,
            vice_captain_id=200,
            bank=5.0,
            free_transfers=2,
            total_points=500,
            gameweek=21,
        )
        data = state.to_dict()
        assert data["captain_id"] == 100
        assert data["bank"] == 5.0
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        now = datetime.now()
        data = {
            "timestamp": now.isoformat(),
            "picks": [{"element": 1}],
            "captain_id": 100,
            "vice_captain_id": 200,
            "bank": 5.0,
            "free_transfers": 2,
            "total_points": 500,
            "gameweek": 21,
        }
        state = TeamState.from_dict(data)
        assert state.captain_id == 100
        assert state.gameweek == 21


class TestAlert:
    """Tests for Alert data class."""

    def test_creation(self):
        """Test Alert creation."""
        alert = Alert(
            level=AlertLevel.WARNING,
            message="Test warning",
            details={"key": "value"},
        )
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test warning"

    def test_str_format(self):
        """Test string representation."""
        alert = Alert(level=AlertLevel.WARNING, message="Test warning")
        assert "[!]" in str(alert)

        alert_critical = Alert(level=AlertLevel.CRITICAL, message="Critical issue")
        assert "[!!!]" in str(alert_critical)

        alert_info = Alert(level=AlertLevel.INFO, message="Info message")
        assert "[i]" in str(alert_info)


class TestTransferDecision:
    """Tests for TransferDecision data class."""

    def test_creation(self):
        """Test TransferDecision creation."""
        decision = TransferDecision(
            player_out_id=100,
            player_out_name="Salah",
            player_in_id=200,
            player_in_name="Palmer",
            expected_gain=5.0,
            hit_cost=0,
            net_gain=5.0,
            reason="Better fixtures",
            confidence=0.85,
        )
        assert decision.player_out_name == "Salah"
        assert decision.net_gain == 5.0


class TestExecutionPlan:
    """Tests for ExecutionPlan data class."""

    def test_creation(self):
        """Test ExecutionPlan creation."""
        plan = ExecutionPlan(id="test_plan")
        assert plan.id == "test_plan"
        assert plan.status == ExecutionStatus.PENDING
        assert plan.transfers == []
        assert plan.alerts == []

    def test_default_values(self):
        """Test default values are set correctly."""
        plan = ExecutionPlan(id="test_plan")
        assert plan.total_expected_gain == 0.0
        assert plan.total_hit_cost == 0
        assert plan.net_expected_gain == 0.0


class TestAuditEntry:
    """Tests for AuditEntry data class."""

    def test_creation(self):
        """Test AuditEntry creation."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            action="TEST_ACTION",
            plan_id="plan_123",
            details={"key": "value"},
            outcome="success",
            user="test_user",
        )
        assert entry.action == "TEST_ACTION"
        assert entry.outcome == "success"

    def test_to_dict(self):
        """Test serialization to dict."""
        now = datetime.now()
        entry = AuditEntry(
            timestamp=now,
            action="TEST",
            plan_id="plan_123",
            details={},
            outcome="success",
        )
        data = entry.to_dict()
        assert data["action"] == "TEST"
        assert "timestamp" in data


# =============================================================================
# Test AuditLogger
# =============================================================================

class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_initialization(self, config, temp_data_dir):
        """Test logger initialization."""
        logger = AuditLogger(config)
        assert logger.config == config
        assert len(logger._entries) == 0

    def test_log_entry(self, config):
        """Test logging an entry."""
        logger = AuditLogger(config)
        logger.log(
            action="TEST_ACTION",
            plan_id="plan_123",
            details={"key": "value"},
            outcome="success",
        )
        assert len(logger._entries) == 1
        assert logger._entries[0].action == "TEST_ACTION"

    def test_get_history(self, config):
        """Test retrieving history."""
        logger = AuditLogger(config)
        logger.log("ACTION_1", "plan_1", {}, "success")
        logger.log("ACTION_2", "plan_2", {}, "failed")
        logger.log("ACTION_3", "plan_1", {}, "success")

        # All history
        history = logger.get_history()
        assert len(history) == 3

        # Filtered by plan
        filtered = logger.get_history(plan_id="plan_1")
        assert len(filtered) == 2

    def test_format_history(self, config):
        """Test formatting history for display."""
        logger = AuditLogger(config)
        logger.log("TEST", "plan_123", {}, "success", user="test_user")

        formatted = logger.format_history()
        assert "DECISION AUDIT TRAIL" in formatted
        assert "TEST" in formatted

    def test_persistence(self, config, temp_data_dir):
        """Test audit trail persistence."""
        # Create logger and add entry
        logger1 = AuditLogger(config)
        logger1.log("TEST", "plan_123", {}, "success")

        # Create new logger and verify entry was loaded
        logger2 = AuditLogger(config)
        assert len(logger2._entries) == 1
        assert logger2._entries[0].action == "TEST"


# =============================================================================
# Test StateManager
# =============================================================================

class TestStateManager:
    """Tests for StateManager class."""

    def test_initialization(self, config):
        """Test StateManager initialization."""
        manager = StateManager(config)
        assert manager._state_dir.exists()

    @pytest.mark.asyncio
    async def test_save_state(self, config, mock_client):
        """Test saving team state."""
        manager = StateManager(config)
        state = await manager.save_state(mock_client, label="test")

        assert state is not None
        assert state.captain_id == 200
        assert state.gameweek == 21

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, config, mock_client):
        """Test save and load round-trip."""
        manager = StateManager(config)

        # Save state
        await manager.save_state(mock_client, label="test")

        # List states
        states = manager.list_states()
        assert len(states) >= 1

        # Load latest state
        latest = manager.get_latest_state()
        assert latest is not None
        assert latest.captain_id == 200

    def test_list_states_empty(self, config):
        """Test listing states when none exist."""
        manager = StateManager(config)
        states = manager.list_states()
        assert states == []

    def test_load_nonexistent_state(self, config):
        """Test loading non-existent state."""
        manager = StateManager(config)
        state = manager.load_state("nonexistent.json")
        assert state is None


# =============================================================================
# Test NotificationService
# =============================================================================

class TestNotificationService:
    """Tests for NotificationService class."""

    def test_initialization(self, config):
        """Test service initialization."""
        service = NotificationService(config)
        assert service.config == config

    def test_format_approval_email(self, config):
        """Test approval email formatting."""
        service = NotificationService(config)

        plan = ExecutionPlan(id="test_plan")
        plan.transfers.append(TransferDecision(
            player_out_id=1,
            player_out_name="Salah",
            player_in_id=2,
            player_in_name="Palmer",
            expected_gain=5.0,
            hit_cost=0,
            net_gain=5.0,
            reason="Better fixtures",
            confidence=0.85,
        ))
        plan.net_expected_gain = 5.0
        plan.overall_confidence = 0.85

        deadline = datetime.now() + timedelta(hours=2)
        body = service._format_approval_email(plan, deadline)

        assert "FPL Agent - Approval Required" in body
        assert "Salah" in body
        assert "Palmer" in body
        assert "Net Expected Gain" in body

    def test_format_result_email(self, config):
        """Test result email formatting."""
        service = NotificationService(config)

        plan = ExecutionPlan(id="test_plan")
        plan.executed_at = datetime.now()

        result = ExecutionResult(
            plan=plan,
            transfers_executed=1,
            captain_set=True,
            success=True,
            messages=["Transfer completed"],
        )

        body = service._format_result_email(result)

        assert "SUCCESS" in body
        assert "Transfer completed" in body


# =============================================================================
# Test Executor - Plan Creation
# =============================================================================

class TestExecutorPlanCreation:
    """Tests for Executor plan creation."""

    def test_create_plan_empty(self, config, mock_client):
        """Test creating empty plan."""
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        plan = executor.create_plan(opt_result)

        assert plan.id.startswith("plan_")
        assert len(plan.transfers) == 0
        assert plan.captain is None

    def test_create_plan_with_transfers(self, config, mock_client):
        """Test creating plan with transfers."""
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Salah", price=13.0, form=5.0)
        player_in = create_player(2, "Palmer", price=11.0, form=7.0)

        transfer = MockTransfer(player_out, player_in, expected_gain=5.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        plan = executor.create_plan(opt_result)

        assert len(plan.transfers) == 1
        assert plan.transfers[0].player_out_name == "Salah"
        assert plan.transfers[0].player_in_name == "Palmer"

    def test_create_plan_with_captain(self, config, mock_client):
        """Test creating plan with captain."""
        executor = Executor(config, mock_client)

        captain = create_player(1, "Haaland")
        vice = create_player(2, "Salah")

        opt_result = MockOptimizationResult(
            captain=MockCaptainChoice(captain, expected_points=10.0),
            vice_captain=MockCaptainChoice(vice, expected_points=8.0),
        )

        plan = executor.create_plan(opt_result)

        assert plan.captain is not None
        assert plan.captain.captain_name == "Haaland"
        assert plan.captain.vice_captain_name == "Salah"

    def test_create_plan_calculates_totals(self, config, mock_client):
        """Test plan totals are calculated correctly."""
        executor = Executor(config, mock_client)

        player_out1 = create_player(1, "Player1")
        player_in1 = create_player(2, "Player2")
        player_out2 = create_player(3, "Player3")
        player_in2 = create_player(4, "Player4")

        transfers = [
            MockTransfer(player_out1, player_in1, expected_gain=5.0, hit_cost=0),
            MockTransfer(player_out2, player_in2, expected_gain=4.0, hit_cost=-4),
        ]
        opt_result = MockOptimizationResult(transfers=transfers)

        plan = executor.create_plan(opt_result)

        assert plan.total_expected_gain == 9.0
        assert plan.total_hit_cost == -4
        assert plan.net_expected_gain == 5.0  # 9.0 - 4


# =============================================================================
# Test Executor - Alerts
# =============================================================================

class TestExecutorAlerts:
    """Tests for Executor alert generation."""

    def test_alert_selling_top_performer(self, config, mock_client):
        """Test alert when selling player in good form."""
        executor = Executor(config, mock_client)

        # Player with form >= 6.0 should trigger alert
        player_out = create_player(1, "Salah", form=7.5)
        player_in = create_player(2, "Palmer", form=5.0)

        transfer = MockTransfer(player_out, player_in)
        opt_result = MockOptimizationResult(transfers=[transfer])

        plan = executor.create_plan(opt_result)

        warning_alerts = [a for a in plan.alerts if a.level == AlertLevel.WARNING]
        assert any("good form" in a.message for a in warning_alerts)

    def test_alert_selling_high_points(self, config, mock_client):
        """Test alert when selling player with high total points."""
        executor = Executor(config, mock_client)

        # Player with >= 100 points should trigger alert
        player_out = create_player(1, "Salah", total_points=150)
        player_in = create_player(2, "Palmer", total_points=80)

        transfer = MockTransfer(player_out, player_in)
        opt_result = MockOptimizationResult(transfers=[transfer])

        plan = executor.create_plan(opt_result)

        warning_alerts = [a for a in plan.alerts if a.level == AlertLevel.WARNING]
        assert any("points" in a.message.lower() for a in warning_alerts)

    def test_alert_buying_low_minutes(self, config, mock_client):
        """Test alert when buying player with low minutes."""
        executor = Executor(config, mock_client)

        # Player with < 500 minutes should trigger info alert
        player_out = create_player(1, "Salah", minutes=1500)
        player_in = create_player(2, "NewPlayer", minutes=300)

        transfer = MockTransfer(player_out, player_in)
        opt_result = MockOptimizationResult(transfers=[transfer])

        plan = executor.create_plan(opt_result)

        info_alerts = [a for a in plan.alerts if a.level == AlertLevel.INFO]
        assert any("minutes" in a.message.lower() for a in info_alerts)

    def test_alert_small_gain_for_hit(self, config, mock_client):
        """Test alert when taking hit for small gain."""
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Player1")
        player_in = create_player(2, "Player2")

        # Hit with expected gain < 6.0 should trigger warning
        transfer = MockTransfer(player_out, player_in, expected_gain=4.0, hit_cost=-4)
        opt_result = MockOptimizationResult(transfers=[transfer])

        plan = executor.create_plan(opt_result)

        warning_alerts = [a for a in plan.alerts if a.level == AlertLevel.WARNING]
        assert any("hit" in a.message.lower() for a in warning_alerts)


# =============================================================================
# Test Executor - Confidence Thresholds
# =============================================================================

class TestExecutorThresholds:
    """Tests for confidence threshold checking."""

    def test_below_min_gain_threshold(self, config, mock_client):
        """Test critical alert when below minimum gain."""
        executor = Executor(config, mock_client, min_expected_gain=5.0)

        player_out = create_player(1, "Player1")
        player_in = create_player(2, "Player2")

        # Net gain below threshold
        transfer = MockTransfer(player_out, player_in, expected_gain=3.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        plan = executor.create_plan(opt_result)

        critical_alerts = [a for a in plan.alerts if a.level == AlertLevel.CRITICAL]
        assert any("below threshold" in a.message.lower() for a in critical_alerts)

    def test_above_min_gain_threshold(self, config, mock_client):
        """Test no alert when above minimum gain."""
        executor = Executor(config, mock_client, min_expected_gain=4.0)

        player_out = create_player(1, "Player1")
        player_in = create_player(2, "Player2")

        # Net gain above threshold
        transfer = MockTransfer(player_out, player_in, expected_gain=6.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        plan = executor.create_plan(opt_result)

        critical_alerts = [a for a in plan.alerts if a.level == AlertLevel.CRITICAL]
        gain_alerts = [a for a in critical_alerts if "below threshold" in a.message.lower()]
        assert len(gain_alerts) == 0

    def test_confidence_for_hits(self, config, mock_client):
        """Test confidence threshold for taking hits."""
        executor = Executor(config, mock_client, min_confidence=0.90)

        player_out = create_player(1, "Player1")
        player_in = create_player(2, "Player2")

        # Taking hit with low confidence
        transfer = MockTransfer(player_out, player_in, expected_gain=8.0, hit_cost=-4)
        opt_result = MockOptimizationResult(transfers=[transfer])

        plan = executor.create_plan(opt_result)

        # Default confidence is 0.8, threshold is 0.9 - should trigger alert
        critical_alerts = [a for a in plan.alerts if a.level == AlertLevel.CRITICAL]
        assert any("confidence" in a.message.lower() for a in critical_alerts)


# =============================================================================
# Test Executor - Dry Run
# =============================================================================

class TestExecutorDryRun:
    """Tests for dry run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_no_changes(self, config, mock_client):
        """Test that dry run doesn't make actual changes."""
        config.dry_run = True
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        transfer = MockTransfer(player_out, player_in, expected_gain=5.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        result = await executor.dry_run(opt_result)

        assert result.plan.status == ExecutionStatus.DRY_RUN
        assert result.success is True
        assert "Dry run completed" in result.messages[0]

        # Verify no actual API calls for transfers
        mock_client.make_transfers.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_saves_state(self, config, mock_client):
        """Test that dry run saves current state for comparison."""
        config.dry_run = True
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        result = await executor.dry_run(opt_result)

        assert result.plan.team_state_before is not None

    @pytest.mark.asyncio
    async def test_dry_run_logs_audit(self, config, mock_client):
        """Test that dry run creates audit entry."""
        config.dry_run = True
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        await executor.dry_run(opt_result)

        history = executor.audit_logger.get_history()
        assert any(e.action == "DRY_RUN" for e in history)


# =============================================================================
# Test Executor - Approval Workflow
# =============================================================================

class TestExecutorApproval:
    """Tests for approval workflow."""

    @pytest.mark.asyncio
    async def test_request_approval(self, config, mock_client):
        """Test requesting approval sets correct status."""
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        plan = await executor.request_approval(opt_result, deadline_hours=2.0)

        assert plan.status == ExecutionStatus.AWAITING_APPROVAL
        assert plan.approval_deadline is not None
        assert plan.approval_deadline > datetime.now()

    @pytest.mark.asyncio
    async def test_approve_executes(self, config, mock_client):
        """Test approving a plan executes it."""
        config.dry_run = False
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        plan = await executor.request_approval(opt_result, deadline_hours=2.0)

        result = await executor.approve(plan.id, approved_by="test_user")

        # Plan should be executed (or at least attempted)
        assert result.plan.approved_by == "test_user"

    @pytest.mark.asyncio
    async def test_reject_sets_status(self, config, mock_client):
        """Test rejecting a plan sets correct status."""
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        plan = await executor.request_approval(opt_result)

        await executor.reject(plan.id, reason="Changed my mind")

        # Reload plan
        loaded = executor._load_plan(plan.id)
        assert loaded.status == ExecutionStatus.REJECTED

    @pytest.mark.asyncio
    async def test_approve_after_deadline_fails(self, config, mock_client):
        """Test approving after deadline raises error."""
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        plan = await executor.request_approval(opt_result, deadline_hours=0.0)

        # Deadline is now in the past
        with pytest.raises(ValueError, match="deadline"):
            await executor.approve(plan.id)

    @pytest.mark.asyncio
    async def test_approve_nonexistent_plan_fails(self, config, mock_client):
        """Test approving non-existent plan raises error."""
        executor = Executor(config, mock_client)

        with pytest.raises(ValueError, match="No plan found"):
            await executor.approve("nonexistent_plan")


# =============================================================================
# Test Executor - Execution
# =============================================================================

class TestExecutorExecution:
    """Tests for plan execution."""

    @pytest.mark.asyncio
    async def test_execute_requires_confirmation(self, config, mock_client):
        """Test execute without confirmation returns pending."""
        config.dry_run = False
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        transfer = MockTransfer(player_out, player_in, expected_gain=6.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        result = await executor.execute(opt_result, confirm=False)

        assert result.plan.status == ExecutionStatus.PENDING
        assert "confirm" in result.messages[0].lower()

    @pytest.mark.asyncio
    async def test_execute_with_confirmation(self, config, mock_client):
        """Test execute with confirmation proceeds."""
        config.dry_run = False
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        transfer = MockTransfer(player_out, player_in, expected_gain=6.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        result = await executor.execute(opt_result, confirm=True)

        # Should have attempted execution
        assert result.plan.status in [ExecutionStatus.SUCCESS, ExecutionStatus.FAILED]

    @pytest.mark.asyncio
    async def test_execute_blocked_by_critical_alerts(self, config, mock_client):
        """Test execution blocked by critical alerts."""
        config.dry_run = False
        executor = Executor(config, mock_client, min_expected_gain=10.0)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        # Low gain will trigger critical alert
        transfer = MockTransfer(player_out, player_in, expected_gain=2.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        result = await executor.execute(opt_result, confirm=False)

        assert result.plan.status == ExecutionStatus.CANCELLED
        assert "critical" in result.messages[0].lower()

    @pytest.mark.asyncio
    async def test_execute_override_critical_alerts(self, config, mock_client):
        """Test execution can override critical alerts with confirm."""
        config.dry_run = False
        executor = Executor(config, mock_client, min_expected_gain=10.0)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        transfer = MockTransfer(player_out, player_in, expected_gain=2.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        result = await executor.execute(opt_result, confirm=True)

        # Should proceed despite critical alerts
        assert result.plan.status != ExecutionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_dry_run_config_triggers_dry_run(self, config, mock_client):
        """Test config.dry_run triggers dry run mode."""
        config.dry_run = True
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        result = await executor.execute(opt_result, confirm=True)

        assert result.plan.status == ExecutionStatus.DRY_RUN


# =============================================================================
# Test Executor - Rollback
# =============================================================================

class TestExecutorRollback:
    """Tests for rollback capability."""

    @pytest.mark.asyncio
    async def test_rollback_provides_instructions(self, config, mock_client):
        """Test rollback provides manual instructions."""
        config.dry_run = False
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        transfer = MockTransfer(player_out, player_in, expected_gain=6.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        # Execute a plan
        result = await executor.execute(opt_result, confirm=True)

        if result.success:
            # Attempt rollback
            rollback_result = await executor.rollback()
            assert "rollback" in rollback_result.messages[0].lower() or "instructions" in rollback_result.messages[0].lower()

    @pytest.mark.asyncio
    async def test_rollback_no_plan_fails(self, config, mock_client):
        """Test rollback without executed plan fails."""
        executor = Executor(config, mock_client)

        with pytest.raises(ValueError, match="No executed plan"):
            await executor.rollback()

    @pytest.mark.asyncio
    async def test_execution_enables_rollback(self, config, mock_client):
        """Test that execution saves state for rollback."""
        config.dry_run = False
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        transfer = MockTransfer(player_out, player_in, expected_gain=6.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        result = await executor.execute(opt_result, confirm=True)

        # Rollback should be available if state was saved
        assert result.rollback_available is True


# =============================================================================
# Test Executor - Preview and History
# =============================================================================

class TestExecutorPreview:
    """Tests for preview functionality."""

    def test_preview_format(self, config, mock_client):
        """Test preview generates formatted output."""
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        transfer = MockTransfer(player_out, player_in, expected_gain=5.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        preview = executor.preview(opt_result)

        assert "EXECUTION PREVIEW" in preview
        assert "TRANSFERS" in preview
        assert "Salah" in preview
        assert "Palmer" in preview
        assert "SUMMARY" in preview

    def test_preview_shows_mode(self, config, mock_client):
        """Test preview shows dry run mode."""
        config.dry_run = True
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        preview = executor.preview(opt_result)

        assert "DRY RUN" in preview

    def test_preview_shows_thresholds(self, config, mock_client):
        """Test preview shows threshold status."""
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        transfer = MockTransfer(player_out, player_in, expected_gain=5.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        preview = executor.preview(opt_result)

        assert "Gain Threshold" in preview
        assert "Confidence Threshold" in preview


class TestExecutorHistory:
    """Tests for execution history."""

    @pytest.mark.asyncio
    async def test_history_tracking(self, config, mock_client):
        """Test execution history is tracked."""
        config.dry_run = True
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        await executor.dry_run(opt_result)

        history = executor.get_history()
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_format_history(self, config, mock_client):
        """Test history formatting."""
        config.dry_run = True
        executor = Executor(config, mock_client)

        opt_result = MockOptimizationResult()
        await executor.dry_run(opt_result)

        formatted = executor.format_history()

        assert "EXECUTION HISTORY" in formatted
        assert "DRY_RUN" in formatted

    def test_format_empty_history(self, config, mock_client):
        """Test formatting empty history."""
        executor = Executor(config, mock_client)

        formatted = executor.format_history()
        assert "No execution history" in formatted


# =============================================================================
# Test Convenience Function
# =============================================================================

class TestSafeExecute:
    """Tests for safe_execute convenience function."""

    @pytest.mark.asyncio
    async def test_safe_execute_dry_run(self, config, mock_client):
        """Test safe_execute with dry run."""
        config.dry_run = True

        opt_result = MockOptimizationResult()
        result = await safe_execute(config, mock_client, opt_result)

        assert result.plan.status == ExecutionStatus.DRY_RUN

    @pytest.mark.asyncio
    async def test_safe_execute_with_approval(self, config, mock_client):
        """Test safe_execute with approval requirement."""
        config.dry_run = False

        opt_result = MockOptimizationResult()
        result = await safe_execute(
            config, mock_client, opt_result,
            require_approval=True
        )

        # When require_approval=True, returns ExecutionPlan awaiting approval
        assert isinstance(result, ExecutionPlan)
        assert result.status == ExecutionStatus.AWAITING_APPROVAL


# =============================================================================
# Test Plan Persistence
# =============================================================================

class TestPlanPersistence:
    """Tests for plan saving and loading."""

    def test_save_and_load_plan(self, config, mock_client):
        """Test plan can be saved and loaded."""
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        transfer = MockTransfer(player_out, player_in, expected_gain=5.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        plan = executor.create_plan(opt_result)

        # Save plan
        executor._save_plan(plan)

        # Load plan
        loaded = executor._load_plan(plan.id)

        assert loaded is not None
        assert loaded.id == plan.id
        assert len(loaded.transfers) == 1
        assert loaded.transfers[0].player_out_name == "Salah"

    def test_load_nonexistent_plan(self, config, mock_client):
        """Test loading non-existent plan returns None."""
        executor = Executor(config, mock_client)

        loaded = executor._load_plan("nonexistent_plan")
        assert loaded is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestExecutorIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_dry_run_workflow(self, config, mock_client):
        """Test complete dry run workflow."""
        config.dry_run = True
        executor = Executor(config, mock_client)

        # Create transfers
        player_out = create_player(1, "Salah", form=5.0, total_points=80)
        player_in = create_player(2, "Palmer", form=7.0, minutes=1200)
        transfer = MockTransfer(player_out, player_in, expected_gain=5.0)

        # Create captain
        captain_player = create_player(3, "Haaland")
        vice_player = create_player(4, "Saka")
        captain = MockCaptainChoice(captain_player, expected_points=10.0)
        vice = MockCaptainChoice(vice_player, expected_points=7.0)

        opt_result = MockOptimizationResult(
            transfers=[transfer],
            captain=captain,
            vice_captain=vice,
        )

        # Run dry run
        result = await executor.execute(opt_result, confirm=True)

        assert result.plan.status == ExecutionStatus.DRY_RUN
        assert result.success is True

        # Check audit trail
        history = executor.audit_logger.get_history()
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_approval_then_execution_workflow(self, config, mock_client):
        """Test approval followed by execution workflow."""
        config.dry_run = False
        executor = Executor(config, mock_client)

        player_out = create_player(1, "Salah")
        player_in = create_player(2, "Palmer")
        transfer = MockTransfer(player_out, player_in, expected_gain=6.0)
        opt_result = MockOptimizationResult(transfers=[transfer])

        # Request approval
        plan = await executor.request_approval(opt_result, deadline_hours=2.0)
        assert plan.status == ExecutionStatus.AWAITING_APPROVAL

        # Approve
        result = await executor.approve(plan.id, approved_by="test_user")

        # Should have been executed
        assert result.plan.approved_by == "test_user"
        assert result.plan.status in [ExecutionStatus.SUCCESS, ExecutionStatus.FAILED]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
