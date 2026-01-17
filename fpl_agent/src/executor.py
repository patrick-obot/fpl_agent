"""
Transfer execution module with comprehensive safety features.

Provides:
- Dry-run mode with detailed comparison
- Confidence thresholds and alerts
- Human approval mode with notifications
- Rollback capability
- Complete decision audit logging
"""

import asyncio
import json
import logging
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import pickle

from .config import Config
from .fpl_client import FPLClient, FPLAPIError, Player


# =============================================================================
# Enums and Constants
# =============================================================================

class ExecutionStatus(Enum):
    """Status of an execution operation."""
    PENDING = "pending"
    DRY_RUN = "dry_run"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"
    TIMEOUT = "timeout"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Confidence thresholds
DEFAULT_MIN_EXPECTED_GAIN = 4.0  # Minimum expected gain after hits
DEFAULT_MIN_CONFIDENCE = 0.80   # Minimum confidence for taking hits
DEFAULT_APPROVAL_TIMEOUT = 2 * 60 * 60  # 2 hours in seconds


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TeamState:
    """Snapshot of team state for rollback capability."""
    timestamp: datetime
    picks: list[dict]  # List of player picks with positions
    captain_id: int
    vice_captain_id: int
    bank: float
    free_transfers: int
    total_points: int
    gameweek: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "picks": self.picks,
            "captain_id": self.captain_id,
            "vice_captain_id": self.vice_captain_id,
            "bank": self.bank,
            "free_transfers": self.free_transfers,
            "total_points": self.total_points,
            "gameweek": self.gameweek,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TeamState":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            picks=data["picks"],
            captain_id=data["captain_id"],
            vice_captain_id=data["vice_captain_id"],
            bank=data["bank"],
            free_transfers=data["free_transfers"],
            total_points=data["total_points"],
            gameweek=data["gameweek"],
        )


@dataclass
class Alert:
    """Represents an alert or warning about a decision."""
    level: AlertLevel
    message: str
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        icon = {"info": "[i]", "warning": "[!]", "critical": "[!!!]"}
        return f"{icon.get(self.level.value, '[?]')} {self.message}"


@dataclass
class TransferDecision:
    """Detailed record of a transfer decision."""
    player_out_id: int
    player_out_name: str
    player_in_id: int
    player_in_name: str
    expected_gain: float
    hit_cost: int
    net_gain: float
    reason: str
    confidence: float
    alerts: list[Alert] = field(default_factory=list)


@dataclass
class CaptainDecision:
    """Detailed record of a captain decision."""
    captain_id: int
    captain_name: str
    vice_captain_id: int
    vice_captain_name: str
    expected_points: float
    confidence: float
    is_differential: bool
    reason: str


@dataclass
class ExecutionPlan:
    """Comprehensive plan for executing changes."""
    id: str  # Unique plan identifier
    transfers: list[TransferDecision] = field(default_factory=list)
    captain: Optional[CaptainDecision] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    approval_deadline: Optional[datetime] = None
    total_expected_gain: float = 0.0
    total_hit_cost: int = 0
    net_expected_gain: float = 0.0
    overall_confidence: float = 0.0
    alerts: list[Alert] = field(default_factory=list)
    team_state_before: Optional[TeamState] = None
    team_state_after: Optional[TeamState] = None
    error_message: Optional[str] = None
    approved_by: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of an execution attempt."""
    plan: ExecutionPlan
    transfers_executed: int = 0
    captain_set: bool = False
    success: bool = False
    messages: list[str] = field(default_factory=list)
    rollback_available: bool = False


@dataclass
class AuditEntry:
    """Single entry in the decision audit trail."""
    timestamp: datetime
    action: str
    plan_id: str
    details: dict
    outcome: str
    user: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "plan_id": self.plan_id,
            "details": self.details,
            "outcome": self.outcome,
            "user": self.user,
        }


# =============================================================================
# Notification System
# =============================================================================

class NotificationService:
    """Handles sending notifications for approval requests."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("fpl_agent.executor.notify")

    async def send_approval_request(
        self,
        plan: ExecutionPlan,
        deadline: datetime
    ) -> bool:
        """
        Send approval request notification.

        Args:
            plan: The execution plan requiring approval.
            deadline: Deadline for approval.

        Returns:
            True if notification was sent successfully.
        """
        subject = f"FPL Agent - Approval Required (Deadline: {deadline.strftime('%H:%M')})"
        body = self._format_approval_email(plan, deadline)

        # Try email first
        if self.config.notification_email:
            try:
                await self._send_email(subject, body)
                self.logger.info(f"Approval request sent to {self.config.notification_email}")
                return True
            except Exception as e:
                self.logger.warning(f"Email notification failed: {e}")

        # Try webhook (Slack, Discord, etc.)
        if self.config.webhook_url:
            try:
                await self._send_webhook(plan, deadline)
                self.logger.info("Approval request sent via webhook")
                return True
            except Exception as e:
                self.logger.warning(f"Webhook notification failed: {e}")

        return False

    async def send_execution_result(
        self,
        result: ExecutionResult
    ) -> bool:
        """Send notification about execution result."""
        status = "SUCCESS" if result.success else "FAILED"
        subject = f"FPL Agent - Execution {status}"
        body = self._format_result_email(result)

        if self.config.notification_email:
            try:
                await self._send_email(subject, body)
                return True
            except Exception as e:
                self.logger.warning(f"Result notification failed: {e}")

        return False

    async def _send_email(self, subject: str, body: str) -> None:
        """Send email notification."""
        if not all([
            self.config.smtp_host,
            self.config.smtp_user,
            self.config.smtp_password,
            self.config.notification_email
        ]):
            raise ValueError("Email configuration incomplete")

        msg = MIMEMultipart()
        msg["From"] = self.config.smtp_user
        msg["To"] = self.config.notification_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_smtp, msg)

    def _send_smtp(self, msg: MIMEMultipart) -> None:
        """Send via SMTP (blocking)."""
        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
            server.starttls()
            server.login(self.config.smtp_user, self.config.smtp_password)
            server.send_message(msg)

    async def _send_webhook(self, plan: ExecutionPlan, deadline: datetime) -> None:
        """Send webhook notification (Slack/Discord format)."""
        import aiohttp

        payload = {
            "text": f"FPL Agent - Approval Required",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "FPL Transfer Approval Required"}
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Deadline:* {deadline.strftime('%Y-%m-%d %H:%M')}\n"
                               f"*Transfers:* {len(plan.transfers)}\n"
                               f"*Net Gain:* {plan.net_expected_gain:.1f} pts"
                    }
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.webhook_url,
                json=payload,
                timeout=30
            ) as response:
                if response.status not in (200, 204):
                    raise Exception(f"Webhook returned {response.status}")

    def _format_approval_email(self, plan: ExecutionPlan, deadline: datetime) -> str:
        """Format approval request email body."""
        lines = [
            "FPL Agent - Approval Required",
            "=" * 50,
            "",
            f"Plan ID: {plan.id}",
            f"Created: {plan.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Deadline: {deadline.strftime('%Y-%m-%d %H:%M')}",
            "",
            "PROPOSED TRANSFERS:",
            "-" * 30,
        ]

        if plan.transfers:
            for t in plan.transfers:
                hit_str = f" (HIT: {t.hit_cost})" if t.hit_cost < 0 else ""
                lines.append(f"  OUT: {t.player_out_name} -> IN: {t.player_in_name}{hit_str}")
                lines.append(f"       Net gain: {t.net_gain:.1f} pts | {t.reason}")
        else:
            lines.append("  No transfers")

        lines.extend([
            "",
            "CAPTAIN CHANGE:",
            "-" * 30,
        ])

        if plan.captain:
            lines.append(f"  Captain: {plan.captain.captain_name}")
            lines.append(f"  Vice-Captain: {plan.captain.vice_captain_name}")
        else:
            lines.append("  No change")

        lines.extend([
            "",
            "SUMMARY:",
            "-" * 30,
            f"  Total Expected Gain: {plan.total_expected_gain:.1f} pts",
            f"  Total Hit Cost: {plan.total_hit_cost} pts",
            f"  Net Expected Gain: {plan.net_expected_gain:.1f} pts",
            f"  Confidence: {plan.overall_confidence:.0%}",
        ])

        if plan.alerts:
            lines.extend([
                "",
                "ALERTS:",
                "-" * 30,
            ])
            for alert in plan.alerts:
                lines.append(f"  {alert}")

        lines.extend([
            "",
            "=" * 50,
            "To approve, respond with 'APPROVE' or use the web interface.",
            "The plan will be auto-rejected after the deadline.",
        ])

        return "\n".join(lines)

    def _format_result_email(self, result: ExecutionResult) -> str:
        """Format execution result email body."""
        status = "SUCCESS" if result.success else "FAILED"

        lines = [
            f"FPL Agent - Execution {status}",
            "=" * 50,
            "",
            f"Plan ID: {result.plan.id}",
            f"Status: {result.plan.status.value}",
            f"Executed: {result.plan.executed_at.strftime('%Y-%m-%d %H:%M') if result.plan.executed_at else 'N/A'}",
            "",
            f"Transfers Executed: {result.transfers_executed}/{len(result.plan.transfers)}",
            f"Captain Set: {'Yes' if result.captain_set else 'No'}",
            "",
            "Messages:",
        ]

        for msg in result.messages:
            lines.append(f"  - {msg}")

        return "\n".join(lines)


# =============================================================================
# Audit Logger
# =============================================================================

class AuditLogger:
    """Maintains decision audit trail."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("fpl_agent.executor.audit")
        self._audit_file = config.data_dir / "audit_trail.json"
        self._entries: list[AuditEntry] = []
        self._load_audit_trail()

    def _load_audit_trail(self) -> None:
        """Load existing audit trail from disk."""
        if self._audit_file.exists():
            try:
                with open(self._audit_file, "r") as f:
                    data = json.load(f)
                    self._entries = [
                        AuditEntry(
                            timestamp=datetime.fromisoformat(e["timestamp"]),
                            action=e["action"],
                            plan_id=e["plan_id"],
                            details=e["details"],
                            outcome=e["outcome"],
                            user=e.get("user"),
                        )
                        for e in data
                    ]
                self.logger.debug(f"Loaded {len(self._entries)} audit entries")
            except Exception as e:
                self.logger.warning(f"Failed to load audit trail: {e}")

    def _save_audit_trail(self) -> None:
        """Save audit trail to disk."""
        try:
            with open(self._audit_file, "w") as f:
                json.dump([e.to_dict() for e in self._entries], f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save audit trail: {e}")

    def log(
        self,
        action: str,
        plan_id: str,
        details: dict,
        outcome: str,
        user: Optional[str] = None
    ) -> None:
        """Log an audit entry."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            action=action,
            plan_id=plan_id,
            details=details,
            outcome=outcome,
            user=user,
        )
        self._entries.append(entry)
        self._save_audit_trail()

        # Also log to standard logger
        self.logger.info(f"AUDIT: {action} | Plan: {plan_id} | Outcome: {outcome}")

    def get_history(self, plan_id: Optional[str] = None, limit: int = 100) -> list[AuditEntry]:
        """Get audit history, optionally filtered by plan ID."""
        entries = self._entries
        if plan_id:
            entries = [e for e in entries if e.plan_id == plan_id]
        return entries[-limit:]

    def format_history(self, limit: int = 20) -> str:
        """Format audit history for display."""
        entries = self._entries[-limit:]

        lines = [
            "DECISION AUDIT TRAIL",
            "=" * 70,
        ]

        for entry in reversed(entries):
            lines.append(
                f"\n{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{entry.action} | {entry.outcome}"
            )
            lines.append(f"  Plan: {entry.plan_id}")
            if entry.user:
                lines.append(f"  User: {entry.user}")

        return "\n".join(lines)


# =============================================================================
# State Manager (Rollback)
# =============================================================================

class StateManager:
    """Manages team state snapshots for rollback capability."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("fpl_agent.executor.state")
        self._state_dir = config.data_dir / "states"
        self._state_dir.mkdir(exist_ok=True)

    async def save_state(self, client: FPLClient, label: str = "auto") -> Optional[TeamState]:
        """
        Save current team state.

        Args:
            client: FPL API client.
            label: Label for the state snapshot.

        Returns:
            TeamState object if successful.
        """
        try:
            my_team = await client.get_my_team()
            current_gw = await client.get_current_gameweek()

            # Convert SquadPick objects to dicts for serialization
            picks_data = [
                {
                    "element": pick.element,
                    "position": pick.position,
                    "is_captain": pick.is_captain,
                    "is_vice_captain": pick.is_vice_captain,
                    "multiplier": pick.multiplier,
                }
                for pick in my_team.picks
            ]

            # Get captain and vice-captain from MyTeam properties
            captain_id = my_team.captain_id or 0
            vice_captain_id = my_team.vice_captain_id or 0

            state = TeamState(
                timestamp=datetime.now(),
                picks=picks_data,
                captain_id=captain_id,
                vice_captain_id=vice_captain_id,
                bank=my_team.bank,
                free_transfers=my_team.free_transfers,
                total_points=0,  # Not available in MyTeam directly
                gameweek=current_gw.id,
            )

            # Save to file
            filename = f"state_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self._state_dir / filename

            with open(filepath, "w") as f:
                json.dump(state.to_dict(), f, indent=2)

            self.logger.info(f"Saved team state: {filename}")
            return state

        except Exception as e:
            self.logger.error(f"Failed to save team state: {e}")
            return None

    def load_state(self, filename: str) -> Optional[TeamState]:
        """Load a saved team state."""
        filepath = self._state_dir / filename
        if not filepath.exists():
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return TeamState.from_dict(data)
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return None

    def list_states(self) -> list[str]:
        """List available state snapshots."""
        return sorted([f.name for f in self._state_dir.glob("state_*.json")])

    def get_latest_state(self) -> Optional[TeamState]:
        """Get the most recent state snapshot."""
        states = self.list_states()
        if not states:
            return None
        return self.load_state(states[-1])


# =============================================================================
# Main Executor Class
# =============================================================================

class Executor:
    """
    Safely executes transfers and captain changes with comprehensive safety features.

    Features:
    - Dry-run mode with detailed before/after comparison
    - Confidence thresholds and alerts
    - Human approval mode with email/webhook notifications
    - Rollback capability with state snapshots
    - Complete decision audit logging
    """

    def __init__(
        self,
        config: Config,
        client: FPLClient,
        min_expected_gain: float = DEFAULT_MIN_EXPECTED_GAIN,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        approval_timeout: int = DEFAULT_APPROVAL_TIMEOUT,
    ):
        """
        Initialize the executor.

        Args:
            config: Application configuration.
            client: FPL API client.
            min_expected_gain: Minimum net gain to execute (default 4 pts).
            min_confidence: Minimum confidence for hits (default 80%).
            approval_timeout: Timeout for approval in seconds (default 2 hours).
        """
        self.config = config
        self.client = client
        self.min_expected_gain = min_expected_gain
        self.min_confidence = min_confidence
        self.approval_timeout = approval_timeout
        self.logger = logging.getLogger("fpl_agent.executor")

        # Components
        self.notification_service = NotificationService(config)
        self.audit_logger = AuditLogger(config)
        self.state_manager = StateManager(config)

        # Session state
        self._history: list[ExecutionResult] = []
        self._pending_approval: Optional[ExecutionPlan] = None
        self._approval_callback: Optional[Callable[[bool], Awaitable[None]]] = None

        # Plans storage
        self._plans_dir = config.data_dir / "plans"
        self._plans_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Plan Creation
    # -------------------------------------------------------------------------

    def create_plan(
        self,
        optimization_result,  # OptimizationResult
        player_df=None,  # For top performer checks
    ) -> ExecutionPlan:
        """
        Create an execution plan from optimization results.

        Args:
            optimization_result: Results from the optimizer.
            player_df: Player DataFrame for validation.

        Returns:
            ExecutionPlan with full details and alerts.
        """
        from .optimizer import OptimizationResult

        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        plan = ExecutionPlan(
            id=plan_id,
            created_at=datetime.now(),
        )

        # Process transfers
        for transfer in optimization_result.transfers:
            decision = TransferDecision(
                player_out_id=transfer.player_out.id,
                player_out_name=transfer.player_out.web_name,
                player_in_id=transfer.player_in.id,
                player_in_name=transfer.player_in.web_name,
                expected_gain=transfer.expected_gain,
                hit_cost=transfer.hit_cost,
                net_gain=transfer.net_gain,
                reason=transfer.reason,
                confidence=0.8,  # Default confidence
            )

            # Check for alerts
            alerts = self._check_transfer_alerts(transfer, player_df)
            decision.alerts = alerts
            plan.alerts.extend(alerts)

            plan.transfers.append(decision)

        # Process captain
        if optimization_result.captain:
            cap = optimization_result.captain
            vice = optimization_result.vice_captain

            plan.captain = CaptainDecision(
                captain_id=cap.player.id,
                captain_name=cap.player.web_name,
                vice_captain_id=vice.player.id if vice else cap.player.id,
                vice_captain_name=vice.player.web_name if vice else cap.player.web_name,
                expected_points=cap.expected_points,
                confidence=cap.confidence,
                is_differential=cap.is_differential,
                reason=cap.reason,
            )

        # Calculate totals
        plan.total_expected_gain = sum(t.expected_gain for t in plan.transfers)
        plan.total_hit_cost = sum(t.hit_cost for t in plan.transfers)
        plan.net_expected_gain = sum(t.net_gain for t in plan.transfers)
        plan.overall_confidence = self._calculate_overall_confidence(plan)

        # Check confidence thresholds
        threshold_alerts = self._check_threshold_alerts(plan)
        plan.alerts.extend(threshold_alerts)

        self.audit_logger.log(
            action="PLAN_CREATED",
            plan_id=plan_id,
            details={
                "transfers": len(plan.transfers),
                "net_gain": plan.net_expected_gain,
                "confidence": plan.overall_confidence,
                "alerts": len(plan.alerts),
            },
            outcome="created",
        )

        return plan

    def _check_transfer_alerts(self, transfer, player_df) -> list[Alert]:
        """Check for alerts on a specific transfer."""
        alerts = []

        # Alert: Selling top performer
        if transfer.player_out.form >= 6.0:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Selling {transfer.player_out.web_name} who is in good form ({transfer.player_out.form})",
                details={"form": transfer.player_out.form},
            ))

        # Alert: Selling high points player
        if transfer.player_out.total_points >= 100:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Selling {transfer.player_out.web_name} with {transfer.player_out.total_points} points",
                details={"total_points": transfer.player_out.total_points},
            ))

        # Alert: Buying player with low minutes
        if transfer.player_in.minutes < 500:
            alerts.append(Alert(
                level=AlertLevel.INFO,
                message=f"Buying {transfer.player_in.web_name} with limited minutes ({transfer.player_in.minutes})",
                details={"minutes": transfer.player_in.minutes},
            ))

        # Alert: Taking a hit for small gain
        if transfer.hit_cost < 0 and transfer.expected_gain < 6.0:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Taking {transfer.hit_cost} hit for only {transfer.expected_gain:.1f} expected gain",
                details={"hit_cost": transfer.hit_cost, "expected_gain": transfer.expected_gain},
            ))

        return alerts

    def _check_threshold_alerts(self, plan: ExecutionPlan) -> list[Alert]:
        """Check confidence threshold alerts."""
        alerts = []

        # Check minimum gain threshold
        if plan.net_expected_gain < self.min_expected_gain and plan.transfers:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Net gain ({plan.net_expected_gain:.1f}) below threshold ({self.min_expected_gain})",
                details={
                    "net_gain": plan.net_expected_gain,
                    "threshold": self.min_expected_gain,
                },
            ))

        # Check confidence for hits
        if plan.total_hit_cost < 0 and plan.overall_confidence < self.min_confidence:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Confidence ({plan.overall_confidence:.0%}) below threshold for taking hits",
                details={
                    "confidence": plan.overall_confidence,
                    "threshold": self.min_confidence,
                },
            ))

        return alerts

    def _calculate_overall_confidence(self, plan: ExecutionPlan) -> float:
        """Calculate overall plan confidence."""
        if not plan.transfers and not plan.captain:
            return 1.0

        confidences = []

        for t in plan.transfers:
            # Reduce confidence for each alert
            conf = t.confidence
            for alert in t.alerts:
                if alert.level == AlertLevel.WARNING:
                    conf *= 0.9
                elif alert.level == AlertLevel.CRITICAL:
                    conf *= 0.7
            confidences.append(conf)

        if plan.captain:
            confidences.append(plan.captain.confidence)

        return sum(confidences) / len(confidences) if confidences else 1.0

    # -------------------------------------------------------------------------
    # Dry Run Mode
    # -------------------------------------------------------------------------

    async def dry_run(
        self,
        optimization_result,
        player_df=None,
    ) -> ExecutionResult:
        """
        Execute in dry-run mode - log all decisions without making changes.

        Args:
            optimization_result: Results from the optimizer.
            player_df: Player DataFrame for comparison.

        Returns:
            ExecutionResult with detailed comparison.
        """
        plan = self.create_plan(optimization_result, player_df)
        plan.status = ExecutionStatus.DRY_RUN

        result = ExecutionResult(plan=plan)

        # Save current state for comparison
        plan.team_state_before = await self.state_manager.save_state(
            self.client, label="dry_run"
        )

        # Log the plan details
        self.logger.info("=" * 70)
        self.logger.info("DRY RUN - No actual changes will be made")
        self.logger.info("=" * 70)

        # Show before/after comparison
        comparison = self._generate_comparison(plan)
        self.logger.info(comparison)

        # Calculate expected points gain
        points_analysis = self._analyze_expected_points(plan)
        self.logger.info(points_analysis)

        result.success = True
        result.messages.append("Dry run completed - no changes made")
        result.messages.append(f"Net expected gain: {plan.net_expected_gain:.1f} pts")

        if plan.alerts:
            result.messages.append(f"{len(plan.alerts)} alert(s) generated")

        self.audit_logger.log(
            action="DRY_RUN",
            plan_id=plan.id,
            details={
                "transfers": len(plan.transfers),
                "net_gain": plan.net_expected_gain,
                "alerts": len(plan.alerts),
            },
            outcome="completed",
        )

        self._history.append(result)
        return result

    def _generate_comparison(self, plan: ExecutionPlan) -> str:
        """Generate before/after comparison string."""
        lines = [
            "",
            "BEFORE/AFTER COMPARISON",
            "-" * 50,
        ]

        if plan.team_state_before:
            state = plan.team_state_before
            lines.append(f"Current Bank: {state.bank:.1f}m")
            lines.append(f"Free Transfers: {state.free_transfers}")

        lines.append("")
        lines.append("PROPOSED CHANGES:")

        if plan.transfers:
            for t in plan.transfers:
                lines.append(f"  OUT: {t.player_out_name}")
                lines.append(f"  IN:  {t.player_in_name}")
                lines.append(f"       Expected: +{t.expected_gain:.1f} pts")
                if t.hit_cost < 0:
                    lines.append(f"       Hit Cost: {t.hit_cost} pts")
                lines.append(f"       Net: +{t.net_gain:.1f} pts")
                lines.append("")
        else:
            lines.append("  No transfers")

        if plan.captain:
            lines.append(f"CAPTAIN: {plan.captain.captain_name}")
            lines.append(f"         Expected: {plan.captain.expected_points:.1f} pts")

        return "\n".join(lines)

    def _analyze_expected_points(self, plan: ExecutionPlan) -> str:
        """Analyze expected points impact."""
        lines = [
            "",
            "EXPECTED POINTS ANALYSIS",
            "-" * 50,
            f"Total Expected Gain: {plan.total_expected_gain:+.1f} pts",
            f"Total Hit Cost:      {plan.total_hit_cost:+d} pts",
            f"Net Expected Gain:   {plan.net_expected_gain:+.1f} pts",
            f"Overall Confidence:  {plan.overall_confidence:.0%}",
        ]

        if plan.net_expected_gain >= self.min_expected_gain:
            lines.append("")
            lines.append("[OK] Net gain meets threshold")
        else:
            lines.append("")
            lines.append(f"[!] Net gain below threshold ({self.min_expected_gain})")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Human Approval Mode
    # -------------------------------------------------------------------------

    async def request_approval(
        self,
        optimization_result,
        player_df=None,
        deadline_hours: float = 2.0,
    ) -> ExecutionPlan:
        """
        Request human approval for execution.

        Args:
            optimization_result: Results from the optimizer.
            player_df: Player DataFrame for validation.
            deadline_hours: Hours until approval deadline.

        Returns:
            ExecutionPlan in AWAITING_APPROVAL status.
        """
        plan = self.create_plan(optimization_result, player_df)
        plan.status = ExecutionStatus.AWAITING_APPROVAL
        plan.approval_deadline = datetime.now() + timedelta(hours=deadline_hours)

        # Save current state
        plan.team_state_before = await self.state_manager.save_state(
            self.client, label="pre_approval"
        )

        # Store pending approval
        self._pending_approval = plan

        # Send notification
        await self.notification_service.send_approval_request(
            plan, plan.approval_deadline
        )

        # Save plan to disk
        self._save_plan(plan)

        self.audit_logger.log(
            action="APPROVAL_REQUESTED",
            plan_id=plan.id,
            details={
                "deadline": plan.approval_deadline.isoformat(),
                "transfers": len(plan.transfers),
                "net_gain": plan.net_expected_gain,
            },
            outcome="awaiting",
        )

        self.logger.info(f"Approval requested. Deadline: {plan.approval_deadline}")
        return plan

    async def approve(
        self,
        plan_id: Optional[str] = None,
        approved_by: str = "user",
    ) -> ExecutionResult:
        """
        Approve and execute a pending plan.

        Args:
            plan_id: Plan ID to approve (uses pending if not specified).
            approved_by: Identifier of approver.

        Returns:
            ExecutionResult from execution.
        """
        plan = self._get_plan(plan_id)

        if not plan:
            raise ValueError(f"No plan found: {plan_id or 'pending'}")

        if plan.status != ExecutionStatus.AWAITING_APPROVAL:
            raise ValueError(f"Plan is not awaiting approval: {plan.status}")

        # Check deadline
        if plan.approval_deadline and datetime.now() > plan.approval_deadline:
            plan.status = ExecutionStatus.TIMEOUT
            self.audit_logger.log(
                action="APPROVAL_TIMEOUT",
                plan_id=plan.id,
                details={},
                outcome="timeout",
            )
            raise ValueError("Approval deadline has passed")

        plan.status = ExecutionStatus.APPROVED
        plan.approved_at = datetime.now()
        plan.approved_by = approved_by

        self.audit_logger.log(
            action="APPROVED",
            plan_id=plan.id,
            details={"approved_by": approved_by},
            outcome="approved",
            user=approved_by,
        )

        # Execute the plan
        return await self._execute_plan(plan)

    async def reject(
        self,
        plan_id: Optional[str] = None,
        rejected_by: str = "user",
        reason: str = "Manual rejection",
    ) -> None:
        """
        Reject a pending plan.

        Args:
            plan_id: Plan ID to reject.
            rejected_by: Identifier of rejector.
            reason: Reason for rejection.
        """
        plan = self._get_plan(plan_id)

        if not plan:
            raise ValueError(f"No plan found: {plan_id or 'pending'}")

        plan.status = ExecutionStatus.REJECTED
        plan.error_message = reason

        self.audit_logger.log(
            action="REJECTED",
            plan_id=plan.id,
            details={"reason": reason, "rejected_by": rejected_by},
            outcome="rejected",
            user=rejected_by,
        )

        self._save_plan(plan)
        self._pending_approval = None
        self.logger.info(f"Plan {plan.id} rejected: {reason}")

    async def wait_for_approval(
        self,
        plan: ExecutionPlan,
        check_interval: int = 60,
    ) -> ExecutionResult:
        """
        Wait for approval with timeout.

        Args:
            plan: Plan awaiting approval.
            check_interval: Seconds between checks.

        Returns:
            ExecutionResult when approved or timed out.
        """
        self.logger.info(f"Waiting for approval until {plan.approval_deadline}")

        while datetime.now() < plan.approval_deadline:
            # Check if approved externally
            loaded_plan = self._load_plan(plan.id)
            if loaded_plan and loaded_plan.status == ExecutionStatus.APPROVED:
                return await self._execute_plan(loaded_plan)

            if loaded_plan and loaded_plan.status == ExecutionStatus.REJECTED:
                return ExecutionResult(
                    plan=loaded_plan,
                    success=False,
                    messages=["Plan was rejected"],
                )

            await asyncio.sleep(check_interval)

        # Timeout reached
        plan.status = ExecutionStatus.TIMEOUT
        self._save_plan(plan)

        self.audit_logger.log(
            action="APPROVAL_TIMEOUT",
            plan_id=plan.id,
            details={"deadline": plan.approval_deadline.isoformat()},
            outcome="timeout",
        )

        return ExecutionResult(
            plan=plan,
            success=False,
            messages=["Approval timeout - no action taken"],
        )

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def execute(
        self,
        optimization_result,
        confirm: bool = False,
        require_approval: bool = False,
        player_df=None,
    ) -> ExecutionResult:
        """
        Execute optimization recommendations with safety checks.

        Args:
            optimization_result: Results from the optimizer.
            confirm: If True, skip final confirmation.
            require_approval: If True, use approval workflow.
            player_df: Player DataFrame for validation.

        Returns:
            ExecutionResult with execution details.
        """
        # Handle dry run mode
        if self.config.dry_run:
            return await self.dry_run(optimization_result, player_df)

        # Create plan with validation
        plan = self.create_plan(optimization_result, player_df)

        # Check for critical alerts
        critical_alerts = [a for a in plan.alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts and not confirm:
            plan.status = ExecutionStatus.CANCELLED
            result = ExecutionResult(plan=plan, success=False)
            result.messages.append("Execution blocked due to critical alerts:")
            for alert in critical_alerts:
                result.messages.append(f"  - {alert.message}")
            result.messages.append("Use confirm=True to override")
            self._history.append(result)
            return result

        # Use approval workflow if required
        if require_approval:
            return await self.request_approval(optimization_result, player_df)

        # Require confirmation for live execution
        if not confirm:
            plan.status = ExecutionStatus.PENDING
            result = ExecutionResult(plan=plan)
            result.messages.append("Execution requires confirmation")
            result.messages.append(f"Net expected gain: {plan.net_expected_gain:.1f} pts")
            result.messages.append("Set confirm=True to execute")
            self._history.append(result)
            return result

        return await self._execute_plan(plan)

    async def _execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute a validated plan."""
        result = ExecutionResult(plan=plan)

        # Save state before execution
        plan.team_state_before = await self.state_manager.save_state(
            self.client, label=f"before_{plan.id}"
        )
        result.rollback_available = plan.team_state_before is not None

        plan.status = ExecutionStatus.EXECUTING
        self.logger.info(f"Executing plan {plan.id}")

        try:
            # Execute transfers
            if plan.transfers:
                await self._execute_transfers(plan, result)

            # Execute captain change
            if plan.captain:
                await self._execute_captain(plan, result)

            # Save state after execution
            plan.team_state_after = await self.state_manager.save_state(
                self.client, label=f"after_{plan.id}"
            )

            # Determine success
            result.success = (
                result.transfers_executed == len(plan.transfers) and
                (plan.captain is None or result.captain_set)
            )

            if result.success:
                plan.status = ExecutionStatus.SUCCESS
                plan.executed_at = datetime.now()
                self.logger.info("Execution completed successfully")
            else:
                plan.status = ExecutionStatus.FAILED
                self.logger.warning("Execution completed with errors")

        except Exception as e:
            plan.status = ExecutionStatus.FAILED
            plan.error_message = str(e)
            result.messages.append(f"Execution error: {e}")
            self.logger.error(f"Execution failed: {e}")

        # Log audit entry
        self.audit_logger.log(
            action="EXECUTED",
            plan_id=plan.id,
            details={
                "transfers_executed": result.transfers_executed,
                "captain_set": result.captain_set,
                "success": result.success,
            },
            outcome=plan.status.value,
        )

        # Send result notification
        await self.notification_service.send_execution_result(result)

        self._save_plan(plan)
        self._history.append(result)
        return result

    async def _execute_transfers(self, plan: ExecutionPlan, result: ExecutionResult) -> None:
        """Execute transfers from plan."""
        from .fpl_client import Transfer

        transfers = []
        for t in plan.transfers:
            transfers.append(Transfer(
                element_in=t.player_in_id,
                element_out=t.player_out_id,
            ))
            self.logger.info(f"Preparing: {t.player_out_name} -> {t.player_in_name}")

        try:
            transfer_result = await self.client.make_transfers(
                transfers=transfers,
                confirm=True,
            )

            if transfer_result.success:
                result.transfers_executed = len(transfers)
                result.messages.append(f"Executed {len(transfers)} transfer(s)")
            else:
                result.messages.append(f"Transfer error: {transfer_result.message}")

        except Exception as e:
            result.messages.append(f"Transfer API error: {e}")
            raise

    async def _execute_captain(self, plan: ExecutionPlan, result: ExecutionResult) -> None:
        """Execute captain change from plan."""
        cap = plan.captain

        self.logger.info(f"Setting captain: {cap.captain_name}")

        try:
            response = await self.client.set_captain(
                captain_id=cap.captain_id,
                vice_captain_id=cap.vice_captain_id,
            )

            if response.get("success") or response.get("status") == "dry_run":
                result.captain_set = True
                result.messages.append(f"Captain set to {cap.captain_name}")
            else:
                result.messages.append(f"Captain error: {response}")

        except Exception as e:
            result.messages.append(f"Captain API error: {e}")
            raise

    # -------------------------------------------------------------------------
    # Rollback
    # -------------------------------------------------------------------------

    async def rollback(
        self,
        plan_id: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Attempt to rollback a previously executed plan.

        Note: Full rollback requires making reverse transfers which costs points.
        This provides the state information needed for manual rollback.

        Args:
            plan_id: Plan ID to rollback (uses latest if not specified).

        Returns:
            ExecutionResult with rollback information.
        """
        # Find the plan
        if plan_id:
            plan = self._load_plan(plan_id)
        else:
            # Find latest executed plan
            executed = [r for r in self._history if r.plan.status == ExecutionStatus.SUCCESS]
            plan = executed[-1].plan if executed else None

        if not plan:
            raise ValueError("No executed plan found to rollback")

        if not plan.team_state_before:
            raise ValueError("No saved state available for rollback")

        result = ExecutionResult(plan=plan)

        self.logger.warning("=" * 70)
        self.logger.warning("ROLLBACK INFORMATION")
        self.logger.warning("=" * 70)
        self.logger.warning("Automatic rollback is not supported by FPL API.")
        self.logger.warning("To undo transfers, you must make reverse transfers manually.")
        self.logger.warning("")
        self.logger.warning("ORIGINAL STATE:")

        state = plan.team_state_before
        self.logger.warning(f"  Bank: {state.bank:.1f}m")
        self.logger.warning(f"  Free Transfers: {state.free_transfers}")
        self.logger.warning(f"  Captain ID: {state.captain_id}")
        self.logger.warning(f"  Vice-Captain ID: {state.vice_captain_id}")

        # Generate reverse transfer instructions
        self.logger.warning("")
        self.logger.warning("TO ROLLBACK TRANSFERS:")
        for t in plan.transfers:
            self.logger.warning(f"  Transfer {t.player_in_name} OUT -> {t.player_out_name} IN")

        plan.status = ExecutionStatus.ROLLED_BACK
        self._save_plan(plan)

        self.audit_logger.log(
            action="ROLLBACK_REQUESTED",
            plan_id=plan.id,
            details={"original_state": state.to_dict()},
            outcome="instructions_provided",
        )

        result.messages.append("Rollback instructions generated")
        result.messages.append("Manual transfers required to complete rollback")

        return result

    # -------------------------------------------------------------------------
    # Plan Persistence
    # -------------------------------------------------------------------------

    def _save_plan(self, plan: ExecutionPlan) -> None:
        """Save plan to disk."""
        filepath = self._plans_dir / f"{plan.id}.json"

        data = {
            "id": plan.id,
            "status": plan.status.value,
            "created_at": plan.created_at.isoformat(),
            "approved_at": plan.approved_at.isoformat() if plan.approved_at else None,
            "executed_at": plan.executed_at.isoformat() if plan.executed_at else None,
            "approval_deadline": plan.approval_deadline.isoformat() if plan.approval_deadline else None,
            "transfers": [
                {
                    "player_out_id": t.player_out_id,
                    "player_out_name": t.player_out_name,
                    "player_in_id": t.player_in_id,
                    "player_in_name": t.player_in_name,
                    "expected_gain": t.expected_gain,
                    "hit_cost": t.hit_cost,
                    "net_gain": t.net_gain,
                    "reason": t.reason,
                    "confidence": t.confidence,
                }
                for t in plan.transfers
            ],
            "captain": {
                "captain_id": plan.captain.captain_id,
                "captain_name": plan.captain.captain_name,
                "vice_captain_id": plan.captain.vice_captain_id,
                "vice_captain_name": plan.captain.vice_captain_name,
                "expected_points": plan.captain.expected_points,
                "confidence": plan.captain.confidence,
                "reason": plan.captain.reason,
            } if plan.captain else None,
            "net_expected_gain": plan.net_expected_gain,
            "overall_confidence": plan.overall_confidence,
            "approved_by": plan.approved_by,
            "error_message": plan.error_message,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _load_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Load plan from disk."""
        filepath = self._plans_dir / f"{plan_id}.json"
        if not filepath.exists():
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            plan = ExecutionPlan(
                id=data["id"],
                status=ExecutionStatus(data["status"]),
                created_at=datetime.fromisoformat(data["created_at"]),
            )

            if data.get("approved_at"):
                plan.approved_at = datetime.fromisoformat(data["approved_at"])
            if data.get("executed_at"):
                plan.executed_at = datetime.fromisoformat(data["executed_at"])
            if data.get("approval_deadline"):
                plan.approval_deadline = datetime.fromisoformat(data["approval_deadline"])

            for t_data in data.get("transfers", []):
                plan.transfers.append(TransferDecision(**t_data))

            if data.get("captain"):
                plan.captain = CaptainDecision(**data["captain"])

            plan.net_expected_gain = data.get("net_expected_gain", 0)
            plan.overall_confidence = data.get("overall_confidence", 0)
            plan.approved_by = data.get("approved_by")
            plan.error_message = data.get("error_message")

            return plan

        except Exception as e:
            self.logger.error(f"Failed to load plan {plan_id}: {e}")
            return None

    def _get_plan(self, plan_id: Optional[str]) -> Optional[ExecutionPlan]:
        """Get plan by ID or return pending plan."""
        if plan_id:
            return self._load_plan(plan_id)
        return self._pending_approval

    # -------------------------------------------------------------------------
    # Preview and History
    # -------------------------------------------------------------------------

    def preview(self, optimization_result) -> str:
        """
        Generate detailed preview of what would be executed.

        Args:
            optimization_result: Results from the optimizer.

        Returns:
            Formatted preview string.
        """
        plan = self.create_plan(optimization_result)

        lines = [
            "=" * 70,
            "EXECUTION PREVIEW",
            f"Mode: {'DRY RUN' if self.config.dry_run else 'LIVE'}",
            f"Plan ID: {plan.id}",
            "=" * 70,
        ]

        # Transfers
        lines.append("\n[TRANSFERS]")
        lines.append("-" * 50)

        if plan.transfers:
            for i, t in enumerate(plan.transfers, 1):
                hit_str = f" (HIT: {t.hit_cost})" if t.hit_cost < 0 else ""
                lines.append(f"  {i}. OUT: {t.player_out_name} -> IN: {t.player_in_name}{hit_str}")
                lines.append(f"     Expected: +{t.expected_gain:.1f} pts | Net: +{t.net_gain:.1f} pts")
                lines.append(f"     Reason: {t.reason}")

                if t.alerts:
                    for alert in t.alerts:
                        lines.append(f"     {alert}")
        else:
            lines.append("  No transfers")

        # Captain
        lines.append("\n[CAPTAIN]")
        lines.append("-" * 50)

        if plan.captain:
            lines.append(f"  Captain: {plan.captain.captain_name}")
            lines.append(f"  Vice-Captain: {plan.captain.vice_captain_name}")
            lines.append(f"  Expected: {plan.captain.expected_points:.1f} pts")
            lines.append(f"  Confidence: {plan.captain.confidence:.0%}")
            if plan.captain.is_differential:
                lines.append("  [DIFFERENTIAL]")
        else:
            lines.append("  No change")

        # Summary
        lines.append("\n[SUMMARY]")
        lines.append("-" * 50)
        lines.append(f"  Total Expected Gain: {plan.total_expected_gain:+.1f} pts")
        lines.append(f"  Total Hit Cost: {plan.total_hit_cost:+d} pts")
        lines.append(f"  Net Expected Gain: {plan.net_expected_gain:+.1f} pts")
        lines.append(f"  Overall Confidence: {plan.overall_confidence:.0%}")

        # Thresholds
        gain_ok = plan.net_expected_gain >= self.min_expected_gain
        conf_ok = plan.overall_confidence >= self.min_confidence or plan.total_hit_cost == 0

        lines.append("")
        lines.append(f"  Gain Threshold ({self.min_expected_gain} pts): {'PASS' if gain_ok else 'FAIL'}")
        lines.append(f"  Confidence Threshold ({self.min_confidence:.0%}): {'PASS' if conf_ok else 'FAIL'}")

        # Alerts
        if plan.alerts:
            lines.append("\n[ALERTS]")
            lines.append("-" * 50)
            for alert in plan.alerts:
                lines.append(f"  {alert}")

        lines.append("\n" + "=" * 70)

        if self.config.dry_run:
            lines.append("[INFO] DRY RUN: No changes will be made")
        else:
            if gain_ok and conf_ok:
                lines.append("[READY] Plan meets all thresholds - ready to execute")
            else:
                lines.append("[BLOCKED] Plan does not meet thresholds")

        lines.append("=" * 70)

        return "\n".join(lines)

    def get_history(self) -> list[ExecutionResult]:
        """Get execution history for this session."""
        return self._history.copy()

    def format_history(self) -> str:
        """Format execution history for display."""
        if not self._history:
            return "No execution history"

        lines = [
            "EXECUTION HISTORY",
            "=" * 70,
        ]

        for i, result in enumerate(self._history, 1):
            status_icon = "[OK]" if result.success else "[FAIL]"
            lines.append(f"\n{i}. {status_icon} {result.plan.status.value.upper()}")
            lines.append(f"   Plan ID: {result.plan.id}")
            lines.append(f"   Created: {result.plan.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

            if result.plan.executed_at:
                lines.append(f"   Executed: {result.plan.executed_at.strftime('%Y-%m-%d %H:%M:%S')}")

            lines.append(f"   Transfers: {result.transfers_executed}/{len(result.plan.transfers)}")
            lines.append(f"   Captain: {'Set' if result.captain_set else 'Not set'}")
            lines.append(f"   Rollback: {'Available' if result.rollback_available else 'N/A'}")

            if result.messages:
                lines.append("   Messages:")
                for msg in result.messages[:3]:
                    lines.append(f"     - {msg}")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

async def safe_execute(
    config: Config,
    client: FPLClient,
    optimization_result,
    confirm: bool = False,
    require_approval: bool = False,
) -> ExecutionResult:
    """
    Convenience function for safe transfer execution.

    Args:
        config: Application configuration.
        client: FPL API client.
        optimization_result: Optimization results to execute.
        confirm: Whether to confirm execution.
        require_approval: Whether to use approval workflow.

    Returns:
        ExecutionResult with execution details.
    """
    executor = Executor(config, client)
    return await executor.execute(
        optimization_result,
        confirm=confirm,
        require_approval=require_approval,
    )
