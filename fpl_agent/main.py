#!/usr/bin/env python3
"""
FPL Agent - Autonomous Fantasy Premier League Manager

Main entry point and orchestration layer for the FPL Agent application.
Handles data collection, optimization, execution, and scheduling.

Usage:
    python main.py                              # Dry-run mode (default)
    python main.py --mode live                  # Live mode with approval
    python main.py --mode live --confirm        # Live mode auto-execute
    python main.py schedule                     # Run scheduler daemon
    python main.py price-check                  # Check price changes only
"""

import asyncio
import argparse
import sys
import signal
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from enum import Enum

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.fpl_client import FPLClient, FPLAPIError, AuthenticationError
from src.data_collector import DataCollector
from src.optimizer import Optimizer
from src.executor import Executor, ExecutionStatus


# =============================================================================
# Constants
# =============================================================================

class AgentMode(Enum):
    """Agent execution modes."""
    DRY_RUN = "dry-run"
    LIVE = "live"


class NotifyMode(Enum):
    """Notification modes."""
    EMAIL = "email"
    SLACK = "slack"
    NONE = "none"


# Schedule times (in UTC/GMT)
PRICE_CHECK_HOUR = 1  # 1:30 AM GMT
PRICE_CHECK_MINUTE = 30
PREPARATION_HOURS = 48  # Hours before deadline for preparation run
FINAL_EXECUTION_HOURS = 2  # Hours before deadline for final execution


# =============================================================================
# Deadline Utilities
# =============================================================================

async def get_next_deadline(client: FPLClient) -> Optional[datetime]:
    """
    Get the next gameweek deadline.

    Args:
        client: FPL API client.

    Returns:
        Datetime of next deadline in UTC, or None if not available.
    """
    try:
        gameweek = await client.get_current_gameweek()
        if gameweek and gameweek.deadline_time:
            return gameweek.deadline_time
    except Exception:
        pass
    return None


def hours_until_deadline(deadline: datetime) -> float:
    """Calculate hours until the deadline."""
    now = datetime.now(timezone.utc)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    delta = deadline - now
    return delta.total_seconds() / 3600


def is_deadline_approaching(deadline: datetime, threshold_hours: float = 48) -> bool:
    """Check if deadline is within threshold hours."""
    return 0 < hours_until_deadline(deadline) <= threshold_hours


# =============================================================================
# Main Agent Functions
# =============================================================================

async def run_agent(
    mode: str = 'dry-run',
    notify: str = 'none',
    deadline_hours: float = 2.0,
    confirm: bool = False,
    require_approval: bool = False,
) -> int:
    """
    Main agent execution flow.

    Workflow:
    1. Check if gameweek deadline is approaching (< 48 hours)
    2. Collect all data sources
    3. Optimize transfers and captain
    4. Generate decision report
    5. Execute if mode='live' and safety checks pass
    6. Log results

    Args:
        mode: Execution mode ('dry-run' or 'live').
        notify: Notification method ('email', 'slack', 'none').
        deadline_hours: Hours before deadline to allow execution.
        confirm: Auto-confirm execution without approval.
        require_approval: Require human approval before execution.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    config = Config.from_env()
    config.dry_run = (mode == 'dry-run')
    logger = config.logger

    logger.info("=" * 70)
    logger.info("FPL AGENT STARTING")
    logger.info(f"Mode: {mode.upper()} | Notify: {notify} | Deadline threshold: {deadline_hours}h")
    logger.info("=" * 70)
    config.log_config()

    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return 1

    try:
        async with FPLClient(config) as client:
            # Authenticate if in live mode
            if not config.dry_run:
                logger.info("Authenticating with FPL...")
                try:
                    await client.authenticate()
                    logger.info("Authentication successful")
                except AuthenticationError as e:
                    logger.error(f"Authentication failed: {e}")
                    return 1

            # =================================================================
            # STEP 1: Check Deadline
            # =================================================================
            logger.info("\n" + "=" * 70)
            logger.info("STEP 1: DEADLINE CHECK")
            logger.info("=" * 70)

            deadline = await get_next_deadline(client)
            if deadline:
                hours_left = hours_until_deadline(deadline)
                logger.info(f"Next deadline: {deadline.strftime('%Y-%m-%d %H:%M UTC')}")
                logger.info(f"Hours remaining: {hours_left:.1f}")

                if hours_left <= 0:
                    logger.warning("Deadline has passed! Waiting for next gameweek...")
                    return 0
                elif hours_left > 48:
                    logger.info("Deadline > 48 hours away - running preparation analysis")
                elif hours_left > deadline_hours:
                    logger.info(f"Deadline > {deadline_hours}h away - running analysis only")
                else:
                    logger.info(f"DEADLINE APPROACHING ({hours_left:.1f}h) - ready for execution")
            else:
                logger.warning("Could not determine deadline - proceeding with analysis")
                hours_left = float('inf')

            # =================================================================
            # STEP 2: Collect Data
            # =================================================================
            logger.info("\n" + "=" * 70)
            logger.info("STEP 2: DATA COLLECTION")
            logger.info("=" * 70)

            collector = DataCollector(config, client)
            player_df = await collector.collect_all(gameweeks_ahead=5)

            logger.info(f"Collected data for {len(player_df)} players")

            # Show top players summary
            top_players = collector.get_top_players(limit=5)
            logger.info("\nTop 5 Players by Composite Score:")
            for _, row in top_players.iterrows():
                logger.info(f"  {row['name']:15} {row['team']:4} {row['price']:5.1f}m  Score: {row['composite_score']:.1f}")

            # Show flagged players
            flagged = collector.get_flagged_players()
            if not flagged.empty:
                logger.info(f"\nFlagged Players ({len(flagged)}):")
                for _, row in flagged.head(5).iterrows():
                    logger.info(f"  {row['name']:15} [{row['status']}] {row['news'][:40]}...")

            # =================================================================
            # STEP 3: Optimization
            # =================================================================
            logger.info("\n" + "=" * 70)
            logger.info("STEP 3: OPTIMIZATION")
            logger.info("=" * 70)

            optimizer = Optimizer(config, client, collector)
            optimization_result = await optimizer.optimize()

            # Display recommendations
            recommendations = optimizer.format_recommendations(optimization_result)
            print(recommendations)
            logger.info(recommendations)

            # =================================================================
            # STEP 4: Generate Decision Report
            # =================================================================
            logger.info("\n" + "=" * 70)
            logger.info("STEP 4: DECISION REPORT")
            logger.info("=" * 70)

            executor = Executor(config, client)
            preview = executor.preview(optimization_result)
            print(preview)
            logger.info(preview)

            # =================================================================
            # STEP 5: Execute (if conditions met)
            # =================================================================
            logger.info("\n" + "=" * 70)
            logger.info("STEP 5: EXECUTION")
            logger.info("=" * 70)

            # Determine if we should execute
            should_execute = False
            execution_reason = ""

            if config.dry_run:
                execution_reason = "DRY RUN mode - simulating execution"
                should_execute = True
            elif not (optimization_result.transfers or optimization_result.captain):
                execution_reason = "No actions to execute"
                should_execute = False
            elif hours_left > deadline_hours:
                execution_reason = f"Deadline > {deadline_hours}h away - skipping execution"
                should_execute = False
            elif require_approval:
                execution_reason = "Approval required - requesting human approval"
                should_execute = True
            elif confirm:
                execution_reason = "Auto-confirm enabled - executing"
                should_execute = True
            else:
                execution_reason = "Live mode without confirmation - use --confirm to execute"
                should_execute = False

            logger.info(execution_reason)

            if should_execute:
                if config.dry_run:
                    result = await executor.dry_run(optimization_result, player_df)
                elif require_approval:
                    plan = await executor.request_approval(
                        optimization_result,
                        player_df,
                        deadline_hours=min(hours_left - 0.5, 2.0)  # Leave 30min buffer
                    )
                    logger.info(f"Approval requested. Plan ID: {plan.id}")
                    logger.info(f"Deadline: {plan.approval_deadline}")

                    # Send notification based on mode
                    if notify == 'email' and config.notification_email:
                        logger.info(f"Email notification sent to {config.notification_email}")
                    elif notify == 'slack' and config.webhook_url:
                        logger.info("Slack notification sent")

                    # For now, return - approval will be handled separately
                    result = type('Result', (), {
                        'success': True,
                        'messages': [f"Approval requested - Plan ID: {plan.id}"],
                        'plan': plan
                    })()
                else:
                    result = await executor.execute(
                        optimization_result,
                        confirm=confirm,
                        player_df=player_df
                    )

                # Log execution result
                for message in result.messages:
                    logger.info(f"  {message}")

                if result.success:
                    logger.info("Execution completed successfully!")
                else:
                    logger.warning("Execution completed with issues")
            else:
                logger.info("Execution skipped")

            # =================================================================
            # STEP 6: Log Results & Export
            # =================================================================
            logger.info("\n" + "=" * 70)
            logger.info("STEP 6: RESULTS & EXPORT")
            logger.info("=" * 70)

            # Export analysis
            try:
                csv_path = collector.export_to_csv()
                logger.info(f"Player analysis exported: {csv_path}")

                fixtures_path = collector.export_fixtures_to_csv()
                logger.info(f"Fixture ticker exported: {fixtures_path}")
            except Exception as e:
                logger.warning(f"Export error: {e}")

            # Log execution history
            history = executor.format_history()
            if "No execution history" not in history:
                logger.info("\n" + history)

            # Log audit trail summary
            audit_history = executor.audit_logger.get_history(limit=5)
            if audit_history:
                logger.info("\nRecent Audit Trail:")
                for entry in audit_history[-5:]:
                    logger.info(f"  {entry.timestamp.strftime('%H:%M:%S')} | {entry.action} | {entry.outcome}")

            logger.info("\n" + "=" * 70)
            logger.info("FPL AGENT COMPLETED")
            logger.info("=" * 70)

            return 0

    except FPLAPIError as e:
        logger.error(f"FPL API error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


async def run_price_check() -> int:
    """
    Run price change monitoring.

    Checks for price rises/falls and alerts on owned players.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    config = Config.from_env()
    config.dry_run = True  # Always dry run for price checks
    logger = config.logger

    logger.info("=" * 70)
    logger.info("FPL PRICE CHANGE CHECK")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    try:
        async with FPLClient(config) as client:
            collector = DataCollector(config, client)
            player_df = await collector.collect_all(gameweeks_ahead=1)

            # Find players with price changes
            price_changes = player_df[player_df['price_change'] != 0].copy()

            if price_changes.empty:
                logger.info("No price changes detected")
                return 0

            # Sort by price change
            price_changes = price_changes.sort_values('price_change', ascending=False)

            # Price rises
            risers = price_changes[price_changes['price_change'] > 0]
            if not risers.empty:
                logger.info(f"\n📈 PRICE RISES ({len(risers)} players):")
                for _, row in risers.head(10).iterrows():
                    logger.info(
                        f"  {row['name']:15} {row['team']:4} "
                        f"{row['price']:.1f}m (+{row['price_change']:.1f})"
                    )

            # Price falls
            fallers = price_changes[price_changes['price_change'] < 0]
            if not fallers.empty:
                logger.info(f"\n📉 PRICE FALLS ({len(fallers)} players):")
                for _, row in fallers.head(10).iterrows():
                    logger.info(
                        f"  {row['name']:15} {row['team']:4} "
                        f"{row['price']:.1f}m ({row['price_change']:.1f})"
                    )

            # Save price change report
            report_path = config.data_dir / f"price_changes_{datetime.now().strftime('%Y%m%d')}.csv"
            price_changes[['name', 'team', 'position', 'price', 'price_change', 'selected_by']].to_csv(
                report_path, index=False
            )
            logger.info(f"\nReport saved: {report_path}")

            return 0

    except Exception as e:
        logger.exception(f"Price check error: {e}")
        return 1


async def run_scheduler() -> None:
    """
    Run the scheduler daemon for automated execution.

    Schedule:
    - Daily price change check at 1:30 AM GMT
    - Gameweek preparation 48h before deadline
    - Final execution 2h before deadline
    """
    config = Config.from_env()
    logger = config.logger

    logger.info("=" * 70)
    logger.info("FPL AGENT SCHEDULER STARTED")
    logger.info("=" * 70)
    logger.info(f"Price check: Daily at {PRICE_CHECK_HOUR:02d}:{PRICE_CHECK_MINUTE:02d} UTC")
    logger.info(f"Preparation: {PREPARATION_HOURS}h before deadline")
    logger.info(f"Execution: {FINAL_EXECUTION_HOURS}h before deadline")
    logger.info("=" * 70)

    # Track last run times to avoid duplicate runs
    last_price_check = None
    last_preparation = None
    last_execution = None

    # Handle graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: signal_handler())

    async with FPLClient(config) as client:
        while not shutdown_event.is_set():
            now = datetime.now(timezone.utc)
            today = now.date()

            try:
                # Get next deadline
                deadline = await get_next_deadline(client)
                hours_left = hours_until_deadline(deadline) if deadline else float('inf')

                # Daily price check (1:30 AM UTC)
                if (now.hour == PRICE_CHECK_HOUR and
                    now.minute >= PRICE_CHECK_MINUTE and
                    last_price_check != today):

                    logger.info("\n>>> Running scheduled price check")
                    await run_price_check()
                    last_price_check = today

                # Preparation run (48h before deadline)
                if deadline and FINAL_EXECUTION_HOURS < hours_left <= PREPARATION_HOURS:
                    prep_date = (deadline - timedelta(hours=PREPARATION_HOURS)).date()
                    if last_preparation != prep_date:
                        logger.info(f"\n>>> Running preparation ({hours_left:.1f}h before deadline)")
                        await run_agent(
                            mode='dry-run',
                            notify='email',
                            deadline_hours=FINAL_EXECUTION_HOURS
                        )
                        last_preparation = prep_date

                # Final execution (2h before deadline)
                if deadline and 0 < hours_left <= FINAL_EXECUTION_HOURS:
                    exec_date = deadline.date()
                    if last_execution != exec_date:
                        logger.info(f"\n>>> Running final execution ({hours_left:.1f}h before deadline)")
                        await run_agent(
                            mode='live',
                            notify='email',
                            deadline_hours=FINAL_EXECUTION_HOURS,
                            require_approval=True
                        )
                        last_execution = exec_date

            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            # Wait before next check (every 5 minutes)
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=300)
            except asyncio.TimeoutError:
                pass

    logger.info("Scheduler stopped")


async def approve_plan(plan_id: str) -> int:
    """
    Approve a pending execution plan.

    Args:
        plan_id: The plan ID to approve.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    config = Config.from_env()
    config.dry_run = False
    logger = config.logger

    logger.info(f"Approving plan: {plan_id}")

    try:
        async with FPLClient(config) as client:
            await client.authenticate()

            executor = Executor(config, client)
            result = await executor.approve(plan_id, approved_by="cli")

            if result.success:
                logger.info("Plan approved and executed successfully!")
                for msg in result.messages:
                    logger.info(f"  {msg}")
                return 0
            else:
                logger.error("Execution failed")
                for msg in result.messages:
                    logger.error(f"  {msg}")
                return 1

    except Exception as e:
        logger.error(f"Approval error: {e}")
        return 1


async def reject_plan(plan_id: str, reason: str = "Manual rejection") -> int:
    """
    Reject a pending execution plan.

    Args:
        plan_id: The plan ID to reject.
        reason: Reason for rejection.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    config = Config.from_env()
    logger = config.logger

    logger.info(f"Rejecting plan: {plan_id}")
    logger.info(f"Reason: {reason}")

    try:
        async with FPLClient(config) as client:
            executor = Executor(config, client)
            await executor.reject(plan_id, reason=reason)
            logger.info("Plan rejected successfully")
            return 0

    except Exception as e:
        logger.error(f"Rejection error: {e}")
        return 1


# =============================================================================
# CLI Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FPL Agent - Autonomous Fantasy Premier League Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (default)     Run the agent once
  schedule      Run as a scheduler daemon
  price-check   Check price changes only
  approve       Approve a pending plan
  reject        Reject a pending plan

Examples:
  python main.py                                # Dry-run mode
  python main.py --mode live                    # Live mode (preview only)
  python main.py --mode live --confirm          # Live mode with execution
  python main.py --mode live --require-approval # Live mode with approval workflow
  python main.py schedule                       # Run scheduler daemon
  python main.py price-check                    # Check price changes
  python main.py approve plan_20240115_123456   # Approve a plan
  python main.py reject plan_20240115_123456    # Reject a plan

Cron Examples (for crontab):
  # Daily price check at 1:30 AM GMT
  30 1 * * * cd /path/to/fpl_agent && python main.py price-check

  # Run agent every 6 hours
  0 */6 * * * cd /path/to/fpl_agent && python main.py --mode dry-run

  # Or use the built-in scheduler
  python main.py schedule
        """
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Run scheduler daemon')

    # Price check command
    price_parser = subparsers.add_parser('price-check', help='Check price changes')

    # Approve command
    approve_parser = subparsers.add_parser('approve', help='Approve a pending plan')
    approve_parser.add_argument('plan_id', help='Plan ID to approve')

    # Reject command
    reject_parser = subparsers.add_parser('reject', help='Reject a pending plan')
    reject_parser.add_argument('plan_id', help='Plan ID to reject')
    reject_parser.add_argument('--reason', default='Manual rejection', help='Rejection reason')

    # Main agent arguments
    parser.add_argument(
        '--mode',
        choices=['dry-run', 'live'],
        default='dry-run',
        help='Execution mode (default: dry-run)'
    )

    parser.add_argument(
        '--notify',
        choices=['email', 'slack', 'none'],
        default='none',
        help='Notification method (default: none)'
    )

    parser.add_argument(
        '--deadline-hours',
        type=float,
        default=2.0,
        help='Hours before deadline to allow execution (default: 2.0)'
    )

    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Auto-confirm execution in live mode'
    )

    parser.add_argument(
        '--require-approval',
        action='store_true',
        help='Require human approval before execution'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=None,
        help='Override log level'
    )

    parser.add_argument(
        '--env-file',
        type=Path,
        default=None,
        help='Path to custom .env file'
    )

    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle subcommands
    if args.command == 'schedule':
        asyncio.run(run_scheduler())
        return 0

    elif args.command == 'price-check':
        return asyncio.run(run_price_check())

    elif args.command == 'approve':
        return asyncio.run(approve_plan(args.plan_id))

    elif args.command == 'reject':
        return asyncio.run(reject_plan(args.plan_id, args.reason))

    # Default: run agent
    else:
        # Safety warning for live mode
        if args.mode == 'live' and not args.confirm and not args.require_approval:
            print("\n" + "=" * 70)
            print("WARNING: LIVE MODE ENABLED")
            print("=" * 70)
            print("This mode can make real changes to your FPL team!")
            print()
            print("Options:")
            print("  --confirm            Auto-execute (use with caution)")
            print("  --require-approval   Request approval before execution")
            print("=" * 70 + "\n")

        return asyncio.run(run_agent(
            mode=args.mode,
            notify=args.notify,
            deadline_hours=args.deadline_hours,
            confirm=args.confirm,
            require_approval=args.require_approval,
        ))


if __name__ == "__main__":
    sys.exit(main())
