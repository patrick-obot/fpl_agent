#!/usr/bin/env python3
"""
FPL Agent - Autonomous Fantasy Premier League Manager

Main entry point for the FPL Agent application.
Orchestrates data collection, optimization, and transfer execution.

Usage:
    python main.py                  # Run in dry-run mode (default)
    python main.py --live           # Run in live mode (requires confirmation)
    python main.py --live --confirm # Run in live mode with auto-confirmation
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.fpl_client import FPLClient, FPLAPIError, AuthenticationError
from src.data_collector import DataCollector
from src.optimizer import Optimizer
from src.executor import Executor


async def run_agent(config: Config, confirm: bool = False) -> int:
    """
    Run the FPL agent workflow.

    Args:
        config: Application configuration.
        confirm: Whether to auto-confirm execution.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger = config.logger

    logger.info("=" * 60)
    logger.info("FPL AGENT STARTING")
    logger.info("=" * 60)
    config.log_config()

    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return 1

    try:
        async with FPLClient(config) as client:
            # Authenticate if not in dry-run mode
            if not config.dry_run:
                logger.info("Authenticating with FPL...")
                try:
                    await client.authenticate()
                except AuthenticationError as e:
                    logger.error(f"Authentication failed: {e}")
                    return 1

            # Initialize components
            collector = DataCollector(config, client)
            optimizer = Optimizer(config, client, collector)
            executor = Executor(config, client)

            # Step 1: Collect data
            logger.info("\n" + "=" * 60)
            logger.info("STEP 1: DATA COLLECTION")
            logger.info("=" * 60)

            await collector.collect_all(gameweeks_ahead=5)

            # Step 2: Run optimization
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: OPTIMIZATION")
            logger.info("=" * 60)

            optimization_result = await optimizer.optimize()

            # Display recommendations
            recommendations = optimizer.format_recommendations(optimization_result)
            print(recommendations)
            logger.info(recommendations)

            # Step 3: Preview execution
            logger.info("\n" + "=" * 60)
            logger.info("STEP 3: EXECUTION PREVIEW")
            logger.info("=" * 60)

            preview = executor.preview(optimization_result)
            print(preview)
            logger.info(preview)

            # Step 4: Execute (if confirmed or dry-run)
            if optimization_result.transfers or optimization_result.captain:
                logger.info("\n" + "=" * 60)
                logger.info("STEP 4: EXECUTION")
                logger.info("=" * 60)

                if config.dry_run:
                    logger.info("Running in DRY RUN mode - no changes will be made")
                    result = await executor.execute(optimization_result)
                elif confirm:
                    logger.info("Auto-confirmation enabled - executing transfers")
                    result = await executor.execute(optimization_result, confirm=True)
                else:
                    logger.info("Live mode without confirmation - skipping execution")
                    logger.info("Use --confirm flag to execute transfers")
                    result = await executor.execute(optimization_result, confirm=False)

                # Log result
                for message in result.messages:
                    logger.info(f"  {message}")

                if result.success:
                    logger.info("Execution completed successfully!")
                else:
                    logger.warning("Execution completed with issues")

            else:
                logger.info("No actions to execute")

            # Export analysis data
            export_path = collector.export_analysis()
            logger.info(f"Analysis exported to: {export_path}")

            logger.info("\n" + "=" * 60)
            logger.info("FPL AGENT COMPLETED")
            logger.info("=" * 60)

            return 0

    except FPLAPIError as e:
        logger.error(f"FPL API error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FPL Agent - Autonomous Fantasy Premier League Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run in dry-run mode (safe, no changes)
  python main.py --live             Run in live mode (shows preview only)
  python main.py --live --confirm   Run in live mode and execute transfers
  python main.py --gameweeks 3      Analyze only 3 gameweeks ahead
        """
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (default is dry-run)"
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Auto-confirm execution in live mode"
    )

    parser.add_argument(
        "--gameweeks",
        type=int,
        default=5,
        help="Number of gameweeks to analyze (default: 5)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level from .env"
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to custom .env file"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = Config.from_env(env_file=args.env_file)

    # Override settings from command line
    if args.live:
        config.dry_run = False

    if args.log_level:
        config.log_level = args.log_level
        # Reinitialize logger with new level
        config.logger.setLevel(args.log_level)

    # Safety check for live mode
    if not config.dry_run and not args.confirm:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: LIVE MODE ENABLED")
        print("=" * 60)
        print("This will make real changes to your FPL team!")
        print("Add --confirm to actually execute transfers.")
        print("=" * 60 + "\n")

    # Run the agent
    return asyncio.run(run_agent(config, confirm=args.confirm))


if __name__ == "__main__":
    sys.exit(main())
