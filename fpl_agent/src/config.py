"""
Configuration management for the FPL Agent.

Handles loading environment variables and providing typed configuration access.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv


def setup_logging(log_level: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging for the application."""
    logger = logging.getLogger("fpl_agent")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplicates on repeated calls
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # FPL Credentials
    fpl_email: str = ""
    fpl_password: str = ""
    fpl_team_id: int = 0

    # Agent Settings
    dry_run: bool = True
    log_level: str = "INFO"

    # API Settings
    api_base_url: str = "https://fantasy.premierleague.com/api"
    request_timeout: int = 30
    max_retries: int = 3

    # Optimization Settings
    max_transfers_per_week: int = 3
    min_bank_balance: float = 0.0
    free_transfers_override: int = 0  # Set > 0 to override API value (for dry-run mode)

    # Notification Settings
    notification_email: str = ""
    webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    # Logger
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize derived paths and logger."""
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = self.logs_dir / "agent.log"
        self.logger = setup_logging(self.log_level, log_file)

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Config":
        """
        Load configuration from environment variables.

        Args:
            env_file: Optional path to .env file. If None, looks in project root.

        Returns:
            Config instance with loaded values.

        Raises:
            ValueError: If required configuration is missing.
        """
        # Determine project root
        project_root = Path(__file__).parent.parent

        # Load .env file
        if env_file is None:
            env_file = project_root / ".env"

        if env_file.exists():
            load_dotenv(env_file)

        # Parse boolean for dry_run
        dry_run_str = os.getenv("DRY_RUN", "true").lower()
        dry_run = dry_run_str in ("true", "1", "yes", "on")

        # Parse team ID
        team_id_str = os.getenv("FPL_TEAM_ID", "0")
        try:
            team_id = int(team_id_str) if team_id_str.isdigit() else 0
        except ValueError:
            team_id = 0

        config = cls(
            fpl_email=os.getenv("FPL_EMAIL", ""),
            fpl_password=os.getenv("FPL_PASSWORD", ""),
            fpl_team_id=team_id,
            dry_run=dry_run,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            api_base_url=os.getenv("API_BASE_URL", "https://fantasy.premierleague.com/api"),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            max_transfers_per_week=int(os.getenv("MAX_TRANSFERS_PER_WEEK", "3")),
            min_bank_balance=float(os.getenv("MIN_BANK_BALANCE", "0.0")),
            free_transfers_override=int(os.getenv("FREE_TRANSFERS_OVERRIDE", "0")),
            notification_email=os.getenv("NOTIFICATION_EMAIL", ""),
            webhook_url=os.getenv("WEBHOOK_URL", ""),
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            project_root=project_root,
        )

        return config

    def validate(self) -> list[str]:
        """
        Validate configuration for required fields.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        if not self.dry_run:
            if not self.fpl_email:
                errors.append("FPL_EMAIL is required when not in dry-run mode")
            if not self.fpl_password:
                errors.append("FPL_PASSWORD is required when not in dry-run mode")
            if not self.fpl_team_id:
                errors.append("FPL_TEAM_ID is required when not in dry-run mode")

        return errors

    def log_config(self) -> None:
        """Log current configuration (excluding sensitive data)."""
        self.logger.info("Configuration loaded:")
        self.logger.info(f"  Dry Run: {self.dry_run}")
        self.logger.info(f"  Team ID: {self.fpl_team_id}")
        self.logger.info(f"  API Base URL: {self.api_base_url}")
        self.logger.info(f"  Max Transfers/Week: {self.max_transfers_per_week}")
        self.logger.info(f"  Min Bank Balance: {self.min_bank_balance}m")
        self.logger.info(f"  Data Directory: {self.data_dir}")
        self.logger.info(f"  Logs Directory: {self.logs_dir}")
