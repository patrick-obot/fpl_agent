"""Tests for configuration management."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


class TestConfig:
    """Test suite for Config class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = Config()

        assert config.dry_run is True
        assert config.log_level == "INFO"
        assert config.api_base_url == "https://fantasy.premierleague.com/api"
        assert config.request_timeout == 30
        assert config.max_retries == 3
        assert config.max_transfers_per_week == 2
        assert config.min_bank_balance == 0.0

    def test_from_env_with_values(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "FPL_EMAIL": "test@example.com",
            "FPL_PASSWORD": "testpass",
            "FPL_TEAM_ID": "12345",
            "DRY_RUN": "false",
            "LOG_LEVEL": "DEBUG",
            "MAX_TRANSFERS_PER_WEEK": "3",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config.from_env()

            assert config.fpl_email == "test@example.com"
            assert config.fpl_password == "testpass"
            assert config.fpl_team_id == 12345
            assert config.dry_run is False
            assert config.log_level == "DEBUG"
            assert config.max_transfers_per_week == 3

    def test_dry_run_parsing(self):
        """Test various dry_run value formats."""
        true_values = ["true", "True", "TRUE", "1", "yes", "on"]
        false_values = ["false", "False", "FALSE", "0", "no", "off"]

        for value in true_values:
            with patch.dict(os.environ, {"DRY_RUN": value}, clear=False):
                config = Config.from_env()
                assert config.dry_run is True, f"Failed for value: {value}"

        for value in false_values:
            with patch.dict(os.environ, {"DRY_RUN": value}, clear=False):
                config = Config.from_env()
                assert config.dry_run is False, f"Failed for value: {value}"

    def test_validate_dry_run_mode(self):
        """Test validation passes in dry-run mode without credentials."""
        config = Config(dry_run=True)
        errors = config.validate()

        assert len(errors) == 0

    def test_validate_live_mode_requires_credentials(self):
        """Test validation fails in live mode without credentials."""
        config = Config(dry_run=False)
        errors = config.validate()

        assert len(errors) == 3
        assert any("FPL_EMAIL" in e for e in errors)
        assert any("FPL_PASSWORD" in e for e in errors)
        assert any("FPL_TEAM_ID" in e for e in errors)

    def test_validate_live_mode_with_credentials(self):
        """Test validation passes in live mode with credentials."""
        config = Config(
            dry_run=False,
            fpl_email="test@example.com",
            fpl_password="password",
            fpl_team_id=12345,
        )
        errors = config.validate()

        assert len(errors) == 0

    def test_directories_created(self):
        """Test that data and logs directories are created."""
        config = Config()

        assert config.data_dir.exists()
        assert config.logs_dir.exists()

    def test_invalid_team_id_defaults_to_zero(self):
        """Test that invalid team ID defaults to 0."""
        with patch.dict(os.environ, {"FPL_TEAM_ID": "invalid"}, clear=False):
            config = Config.from_env()
            assert config.fpl_team_id == 0
