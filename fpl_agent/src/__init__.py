"""
FPL Agent - Autonomous Fantasy Premier League Manager

This package provides tools for automated FPL team management including:
- API interactions with the FPL platform
- Data collection and analysis
- Transfer and captain optimization
- Safe execution of transfers
"""

__version__ = "1.0.0"

from .config import Config
from .fpl_client import FPLClient
from .data_collector import DataCollector
from .optimizer import Optimizer
from .executor import Executor

__all__ = [
    "Config",
    "FPLClient",
    "DataCollector",
    "Optimizer",
    "Executor",
]
