"""
FPL Review Client — STUB.

CSV download has been migrated to the Raspberry Pi.
See pi/download_and_upload.py for the active implementation.
The Pi downloads the CSV and SCPs it to the VPS data/ directory.
"""

from pathlib import Path
from typing import Optional


class FPLReviewClient:
    """Stub — download migrated to Raspberry Pi."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FPL Review CSV download has been migrated to the Raspberry Pi. "
            "See pi/download_and_upload.py"
        )


async def download_fplreview_projections(
    email: str,
    password: str,
    download_dir: Path,
    headless: bool = True,
    logger=None,
    team_id: str = "",
) -> Optional[Path]:
    """Stub — download migrated to Raspberry Pi."""
    raise NotImplementedError(
        "FPL Review CSV download has been migrated to the Raspberry Pi. "
        "See pi/download_and_upload.py"
    )
