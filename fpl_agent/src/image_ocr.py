"""
OCR module for extracting FPL data from @robtFPL tweet images.

Extracts:
- Team projected goals and clean sheet odds
- Player projected FPL points

Supports:
- easyocr (preferred, pure Python)
- pytesseract (requires Tesseract installed)
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Try easyocr first (preferred)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Fallback to pytesseract
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

OCR_AVAILABLE = EASYOCR_AVAILABLE or TESSERACT_AVAILABLE

logger = logging.getLogger("fpl_agent.ocr")

# Lazy-loaded easyocr reader
_easyocr_reader = None


@dataclass
class TeamOdds:
    """Projected goals and clean sheet odds for a team."""
    team_code: str  # 3-letter code (e.g., ARS, CHE)
    projected_goals: float
    clean_sheet_pct: float
    gameweek: Optional[int] = None


@dataclass
class PlayerProjection:
    """Projected FPL points for a player."""
    rank: int
    player_name: str
    team_code: str
    projected_points: float
    gameweek: Optional[int] = None


@dataclass
class OCRResult:
    """Combined OCR extraction result."""
    gameweek: Optional[int] = None
    team_odds: list[TeamOdds] = field(default_factory=list)
    player_projections: list[PlayerProjection] = field(default_factory=list)
    raw_text: str = ""
    success: bool = False
    error: Optional[str] = None


# Premier League team codes
TEAM_CODES = {
    'ARS': 'Arsenal', 'AVL': 'Aston Villa', 'BOU': 'Bournemouth',
    'BRE': 'Brentford', 'BHA': 'Brighton', 'CHE': 'Chelsea',
    'CRY': 'Crystal Palace', 'EVE': 'Everton', 'FUL': 'Fulham',
    'IPS': 'Ipswich', 'LEI': 'Leicester', 'LIV': 'Liverpool',
    'MCI': 'Man City', 'MUN': 'Man Utd', 'NEW': 'Newcastle',
    'NFO': "Nott'm Forest", 'SOU': 'Southampton', 'TOT': 'Spurs',
    'WHU': 'West Ham', 'WOL': 'Wolves', 'LEE': 'Leeds',
    'BUR': 'Burnley'
}


def _get_easyocr_reader():
    """Get or create easyocr reader (lazy loaded)."""
    global _easyocr_reader
    if _easyocr_reader is None and EASYOCR_AVAILABLE:
        import io
        import sys

        logger.info("Initializing easyocr reader (first run may download models)...")

        # Suppress output during initialization
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return _easyocr_reader


def check_ocr() -> tuple[bool, str]:
    """Check if OCR is available and return engine name."""
    if EASYOCR_AVAILABLE:
        return True, "easyocr"
    if TESSERACT_AVAILABLE:
        try:
            pytesseract.get_tesseract_version()
            return True, "tesseract"
        except Exception:
            pass
    return False, "none"


def check_tesseract() -> bool:
    """Check if Tesseract OCR is available (legacy)."""
    available, engine = check_ocr()
    return available


def extract_gameweek(text: str) -> Optional[int]:
    """Extract gameweek number from text."""
    # Match patterns like "GW26", "DGW26", "GW 26"
    match = re.search(r'(?:D)?GW\s*(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_team_odds(text: str) -> list[TeamOdds]:
    """
    Extract team projected goals and CS odds from OCR text.

    Looks for patterns like:
    - "ARS 2.13 52%"
    - "CHE 2.07 36%"
    """
    results = []
    seen_teams = {}  # Track best match per team

    # Pattern: team code followed by decimal (goals) and percentage (CS)
    # Allow for various separators and noise
    pattern = r'\b([A-Z]{3})\b[^\d]*(\d+\.?\d*)[^\d]*(\d+)\s*%'

    for match in re.finditer(pattern, text):
        team_code = match.group(1)
        if team_code in TEAM_CODES:
            try:
                goals = float(match.group(2))
                cs_pct = float(match.group(3))

                # Sanity checks
                if 0.5 <= goals <= 4.0 and 0 <= cs_pct <= 100:
                    # Keep the match with higher goals (likely the actual projection)
                    if team_code not in seen_teams or goals > seen_teams[team_code].projected_goals:
                        seen_teams[team_code] = TeamOdds(
                            team_code=team_code,
                            projected_goals=goals,
                            clean_sheet_pct=cs_pct
                        )
            except ValueError:
                continue

    # Alternative pattern for different formatting
    if not seen_teams:
        # Try line-by-line parsing
        lines = text.split('\n')
        for line in lines:
            for code in TEAM_CODES:
                if code in line.upper() and code not in seen_teams:
                    # Find numbers in the line
                    numbers = re.findall(r'(\d+\.?\d*)', line)
                    if len(numbers) >= 2:
                        try:
                            goals = float(numbers[0])
                            cs_pct = float(numbers[1])
                            if 0.5 <= goals <= 4.0 and 0 <= cs_pct <= 100:
                                seen_teams[code] = TeamOdds(
                                    team_code=code,
                                    projected_goals=goals,
                                    clean_sheet_pct=cs_pct
                                )
                                break
                        except ValueError:
                            continue

    results = list(seen_teams.values())
    return results


def extract_player_projections(text: str) -> list[PlayerProjection]:
    """
    Extract player projected points from OCR text.

    Looks for patterns like:
    - "Gabriel 10.2"
    - "Rice 9.7"
    - "J.Timber 9.1"
    """
    results = []
    seen_names = set()

    # Known FPL player names to look for (more comprehensive)
    known_players = [
        # Attackers
        'Haaland', 'Isak', 'Watkins', 'Solanke', 'Cunha', 'Wood', 'Vardy',
        'Jackson', 'Nkunku', 'Wissa', 'Mateta', 'Jimenez', 'Darwin', 'Richarlison',
        'Gyokeres', 'Mbeumo', 'Raul', 'Joao Pedro', 'Delap', 'Muniz',
        # Midfielders
        'Salah', 'Palmer', 'Saka', 'Son', 'Gordon', 'Eze', 'Diaz', 'Jota',
        'Fernandes', 'B.Fernandes', 'Bruno', 'Kulusevski', 'Maddison', 'Foden',
        'Rice', 'Trossard', 'Havertz', 'Martinelli', 'Sterling', 'Bowen',
        'McNeil', 'Elanga', 'Gakpo', 'Rogers', 'Zubimendi', 'Kudus', 'Luis Diaz',
        'Madueke', 'Morgan Gibbs', 'Gibbs-White', 'Murphy', 'Barnes',
        # Defenders
        'Gabriel', 'Saliba', 'Timber', 'J.Timber', 'Calafiori', 'Van Dijk',
        'Gvardiol', 'Dias', 'Stones', 'Walker', 'Schar', 'Trippier', 'Hall',
        'Porro', 'Udogie', 'Estupinan', 'Munoz', 'Lewis', 'Ait-Nouri',
        'Konsa', 'Cash', 'Digne', 'Colwill', 'Cucurella', 'James',
        'Mykolenko', 'Branthwaite', 'Tarkowski', 'Robinson', 'Burn',
        # Goalkeepers
        'Raya', 'Martinez', 'Pickford', 'Henderson', 'Sanchez', 'Vicario',
        'Ramsdale', 'Areola', 'Flekken', 'Pope', 'Sa', 'Leno',
        # More players
        'Angel', 'Gomes', 'J.Gomes', 'Onana', 'Garnacho', 'Rashford',
    ]

    # First pass: look for known player names followed by points
    for player in known_players:
        # Escape dots in names
        escaped_name = player.replace('.', r'\.?')
        # Pattern: name followed by decimal points (4.0-15.0 typical range)
        pattern = rf'\b{escaped_name}\b[^\d]*(\d{{1,2}}\.\d)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                points = float(match.group(1))
                if 3.0 <= points <= 15.0 and player.lower() not in seen_names:
                    results.append(PlayerProjection(
                        rank=0,
                        player_name=player,
                        team_code="",
                        projected_points=points
                    ))
                    seen_names.add(player.lower())
            except ValueError:
                continue

    # Noise words to filter (OCR artifacts, common words)
    noise_words = {
        'the', 'and', 'for', 'from', 'with', 'fpl', 'pts', 'gw', 'dgw',
        'projected', 'solio', 'picks', 'team', 'live', 'market', 'odds',
        'goals', 'update', 'final', 'rob', 'felnl', 'ella', 'meftakd',
        'bffaoa', 'cawtu', 'tfcmro', 'elef', 'pruou', 'tocttnt', 'tnj',
        'oman', 'complete', 'progress', 'calculated', 'pinnacle', 'example',
    }

    # Second pass: only add known-player-like names
    # Skip the generic pattern since it picks up too much noise

    # Sort by points descending and assign ranks
    results.sort(key=lambda x: x.projected_points, reverse=True)
    for i, proj in enumerate(results, 1):
        proj.rank = i

    return results


def _run_ocr(image_path: Path) -> str:
    """Run OCR on image and return raw text."""
    if EASYOCR_AVAILABLE:
        import io
        import sys

        # Suppress easyocr's progress bar (causes encoding issues on Windows)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            reader = _get_easyocr_reader()
            results = reader.readtext(str(image_path), detail=1, paragraph=False)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Combine all detected text
        return '\n'.join([text for _, text, _ in results])
    elif TESSERACT_AVAILABLE:
        from PIL import Image
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    else:
        raise RuntimeError("No OCR engine available")


def ocr_image(image_path: str | Path) -> OCRResult:
    """
    Extract FPL data from an image using OCR.

    Args:
        image_path: Path to the image file.

    Returns:
        OCRResult with extracted data.
    """
    available, engine = check_ocr()
    if not available:
        return OCRResult(
            success=False,
            error="No OCR engine available. Install easyocr: pip install easyocr"
        )

    try:
        image_path = Path(image_path)
        if not image_path.exists():
            return OCRResult(success=False, error=f"Image not found: {image_path}")

        logger.info(f"Running OCR with {engine} on {image_path.name}...")

        # Run OCR
        raw_text = _run_ocr(image_path)
        logger.debug(f"OCR raw text:\n{raw_text}")

        # Extract gameweek
        gameweek = extract_gameweek(raw_text)

        # Extract team odds
        team_odds = extract_team_odds(raw_text)

        # Extract player projections
        player_projections = extract_player_projections(raw_text)

        # Set gameweek on all results
        for odds in team_odds:
            odds.gameweek = gameweek
        for proj in player_projections:
            proj.gameweek = gameweek

        result = OCRResult(
            gameweek=gameweek,
            team_odds=team_odds,
            player_projections=player_projections,
            raw_text=raw_text,
            success=True
        )

        logger.info(
            f"OCR extracted: GW{gameweek}, "
            f"{len(team_odds)} teams, {len(player_projections)} players"
        )

        return result

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return OCRResult(success=False, error=str(e))


def ocr_multiple_images(image_paths: list[str | Path]) -> OCRResult:
    """
    Extract and combine FPL data from multiple images.

    Args:
        image_paths: List of image file paths.

    Returns:
        Combined OCRResult.
    """
    combined = OCRResult(success=True)

    for path in image_paths:
        result = ocr_image(path)
        if result.success:
            if result.gameweek and not combined.gameweek:
                combined.gameweek = result.gameweek
            combined.team_odds.extend(result.team_odds)
            combined.player_projections.extend(result.player_projections)
            combined.raw_text += f"\n---\n{result.raw_text}"
        else:
            logger.warning(f"Failed to OCR {path}: {result.error}")

    # Deduplicate by team code / player name
    seen_teams = set()
    unique_teams = []
    for odds in combined.team_odds:
        if odds.team_code not in seen_teams:
            seen_teams.add(odds.team_code)
            unique_teams.append(odds)
    combined.team_odds = unique_teams

    seen_players = set()
    unique_players = []
    for proj in combined.player_projections:
        if proj.player_name not in seen_players:
            seen_players.add(proj.player_name)
            unique_players.append(proj)
    combined.player_projections = unique_players

    return combined
