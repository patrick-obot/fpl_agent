"""
Data collection module for fixture difficulty, news scraping, and projected points.

Collects and processes data from various sources to inform transfer decisions.
Includes 6-hour caching to avoid excessive API requests.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import re
import time
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

from .config import Config
from .fpl_client import FPLClient, Player, Team, Fixture, Gameweek


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FixtureDifficulty:
    """Represents fixture difficulty for a team in a gameweek."""
    team_id: int
    team_name: str
    team_short: str
    gameweek: int
    opponent_id: int
    opponent_name: str
    opponent_short: str
    is_home: bool
    difficulty: int  # 1-5 scale from FPL
    kickoff_time: Optional[datetime] = None

    @property
    def adjusted_difficulty(self) -> float:
        """Calculate adjusted difficulty (lower is better). Home advantage reduces by 0.5."""
        adjustment = -0.5 if self.is_home else 0.5
        return self.difficulty + adjustment

    @property
    def fixture_string(self) -> str:
        """Format fixture as string like 'ARS (H)' or 'liv (A)'."""
        venue = "H" if self.is_home else "A"
        return f"{self.opponent_short} ({venue})"

    def __str__(self) -> str:
        return f"GW{self.gameweek}: {self.fixture_string} [{self.difficulty}]"


@dataclass
class TeamFixtures:
    """Collection of fixtures for a team across multiple gameweeks."""
    team_id: int
    team_name: str
    team_short: str
    fixtures: list[FixtureDifficulty] = field(default_factory=list)

    @property
    def average_difficulty(self) -> float:
        """Average fixture difficulty across all fixtures."""
        if not self.fixtures:
            return 3.0
        return sum(f.adjusted_difficulty for f in self.fixtures) / len(self.fixtures)

    @property
    def fixture_difficulty_rating(self) -> str:
        """Rate fixture difficulty as Easy/Medium/Hard."""
        avg = self.average_difficulty
        if avg <= 2.5:
            return "Easy"
        elif avg <= 3.5:
            return "Medium"
        else:
            return "Hard"

    def get_fixture(self, gameweek: int) -> Optional[FixtureDifficulty]:
        """Get fixture for specific gameweek."""
        for f in self.fixtures:
            if f.gameweek == gameweek:
                return f
        return None


@dataclass
class PlayerNews:
    """Represents news/injury information for a player."""
    player_id: int
    player_name: str
    team_id: int
    team_name: str
    news: str
    chance_of_playing: Optional[int]
    status: str
    source: str  # 'fpl', 'reddit', 'external'
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_available(self) -> bool:
        """Check if player is likely to play."""
        if self.chance_of_playing is not None:
            return self.chance_of_playing >= 75
        return self.status == 'a'

    @property
    def availability_score(self) -> float:
        """Return availability as 0-1 score."""
        if self.chance_of_playing is not None:
            return self.chance_of_playing / 100
        status_scores = {'a': 1.0, 'd': 0.5, 'i': 0.0, 's': 0.0, 'u': 0.0, 'n': 0.0}
        return status_scores.get(self.status, 0.5)


@dataclass
class ProjectedPoints:
    """Projected points for a player across gameweeks."""
    player_id: int
    player_name: str
    team: str
    position: str
    gameweek_points: dict[int, float] = field(default_factory=dict)
    gameweek_xmins: dict[int, float] = field(default_factory=dict)

    def get_points(self, gameweek: int) -> float:
        """Get projected points for a gameweek."""
        return self.gameweek_points.get(gameweek, 0.0)

    def get_xmins(self, gameweek: int) -> float:
        """Get expected minutes for a gameweek."""
        return self.gameweek_xmins.get(gameweek, 0.0)

    def get_minutes_weight(self, gameweek: int, min_threshold: int = 60) -> float:
        """Get weight based on expected minutes. Returns 0-1 scale."""
        xmins = self.get_xmins(gameweek)
        if xmins >= min_threshold:
            return 1.0
        elif xmins > 0:
            return xmins / min_threshold  # Proportional weight
        return 0.0

    def total_points(self, gameweeks: list[int]) -> float:
        """Get total projected points across gameweeks."""
        return sum(self.get_points(gw) for gw in gameweeks)


@dataclass
class CacheEntry:
    """Cache entry with timestamp."""
    data: Any
    timestamp: float
    ttl: float  # Time to live in seconds

    @property
    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        return (time.time() - self.timestamp) < self.ttl


# =============================================================================
# Data Collector
# =============================================================================

class DataCollector:
    """
    Collects and processes FPL data from various sources.

    Features:
    - Fixture difficulty from FPL API
    - Injury news from FPL API and Reddit RSS
    - Projected points from CSV
    - 6-hour caching to reduce API calls
    """

    # Cache settings
    CACHE_TTL = 6 * 60 * 60  # 6 hours in seconds
    CACHE_FILE = "data_cache.pkl"

    # Reddit RSS feed for r/FantasyPL
    REDDIT_RSS_URL = "https://www.reddit.com/r/FantasyPL/search.rss?q=flair%3A%22Injury%22+OR+flair%3A%22News%22&restrict_sr=on&sort=new&t=week"

    def __init__(self, config: Config, client: FPLClient):
        """
        Initialize the data collector.

        Args:
            config: Application configuration.
            client: FPL API client.
        """
        self.config = config
        self.client = client
        self.logger = logging.getLogger("fpl_agent.collector")

        # Data storage
        self._team_fixtures: dict[int, TeamFixtures] = {}
        self._player_news: dict[int, PlayerNews] = {}
        self._projected_points: dict[int, ProjectedPoints] = {}
        self._players_df: Optional[pd.DataFrame] = None

        # Cache
        self._cache: dict[str, CacheEntry] = {}
        self._cache_path = config.data_dir / self.CACHE_FILE

        # Load cache from disk
        self._load_cache()

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def _load_cache(self) -> None:
        """Load cache from disk if exists."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, 'rb') as f:
                    self._cache = pickle.load(f)
                    # Clean expired entries
                    self._cache = {k: v for k, v in self._cache.items() if v.is_valid}
                self.logger.debug(f"Loaded {len(self._cache)} cache entries from disk")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            # Clean expired entries before saving
            self._cache = {k: v for k, v in self._cache.items() if v.is_valid}
            with open(self._cache_path, 'wb') as f:
                pickle.dump(self._cache, f)
            self.logger.debug(f"Saved {len(self._cache)} cache entries to disk")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if valid."""
        entry = self._cache.get(key)
        if entry and entry.is_valid:
            self.logger.debug(f"Cache hit: {key}")
            return entry.data
        return None

    def _set_cached(self, key: str, data: Any, ttl: Optional[float] = None) -> None:
        """Set cached data."""
        ttl = ttl or self.CACHE_TTL
        self._cache[key] = CacheEntry(data=data, timestamp=time.time(), ttl=ttl)
        self._save_cache()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache = {}
        if self._cache_path.exists():
            self._cache_path.unlink()
        self.logger.info("Cache cleared")

    # -------------------------------------------------------------------------
    # Main Collection Methods
    # -------------------------------------------------------------------------

    async def collect_all(self, gameweeks_ahead: int = 5) -> pd.DataFrame:
        """
        Collect all data needed for optimization.

        Args:
            gameweeks_ahead: Number of future gameweeks to analyze.

        Returns:
            Unified DataFrame with all player data and analysis.
        """
        self.logger.info("Starting comprehensive data collection...")

        # Get current gameweek
        current_gw = await self.client.get_current_gameweek()
        target_gws = list(range(current_gw.id, current_gw.id + gameweeks_ahead))

        # Run collections (use cached if available)
        await asyncio.gather(
            self.collect_fixture_difficulties(target_gws),
            self.collect_player_news(),
            self.collect_reddit_news(),
            return_exceptions=True
        )

        # Load projected points (sync operation)
        self.load_projected_points(target_gws)

        # Build unified DataFrame
        self._players_df = await self.build_unified_dataframe(target_gws)

        self.logger.info(f"Data collection complete. {len(self._players_df)} players analyzed.")
        return self._players_df

    # -------------------------------------------------------------------------
    # Fixture Difficulty Collection
    # -------------------------------------------------------------------------

    async def collect_fixture_difficulties(self, gameweeks: list[int]) -> dict[int, TeamFixtures]:
        """
        Collect fixture difficulty ratings for all teams.

        Args:
            gameweeks: List of gameweek numbers to analyze.

        Returns:
            Dictionary mapping team_id to TeamFixtures object.
        """
        cache_key = f"fixtures_{min(gameweeks)}_{max(gameweeks)}"
        cached = self._get_cached(cache_key)
        if cached:
            self._team_fixtures = cached
            return self._team_fixtures

        self.logger.info(f"Collecting fixture difficulties for GW{min(gameweeks)}-{max(gameweeks)}")

        # Get teams from API
        teams = await self.client.get_teams()
        team_map = {t.id: t for t in teams}

        # Initialize team fixtures
        self._team_fixtures = {
            t.id: TeamFixtures(
                team_id=t.id,
                team_name=t.name,
                team_short=t.short_name
            )
            for t in teams
        }

        # Get fixtures for each gameweek
        for gw in gameweeks:
            try:
                fixtures = await self.client.get_fixtures(gameweek=gw)

                for fixture in fixtures:
                    home_id = fixture.home_team
                    away_id = fixture.away_team
                    home_team = team_map.get(home_id)
                    away_team = team_map.get(away_id)

                    if not home_team or not away_team:
                        continue

                    # Add home fixture
                    self._team_fixtures[home_id].fixtures.append(
                        FixtureDifficulty(
                            team_id=home_id,
                            team_name=home_team.name,
                            team_short=home_team.short_name,
                            gameweek=gw,
                            opponent_id=away_id,
                            opponent_name=away_team.name,
                            opponent_short=away_team.short_name,
                            is_home=True,
                            difficulty=fixture.home_team_difficulty,
                            kickoff_time=fixture.kickoff_time,
                        )
                    )

                    # Add away fixture
                    self._team_fixtures[away_id].fixtures.append(
                        FixtureDifficulty(
                            team_id=away_id,
                            team_name=away_team.name,
                            team_short=away_team.short_name,
                            gameweek=gw,
                            opponent_id=home_id,
                            opponent_name=home_team.name,
                            opponent_short=home_team.short_name,
                            is_home=False,
                            difficulty=fixture.away_team_difficulty,
                            kickoff_time=fixture.kickoff_time,
                        )
                    )

            except Exception as e:
                self.logger.warning(f"Failed to get fixtures for GW{gw}: {e}")

        # Sort fixtures by gameweek
        for team_fixtures in self._team_fixtures.values():
            team_fixtures.fixtures.sort(key=lambda f: f.gameweek)

        self._set_cached(cache_key, self._team_fixtures)
        self.logger.info(f"Collected fixtures for {len(self._team_fixtures)} teams")

        return self._team_fixtures

    def get_team_fixtures(self, team_id: int) -> Optional[TeamFixtures]:
        """Get fixtures for a specific team."""
        return self._team_fixtures.get(team_id)

    def get_fixture_difficulty_score(self, team_id: int, gameweeks: Optional[list[int]] = None) -> float:
        """
        Calculate weighted fixture difficulty score.

        Lower scores indicate easier fixtures.

        Args:
            team_id: Team ID to evaluate.
            gameweeks: Specific gameweeks to consider (defaults to all).

        Returns:
            Weighted fixture difficulty score.
        """
        team_fixtures = self._team_fixtures.get(team_id)
        if not team_fixtures or not team_fixtures.fixtures:
            return 3.0  # Default medium difficulty

        fixtures = team_fixtures.fixtures
        if gameweeks:
            fixtures = [f for f in fixtures if f.gameweek in gameweeks]

        if not fixtures:
            return 3.0

        # Weight recent gameweeks more heavily (exponential decay)
        total_weight = 0.0
        weighted_sum = 0.0

        for i, fixture in enumerate(fixtures):
            weight = 0.8 ** i  # Decay factor
            weighted_sum += fixture.adjusted_difficulty * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 3.0

    # -------------------------------------------------------------------------
    # Player News Collection (FPL API)
    # -------------------------------------------------------------------------

    async def collect_player_news(self) -> dict[int, PlayerNews]:
        """
        Collect injury and availability news from FPL API.

        Returns:
            Dictionary mapping player_id to PlayerNews objects.
        """
        cache_key = "player_news_fpl"
        cached = self._get_cached(cache_key)
        if cached:
            self._player_news = cached
            return self._player_news

        self.logger.info("Collecting player news from FPL API")

        # Get bootstrap data
        bootstrap = await self.client.get_bootstrap_static()
        teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}

        for element in bootstrap["elements"]:
            news = element.get("news", "")
            chance = element.get("chance_of_playing_next_round")
            status = element.get("status", "a")

            # Only track players with news or availability issues
            if news or status != "a" or chance is not None:
                self._player_news[element["id"]] = PlayerNews(
                    player_id=element["id"],
                    player_name=element.get("web_name", "Unknown"),
                    team_id=element.get("team", 0),
                    team_name=teams.get(element.get("team", 0), "Unknown"),
                    news=news,
                    chance_of_playing=chance,
                    status=status,
                    source="fpl",
                )

        self._set_cached(cache_key, self._player_news)
        flagged = len([p for p in self._player_news.values() if not p.is_available])
        self.logger.info(f"Found {flagged} players with availability concerns")

        return self._player_news

    # -------------------------------------------------------------------------
    # Reddit News Collection (RSS)
    # -------------------------------------------------------------------------

    async def collect_reddit_news(self) -> list[dict]:
        """
        Fetch injury/news from r/FantasyPL RSS feed.

        Returns:
            List of news items from Reddit.
        """
        if not FEEDPARSER_AVAILABLE:
            self.logger.warning("feedparser not installed. Skipping Reddit news.")
            return []

        cache_key = "reddit_news"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self.logger.info("Fetching news from r/FantasyPL RSS feed")

        news_items = []

        try:
            # Fetch RSS feed asynchronously
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.REDDIT_RSS_URL,
                    headers={"User-Agent": "FPL-Agent/1.0"},
                    timeout=30
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"Reddit RSS returned status {response.status}")
                        return []

                    content = await response.text()

            # Parse RSS feed
            feed = feedparser.parse(content)

            for entry in feed.entries[:20]:  # Limit to 20 most recent
                # Extract relevant info
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                link = entry.get("link", "")
                published = entry.get("published", "")

                # Try to extract player names from title
                player_mentions = self._extract_player_mentions(title + " " + summary)

                news_items.append({
                    "title": title,
                    "summary": self._clean_html(summary),
                    "link": link,
                    "published": published,
                    "player_mentions": player_mentions,
                    "source": "reddit",
                })

            self._set_cached(cache_key, news_items, ttl=3600)  # 1 hour cache for Reddit
            self.logger.info(f"Fetched {len(news_items)} news items from Reddit")

        except Exception as e:
            self.logger.warning(f"Failed to fetch Reddit news: {e}")

        return news_items

    def _extract_player_mentions(self, text: str) -> list[str]:
        """Extract potential player names from text."""
        # Common FPL player name patterns
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b(?:\s+(?:injury|injured|out|doubtful|fit|available|benched|starts|starting))',
            r'(?:injury|update|news).*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]

        mentions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentions.extend(matches)

        return list(set(mentions))

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:500]

    # -------------------------------------------------------------------------
    # Projected Points Loading
    # -------------------------------------------------------------------------

    def load_projected_points(self, gameweeks: list[int], min_xmins: int = 60) -> dict[int, ProjectedPoints]:
        """
        Load projected points from CSV file.

        Supports format with columns:
        - ID: Player ID
        - Name: Player name
        - Team: Team name
        - Pos: Position (G/D/M/F)
        - {gw}_xMins: Expected minutes for gameweek
        - {gw}_Pts: Expected points for gameweek

        Players with xMins < min_xmins are filtered out for that gameweek.

        Args:
            gameweeks: List of gameweeks to load projections for.
            min_xmins: Minimum expected minutes to consider player (default 60).

        Returns:
            Dictionary mapping player_id to ProjectedPoints.
        """
        # Try multiple possible filenames
        possible_files = [
            self.config.data_dir / "projected_points.csv",
            self.config.data_dir / "projected_points_14012026.csv.csv",
        ]

        csv_path = None
        for path in possible_files:
            if path.exists():
                csv_path = path
                break

        if not csv_path:
            self.logger.warning(f"Projected points file not found in {self.config.data_dir}")
            return self._projected_points

        # Check CSV freshness (file is SCP'd from Raspberry Pi)
        file_age_hours = (time.time() - csv_path.stat().st_mtime) / 3600
        if file_age_hours > 24:
            self.logger.warning(
                f"projected_points.csv is {file_age_hours:.0f}h old — "
                "check if Raspberry Pi download cron is working"
            )
        elif file_age_hours > 12:
            self.logger.info(f"projected_points.csv is {file_age_hours:.0f}h old")

        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')  # Handle BOM
            self.logger.info(f"Loading projected points from {csv_path}")

            columns = df.columns.tolist()

            for _, row in df.iterrows():
                # Get player ID - try both 'ID' and 'player_id'
                player_id = int(row.get("ID", row.get("player_id", 0)))
                if player_id == 0:
                    continue

                gw_points = {}
                gw_xmins = {}

                # Parse gameweek columns (format: 22_Pts, 22_xMins, etc.)
                for gw in gameweeks:
                    pts_col = f"{gw}_Pts"
                    xmins_col = f"{gw}_xMins"

                    if pts_col in columns and xmins_col in columns:
                        try:
                            xmins = float(row.get(xmins_col, 0) or 0)
                            pts = float(row.get(pts_col, 0) or 0)

                            gw_xmins[gw] = xmins
                            gw_points[gw] = pts
                        except (ValueError, TypeError):
                            continue

                # Also check for older format (gw_points, gw+1_points)
                if not gw_points:
                    for col in columns:
                        if col.startswith("gw") and "_points" in col.lower():
                            try:
                                match = re.search(r'gw\+?(\d+)', col)
                                if match:
                                    gw_offset = int(match.group(1))
                                    if gw_offset < len(gameweeks):
                                        gw_points[gameweeks[gw_offset]] = float(row.get(col, 0) or 0)
                            except (ValueError, TypeError):
                                continue

                if gw_points:
                    self._projected_points[player_id] = ProjectedPoints(
                        player_id=player_id,
                        player_name=str(row.get("Name", row.get("name", "Unknown"))),
                        team=str(row.get("Team", row.get("team", ""))),
                        position=str(row.get("Pos", row.get("position", ""))),
                        gameweek_points=gw_points,
                        gameweek_xmins=gw_xmins,
                    )

            self.logger.info(f"Loaded projections for {len(self._projected_points)} players (min xMins: {min_xmins})")

        except Exception as e:
            self.logger.error(f"Failed to load projected points: {e}")

        return self._projected_points

    def get_projected_points(self, player_id: int, gameweek: Optional[int] = None) -> float:
        """Get projected points for a player."""
        projection = self._projected_points.get(player_id)
        if not projection:
            return 0.0

        if gameweek:
            return projection.get_points(gameweek)

        # Return total if no specific gameweek
        return sum(projection.gameweek_points.values())

    # -------------------------------------------------------------------------
    # Unified DataFrame Builder
    # -------------------------------------------------------------------------

    async def build_unified_dataframe(self, gameweeks: list[int]) -> pd.DataFrame:
        """
        Build a comprehensive DataFrame with all collected data.

        Args:
            gameweeks: List of gameweeks to include projections for.

        Returns:
            DataFrame with player data, fixtures, news, and projections.
        """
        self.logger.info("Building unified player DataFrame")

        players = await self.client.get_players()
        teams = await self.client.get_teams()
        team_map = {t.id: t for t in teams}

        data = []

        for player in players:
            team = team_map.get(player.team)

            # Base player data
            row = {
                "player_id": player.id,
                "name": player.web_name,
                "full_name": f"{player.first_name} {player.second_name}",
                "team_id": player.team,
                "team": team.short_name if team else "UNK",
                "team_name": team.name if team else "Unknown",
                "position": player.position.name,
                "position_id": player.element_type,
                "price": player.now_cost,
                "total_points": player.total_points,
                "form": player.form,
                "points_per_game": player.points_per_game,
                "selected_by": player.selected_by_percent,
                "minutes": player.minutes,
                "goals": player.goals_scored,
                "assists": player.assists,
                "clean_sheets": player.clean_sheets,
                "bonus": player.bonus,
                "xG": player.expected_goals,
                "xA": player.expected_assists,
                "xGI": player.expected_goal_involvements,
                "transfers_in": player.transfers_in_event,
                "transfers_out": player.transfers_out_event,
                "price_change": player.cost_change_event / 10,
            }

            # Availability data
            news = self._player_news.get(player.id)
            row["status"] = player.status
            row["news"] = player.news if player.news else (news.news if news else "")
            row["chance_of_playing"] = player.chance_of_playing_next_round
            row["availability"] = news.availability_score if news else (1.0 if player.is_available else 0.0)

            # Fixture difficulty
            team_fixtures = self._team_fixtures.get(player.team)
            if team_fixtures:
                row["fixture_difficulty"] = self.get_fixture_difficulty_score(player.team, gameweeks)
                row["fixture_rating"] = team_fixtures.fixture_difficulty_rating

                # Add individual gameweek fixtures
                for i, gw in enumerate(gameweeks[:5]):
                    fixture = team_fixtures.get_fixture(gw)
                    if fixture:
                        row[f"gw{i+1}_fixture"] = fixture.fixture_string
                        row[f"gw{i+1}_difficulty"] = fixture.difficulty
                    else:
                        row[f"gw{i+1}_fixture"] = "BGW"
                        row[f"gw{i+1}_difficulty"] = 0
            else:
                row["fixture_difficulty"] = 3.0
                row["fixture_rating"] = "Medium"

            # Projected points and expected minutes
            projection = self._projected_points.get(player.id)
            if projection:
                row["projected_total"] = projection.total_points(gameweeks)
                for i, gw in enumerate(gameweeks[:5]):
                    row[f"gw{i+1}_projected"] = projection.get_points(gw)
                    row[f"gw{i+1}_xmins"] = projection.get_xmins(gw)
                    row[f"gw{i+1}_mins_weight"] = projection.get_minutes_weight(gw)
            else:
                row["projected_total"] = 0.0
                for i in range(5):
                    row[f"gw{i+1}_projected"] = 0.0
                    row[f"gw{i+1}_xmins"] = 0.0
                    row[f"gw{i+1}_mins_weight"] = 0.0

            # Calculated metrics
            row["value"] = player.total_points / player.now_cost if player.now_cost > 0 else 0
            row["form_value"] = player.form / player.now_cost if player.now_cost > 0 else 0

            # Composite score
            row["composite_score"] = self._calculate_composite_score(row)

            data.append(row)

        df = pd.DataFrame(data)

        # Sort by composite score
        df = df.sort_values("composite_score", ascending=False)

        self.logger.info(f"Built DataFrame with {len(df)} players and {len(df.columns)} columns")
        return df

    def _calculate_composite_score(self, row: dict) -> float:
        """Calculate composite player score for ranking."""
        score = 0.0

        # Form (weight: 2.0)
        score += row.get("form", 0) * 2.0

        # Fixture difficulty inverted (weight: 1.5, lower difficulty = higher score)
        fixture_diff = row.get("fixture_difficulty", 3.0)
        score += (5 - fixture_diff) * 1.5

        # xGI (weight: 2.5)
        score += row.get("xGI", 0) * 2.5

        # Value score (weight: 1.0)
        score += row.get("value", 0) * 1.0

        # Projected points (weight: 3.0)
        score += row.get("projected_total", 0) * 0.5

        # Availability penalty
        availability = row.get("availability", 1.0)
        score *= availability

        return score

    # -------------------------------------------------------------------------
    # Data Access Methods
    # -------------------------------------------------------------------------

    def get_players_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the unified players DataFrame."""
        return self._players_df

    def get_top_players(
        self,
        position: Optional[str] = None,
        max_price: Optional[float] = None,
        min_form: float = 0.0,
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Get top players filtered by criteria.

        Args:
            position: Filter by position (GK, DEF, MID, FWD)
            max_price: Maximum price filter
            min_form: Minimum form filter
            limit: Maximum number of players to return

        Returns:
            Filtered and sorted DataFrame.
        """
        if self._players_df is None:
            return pd.DataFrame()

        df = self._players_df.copy()

        if position:
            df = df[df["position"].str.upper() == position.upper()]

        if max_price:
            df = df[df["price"] <= max_price]

        if min_form > 0:
            df = df[df["form"] >= min_form]

        return df.head(limit)

    def get_flagged_players(self) -> pd.DataFrame:
        """Get players with availability concerns."""
        if self._players_df is None:
            return pd.DataFrame()

        return self._players_df[
            (self._players_df["availability"] < 1.0) |
            (self._players_df["news"].str.len() > 0)
        ].copy()

    def get_fixture_ticker(self, gameweeks: int = 5) -> pd.DataFrame:
        """
        Get fixture difficulty ticker for all teams.

        Returns:
            DataFrame with team fixtures across gameweeks.
            Includes DGW detection (gw{N}_dgw = True if 2+ fixtures).
        """
        data = []

        for team_id, team_fixtures in self._team_fixtures.items():
            row = {
                "team_id": team_id,
                "team": team_fixtures.team_short,
                "team_name": team_fixtures.team_name,
                "avg_difficulty": team_fixtures.average_difficulty,
                "rating": team_fixtures.fixture_difficulty_rating,
            }

            # Group fixtures by gameweek to detect DGWs
            gw_fixtures: dict[int, list] = {}
            for fixture in team_fixtures.fixtures:
                gw = fixture.gameweek
                if gw not in gw_fixtures:
                    gw_fixtures[gw] = []
                gw_fixtures[gw].append(fixture)

            # Add fixture info for each GW
            first_gw = None
            for gw, fixtures in sorted(gw_fixtures.items())[:gameweeks]:
                if first_gw is None:
                    first_gw = gw

                is_dgw = len(fixtures) >= 2
                is_bgw = len(fixtures) == 0

                if is_dgw:
                    # Combine fixture strings for DGW
                    fixture_strs = [f.fixture_string for f in fixtures]
                    row[f"gw{gw}"] = " + ".join(fixture_strs)
                    row[f"gw{gw}_diff"] = sum(f.difficulty for f in fixtures) / len(fixtures)
                elif is_bgw:
                    row[f"gw{gw}"] = "BGW"
                    row[f"gw{gw}_diff"] = 0
                else:
                    row[f"gw{gw}"] = fixtures[0].fixture_string
                    row[f"gw{gw}_diff"] = fixtures[0].difficulty

                row[f"gw{gw}_dgw"] = is_dgw
                row[f"gw{gw}_fixture_count"] = len(fixtures)

            # Add flag for next GW DGW
            if first_gw and first_gw in gw_fixtures:
                row["next_gw"] = first_gw
                row["next_gw_dgw"] = len(gw_fixtures[first_gw]) >= 2
                row["next_gw_fixture_count"] = len(gw_fixtures[first_gw])
            else:
                row["next_gw"] = None
                row["next_gw_dgw"] = False
                row["next_gw_fixture_count"] = 0

            data.append(row)

        df = pd.DataFrame(data)
        return df.sort_values("avg_difficulty")

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def export_to_csv(self, filename: Optional[str] = None) -> Path:
        """Export unified DataFrame to CSV."""
        if self._players_df is None:
            raise ValueError("No data to export. Run collect_all() first.")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"player_analysis_{timestamp}.csv"

        output_path = self.config.data_dir / filename
        self._players_df.to_csv(output_path, index=False)
        self.logger.info(f"Exported data to {output_path}")

        return output_path

    def export_fixtures_to_csv(self, filename: Optional[str] = None) -> Path:
        """Export fixture ticker to CSV."""
        if filename is None:
            filename = "fixture_ticker.csv"

        output_path = self.config.data_dir / filename
        ticker = self.get_fixture_ticker()
        ticker.to_csv(output_path, index=False)
        self.logger.info(f"Exported fixtures to {output_path}")

        return output_path
