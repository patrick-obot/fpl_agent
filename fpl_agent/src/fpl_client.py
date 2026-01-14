"""
FPL API Client for interacting with the Fantasy Premier League API.

Handles authentication, session management, and all API requests.
"""

import asyncio
import logging
import time
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum

import aiohttp
from aiolimiter import AsyncLimiter

from .config import Config


# =============================================================================
# Exceptions
# =============================================================================

class FPLAPIError(Exception):
    """Base exception for FPL API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(FPLAPIError):
    """Raised when authentication fails."""
    pass


class RateLimitError(FPLAPIError):
    """Raised when rate limit is exceeded."""
    pass


class TransferError(FPLAPIError):
    """Raised when transfer operation fails."""
    pass


class SessionExpiredError(FPLAPIError):
    """Raised when the session has expired."""
    pass


# =============================================================================
# Enums
# =============================================================================

class Position(IntEnum):
    """Player positions."""
    GOALKEEPER = 1
    DEFENDER = 2
    MIDFIELDER = 3
    FORWARD = 4

    @classmethod
    def from_int(cls, value: int) -> "Position":
        return cls(value)

    def __str__(self) -> str:
        return self.name.title()


class PlayerStatus(str):
    """Player availability status codes."""
    AVAILABLE = 'a'
    DOUBTFUL = 'd'
    INJURED = 'i'
    SUSPENDED = 's'
    UNAVAILABLE = 'u'
    NOT_IN_SQUAD = 'n'


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Player:
    """Represents an FPL player with comprehensive stats."""
    id: int
    web_name: str
    first_name: str
    second_name: str
    team: int
    team_name: str
    element_type: int  # 1=GK, 2=DEF, 3=MID, 4=FWD
    now_cost: float  # Price in millions
    total_points: int
    form: float
    points_per_game: float
    selected_by_percent: float
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    bonus: int
    bps: int
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    # Availability
    status: str
    news: str
    news_added: Optional[datetime]
    chance_of_playing_this_round: Optional[int]
    chance_of_playing_next_round: Optional[int]
    # Ownership
    transfers_in_event: int
    transfers_out_event: int
    cost_change_event: int
    cost_change_start: int

    @property
    def position(self) -> Position:
        return Position.from_int(self.element_type)

    @property
    def is_available(self) -> bool:
        return self.status == PlayerStatus.AVAILABLE

    @property
    def availability_percent(self) -> int:
        if self.chance_of_playing_next_round is not None:
            return self.chance_of_playing_next_round
        return 100 if self.is_available else 0

    @classmethod
    def from_api(cls, data: dict, teams_map: dict[int, str] = None) -> "Player":
        """Create Player from API response data."""
        teams_map = teams_map or {}
        news_added = None
        if data.get("news_added"):
            try:
                news_added = datetime.fromisoformat(data["news_added"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return cls(
            id=data["id"],
            web_name=data["web_name"],
            first_name=data.get("first_name", ""),
            second_name=data.get("second_name", ""),
            team=data["team"],
            team_name=teams_map.get(data["team"], f"Team {data['team']}"),
            element_type=data["element_type"],
            now_cost=data["now_cost"] / 10,
            total_points=data["total_points"],
            form=float(data.get("form", 0) or 0),
            points_per_game=float(data.get("points_per_game", 0) or 0),
            selected_by_percent=float(data.get("selected_by_percent", 0) or 0),
            minutes=data.get("minutes", 0),
            goals_scored=data.get("goals_scored", 0),
            assists=data.get("assists", 0),
            clean_sheets=data.get("clean_sheets", 0),
            goals_conceded=data.get("goals_conceded", 0),
            bonus=data.get("bonus", 0),
            bps=data.get("bps", 0),
            expected_goals=float(data.get("expected_goals", 0) or 0),
            expected_assists=float(data.get("expected_assists", 0) or 0),
            expected_goal_involvements=float(data.get("expected_goal_involvements", 0) or 0),
            expected_goals_conceded=float(data.get("expected_goals_conceded", 0) or 0),
            status=data.get("status", "a"),
            news=data.get("news", ""),
            news_added=news_added,
            chance_of_playing_this_round=data.get("chance_of_playing_this_round"),
            chance_of_playing_next_round=data.get("chance_of_playing_next_round"),
            transfers_in_event=data.get("transfers_in_event", 0),
            transfers_out_event=data.get("transfers_out_event", 0),
            cost_change_event=data.get("cost_change_event", 0),
            cost_change_start=data.get("cost_change_start", 0),
        )


@dataclass
class Team:
    """Represents a Premier League team."""
    id: int
    name: str
    short_name: str
    code: int
    strength: int
    strength_overall_home: int
    strength_overall_away: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int

    @classmethod
    def from_api(cls, data: dict) -> "Team":
        """Create Team from API response data."""
        return cls(
            id=data["id"],
            name=data["name"],
            short_name=data["short_name"],
            code=data.get("code", 0),
            strength=data["strength"],
            strength_overall_home=data["strength_overall_home"],
            strength_overall_away=data["strength_overall_away"],
            strength_attack_home=data["strength_attack_home"],
            strength_attack_away=data["strength_attack_away"],
            strength_defence_home=data["strength_defence_home"],
            strength_defence_away=data["strength_defence_away"],
        )


@dataclass
class Gameweek:
    """Represents an FPL gameweek/event."""
    id: int
    name: str
    deadline_time: datetime
    is_previous: bool
    is_current: bool
    is_next: bool
    finished: bool
    average_entry_score: Optional[int]
    highest_score: Optional[int]
    chip_plays: list[dict] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.is_current and not self.finished

    @property
    def time_until_deadline(self) -> Optional[float]:
        """Returns seconds until deadline, negative if passed."""
        if self.deadline_time:
            return (self.deadline_time - datetime.now(self.deadline_time.tzinfo)).total_seconds()
        return None

    @classmethod
    def from_api(cls, data: dict) -> "Gameweek":
        """Create Gameweek from API response data."""
        deadline = None
        if data.get("deadline_time"):
            try:
                deadline = datetime.fromisoformat(data["deadline_time"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return cls(
            id=data["id"],
            name=data["name"],
            deadline_time=deadline,
            is_previous=data.get("is_previous", False),
            is_current=data.get("is_current", False),
            is_next=data.get("is_next", False),
            finished=data.get("finished", False),
            average_entry_score=data.get("average_entry_score"),
            highest_score=data.get("highest_score"),
            chip_plays=data.get("chip_plays", []),
        )


@dataclass
class Fixture:
    """Represents a Premier League fixture."""
    id: int
    gameweek: Optional[int]
    home_team: int
    away_team: int
    home_team_difficulty: int
    away_team_difficulty: int
    kickoff_time: Optional[datetime]
    finished: bool
    started: bool
    home_score: Optional[int]
    away_score: Optional[int]

    @classmethod
    def from_api(cls, data: dict) -> "Fixture":
        """Create Fixture from API response data."""
        kickoff = None
        if data.get("kickoff_time"):
            try:
                kickoff = datetime.fromisoformat(data["kickoff_time"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return cls(
            id=data["id"],
            gameweek=data.get("event"),
            home_team=data["team_h"],
            away_team=data["team_a"],
            home_team_difficulty=data.get("team_h_difficulty", 3),
            away_team_difficulty=data.get("team_a_difficulty", 3),
            kickoff_time=kickoff,
            finished=data.get("finished", False),
            started=data.get("started", False),
            home_score=data.get("team_h_score"),
            away_score=data.get("team_a_score"),
        )


@dataclass
class SquadPick:
    """Represents a player pick in the squad."""
    element: int  # Player ID
    position: int  # Squad position (1-15)
    is_captain: bool
    is_vice_captain: bool
    multiplier: int  # 0=benched, 1=playing, 2=captain, 3=triple captain
    selling_price: float
    purchase_price: float

    @property
    def is_starter(self) -> bool:
        return self.position <= 11

    @classmethod
    def from_api(cls, data: dict) -> "SquadPick":
        return cls(
            element=data["element"],
            position=data["position"],
            is_captain=data.get("is_captain", False),
            is_vice_captain=data.get("is_vice_captain", False),
            multiplier=data.get("multiplier", 1),
            selling_price=data.get("selling_price", 0) / 10,
            purchase_price=data.get("purchase_price", 0) / 10,
        )


@dataclass
class MyTeam:
    """Represents the authenticated user's team data."""
    picks: list[SquadPick]
    chips: list[dict]
    transfers: dict
    # Computed properties
    bank: float = 0.0
    total_value: float = 0.0
    free_transfers: int = 0
    transfers_made: int = 0
    wildcard_available: bool = False
    freehit_available: bool = False
    benchboost_available: bool = False
    triplecaptain_available: bool = False

    @property
    def captain_id(self) -> Optional[int]:
        for pick in self.picks:
            if pick.is_captain:
                return pick.element
        return None

    @property
    def vice_captain_id(self) -> Optional[int]:
        for pick in self.picks:
            if pick.is_vice_captain:
                return pick.element
        return None

    @property
    def starters(self) -> list[SquadPick]:
        return [p for p in self.picks if p.is_starter]

    @property
    def bench(self) -> list[SquadPick]:
        return [p for p in self.picks if not p.is_starter]

    @classmethod
    def from_api(cls, data: dict) -> "MyTeam":
        picks = [SquadPick.from_api(p) for p in data.get("picks", [])]
        transfers = data.get("transfers", {})
        chips = data.get("chips", [])

        # Calculate chip availability
        available_chips = {c["name"] for c in chips if c["status_for_entry"] == "available"}

        return cls(
            picks=picks,
            chips=chips,
            transfers=transfers,
            bank=transfers.get("bank", 0) / 10,
            total_value=transfers.get("value", 0) / 10,
            free_transfers=transfers.get("limit", 1) - transfers.get("made", 0),
            transfers_made=transfers.get("made", 0),
            wildcard_available="wildcard" in available_chips,
            freehit_available="freehit" in available_chips,
            benchboost_available="bboost" in available_chips,
            triplecaptain_available="3xc" in available_chips,
        )


@dataclass
class ManagerInfo:
    """Represents FPL manager information."""
    id: int
    name: str
    player_first_name: str
    player_last_name: str
    player_region_name: str
    summary_overall_points: int
    summary_overall_rank: Optional[int]
    summary_event_points: int
    summary_event_rank: Optional[int]
    current_event: int
    started_event: int
    favourite_team: Optional[int]
    joined_time: Optional[datetime]

    @property
    def full_name(self) -> str:
        return f"{self.player_first_name} {self.player_last_name}"

    @classmethod
    def from_api(cls, data: dict) -> "ManagerInfo":
        joined = None
        if data.get("joined_time"):
            try:
                joined = datetime.fromisoformat(data["joined_time"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return cls(
            id=data["id"],
            name=data.get("name", ""),
            player_first_name=data.get("player_first_name", ""),
            player_last_name=data.get("player_last_name", ""),
            player_region_name=data.get("player_region_name", ""),
            summary_overall_points=data.get("summary_overall_points", 0),
            summary_overall_rank=data.get("summary_overall_rank"),
            summary_event_points=data.get("summary_event_points", 0),
            summary_event_rank=data.get("summary_event_rank"),
            current_event=data.get("current_event", 0),
            started_event=data.get("started_event", 1),
            favourite_team=data.get("favourite_team"),
            joined_time=joined,
        )


@dataclass
class Transfer:
    """Represents a transfer to be made."""
    element_in: int
    element_out: int
    purchase_price: Optional[float] = None
    selling_price: Optional[float] = None


@dataclass
class TransferResult:
    """Result of a transfer operation."""
    success: bool
    transfers: list[Transfer]
    message: str
    cost: int = 0  # Points cost for extra transfers


# =============================================================================
# FPL Client
# =============================================================================

class FPLClient:
    """
    Async client for FPL API interactions.

    Handles authentication, rate limiting, session management, and provides
    methods for all necessary API endpoints.

    Usage:
        async with FPLClient(config) as client:
            await client.login()
            team = await client.get_my_team()
    """

    # URLs
    BASE_URL = "https://fantasy.premierleague.com/api"
    LOGIN_URL = "https://users.premierleague.com/accounts/login/"

    # Rate limit: 1 request per second (strict)
    _rate_limiter = AsyncLimiter(1, 1.0)

    def __init__(self, config: Config):
        """
        Initialize the FPL client.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.logger = logging.getLogger("fpl_agent.client")
        self._session: Optional[aiohttp.ClientSession] = None
        self._authenticated = False
        self._last_request_time: float = 0

        # Caches
        self._bootstrap_cache: Optional[dict] = None
        self._bootstrap_cache_time: float = 0
        self._cache_ttl: float = 300  # 5 minutes cache TTL

        self._players_cache: Optional[list[Player]] = None
        self._teams_cache: Optional[list[Team]] = None
        self._teams_map: Optional[dict[int, str]] = None
        self._gameweeks_cache: Optional[list[Gameweek]] = None

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "FPLClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialize the HTTP session with cookie jar for authentication."""
        if self._session is None:
            # Create cookie jar to persist session cookies
            cookie_jar = aiohttp.CookieJar()

            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                cookie_jar=cookie_jar,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Origin": "https://fantasy.premierleague.com",
                    "Referer": "https://fantasy.premierleague.com/",
                }
            )
            self.logger.debug("HTTP session created with cookie jar")

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            self._authenticated = False
            self._clear_caches()
            self.logger.debug("HTTP session closed")

    def _clear_caches(self) -> None:
        """Clear all cached data."""
        self._bootstrap_cache = None
        self._bootstrap_cache_time = 0
        self._players_cache = None
        self._teams_cache = None
        self._teams_map = None
        self._gameweeks_cache = None

    @property
    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        return self._authenticated

    # -------------------------------------------------------------------------
    # Core Request Method
    # -------------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        url: str,
        retry_count: int = 0,
        **kwargs: Any
    ) -> dict:
        """
        Make a rate-limited request with comprehensive error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL or endpoint path
            retry_count: Current retry attempt number
            **kwargs: Additional arguments for aiohttp request

        Returns:
            JSON response as dictionary.

        Raises:
            FPLAPIError: If request fails after all retries.
            AuthenticationError: If authentication is required.
            RateLimitError: If rate limit is exceeded.
        """
        if self._session is None:
            await self.connect()

        # Build full URL if needed
        if not url.startswith("http"):
            url = f"{self.BASE_URL}/{url.lstrip('/')}"

        # Rate limiting - strict 1 request per second
        async with self._rate_limiter:
            # Additional time-based rate limiting
            elapsed = time.time() - self._last_request_time
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)

            self._last_request_time = time.time()

            try:
                self.logger.debug(f"Request: {method} {url} (attempt {retry_count + 1})")

                async with self._session.request(method, url, **kwargs) as response:
                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(f"Rate limited. Waiting {retry_after}s")

                        if retry_count < self.config.max_retries:
                            await asyncio.sleep(retry_after)
                            return await self._request(method, url, retry_count + 1, **kwargs)

                        raise RateLimitError(f"Rate limit exceeded after {retry_count + 1} attempts", 429)

                    # Handle authentication errors
                    if response.status == 401:
                        self._authenticated = False
                        raise AuthenticationError("Authentication required or session expired", 401)

                    if response.status == 403:
                        self._authenticated = False
                        raise AuthenticationError("Access forbidden - check credentials", 403)

                    # Handle not found
                    if response.status == 404:
                        raise FPLAPIError(f"Resource not found: {url}", 404)

                    # Handle server errors with retry
                    if response.status >= 500:
                        if retry_count < self.config.max_retries:
                            wait_time = 2 ** retry_count
                            self.logger.warning(f"Server error {response.status}. Retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            return await self._request(method, url, retry_count + 1, **kwargs)

                        text = await response.text()
                        raise FPLAPIError(f"Server error {response.status}: {text}", response.status)

                    # Handle other client errors
                    if response.status >= 400:
                        text = await response.text()
                        raise FPLAPIError(f"API error {response.status}: {text}", response.status)

                    # Success - parse JSON
                    try:
                        return await response.json()
                    except aiohttp.ContentTypeError:
                        # Some endpoints return empty response on success
                        return {"success": True}

            except aiohttp.ClientError as e:
                self.logger.error(f"Request failed: {type(e).__name__}: {e}")

                if retry_count < self.config.max_retries:
                    wait_time = 2 ** retry_count
                    self.logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    return await self._request(method, url, retry_count + 1, **kwargs)

                raise FPLAPIError(f"Request failed after {retry_count + 1} attempts: {e}")

            except asyncio.TimeoutError:
                self.logger.error(f"Request timed out: {url}")

                if retry_count < self.config.max_retries:
                    wait_time = 2 ** retry_count
                    await asyncio.sleep(wait_time)
                    return await self._request(method, url, retry_count + 1, **kwargs)

                raise FPLAPIError(f"Request timed out after {retry_count + 1} attempts")

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    async def login(self) -> bool:
        """
        Authenticate with FPL using credentials from config.

        Establishes a session that maintains authentication cookies.

        Returns:
            True if authentication successful.

        Raises:
            AuthenticationError: If authentication fails.
        """
        if self.config.dry_run:
            self.logger.info("Dry run mode - skipping authentication")
            self._authenticated = True
            return True

        if not self.config.fpl_email or not self.config.fpl_password:
            raise AuthenticationError("Email and password required for authentication")

        self.logger.info(f"Authenticating as {self.config.fpl_email}...")

        if self._session is None:
            await self.connect()

        # Prepare login payload
        payload = {
            "login": self.config.fpl_email,
            "password": self.config.fpl_password,
            "redirect_uri": "https://fantasy.premierleague.com/",
            "app": "plfpl-web",
        }

        try:
            # Rate limit the login request too
            async with self._rate_limiter:
                async with self._session.post(
                    self.LOGIN_URL,
                    data=payload,
                    allow_redirects=False
                ) as response:

                    self.logger.debug(f"Login response status: {response.status}")

                    # Check for successful login (redirect or 200)
                    if response.status in (200, 302):
                        # Verify we got session cookies
                        cookies = self._session.cookie_jar.filter_cookies(self.BASE_URL)

                        if cookies:
                            self._authenticated = True
                            self.logger.info("Authentication successful")
                            return True
                        else:
                            # Try to detect login failure from response
                            try:
                                text = await response.text()
                                if "incorrect" in text.lower() or "invalid" in text.lower():
                                    raise AuthenticationError("Invalid email or password")
                            except:
                                pass

                            # Assume success if we got 200/302
                            self._authenticated = True
                            self.logger.info("Authentication successful (no cookies visible)")
                            return True

                    elif response.status == 400:
                        raise AuthenticationError("Invalid credentials")

                    else:
                        text = await response.text()
                        raise AuthenticationError(f"Login failed with status {response.status}: {text[:200]}")

        except aiohttp.ClientError as e:
            raise AuthenticationError(f"Login request failed: {e}")

    async def ensure_authenticated(self) -> None:
        """Ensure the client is authenticated, logging in if necessary."""
        if not self._authenticated:
            await self.login()

    # -------------------------------------------------------------------------
    # Bootstrap Static Data
    # -------------------------------------------------------------------------

    async def get_bootstrap_static(self, force_refresh: bool = False) -> dict:
        """
        Get the main bootstrap-static data (cached).

        Contains all players, teams, gameweeks, and game settings.

        Args:
            force_refresh: Force refresh even if cached.

        Returns:
            Dictionary containing players, teams, events, etc.
        """
        now = time.time()

        # Return cached if valid
        if (not force_refresh and
            self._bootstrap_cache and
            (now - self._bootstrap_cache_time) < self._cache_ttl):
            return self._bootstrap_cache

        self._bootstrap_cache = await self._request("GET", "bootstrap-static/")
        self._bootstrap_cache_time = now

        # Update teams map for player lookups
        self._teams_map = {t["id"]: t["short_name"] for t in self._bootstrap_cache.get("teams", [])}

        return self._bootstrap_cache

    # -------------------------------------------------------------------------
    # Player Data
    # -------------------------------------------------------------------------

    async def get_players(self, force_refresh: bool = False) -> list[Player]:
        """
        Get all players with comprehensive statistics.

        Args:
            force_refresh: Force refresh of cached data.

        Returns:
            List of Player objects with full stats.
        """
        if self._players_cache and not force_refresh:
            return self._players_cache

        data = await self.get_bootstrap_static(force_refresh)

        # Ensure we have teams map
        if not self._teams_map:
            self._teams_map = {t["id"]: t["short_name"] for t in data.get("teams", [])}

        self._players_cache = [
            Player.from_api(p, self._teams_map)
            for p in data["elements"]
        ]

        self.logger.info(f"Loaded {len(self._players_cache)} players")
        return self._players_cache

    async def get_player(self, player_id: int) -> Optional[Player]:
        """Get a specific player by ID."""
        players = await self.get_players()
        for player in players:
            if player.id == player_id:
                return player
        return None

    async def get_player_history(self, player_id: int) -> dict:
        """
        Get detailed history for a specific player.

        Args:
            player_id: The player's element ID.

        Returns:
            Dictionary with gameweek history and fixture history.
        """
        return await self._request("GET", f"element-summary/{player_id}/")

    # -------------------------------------------------------------------------
    # Team Data
    # -------------------------------------------------------------------------

    async def get_teams(self, force_refresh: bool = False) -> list[Team]:
        """
        Get all Premier League teams.

        Args:
            force_refresh: Force refresh of cached data.

        Returns:
            List of Team objects.
        """
        if self._teams_cache and not force_refresh:
            return self._teams_cache

        data = await self.get_bootstrap_static(force_refresh)
        self._teams_cache = [Team.from_api(t) for t in data["teams"]]
        self.logger.info(f"Loaded {len(self._teams_cache)} teams")
        return self._teams_cache

    # -------------------------------------------------------------------------
    # Gameweek Data
    # -------------------------------------------------------------------------

    async def get_gameweeks(self, force_refresh: bool = False) -> list[Gameweek]:
        """
        Get all gameweeks with deadlines.

        Args:
            force_refresh: Force refresh of cached data.

        Returns:
            List of Gameweek objects.
        """
        if self._gameweeks_cache and not force_refresh:
            return self._gameweeks_cache

        data = await self.get_bootstrap_static(force_refresh)
        self._gameweeks_cache = [Gameweek.from_api(e) for e in data["events"]]
        return self._gameweeks_cache

    async def get_current_gameweek(self, force_refresh: bool = False) -> Gameweek:
        """
        Get the current/upcoming gameweek for transfers.

        Returns the next gameweek with an upcoming deadline, not the one
        marked as 'current' by the API if its deadline has already passed.

        Returns:
            Current or next Gameweek object.

        Raises:
            FPLAPIError: If no current gameweek found.
        """
        gameweeks = await self.get_gameweeks(force_refresh)
        now = datetime.now(timezone.utc)

        # Find the gameweek marked as current
        current_gw = None
        next_gw = None
        for gw in gameweeks:
            if gw.is_current:
                current_gw = gw
            if gw.is_next:
                next_gw = gw

        # If current gameweek's deadline has passed or it's finished, use next
        if current_gw:
            deadline = current_gw.deadline_time
            if deadline and deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)

            if current_gw.finished or (deadline and deadline < now):
                if next_gw:
                    self.logger.info(f"GW{current_gw.id} finished/passed, using next: GW{next_gw.id} ({next_gw.name})")
                    return next_gw

            self.logger.info(f"Current gameweek: {current_gw.id} ({current_gw.name})")
            return current_gw

        # Fall back to next gameweek
        if next_gw:
            self.logger.info(f"Using next gameweek: {next_gw.id} ({next_gw.name})")
            return next_gw

        raise FPLAPIError("Could not determine current gameweek")

    async def get_next_deadline(self) -> Optional[datetime]:
        """Get the next gameweek deadline."""
        gameweeks = await self.get_gameweeks()

        for gw in gameweeks:
            if gw.is_next or gw.is_current:
                return gw.deadline_time

        return None

    # -------------------------------------------------------------------------
    # Fixture Data
    # -------------------------------------------------------------------------

    async def get_fixtures(self, gameweek: Optional[int] = None) -> list[Fixture]:
        """
        Get fixtures, optionally filtered by gameweek.

        Args:
            gameweek: Specific gameweek to get fixtures for.

        Returns:
            List of Fixture objects.
        """
        endpoint = "fixtures/"
        if gameweek:
            endpoint += f"?event={gameweek}"

        data = await self._request("GET", endpoint)
        fixtures = [Fixture.from_api(f) for f in data]
        self.logger.debug(f"Loaded {len(fixtures)} fixtures")
        return fixtures

    # -------------------------------------------------------------------------
    # My Team Data
    # -------------------------------------------------------------------------

    async def get_my_team(self) -> MyTeam:
        """
        Get the authenticated user's current team.

        Returns:
            MyTeam object with picks, transfers, and chips.

        Raises:
            AuthenticationError: If not authenticated.
        """
        if self.config.dry_run:
            # Try to fetch real team from public API first
            if self.config.fpl_team_id:
                try:
                    team = await self._get_team_from_public_api()
                    self.logger.info("Dry run mode - fetched real team from public API")
                    return team
                except Exception as e:
                    self.logger.warning(f"Failed to fetch team from public API: {e}")
                    self.logger.info("Dry run mode - falling back to mock team data")
            return self._get_mock_team()

        await self.ensure_authenticated()
        data = await self._request("GET", f"my-team/{self.config.fpl_team_id}/")
        return MyTeam.from_api(data)

    async def _get_team_from_public_api(self) -> MyTeam:
        """
        Fetch team data from public API endpoints (no authentication required).

        This allows fetching real team data in dry-run mode.
        """
        team_id = self.config.fpl_team_id

        # Get the last completed gameweek for picks (not the upcoming one)
        # Picks are only available for gameweeks that have started
        gameweeks = await self.get_gameweeks()
        picks_gw_id = 1
        for gw in gameweeks:
            if gw.finished or gw.is_current:
                picks_gw_id = gw.id
            if gw.is_next:
                break

        # Fetch picks from public endpoint using last completed/current GW
        picks_data = await self._request("GET", f"entry/{team_id}/event/{picks_gw_id}/picks/")

        # Fetch entry history for transfer info
        history_data = await self._request("GET", f"entry/{team_id}/history/")

        # Fetch entry info for chips
        entry_data = await self._request("GET", f"entry/{team_id}/")

        # Build picks list
        picks = []
        for p in picks_data.get("picks", []):
            picks.append(SquadPick(
                element=p["element"],
                position=p["position"],
                is_captain=p.get("is_captain", False),
                is_vice_captain=p.get("is_vice_captain", False),
                multiplier=p.get("multiplier", 1),
                selling_price=p.get("selling_price", 0) / 10 if p.get("selling_price") else 5.0,
                purchase_price=p.get("purchase_price", 0) / 10 if p.get("purchase_price") else 5.0,
            ))

        # Get current event history for bank and value
        current_history = None
        for event in history_data.get("current", []):
            if event.get("event") == gw_id:
                current_history = event
                break

        # Get latest if current not found
        if not current_history and history_data.get("current"):
            current_history = history_data["current"][-1]

        bank = (current_history.get("bank", 0) / 10) if current_history else 0.0
        total_value = (current_history.get("value", 1000) / 10) if current_history else 100.0
        transfers_made = current_history.get("event_transfers", 0) if current_history else 0

        # Determine free transfers (default 1-2, max 5 with rollover)
        # This is an approximation since we can't see exact free transfers from public API
        free_transfers = max(1, 2 - transfers_made)

        # Check chip usage from history
        chips_used = set()
        for chip in history_data.get("chips", []):
            chips_used.add(chip.get("name"))

        return MyTeam(
            picks=picks,
            chips=history_data.get("chips", []),
            transfers={"bank": int(bank * 10), "value": int(total_value * 10), "made": transfers_made},
            bank=bank,
            total_value=total_value,
            free_transfers=free_transfers,
            transfers_made=transfers_made,
            wildcard_available="wildcard" not in chips_used,
            freehit_available="freehit" not in chips_used,
            benchboost_available="bboost" not in chips_used,
            triplecaptain_available="3xc" not in chips_used,
        )

    async def get_manager_info(self) -> ManagerInfo:
        """
        Get information about the authenticated manager.

        Returns:
            ManagerInfo object with overall stats.
        """
        if self.config.dry_run:
            return ManagerInfo(
                id=self.config.fpl_team_id,
                name="Dry Run Team",
                player_first_name="Dry",
                player_last_name="Run",
                player_region_name="Test",
                summary_overall_points=0,
                summary_overall_rank=None,
                summary_event_points=0,
                summary_event_rank=None,
                current_event=1,
                started_event=1,
                favourite_team=None,
                joined_time=None,
            )

        await self.ensure_authenticated()
        data = await self._request("GET", f"entry/{self.config.fpl_team_id}/")
        return ManagerInfo.from_api(data)

    async def get_team_with_players(self) -> tuple[MyTeam, dict[int, Player]]:
        """
        Get team data with full player details.

        Returns:
            Tuple of (MyTeam, dict mapping player_id to Player)
        """
        team = await self.get_my_team()
        all_players = await self.get_players()

        player_map = {p.id: p for p in all_players}

        return team, player_map

    # -------------------------------------------------------------------------
    # Transfers
    # -------------------------------------------------------------------------

    async def make_transfers(
        self,
        transfers: list[Transfer],
        confirm: bool = False,
        use_wildcard: bool = False,
        use_freehit: bool = False,
    ) -> TransferResult:
        """
        Execute transfers with confirmation step.

        Args:
            transfers: List of Transfer objects to execute.
            confirm: If True, actually execute. If False, just validate.
            use_wildcard: Use wildcard chip.
            use_freehit: Use free hit chip.

        Returns:
            TransferResult with success status and details.

        Raises:
            TransferError: If transfer validation or execution fails.
        """
        if not transfers:
            return TransferResult(
                success=True,
                transfers=[],
                message="No transfers to make"
            )

        if self.config.dry_run:
            self.logger.info(f"Dry run mode - would execute {len(transfers)} transfers")
            return TransferResult(
                success=True,
                transfers=transfers,
                message=f"[DRY RUN] Would execute {len(transfers)} transfer(s)"
            )

        await self.ensure_authenticated()

        # Get current gameweek
        gw = await self.get_current_gameweek()

        # Build transfer payload
        transfer_data = [
            {"element_in": t.element_in, "element_out": t.element_out}
            for t in transfers
        ]

        # Determine chip
        chip = None
        if use_wildcard:
            chip = "wildcard"
        elif use_freehit:
            chip = "freehit"

        payload = {
            "chip": chip,
            "entry": self.config.fpl_team_id,
            "event": gw.id,
            "transfers": transfer_data,
        }

        if not confirm:
            # Validation only - don't actually submit
            self.logger.info("Transfer validation (not confirmed)")
            return TransferResult(
                success=True,
                transfers=transfers,
                message=f"Ready to execute {len(transfers)} transfer(s). Call with confirm=True to proceed."
            )

        # Execute transfers
        self.logger.info(f"Executing {len(transfers)} transfer(s)...")

        try:
            response = await self._request(
                "POST",
                "transfers/",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            # Check response for errors
            if "error" in response:
                raise TransferError(f"Transfer failed: {response['error']}")

            self.logger.info("Transfers executed successfully")
            return TransferResult(
                success=True,
                transfers=transfers,
                message=f"Successfully executed {len(transfers)} transfer(s)",
                cost=response.get("spent_points", 0)
            )

        except FPLAPIError as e:
            raise TransferError(f"Transfer failed: {e}")

    # -------------------------------------------------------------------------
    # Captain Selection
    # -------------------------------------------------------------------------

    async def set_captain(
        self,
        captain_id: int,
        vice_captain_id: int,
        confirm: bool = False
    ) -> dict:
        """
        Set the captain and vice-captain.

        Args:
            captain_id: Element ID of the captain.
            vice_captain_id: Element ID of the vice-captain.
            confirm: If True, actually execute.

        Returns:
            Result dictionary.
        """
        if self.config.dry_run:
            self.logger.info(f"Dry run mode - would set captain to {captain_id}")
            return {
                "success": True,
                "captain": captain_id,
                "vice_captain": vice_captain_id,
                "message": "[DRY RUN] Captain changes not applied"
            }

        await self.ensure_authenticated()

        if not confirm:
            return {
                "success": True,
                "captain": captain_id,
                "vice_captain": vice_captain_id,
                "message": "Ready to set captain. Call with confirm=True to proceed."
            }

        # Get current team to build picks payload
        team = await self.get_my_team()

        picks = []
        for pick in team.picks:
            picks.append({
                "element": pick.element,
                "position": pick.position,
                "is_captain": pick.element == captain_id,
                "is_vice_captain": pick.element == vice_captain_id,
            })

        response = await self._request(
            "POST",
            f"my-team/{self.config.fpl_team_id}/",
            json={"picks": picks},
            headers={"Content-Type": "application/json"}
        )

        self.logger.info(f"Captain set to player {captain_id}")
        return {
            "success": True,
            "captain": captain_id,
            "vice_captain": vice_captain_id,
            "message": "Captain updated successfully"
        }

    # -------------------------------------------------------------------------
    # Mock Data for Dry Run
    # -------------------------------------------------------------------------

    def _get_mock_team(self) -> MyTeam:
        """Generate mock team data for dry run mode."""
        picks = [
            SquadPick(element=i, position=i, is_captain=(i == 8),
                      is_vice_captain=(i == 9), multiplier=2 if i == 8 else 1,
                      selling_price=5.0, purchase_price=5.0)
            for i in range(1, 16)
        ]

        return MyTeam(
            picks=picks,
            chips=[],
            transfers={"bank": 50, "limit": 2, "made": 0, "value": 1000},
            bank=5.0,
            total_value=100.0,
            free_transfers=2,
            transfers_made=0,
            wildcard_available=True,
            freehit_available=True,
            benchboost_available=True,
            triplecaptain_available=True,
        )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    async def get_squad_value(self) -> float:
        """Get total squad value including bank."""
        team = await self.get_my_team()
        return team.total_value + team.bank

    async def get_budget(self) -> float:
        """Get available budget (bank balance)."""
        team = await self.get_my_team()
        return team.bank

    async def get_transfer_info(self) -> dict:
        """Get transfer information."""
        team = await self.get_my_team()
        return {
            "free_transfers": team.free_transfers,
            "transfers_made": team.transfers_made,
            "bank": team.bank,
            "wildcard_available": team.wildcard_available,
            "freehit_available": team.freehit_available,
        }
