"""
Twitter/X data collector for FPL betting odds and projections.

Fetches tweets from @robtFPL and extracts relevant FPL data signals.
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import tweepy


@dataclass
class FPLTweet:
    """Represents an FPL-relevant tweet."""
    tweet_id: str
    text: str
    created_at: datetime
    gameweek: Optional[int] = None
    has_image: bool = False
    image_urls: list[str] = field(default_factory=list)
    url: str = ""

    @property
    def is_projection_tweet(self) -> bool:
        """Check if tweet contains projection data."""
        keywords = ['projected', 'projections', 'odds', 'matchups', 'xpts']
        return any(kw in self.text.lower() for kw in keywords)

    @property
    def is_dgw_tweet(self) -> bool:
        """Check if tweet mentions Double Gameweek."""
        return 'dgw' in self.text.lower() or 'double' in self.text.lower()

    @property
    def is_bgw_tweet(self) -> bool:
        """Check if tweet mentions Blank Gameweek."""
        return 'bgw' in self.text.lower() or 'blank' in self.text.lower()


@dataclass
class GameweekSignals:
    """Extracted signals for a specific gameweek."""
    gameweek: int
    tweet_id: str
    tweet_url: str
    created_at: datetime
    is_dgw: bool = False
    is_bgw: bool = False
    teams_mentioned: list[str] = field(default_factory=list)
    has_projections: bool = False
    has_odds: bool = False
    raw_text: str = ""


class TwitterCollector:
    """
    Collects FPL data from Twitter/X accounts.

    Primary source: @robtFPL for betting odds and projections.
    """

    # Premier League team codes for detection
    TEAM_CODES = [
        'ARS', 'AVL', 'BOU', 'BRE', 'BHA', 'CHE', 'CRY',
        'EVE', 'FUL', 'IPS', 'LEI', 'LIV', 'MCI', 'MUN',
        'NEW', 'NFO', 'SOU', 'TOT', 'WHU', 'WOL'
    ]

    def __init__(self, bearer_token: str):
        """
        Initialize Twitter collector.

        Args:
            bearer_token: Twitter API v2 Bearer Token for read-only access.
        """
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.logger = logging.getLogger("fpl_agent.twitter")
        self._user_id_cache: dict[str, str] = {}

    def _get_user_id(self, username: str) -> Optional[str]:
        """Get Twitter user ID from username."""
        if username in self._user_id_cache:
            return self._user_id_cache[username]

        try:
            user = self.client.get_user(username=username)
            if user.data:
                self._user_id_cache[username] = str(user.data.id)
                return self._user_id_cache[username]
        except tweepy.TweepyException as e:
            self.logger.error(f"Failed to get user ID for @{username}: {e}")

        return None

    def fetch_recent_tweets(
        self,
        username: str = "robtFPL",
        max_results: int = 20
    ) -> list[FPLTweet]:
        """
        Fetch recent tweets from a user.

        Args:
            username: Twitter username (without @).
            max_results: Maximum tweets to fetch (10-100).

        Returns:
            List of FPLTweet objects.
        """
        user_id = self._get_user_id(username)
        if not user_id:
            self.logger.error(f"Could not find user @{username}")
            return []

        try:
            self.logger.info(f"Fetching tweets from @{username}...")

            response = self.client.get_users_tweets(
                id=user_id,
                max_results=min(max(max_results, 10), 100),
                tweet_fields=["created_at", "attachments"],
                expansions=["attachments.media_keys"],
                media_fields=["url", "preview_image_url", "type"]
            )

            if not response.data:
                self.logger.warning(f"No tweets found for @{username}")
                return []

            # Build media lookup from includes
            media_lookup = {}
            if response.includes and "media" in response.includes:
                for media in response.includes["media"]:
                    media_lookup[media.media_key] = media

            tweets = []
            for tweet in response.data:
                # Extract gameweek from text
                gw_match = re.search(r'GW\s*(\d+)', tweet.text, re.IGNORECASE)
                gameweek = int(gw_match.group(1)) if gw_match else None

                # Extract image URLs
                image_urls = []
                if hasattr(tweet, 'attachments') and tweet.attachments:
                    media_keys = tweet.attachments.get('media_keys', [])
                    for key in media_keys:
                        media = media_lookup.get(key)
                        if media and media.type == "photo":
                            url = getattr(media, 'url', None) or getattr(media, 'preview_image_url', None)
                            if url:
                                image_urls.append(url)

                tweets.append(FPLTweet(
                    tweet_id=str(tweet.id),
                    text=tweet.text,
                    created_at=tweet.created_at,
                    gameweek=gameweek,
                    has_image=len(image_urls) > 0,
                    image_urls=image_urls,
                    url=f"https://x.com/{username}/status/{tweet.id}"
                ))

            self.logger.info(f"Fetched {len(tweets)} tweets from @{username}")
            return tweets

        except tweepy.TooManyRequests as e:
            self.logger.warning(f"Twitter API rate limited. Try again later.")
            return []
        except tweepy.TweepyException as e:
            self.logger.error(f"Twitter API error: {e}")
            return []

    def extract_signals(self, tweet: FPLTweet) -> Optional[GameweekSignals]:
        """
        Extract FPL signals from a tweet.

        Args:
            tweet: FPLTweet to analyze.

        Returns:
            GameweekSignals if relevant, None otherwise.
        """
        if not tweet.gameweek:
            return None

        # Detect team mentions
        teams_mentioned = []
        text_upper = tweet.text.upper()
        for code in self.TEAM_CODES:
            if code in text_upper:
                teams_mentioned.append(code)

        return GameweekSignals(
            gameweek=tweet.gameweek,
            tweet_id=tweet.tweet_id,
            tweet_url=tweet.url,
            created_at=tweet.created_at,
            is_dgw=tweet.is_dgw_tweet,
            is_bgw=tweet.is_bgw_tweet,
            teams_mentioned=teams_mentioned,
            has_projections=tweet.is_projection_tweet,
            has_odds='odds' in tweet.text.lower() or '%' in tweet.text,
            raw_text=tweet.text
        )

    def get_gameweek_signals(
        self,
        target_gameweek: int,
        username: str = "robtFPL",
        max_results: int = 50
    ) -> list[GameweekSignals]:
        """
        Get all signals for a specific gameweek.

        Args:
            target_gameweek: Gameweek number to filter for.
            username: Twitter username to fetch from.
            max_results: Maximum tweets to search through.

        Returns:
            List of GameweekSignals for the target gameweek.
        """
        tweets = self.fetch_recent_tweets(username, max_results)
        signals = []

        for tweet in tweets:
            signal = self.extract_signals(tweet)
            if signal and signal.gameweek == target_gameweek:
                signals.append(signal)

        self.logger.info(
            f"Found {len(signals)} signal(s) for GW{target_gameweek} from @{username}"
        )
        return signals

    def get_latest_projection_tweet(
        self,
        username: str = "robtFPL",
        max_results: int = 20
    ) -> Optional[FPLTweet]:
        """
        Get the most recent projection tweet.

        Args:
            username: Twitter username.
            max_results: Maximum tweets to search.

        Returns:
            Most recent projection tweet, or None.
        """
        tweets = self.fetch_recent_tweets(username, max_results)

        for tweet in tweets:
            if tweet.is_projection_tweet:
                self.logger.info(
                    f"Found projection tweet for GW{tweet.gameweek}: {tweet.url}"
                )
                return tweet

        return None
