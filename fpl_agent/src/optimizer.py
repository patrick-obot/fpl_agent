"""
Transfer and captain optimization module.

Provides algorithms for selecting optimal transfers and captains using
greedy optimization with constraint handling.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import combinations

import pandas as pd
import numpy as np

from .config import Config
from .fpl_client import FPLClient, Player
from .data_collector import DataCollector


# =============================================================================
# Constants and Enums
# =============================================================================

class Position(IntEnum):
    """Player positions."""
    GOALKEEPER = 1
    DEFENDER = 2
    MIDFIELDER = 3
    FORWARD = 4


POSITION_NAMES = {
    Position.GOALKEEPER: "GK",
    Position.DEFENDER: "DEF",
    Position.MIDFIELDER: "MID",
    Position.FORWARD: "FWD",
}


# Valid formations for starting XI
VALID_FORMATIONS = [
    (1, 3, 5, 2),  # 3-5-2
    (1, 3, 4, 3),  # 3-4-3
    (1, 4, 5, 1),  # 4-5-1
    (1, 4, 4, 2),  # 4-4-2
    (1, 4, 3, 3),  # 4-3-3
    (1, 5, 4, 1),  # 5-4-1
    (1, 5, 3, 2),  # 5-3-2
    (1, 5, 2, 3),  # 5-2-3
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TransferRecommendation:
    """Represents a recommended transfer."""
    player_out: Player
    player_in: Player
    expected_gain: float
    reason: str
    priority: int = 1  # 1 = highest priority
    hit_cost: int = 0  # Transfer hit cost (-4 per extra transfer)
    net_gain: float = 0.0  # expected_gain - hit_cost

    def __post_init__(self):
        self.net_gain = self.expected_gain - abs(self.hit_cost)

    def __str__(self) -> str:
        hit_str = f" (HIT: {self.hit_cost})" if self.hit_cost < 0 else ""
        return (
            f"OUT: {self.player_out.web_name} ({self.player_out.now_cost:.1f}m) -> "
            f"IN: {self.player_in.web_name} ({self.player_in.now_cost:.1f}m){hit_str} | "
            f"Net gain: {self.net_gain:.2f} pts | {self.reason}"
        )


@dataclass
class CaptainRecommendation:
    """Represents a captain recommendation."""
    player: Player
    expected_points: float
    fixture_score: float
    confidence: float  # 0-1 scale
    is_home: bool
    ownership: float
    reason: str
    is_differential: bool = False  # True if < 10% ownership

    def __post_init__(self):
        self.is_differential = self.ownership < 10.0

    def __str__(self) -> str:
        venue = "(H)" if self.is_home else "(A)"
        diff_str = " [DIFF]" if self.is_differential else ""
        return (
            f"{self.player.web_name} {venue}{diff_str} | "
            f"Expected: {self.expected_points:.2f} pts | "
            f"Confidence: {self.confidence:.0%} | {self.reason}"
        )


@dataclass
class ChipRecommendation:
    """Represents a chip usage recommendation."""
    chip_name: str  # wildcard, bench_boost, free_hit, triple_captain
    gameweek: int
    score: float  # 0-100 scale
    reasons: list[str] = field(default_factory=list)
    is_recommended: bool = False

    def __str__(self) -> str:
        status = "RECOMMENDED" if self.is_recommended else "Consider"
        return f"{self.chip_name.upper()} GW{self.gameweek}: {status} (Score: {self.score:.0f}/100)"


@dataclass
class BenchOrder:
    """Recommended bench order."""
    bench_gk: Optional[int] = None  # Player ID for bench GK
    bench_1: Optional[int] = None   # First sub (highest xPts outfield)
    bench_2: Optional[int] = None   # Second sub
    bench_3: Optional[int] = None   # Third sub (lowest xPts outfield)

    def to_list(self) -> list[int]:
        """Return bench order as list of player IDs."""
        return [p for p in [self.bench_gk, self.bench_1, self.bench_2, self.bench_3] if p]


@dataclass
class OptimizationResult:
    """Results of the optimization process."""
    transfers: list[TransferRecommendation] = field(default_factory=list)
    captain: Optional[CaptainRecommendation] = None
    vice_captain: Optional[CaptainRecommendation] = None
    differential_captain: Optional[CaptainRecommendation] = None
    chip_recommendations: list[ChipRecommendation] = field(default_factory=list)
    starting_xi: list[int] = field(default_factory=list)  # Player IDs for starting 11
    bench_order: Optional[BenchOrder] = None
    total_expected_gain: float = 0.0
    total_hit_cost: int = 0
    net_expected_gain: float = 0.0
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Transfer Optimizer
# =============================================================================

class TransferOptimizer:
    """
    Optimizes transfers using greedy algorithm with constraint handling.

    Considers:
    - Free transfers available
    - Hit penalties (-4 per extra transfer)
    - Team composition limits (max 3 per team)
    - Formation validity
    - Budget constraints
    - Multi-week projected points
    """

    HIT_PENALTY = -4
    MAX_FROM_TEAM = 3
    MIN_IMPROVEMENT_THRESHOLD = 4.0  # Minimum improvement to suggest transfer
    INJURY_REPLACEMENT_THRESHOLD = 2.0  # Lower threshold for replacing injured players
    HIT_WORTHWHILE_THRESHOLD = 8.0  # Net gain must exceed this to justify a hit

    def __init__(
        self,
        current_squad: list[Player],
        player_df: pd.DataFrame,
        budget: float,
        free_transfers: int,
        max_transfers: int = 2,
        planning_horizon: int = 5,
    ):
        """
        Initialize transfer optimizer.

        Args:
            current_squad: Current squad players.
            player_df: DataFrame with all player data and projections.
            budget: Available budget in millions.
            free_transfers: Number of free transfers available.
            max_transfers: Maximum transfers to recommend.
            planning_horizon: Number of gameweeks to consider.
        """
        self.current_squad = current_squad
        self.player_df = player_df
        self.budget = budget
        self.free_transfers = free_transfers
        self.max_transfers = max_transfers
        self.planning_horizon = planning_horizon
        self.logger = logging.getLogger("fpl_agent.optimizer.transfer")

        # Build lookup structures
        self.squad_ids = {p.id for p in current_squad}
        self.team_counts = self._count_teams()
        self.position_counts = self._count_positions()

    def _count_teams(self) -> dict[int, int]:
        """Count players per team in current squad."""
        counts = {}
        for player in self.current_squad:
            counts[player.team] = counts.get(player.team, 0) + 1
        return counts

    def _count_positions(self) -> dict[int, int]:
        """Count players per position in current squad."""
        counts = {}
        for player in self.current_squad:
            counts[player.element_type] = counts.get(player.element_type, 0) + 1
        return counts

    def optimize(self) -> list[TransferRecommendation]:
        """
        Find optimal transfers using greedy algorithm.

        Returns:
            List of TransferRecommendation objects sorted by net gain.
        """
        self.logger.info(
            f"Optimizing transfers: budget={self.budget:.1f}m, "
            f"free_transfers={self.free_transfers}, max={self.max_transfers}"
        )

        all_recommendations = []

        # Evaluate each player in squad for potential replacement
        for player_out in self.current_squad:
            candidates = self._find_replacement_candidates(player_out)

            if candidates.empty:
                continue

            # Score and rank candidates
            candidates = self._score_candidates(candidates, player_out)

            # Get best candidate
            if not candidates.empty:
                best = candidates.iloc[0]
                gain = best["improvement"]

                if gain > self.MIN_IMPROVEMENT_THRESHOLD:
                    recommendation = self._create_recommendation(player_out, best)
                    all_recommendations.append(recommendation)

        # Sort by priority and net gain
        all_recommendations.sort(key=lambda x: (x.priority, -x.net_gain))

        # Apply transfer limits and hit penalties
        final_recommendations = self._apply_transfer_limits(all_recommendations)

        return final_recommendations

    def _find_replacement_candidates(self, player_out: Player) -> pd.DataFrame:
        """Find valid replacement candidates for a player."""
        position = player_out.element_type
        max_price = player_out.now_cost + self.budget

        # Base filters
        candidates = self.player_df[
            (self.player_df["position_id"] == position) &
            (self.player_df["price"] <= max_price) &
            (~self.player_df["player_id"].isin(self.squad_ids)) &
            (self.player_df["availability"] >= 0.75) &
            (self.player_df["minutes"] > 0)
        ].copy()

        # Filter by team limits
        valid_mask = candidates["team_id"].apply(
            lambda t: self._can_add_from_team(t, player_out.team)
        )
        candidates = candidates[valid_mask]

        return candidates

    def _can_add_from_team(self, new_team: int, replacing_team: int) -> bool:
        """Check if adding player from team would violate limits."""
        current_count = self.team_counts.get(new_team, 0)
        # If replacing someone from same team, don't count them
        if new_team == replacing_team:
            current_count -= 1
        return current_count < self.MAX_FROM_TEAM

    def _score_candidates(self, candidates: pd.DataFrame, player_out: Player) -> pd.DataFrame:
        """Score candidates based on expected improvement."""
        out_score = self._calculate_player_score(player_out)

        candidates["score"] = candidates.apply(
            lambda row: self._calculate_row_score(row),
            axis=1
        )
        candidates["improvement"] = candidates["score"] - out_score
        candidates["price_diff"] = candidates["price"] - player_out.now_cost

        return candidates.sort_values("improvement", ascending=False)

    def _calculate_player_score(self, player: Player) -> float:
        """Calculate score for a player from Player object."""
        row = self.player_df[self.player_df["player_id"] == player.id]
        if row.empty:
            return 0.0
        return self._calculate_row_score(row.iloc[0])

    def _calculate_row_score(self, row: pd.Series) -> float:
        """
        Calculate score for player using weighted formula + projected points.

        Components:
        - Projected points from CSV (weighted by xMins)
        - Form, fixture difficulty, xGI, points per game
        - xMins weight: if < 60, reduce the projected points contribution
        """
        score = 0.0

        # 1. Projected points (primary factor when available)
        projected_pts = float(row.get("gw1_projected", 0) or 0)
        xmins_weight = float(row.get("gw1_mins_weight", 1.0) or 1.0)

        if projected_pts > 0:
            # Weight projected points by expected minutes
            # Full weight if xMins >= 60, reduced if less
            score += projected_pts * xmins_weight * 3.0

            # Add future gameweeks with decay
            for i in range(2, min(6, self.planning_horizon + 1)):
                pts_col = f"gw{i}_projected"
                weight_col = f"gw{i}_mins_weight"
                if pts_col in row.index:
                    future_pts = float(row.get(pts_col, 0) or 0)
                    future_weight = float(row.get(weight_col, 1.0) or 1.0)
                    decay = 0.5 / i  # Decay for future weeks
                    score += future_pts * future_weight * decay

        # 2. Form (recent performance) - small bonus
        score += float(row.get("form", 0)) * 0.5

        # 3. Fixture difficulty (inverted - lower is better) - small bonus
        fixture = float(row.get("fixture_difficulty", 3.0))
        score += (5 - fixture) * 0.5

        # 4. Points per game (historical) - small bonus
        score += float(row.get("points_per_game", 0)) * 0.3

        # 7. Availability penalty (injury/suspension)
        availability = float(row.get("availability", 1.0))
        score *= availability

        return score

    def _create_recommendation(
        self, player_out: Player, candidate: pd.Series
    ) -> TransferRecommendation:
        """Create transfer recommendation from candidate data."""
        # Find Player object for candidate
        player_in = Player(
            id=int(candidate["player_id"]),
            web_name=str(candidate["name"]),
            first_name=str(candidate.get("full_name", candidate["name"])).split()[0] if candidate.get("full_name") else str(candidate["name"]),
            second_name=str(candidate.get("full_name", candidate["name"])).split()[-1] if candidate.get("full_name") else str(candidate["name"]),
            team=int(candidate["team_id"]),
            team_name=str(candidate.get("team_name", f"Team{candidate['team_id']}")),
            element_type=int(candidate["position_id"]),
            now_cost=float(candidate["price"]),
            total_points=int(candidate.get("total_points", 0)),
            form=float(candidate.get("form", 0)),
            points_per_game=float(candidate.get("points_per_game", 0)),
            selected_by_percent=float(candidate.get("selected_by", 0)),
            minutes=int(candidate.get("minutes", 0)),
            goals_scored=int(candidate.get("goals", 0)),
            assists=int(candidate.get("assists", 0)),
            clean_sheets=int(candidate.get("clean_sheets", 0)),
            goals_conceded=int(candidate.get("goals_conceded", 0)),
            bonus=int(candidate.get("bonus", 0)),
            bps=int(candidate.get("bps", 0)),
            expected_goals=float(candidate.get("xG", 0)),
            expected_assists=float(candidate.get("xA", 0)),
            expected_goal_involvements=float(candidate.get("xGI", 0)),
            expected_goals_conceded=float(candidate.get("xGC", 0)),
            status=str(candidate.get("status", "a")),
            news=str(candidate.get("news", "")),
            news_added=None,
            chance_of_playing_this_round=None,
            chance_of_playing_next_round=int(candidate["chance_of_playing"]) if pd.notna(candidate.get("chance_of_playing")) else None,
            transfers_in_event=int(candidate.get("transfers_in", 0) or 0),
            transfers_out_event=int(candidate.get("transfers_out", 0) or 0),
            cost_change_event=int((candidate.get("price_change", 0) or 0) * 10),
            cost_change_start=0,
        )

        # Determine priority and reason
        out_availability = self.player_df[
            self.player_df["player_id"] == player_out.id
        ].iloc[0].get("availability", 1.0) if not self.player_df[
            self.player_df["player_id"] == player_out.id
        ].empty else 1.0

        if out_availability < 0.5:
            priority = 1
            reason = f"{player_out.web_name} injured/unavailable"
        elif out_availability < 0.75:
            priority = 2
            reason = f"{player_out.web_name} doubtful"
        elif candidate["improvement"] > 5.0:
            priority = 2
            reason = f"Significant upgrade ({candidate['improvement']:.1f} pts)"
        else:
            priority = 3
            reason = self._generate_reason(player_out, candidate)

        return TransferRecommendation(
            player_out=player_out,
            player_in=player_in,
            expected_gain=candidate["improvement"],
            reason=reason,
            priority=priority,
            hit_cost=0,  # Will be set by _apply_transfer_limits
        )

    def _generate_reason(self, player_out: Player, candidate: pd.Series) -> str:
        """Generate human-readable reason for transfer."""
        reasons = []

        # Form comparison
        if candidate["form"] > player_out.form + 1.5:
            reasons.append(f"Better form ({candidate['form']:.1f} vs {player_out.form:.1f})")

        # Fixture comparison
        fixture_diff = candidate.get("fixture_difficulty", 3.0)
        if fixture_diff <= 2.5:
            reasons.append("Favorable fixtures")

        # Price difference
        price_diff = candidate["price"] - player_out.now_cost
        if price_diff < -0.5:
            reasons.append(f"Saves {abs(price_diff):.1f}m")
        elif price_diff > 0.5:
            reasons.append(f"Premium upgrade (+{price_diff:.1f}m)")

        # xGI comparison
        if candidate.get("xGI", 0) > player_out.expected_goal_involvements + 0.3:
            reasons.append(f"Higher xGI ({candidate['xGI']:.2f})")

        if not reasons:
            reasons.append(f"Expected improvement of {candidate['improvement']:.1f} pts")

        return "; ".join(reasons[:2])

    def _apply_transfer_limits(
        self, recommendations: list[TransferRecommendation]
    ) -> list[TransferRecommendation]:
        """
        Apply transfer limits and calculate hit costs.

        Only uses free transfers by default. Hits are only taken when:
        - Replacing injured/unavailable players (priority 1)
        - Exceptional gain that justifies the -4 cost
        """
        final = []
        transfers_used = 0
        players_in = set()  # Track players already being transferred in
        players_out = set()  # Track players already being transferred out
        # Track team counts including pending transfers
        pending_team_counts = dict(self.team_counts)

        for rec in recommendations:
            # Skip if this player is already being transferred in or out
            if rec.player_in.id in players_in:
                continue
            if rec.player_out.id in players_out:
                continue

            # Check if this transfer would violate team limits
            player_in_team = rec.player_in.team
            player_out_team = rec.player_out.team
            new_count = pending_team_counts.get(player_in_team, 0)
            if player_in_team != player_out_team:
                # Adding player from different team
                if new_count >= self.MAX_FROM_TEAM:
                    self.logger.debug(
                        f"Skipping {rec.player_in.web_name}: would exceed {self.MAX_FROM_TEAM} "
                        f"players from team {player_in_team}"
                    )
                    continue

            # Calculate hit cost
            if transfers_used < self.free_transfers:
                rec.hit_cost = 0
            else:
                rec.hit_cost = self.HIT_PENALTY

            rec.net_gain = rec.expected_gain + rec.hit_cost

            # Determine if transfer should be included
            is_injury_replacement = rec.priority == 1  # Priority 1 = injury/unavailable
            is_free_transfer = transfers_used < self.free_transfers

            def accept_transfer():
                """Accept the transfer and update tracking."""
                nonlocal transfers_used
                final.append(rec)
                transfers_used += 1
                players_in.add(rec.player_in.id)
                players_out.add(rec.player_out.id)
                # Update team counts
                if player_in_team != player_out_team:
                    pending_team_counts[player_in_team] = pending_team_counts.get(player_in_team, 0) + 1
                    pending_team_counts[player_out_team] = pending_team_counts.get(player_out_team, 1) - 1

            if is_free_transfer:
                # Free transfers: include if net gain is positive
                if rec.net_gain > 0:
                    accept_transfer()
            else:
                # Hit transfer: only for injuries or exceptional gains
                if is_injury_replacement and rec.net_gain > 0:
                    accept_transfer()
                elif rec.net_gain >= self.HIT_WORTHWHILE_THRESHOLD:
                    # Exceptional gain that justifies the hit
                    accept_transfer()

        return final


# =============================================================================
# Captain Selector
# =============================================================================

class CaptainSelector:
    """
    Selects optimal captain and vice-captain.

    Considers:
    - Home/away fixture
    - Fixture difficulty rating
    - Ownership percentage
    - Form and projected points
    - Historical captain points
    """

    def __init__(
        self,
        squad: list[Player],
        player_df: pd.DataFrame,
        fixtures_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize captain selector.

        Args:
            squad: Current squad (starting XI).
            player_df: DataFrame with player analysis.
            fixtures_df: Optional DataFrame with fixture details.
        """
        self.squad = squad
        self.player_df = player_df
        self.fixtures_df = fixtures_df
        self.logger = logging.getLogger("fpl_agent.optimizer.captain")

    def select(self) -> list[CaptainRecommendation]:
        """
        Rank players for captaincy.

        Returns:
            Sorted list of CaptainRecommendation objects.
        """
        recommendations = []

        for player in self.squad:
            row = self.player_df[self.player_df["player_id"] == player.id]
            if row.empty:
                continue

            row = row.iloc[0]

            # Skip unavailable players
            availability = float(row.get("availability", 1.0))
            if availability < 0.75:
                continue

            # Determine home/away
            is_home = self._is_home_fixture(row)

            # Get fixture difficulty
            fixture_score = float(row.get("fixture_difficulty", 3.0))

            # Calculate expected points
            expected_points = self._calculate_captain_score(player, row, is_home)

            # Calculate confidence
            confidence = self._calculate_confidence(row, is_home, fixture_score)

            # Get ownership
            ownership = float(row.get("selected_by", player.selected_by_percent))

            # Generate reason
            reason = self._generate_captain_reason(player, row, is_home, fixture_score)

            recommendations.append(
                CaptainRecommendation(
                    player=player,
                    expected_points=expected_points,
                    fixture_score=fixture_score,
                    confidence=confidence,
                    is_home=is_home,
                    ownership=ownership,
                    reason=reason,
                )
            )

        # Sort by expected points
        recommendations.sort(key=lambda x: -x.expected_points)

        return recommendations

    def _is_home_fixture(self, row: pd.Series) -> bool:
        """Determine if player has home fixture."""
        gw1_fixture = str(row.get("gw1_fixture", ""))
        return "(H)" in gw1_fixture

    def _calculate_captain_score(
        self, player: Player, row: pd.Series, is_home: bool
    ) -> float:
        """
        Calculate expected captain score.

        Returns a score close to expected points (doubled for captain).
        Primary factor is projected points from CSV, with small bonuses.
        """
        # Projected points is the primary factor (x2 for captain effect)
        projected = float(row.get("gw1_projected", 0) or 0)
        if projected > 0:
            base_score = projected * 2.0  # Captain doubles points
        else:
            # Fallback to form-based estimation
            base_score = float(row.get("form", 0)) * 2.0

        # Small fixture difficulty bonus (max +1.5 pts for easiest fixture)
        fixture = float(row.get("fixture_difficulty", 3.0))
        base_score += (5 - fixture) * 0.3

        # Small home advantage bonus (+5%)
        if is_home:
            base_score *= 1.05

        # Small form bonus (max +5%)
        if player.form >= 7.0:
            base_score *= 1.05
        elif player.form >= 5.0:
            base_score *= 1.02

        # Availability penalty
        availability = float(row.get("availability", 1.0))
        base_score *= availability

        return base_score

    def _calculate_confidence(
        self, row: pd.Series, is_home: bool, fixture_score: float
    ) -> float:
        """Calculate confidence score (0-1)."""
        confidence = 0.5  # Base confidence

        # Fixture difficulty factor
        confidence += (5 - fixture_score) * 0.1

        # Home advantage
        if is_home:
            confidence += 0.1

        # Form factor
        form = float(row.get("form", 0))
        if form >= 7.0:
            confidence += 0.15
        elif form >= 5.0:
            confidence += 0.1
        elif form < 3.0:
            confidence -= 0.15

        # Projected points available
        projected = float(row.get("gw1_projected", 0) or 0)
        if projected > 0:
            confidence += 0.1

        # Availability
        availability = float(row.get("availability", 1.0))
        confidence *= availability

        return max(0.0, min(1.0, confidence))

    def _generate_captain_reason(
        self,
        player: Player,
        row: pd.Series,
        is_home: bool,
        fixture_score: float,
    ) -> str:
        """Generate reason for captain choice."""
        reasons = []

        # Fixture
        if fixture_score <= 2:
            reasons.append("Easy fixture")
        elif fixture_score <= 2.5:
            reasons.append("Favorable fixture")

        # Home advantage
        if is_home:
            reasons.append("Home game")

        # Form
        if player.form >= 7.0:
            reasons.append("Excellent form")
        elif player.form >= 5.5:
            reasons.append("Good form")

        # Projected points
        projected = float(row.get("gw1_projected", 0) or 0)
        if projected >= 6.0:
            reasons.append(f"High projected ({projected:.1f} pts)")

        # Premium
        if player.now_cost >= 12.0:
            reasons.append("Premium asset")

        # Ownership (differential)
        ownership = float(row.get("selected_by", player.selected_by_percent))
        if ownership < 10.0:
            reasons.append(f"Differential ({ownership:.1f}%)")
        elif ownership > 40.0:
            reasons.append(f"Template pick ({ownership:.1f}%)")

        if not reasons:
            reasons.append("Consistent performer")

        return "; ".join(reasons[:3])


# =============================================================================
# Chip Strategy Advisor
# =============================================================================

class ChipStrategyAdvisor:
    """
    Advises on optimal chip usage timing.

    Detects:
    - Wildcard opportunities (injuries, fixture swings, DGW preparation)
    - Bench boost timing (double gameweeks)
    - Free hit opportunities (blank gameweeks)
    - Triple captain opportunities

    Proactive strategy:
    - If DGW coming in GW+1 and BB available, recommend WC now to prepare
    - On DGW, compare BB vs TC expected points and recommend the better one
    """

    # Thresholds for recommendations
    WILDCARD_THRESHOLD = 60  # Score out of 100 to recommend
    BENCH_BOOST_THRESHOLD = 70
    FREE_HIT_THRESHOLD = 50
    TRIPLE_CAPTAIN_THRESHOLD = 65

    def __init__(
        self,
        squad: list[Player],
        player_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        available_chips: list[str],
        free_transfers: int = 1,
        current_gameweek: int = 1,
    ):
        """
        Initialize chip strategy advisor.

        Args:
            squad: Current squad.
            player_df: Player analysis DataFrame.
            fixtures_df: Fixture ticker DataFrame.
            available_chips: List of available chip names.
            free_transfers: Current free transfers.
            current_gameweek: Current gameweek number.
        """
        self.squad = squad
        self.player_df = player_df
        self.fixtures_df = fixtures_df
        self.available_chips = available_chips
        self.free_transfers = free_transfers
        self.current_gameweek = current_gameweek
        self.logger = logging.getLogger("fpl_agent.optimizer.chips")

    def analyze(self, gameweeks_ahead: int = 5) -> list[ChipRecommendation]:
        """
        Analyze and recommend chip usage.

        Proactive strategy:
        - If DGW coming in GW+1 and BB available, boost WC recommendation
        - On DGW, compare BB vs TC expected points and recommend the better one
        - End-of-season urgency: force chip usage as GW38 approaches
        - If no DGWs available, compare chips purely on expected points

        Args:
            gameweeks_ahead: Number of gameweeks to analyze.

        Returns:
            List of ChipRecommendation objects.
        """
        recommendations = []

        # Detect DGW in current and future GWs
        dgw_this_gw = self._detect_double_gameweek()
        dgw_next_gw = self._detect_upcoming_dgw(1)  # Check GW+1
        gws_remaining = self._get_gameweeks_remaining()

        # Check for any upcoming DGWs (scan up to 5 GWs ahead)
        any_dgw_upcoming = dgw_this_gw > 0 or dgw_next_gw > 0
        for gw_ahead in range(2, min(6, gws_remaining + 1)):
            if self._detect_upcoming_dgw(gw_ahead) > 0:
                any_dgw_upcoming = True
                break

        # End-of-season urgency: chips must be used by GW38
        end_of_season_urgency = 0
        if gws_remaining <= 3:
            end_of_season_urgency = 30  # Very urgent
            self.logger.info(f"End-of-season urgency: {gws_remaining} GWs remaining")
        elif gws_remaining <= 5:
            end_of_season_urgency = 15  # Moderately urgent

        # Analyze each chip type
        if "wildcard" in self.available_chips:
            wc_rec = self._analyze_wildcard(dgw_next_gw=dgw_next_gw)
            if wc_rec:
                # Boost WC if end-of-season
                if end_of_season_urgency > 0:
                    wc_rec.score = min(100, wc_rec.score + end_of_season_urgency)
                    wc_rec.reasons.insert(0, f"Only {gws_remaining} GWs left - use WC")
                    wc_rec.is_recommended = wc_rec.score >= self.WILDCARD_THRESHOLD
                recommendations.append(wc_rec)

        # On DGW, compare BB vs TC and only recommend the better one
        bb_rec = None
        tc_rec = None

        if "bench_boost" in self.available_chips:
            bb_rec = self._analyze_bench_boost()

        if "triple_captain" in self.available_chips:
            tc_rec = self._analyze_triple_captain()

        # Calculate expected points for comparison
        bb_xpts = self._calculate_bb_expected_points() if bb_rec else 0
        tc_xpts = self._calculate_tc_expected_points() if tc_rec else 0

        # BB vs TC decision based on expected points
        if bb_rec and tc_rec:
            # Apply end-of-season urgency first
            if end_of_season_urgency > 0:
                bb_rec.score = min(100, bb_rec.score + end_of_season_urgency)
                bb_rec.reasons.insert(0, f"Only {gws_remaining} GWs left")
                tc_rec.score = min(100, tc_rec.score + end_of_season_urgency)
                tc_rec.reasons.insert(0, f"Only {gws_remaining} GWs left")

            # Compare chips on expected points (DGW or no DGW)
            if dgw_this_gw > 0 or not any_dgw_upcoming or gws_remaining <= 3:
                self.logger.info(
                    f"Chip comparison (GW{self.current_gameweek}): BB xPts={bb_xpts:.1f}, TC xPts={tc_xpts:.1f}"
                )

                if bb_xpts > tc_xpts:
                    # BB is better - boost BB score, reduce TC score
                    bb_rec.score = min(100, bb_rec.score + 15)
                    bb_rec.reasons.insert(0, f"BB ({bb_xpts:.1f} xPts) > TC ({tc_xpts:.1f} xPts)")
                    bb_rec.is_recommended = bb_rec.score >= self.BENCH_BOOST_THRESHOLD

                    if dgw_this_gw > 0 or gws_remaining <= 3:
                        tc_rec.score = max(0, tc_rec.score - 20)
                        tc_rec.reasons.insert(0, f"TC ({tc_xpts:.1f} xPts) < BB ({bb_xpts:.1f} xPts)")
                        tc_rec.is_recommended = False
                else:
                    # TC is better - boost TC score, reduce BB score
                    tc_rec.score = min(100, tc_rec.score + 15)
                    tc_rec.reasons.insert(0, f"TC ({tc_xpts:.1f} xPts) > BB ({bb_xpts:.1f} xPts)")
                    tc_rec.is_recommended = tc_rec.score >= self.TRIPLE_CAPTAIN_THRESHOLD

                    if dgw_this_gw > 0 or gws_remaining <= 3:
                        bb_rec.score = max(0, bb_rec.score - 20)
                        bb_rec.reasons.insert(0, f"BB ({bb_xpts:.1f} xPts) < TC ({tc_xpts:.1f} xPts)")
                        bb_rec.is_recommended = False

        if bb_rec:
            recommendations.append(bb_rec)
        if tc_rec:
            recommendations.append(tc_rec)

        if "free_hit" in self.available_chips:
            fh_rec = self._analyze_free_hit()
            if fh_rec:
                # Boost FH if end-of-season
                if end_of_season_urgency > 0:
                    fh_rec.score = min(100, fh_rec.score + end_of_season_urgency)
                    fh_rec.reasons.insert(0, f"Only {gws_remaining} GWs left - use FH")
                    fh_rec.is_recommended = fh_rec.score >= self.FREE_HIT_THRESHOLD
                recommendations.append(fh_rec)

        # Sort by score
        recommendations.sort(key=lambda x: -x.score)

        return recommendations

    def _analyze_wildcard(self, dgw_next_gw: int = 0) -> Optional[ChipRecommendation]:
        """
        Analyze wildcard opportunity.

        Proactive strategy: If DGW coming in GW+1 and BB available,
        recommend WC now to maximize DGW players for BB.

        Args:
            dgw_next_gw: Number of teams with DGW in GW+1.
        """
        score = 0.0
        reasons = []

        # PROACTIVE FACTOR: DGW coming next week and BB available
        if dgw_next_gw >= 4 and "bench_boost" in self.available_chips:
            score += 40
            reasons.append(f"DGW in GW+1 ({dgw_next_gw} teams) - prepare for BB")
        elif dgw_next_gw >= 2 and "bench_boost" in self.available_chips:
            score += 20
            reasons.append(f"Partial DGW in GW+1 ({dgw_next_gw} teams)")

        # Factor 1: Squad injuries
        injured_count = self._count_injured_players()
        if injured_count >= 4:
            score += 30
            reasons.append(f"{injured_count} injured/doubtful players")
        elif injured_count >= 2:
            score += 15
            reasons.append(f"{injured_count} players with concerns")

        # Factor 2: Fixture swing
        fixture_score = self._calculate_fixture_swing()
        if fixture_score > 20:
            score += 25
            reasons.append("Major fixture swing opportunity")
        elif fixture_score > 10:
            score += 15
            reasons.append("Moderate fixture swing")

        # Factor 3: Squad value (underperforming assets)
        underperformers = self._count_underperformers()
        if underperformers >= 5:
            score += 25
            reasons.append(f"{underperformers} underperforming assets")
        elif underperformers >= 3:
            score += 15
            reasons.append(f"{underperformers} transfers needed")

        # Factor 4: Free transfer bank
        if self.free_transfers <= 1:
            score += 10
            reasons.append("Low free transfers")

        is_recommended = score >= self.WILDCARD_THRESHOLD

        return ChipRecommendation(
            chip_name="wildcard",
            gameweek=self.current_gameweek,
            score=min(100, score),
            reasons=reasons,
            is_recommended=is_recommended,
        )

    def _analyze_bench_boost(self) -> Optional[ChipRecommendation]:
        """Analyze bench boost opportunity."""
        score = 0.0
        reasons = []

        # Factor 1: Double gameweek detection
        dgw_teams = self._detect_double_gameweek()
        if dgw_teams >= 4:
            score += 40
            reasons.append(f"Double GW with {dgw_teams} teams")
        elif dgw_teams >= 2:
            score += 20
            reasons.append(f"Partial DGW ({dgw_teams} teams)")

        # Factor 2: Bench strength
        bench_score = self._calculate_bench_strength()
        if bench_score >= 20:
            score += 30
            reasons.append("Strong bench players")
        elif bench_score >= 12:
            score += 15
            reasons.append("Decent bench")

        # Factor 3: Bench fixtures
        bench_fixtures = self._calculate_bench_fixtures()
        if bench_fixtures <= 2.5:
            score += 20
            reasons.append("Favorable bench fixtures")
        elif bench_fixtures <= 3.0:
            score += 10
            reasons.append("Okay bench fixtures")

        # Factor 4: Squad DGW coverage (how many of your 15 have DGW)
        squad_dgw_count = self._count_squad_dgw_players()
        if squad_dgw_count >= 10:
            score += 25
            reasons.append(f"{squad_dgw_count} squad players with DGW")
        elif squad_dgw_count >= 6:
            score += 15
            reasons.append(f"{squad_dgw_count} players doubling")

        is_recommended = score >= self.BENCH_BOOST_THRESHOLD

        return ChipRecommendation(
            chip_name="bench_boost",
            gameweek=self.current_gameweek,
            score=min(100, score),
            reasons=reasons,
            is_recommended=is_recommended,
        )

    def _analyze_free_hit(self) -> Optional[ChipRecommendation]:
        """Analyze free hit opportunity."""
        score = 0.0
        reasons = []

        # Factor 1: Blank gameweek (many teams not playing)
        blanks = self._detect_blank_gameweek()
        if blanks >= 6:
            score += 50
            reasons.append(f"Blank GW - {blanks} teams not playing")
        elif blanks >= 3:
            score += 25
            reasons.append(f"Partial blank ({blanks} teams)")

        # Factor 2: Squad players affected
        squad_blanks = self._count_squad_blanks()
        if squad_blanks >= 5:
            score += 30
            reasons.append(f"{squad_blanks} squad players blanking")
        elif squad_blanks >= 3:
            score += 15
            reasons.append(f"{squad_blanks} players affected")

        # Factor 3: Alternative attractive fixtures
        attractive = self._count_attractive_fixtures()
        if attractive >= 5:
            score += 20
            reasons.append("Many attractive alternatives")

        is_recommended = score >= self.FREE_HIT_THRESHOLD

        return ChipRecommendation(
            chip_name="free_hit",
            gameweek=self.current_gameweek,
            score=min(100, score),
            reasons=reasons,
            is_recommended=is_recommended,
        )

    def _analyze_triple_captain(self) -> Optional[ChipRecommendation]:
        """Analyze triple captain opportunity."""
        score = 0.0
        reasons = []

        # Factor 1: DGW premium player
        premium_dgw = self._has_premium_dgw_player()
        if premium_dgw:
            score += 40
            reasons.append("Premium player with DGW")

        # Factor 2: Easy fixture for premium
        premium_fixture = self._get_premium_fixture_score()
        if premium_fixture <= 2.0:
            score += 30
            reasons.append("Very easy premium fixture")
        elif premium_fixture <= 2.5:
            score += 20
            reasons.append("Favorable premium fixture")

        # Factor 3: Premium form
        premium_form = self._get_premium_form()
        if premium_form >= 7.0:
            score += 25
            reasons.append("Premium in excellent form")
        elif premium_form >= 5.5:
            score += 15
            reasons.append("Premium in good form")

        is_recommended = score >= self.TRIPLE_CAPTAIN_THRESHOLD

        return ChipRecommendation(
            chip_name="triple_captain",
            gameweek=self.current_gameweek,
            score=min(100, score),
            reasons=reasons,
            is_recommended=is_recommended,
        )

    # Helper methods for chip analysis

    def _count_injured_players(self) -> int:
        """Count injured/doubtful players in squad."""
        count = 0
        for player in self.squad:
            row = self.player_df[self.player_df["player_id"] == player.id]
            if not row.empty:
                availability = float(row.iloc[0].get("availability", 1.0))
                if availability < 0.75:
                    count += 1
        return count

    def _calculate_fixture_swing(self) -> float:
        """Calculate fixture difficulty swing score."""
        if self.fixtures_df is None or self.fixtures_df.empty:
            return 0.0

        # Compare current squad's fixtures vs best available
        squad_teams = {p.team for p in self.squad}
        squad_fixtures = self.fixtures_df[
            self.fixtures_df["team_id"].isin(squad_teams)
        ]
        if squad_fixtures.empty:
            return 0.0

        squad_avg = squad_fixtures["avg_difficulty"].mean()
        best_avg = self.fixtures_df["avg_difficulty"].min()

        return (squad_avg - best_avg) * 10

    def _count_underperformers(self) -> int:
        """Count underperforming squad players."""
        count = 0
        for player in self.squad:
            row = self.player_df[self.player_df["player_id"] == player.id]
            if not row.empty:
                row = row.iloc[0]
                form = float(row.get("form", 0))
                value = float(row.get("value", 0))
                if form < 3.5 or value < 3.0:
                    count += 1
        return count

    def _detect_double_gameweek(self) -> int:
        """Detect teams with double gameweek fixtures."""
        if self.fixtures_df is None or self.fixtures_df.empty:
            return 0

        # Check for next_gw_dgw column (added by get_fixture_ticker)
        if "next_gw_dgw" in self.fixtures_df.columns:
            return int(self.fixtures_df["next_gw_dgw"].sum())

        # Fallback: check for " + " in fixture strings (indicates DGW)
        dgw_count = 0
        for col in self.fixtures_df.columns:
            if col.startswith("gw") and not col.endswith("_diff") and not col.endswith("_dgw"):
                dgw_count += self.fixtures_df[col].str.contains(r"\+", na=False).sum()
                break  # Only check first GW column

        return dgw_count

    def _calculate_bench_strength(self) -> float:
        """Calculate total expected points from bench."""
        if len(self.squad) < 15:
            return 0.0

        bench = self.squad[11:15]  # Last 4 players
        total = 0.0

        for player in bench:
            row = self.player_df[self.player_df["player_id"] == player.id]
            if not row.empty:
                total += float(row.iloc[0].get("gw1_projected", 0) or row.iloc[0].get("form", 0))

        return total

    def _calculate_bench_fixtures(self) -> float:
        """Calculate average fixture difficulty for bench."""
        if len(self.squad) < 15:
            return 3.0

        bench = self.squad[11:15]
        difficulties = []

        for player in bench:
            row = self.player_df[self.player_df["player_id"] == player.id]
            if not row.empty:
                difficulties.append(float(row.iloc[0].get("fixture_difficulty", 3.0)))

        return sum(difficulties) / len(difficulties) if difficulties else 3.0

    def _detect_blank_gameweek(self) -> int:
        """Detect number of teams with blank gameweek."""
        if self.fixtures_df is None:
            return 0
        # Check for teams with no fixture (BGW)
        bgw_count = 0
        for _, row in self.fixtures_df.iterrows():
            if row.get("gw1_fixture", "") == "BGW":
                bgw_count += 1
        return bgw_count

    def _count_squad_blanks(self) -> int:
        """Count squad players with blank gameweek."""
        count = 0
        for player in self.squad:
            row = self.player_df[self.player_df["player_id"] == player.id]
            if not row.empty:
                fixture = str(row.iloc[0].get("gw1_fixture", ""))
                if fixture == "BGW" or not fixture:
                    count += 1
        return count

    def _count_attractive_fixtures(self) -> int:
        """Count teams with very attractive fixtures."""
        if self.fixtures_df is None:
            return 0
        return len(self.fixtures_df[self.fixtures_df["avg_difficulty"] <= 2.5])

    def _count_squad_dgw_players(self) -> int:
        """Count squad players with DGW fixtures."""
        if self.fixtures_df is None or self.fixtures_df.empty:
            return 0

        # Get teams with DGW
        dgw_teams = set()
        if "next_gw_dgw" in self.fixtures_df.columns:
            dgw_teams = set(
                self.fixtures_df[self.fixtures_df["next_gw_dgw"] == True]["team_id"].values
            )

        if not dgw_teams:
            return 0

        return sum(1 for player in self.squad if player.team in dgw_teams)

    def _has_premium_dgw_player(self) -> bool:
        """Check if any premium player (>= 10m) has DGW."""
        if self.fixtures_df is None or self.fixtures_df.empty:
            return False

        # Get teams with DGW
        dgw_teams = set()
        if "next_gw_dgw" in self.fixtures_df.columns:
            dgw_teams = set(
                self.fixtures_df[self.fixtures_df["next_gw_dgw"] == True]["team_id"].values
            )

        if not dgw_teams:
            return False

        # Check if any premium player (>= 10m) is from a DGW team
        for player in self.squad:
            if player.now_cost >= 10.0 and player.team in dgw_teams:
                return True

        return False

    def _get_premium_fixture_score(self) -> float:
        """Get fixture difficulty for best premium player."""
        premiums = [p for p in self.squad if p.now_cost >= 10.0]
        if not premiums:
            return 3.0

        best_fixture = 5.0
        for player in premiums:
            row = self.player_df[self.player_df["player_id"] == player.id]
            if not row.empty:
                fixture = float(row.iloc[0].get("fixture_difficulty", 3.0))
                best_fixture = min(best_fixture, fixture)

        return best_fixture

    def _get_premium_form(self) -> float:
        """Get form of best premium player."""
        premiums = [p for p in self.squad if p.now_cost >= 10.0]
        if not premiums:
            return 0.0

        return max(p.form for p in premiums)

    def _detect_upcoming_dgw(self, gameweeks_ahead: int = 1) -> int:
        """
        Detect teams with DGW in future gameweeks.

        Args:
            gameweeks_ahead: How many GWs ahead to check (1 = next GW).

        Returns:
            Number of teams with DGW in that gameweek.
        """
        if self.fixtures_df is None or self.fixtures_df.empty:
            return 0

        target_gw = self.current_gameweek + gameweeks_ahead
        dgw_col = f"gw{target_gw}_dgw"

        if dgw_col in self.fixtures_df.columns:
            return int(self.fixtures_df[dgw_col].sum())

        return 0

    def _calculate_bb_expected_points(self) -> float:
        """
        Calculate expected CHIP VALUE if Bench Boost is played.

        BB chip value = sum of bench 4 players' xPts (extra points beyond normal 11).

        Returns:
            Sum of bench 4 players' projected points.
        """
        # Get xPts for all players
        player_xpts = []
        for player in self.squad:
            row = self.player_df[self.player_df["player_id"] == player.id]
            if not row.empty:
                row = row.iloc[0]
                xpts = float(row.get("gw1_projected", 0) or 0)
                # Fallback to form if no projection
                if xpts == 0:
                    xpts = float(row.get("form", 0) or 0)
                player_xpts.append((player, xpts))

        # Sort by xPts to identify bench (lowest 4)
        player_xpts.sort(key=lambda x: x[1], reverse=True)

        # BB value = sum of bench 4 players' xPts
        if len(player_xpts) >= 15:
            bench_xpts = sum(xpts for _, xpts in player_xpts[11:15])
            return bench_xpts

        return 0.0

    def _calculate_tc_expected_points(self) -> float:
        """
        Calculate expected CHIP VALUE if Triple Captain is played.

        TC chip value = best captain's xPts * 1 (extra points beyond normal captain 2x).

        Returns:
            Best captain's projected points (the extra from TC).
        """
        best_captain_xpts = 0.0

        for player in self.squad:
            row = self.player_df[self.player_df["player_id"] == player.id]
            if not row.empty:
                row = row.iloc[0]
                xpts = float(row.get("gw1_projected", 0) or 0)
                # Fallback to form if no projection
                if xpts == 0:
                    xpts = float(row.get("form", 0) or 0)

                # Only consider available players
                availability = float(row.get("availability", 1.0))
                if availability >= 0.75 and xpts > best_captain_xpts:
                    best_captain_xpts = xpts

        # TC gives captain * 3 total, normal captain gives captain * 2
        # Chip value = extra points = captain_xpts * 1
        return best_captain_xpts

    def _get_gameweeks_remaining(self) -> int:
        """Get number of gameweeks remaining in the season."""
        return max(0, 38 - self.current_gameweek)


# =============================================================================
# Main Optimizer Class
# =============================================================================

class Optimizer:
    """
    Main optimizer combining transfer, captain, and chip strategies.

    Usage:
        optimizer = Optimizer(config, client, collector)
        result = await optimizer.optimize()
        print(optimizer.format_recommendations(result))
    """

    # Squad constraints
    SQUAD_SIZE = 15
    MAX_FROM_TEAM = 3
    POSITION_LIMITS = {
        Position.GOALKEEPER: (2, 2),
        Position.DEFENDER: (5, 5),
        Position.MIDFIELDER: (5, 5),
        Position.FORWARD: (3, 3),
    }

    def __init__(self, config: Config, client: FPLClient, collector: DataCollector):
        """
        Initialize the optimizer.

        Args:
            config: Application configuration.
            client: FPL API client.
            collector: Data collector instance.
        """
        self.config = config
        self.client = client
        self.collector = collector
        self.logger = logging.getLogger("fpl_agent.optimizer")

    async def optimize(
        self,
        include_chips: bool = True,
        available_chips: Optional[list[str]] = None,
    ) -> OptimizationResult:
        """
        Run full optimization for transfers, captain, and chips.

        Args:
            include_chips: Whether to include chip recommendations.
            available_chips: List of available chips (defaults to all).

        Returns:
            OptimizationResult with all recommendations.
        """
        self.logger.info("Starting optimization process")

        result = OptimizationResult()

        try:
            # Get current team data
            my_team = await self.client.get_my_team()
            players = await self.client.get_players()
            player_map = {p.id: p for p in players}

            # Get current squad from MyTeam dataclass
            current_squad = [
                player_map[pick.element]
                for pick in my_team.picks
                if pick.element in player_map
            ]

            # Get transfer info from MyTeam attributes
            bank = my_team.bank
            free_transfers = my_team.free_transfers

            self.logger.info(f"Bank: {bank:.1f}m, Free transfers: {free_transfers}")

            # Get player DataFrame
            player_df = self.collector.get_players_dataframe()
            if player_df is None:
                player_df = await self.collector.collect_all()

            # 1. Optimize Transfers
            # max_transfers allows buffer for injury replacements, but _apply_transfer_limits
            # will only use free transfers unless there's injury or exceptional gain
            transfer_optimizer = TransferOptimizer(
                current_squad=current_squad,
                player_df=player_df,
                budget=bank,
                free_transfers=free_transfers,
                max_transfers=free_transfers + 2,  # Buffer for injury hits
            )
            result.transfers = transfer_optimizer.optimize()

            # Calculate totals
            result.total_expected_gain = sum(t.expected_gain for t in result.transfers)
            result.total_hit_cost = sum(t.hit_cost for t in result.transfers)
            result.net_expected_gain = sum(t.net_gain for t in result.transfers)

            # Build post-transfer squad
            transferred_in_ids = set()
            post_transfer_squad = list(current_squad)
            for transfer in result.transfers:
                # Remove outgoing player
                post_transfer_squad = [p for p in post_transfer_squad if p.id != transfer.player_out.id]
                # Add incoming player
                post_transfer_squad.append(transfer.player_in)
                transferred_in_ids.add(transfer.player_in.id)

            # 2. Select Starting XI and Bench Order
            result.starting_xi, result.bench_order = self.select_starting_xi_and_bench(
                squad=post_transfer_squad,
                player_df=player_df,
                transferred_in_ids=transferred_in_ids,
                gameweek=1,
            )

            # Get starting XI players for captain selection
            starting_xi_players = [p for p in post_transfer_squad if p.id in result.starting_xi]
            captain_selector = CaptainSelector(starting_xi_players, player_df)
            captain_recs = captain_selector.select()

            if captain_recs:
                result.captain = captain_recs[0]
                if len(captain_recs) > 1:
                    result.vice_captain = captain_recs[1]

                # Find best differential captain
                differentials = [c for c in captain_recs if c.is_differential]
                if differentials:
                    result.differential_captain = differentials[0]

            # 3. Chip Strategy
            if include_chips:
                if available_chips is None:
                    available_chips = ["wildcard", "bench_boost", "free_hit", "triple_captain"]

                fixtures_df = self.collector.get_fixture_ticker()

                # Get current gameweek
                current_gw = 1
                try:
                    gameweeks = await self.client.get_gameweeks()
                    for gw in gameweeks:
                        if gw.is_current:
                            current_gw = gw.id
                            break
                except Exception:
                    pass  # Default to GW1 if can't fetch

                chip_advisor = ChipStrategyAdvisor(
                    squad=current_squad,
                    player_df=player_df,
                    fixtures_df=fixtures_df,
                    available_chips=available_chips,
                    free_transfers=free_transfers,
                    current_gameweek=current_gw,
                )
                result.chip_recommendations = chip_advisor.analyze()

            self.logger.info(
                f"Optimization complete: {len(result.transfers)} transfers, "
                f"{len(result.chip_recommendations)} chip suggestions"
            )

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            result.warnings.append(f"Optimization error: {str(e)}")

        return result

    def optimize_transfers_offline(
        self,
        current_squad: list[Player],
        player_df: pd.DataFrame,
        budget: float,
        free_transfers: int,
        max_transfers: int = 2,
    ) -> list[TransferRecommendation]:
        """
        Optimize transfers without API calls (for testing).

        Args:
            current_squad: Current squad players.
            player_df: Player DataFrame.
            budget: Available budget.
            free_transfers: Free transfers available.
            max_transfers: Maximum transfers to make.

        Returns:
            List of TransferRecommendation objects.
        """
        optimizer = TransferOptimizer(
            current_squad=current_squad,
            player_df=player_df,
            budget=budget,
            free_transfers=free_transfers,
            max_transfers=max_transfers,
        )
        return optimizer.optimize()

    def select_captain_offline(
        self,
        squad: list[Player],
        player_df: pd.DataFrame,
    ) -> list[CaptainRecommendation]:
        """
        Select captain without API calls (for testing).

        Args:
            squad: Starting XI.
            player_df: Player DataFrame.

        Returns:
            List of CaptainRecommendation objects.
        """
        selector = CaptainSelector(squad, player_df)
        return selector.select()

    def validate_squad(self, squad: list[Player]) -> list[str]:
        """
        Validate squad meets FPL rules.

        Args:
            squad: List of players in squad.

        Returns:
            List of validation error messages.
        """
        errors = []

        # Check squad size
        if len(squad) != self.SQUAD_SIZE:
            errors.append(f"Squad must have {self.SQUAD_SIZE} players, has {len(squad)}")

        # Check position limits
        position_counts = {}
        for player in squad:
            pos = Position(player.element_type)
            position_counts[pos] = position_counts.get(pos, 0) + 1

        for position, (min_count, max_count) in self.POSITION_LIMITS.items():
            count = position_counts.get(position, 0)
            if count < min_count:
                errors.append(f"Need at least {min_count} {position.name}s, have {count}")
            if count > max_count:
                errors.append(f"Maximum {max_count} {position.name}s allowed, have {count}")

        # Check team limits
        team_counts = {}
        for player in squad:
            team_counts[player.team] = team_counts.get(player.team, 0) + 1
            if team_counts[player.team] > self.MAX_FROM_TEAM:
                errors.append(f"Maximum {self.MAX_FROM_TEAM} players from same team")

        return errors

    def select_starting_xi_and_bench(
        self,
        squad: list[Player],
        player_df: pd.DataFrame,
        transferred_in_ids: Optional[set[int]] = None,
        gameweek: int = 1,
    ) -> tuple[list[int], BenchOrder]:
        """
        Select optimal starting XI and bench order based on xPts.

        Args:
            squad: Full 15-player squad.
            player_df: Player DataFrame with projections.
            transferred_in_ids: Set of player IDs that were transferred in (must start).
            gameweek: Gameweek number for xPts lookup.

        Returns:
            Tuple of (starting_xi player IDs, BenchOrder).
        """
        if transferred_in_ids is None:
            transferred_in_ids = set()

        # Get xPts for each player
        player_xpts = {}
        for player in squad:
            row = player_df[player_df["player_id"] == player.id]
            if not row.empty:
                row = row.iloc[0]
                xpts_col = f"gw{gameweek}_projected"
                xpts = float(row.get(xpts_col, 0) or row.get("gw1_projected", 0) or 0)
                # Fallback to form if no projection
                if xpts == 0:
                    xpts = float(row.get("form", 0) or 0)
                player_xpts[player.id] = xpts
            else:
                player_xpts[player.id] = 0.0

        # Group players by position
        by_position = {1: [], 2: [], 3: [], 4: []}  # GK, DEF, MID, FWD
        for player in squad:
            by_position[player.element_type].append(player)

        # Sort each position by xPts (highest first)
        for pos in by_position:
            by_position[pos].sort(key=lambda p: player_xpts[p.id], reverse=True)

        # Find best valid formation that includes all transferred-in players
        best_xi = None
        best_xi_xpts = -1

        for formation in VALID_FORMATIONS:
            gk_count, def_count, mid_count, fwd_count = formation

            # Select top N from each position
            xi_gks = by_position[1][:gk_count]
            xi_defs = by_position[2][:def_count]
            xi_mids = by_position[3][:mid_count]
            xi_fwds = by_position[4][:fwd_count]

            xi_ids = set(p.id for p in xi_gks + xi_defs + xi_mids + xi_fwds)

            # Check if all transferred-in players are in starting XI
            if not transferred_in_ids.issubset(xi_ids):
                # Try to include transferred-in players by swapping
                xi_candidates = xi_gks + xi_defs + xi_mids + xi_fwds
                missing = transferred_in_ids - xi_ids

                # For each missing transferred player, swap with lowest xPts player of same position
                can_include_all = True
                for missing_id in missing:
                    missing_player = next((p for p in squad if p.id == missing_id), None)
                    if missing_player is None:
                        can_include_all = False
                        break

                    pos = missing_player.element_type
                    pos_players_in_xi = [p for p in xi_candidates if p.element_type == pos]
                    pos_players_not_in_xi = [p for p in by_position[pos] if p not in pos_players_in_xi]

                    # Find if we can swap
                    if missing_player in pos_players_not_in_xi:
                        # Swap with lowest xPts player of same position
                        lowest = min(pos_players_in_xi, key=lambda p: player_xpts[p.id])
                        if lowest.id not in transferred_in_ids:
                            xi_candidates.remove(lowest)
                            xi_candidates.append(missing_player)
                        else:
                            can_include_all = False
                            break
                    else:
                        can_include_all = False
                        break

                if not can_include_all:
                    continue

                # Verify formation is still valid
                pos_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                for p in xi_candidates:
                    pos_counts[p.element_type] += 1

                if (pos_counts[1], pos_counts[2], pos_counts[3], pos_counts[4]) != formation:
                    continue

                xi_ids = set(p.id for p in xi_candidates)

            # Check max 3 players per team constraint
            xi_players = [p for p in squad if p.id in xi_ids]
            team_counts = {}
            for p in xi_players:
                team_counts[p.team] = team_counts.get(p.team, 0) + 1
            if any(count > self.MAX_FROM_TEAM for count in team_counts.values()):
                continue  # Skip this formation - violates team limit

            # Calculate total xPts for this XI
            total_xpts = sum(player_xpts[pid] for pid in xi_ids)

            if total_xpts > best_xi_xpts:
                best_xi_xpts = total_xpts
                best_xi = list(xi_ids)

        # Fallback: if no valid formation found, use first 11
        if best_xi is None:
            self.logger.warning("Could not find valid formation including all transfers, using default")
            best_xi = [p.id for p in squad[:11]]

        # Validate final XI doesn't violate team limit
        final_xi_players = [p for p in squad if p.id in best_xi]
        final_team_counts = {}
        for p in final_xi_players:
            final_team_counts[p.team] = final_team_counts.get(p.team, 0) + 1
        for team_id, count in final_team_counts.items():
            if count > self.MAX_FROM_TEAM:
                self.logger.error(
                    f"Starting XI violates team limit: {count} players from team {team_id} "
                    f"(max {self.MAX_FROM_TEAM}). Squad may have invalid composition."
                )

        # Build bench (remaining players not in starting XI)
        bench_players = [p for p in squad if p.id not in best_xi]

        # Sort bench: GK first, then outfield by xPts descending
        bench_gk = next((p for p in bench_players if p.element_type == 1), None)
        bench_outfield = [p for p in bench_players if p.element_type != 1]
        bench_outfield.sort(key=lambda p: player_xpts[p.id], reverse=True)

        # Create BenchOrder
        bench_order = BenchOrder(
            bench_gk=bench_gk.id if bench_gk else None,
            bench_1=bench_outfield[0].id if len(bench_outfield) > 0 else None,
            bench_2=bench_outfield[1].id if len(bench_outfield) > 1 else None,
            bench_3=bench_outfield[2].id if len(bench_outfield) > 2 else None,
        )

        return best_xi, bench_order

    def validate_formation(self, starting_xi: list[Player]) -> tuple[bool, str]:
        """
        Validate starting XI has valid formation.

        Args:
            starting_xi: List of 11 starting players.

        Returns:
            Tuple of (is_valid, formation_string).
        """
        if len(starting_xi) != 11:
            return False, "Invalid XI size"

        pos_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for player in starting_xi:
            pos_counts[player.element_type] = pos_counts.get(player.element_type, 0) + 1

        formation = (pos_counts[1], pos_counts[2], pos_counts[3], pos_counts[4])

        if formation in VALID_FORMATIONS:
            return True, f"{formation[1]}-{formation[2]}-{formation[3]}"

        return False, f"Invalid: {formation[1]}-{formation[2]}-{formation[3]}"

    def format_recommendations(
        self,
        result: OptimizationResult,
        player_map: Optional[dict[int, Player]] = None,
    ) -> str:
        """
        Format optimization results for display.

        Args:
            result: Optimization result to format.
            player_map: Optional mapping of player ID to Player for name lookup.

        Returns:
            Formatted string.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("FPL AGENT RECOMMENDATIONS")
        lines.append("=" * 70)

        # Transfers
        lines.append("\n[TRANSFERS] TRANSFER RECOMMENDATIONS:")
        lines.append("-" * 70)
        if result.transfers:
            for i, transfer in enumerate(result.transfers, 1):
                lines.append(f"  {i}. {transfer}")

            lines.append("")
            lines.append(f"  Expected gain: {result.total_expected_gain:+.2f} pts")
            if result.total_hit_cost < 0:
                lines.append(f"  Hit cost: {result.total_hit_cost} pts")
            lines.append(f"  Net expected gain: {result.net_expected_gain:+.2f} pts")
        else:
            lines.append("  No transfers recommended - squad is strong")

        # Starting XI and Bench
        if result.starting_xi or result.bench_order:
            lines.append("\n[LINEUP] STARTING XI & BENCH ORDER:")
            lines.append("-" * 70)

            # Helper to get player name
            def get_name(player_id: int) -> str:
                if player_map and player_id in player_map:
                    return player_map[player_id].web_name
                # Check if player is in transfers
                for t in result.transfers:
                    if t.player_in.id == player_id:
                        return t.player_in.web_name
                return f"ID:{player_id}"

            if result.starting_xi:
                xi_names = [get_name(pid) for pid in result.starting_xi]
                lines.append(f"  Starting XI: {', '.join(xi_names)}")

            if result.bench_order:
                bench_items = []
                if result.bench_order.bench_gk:
                    bench_items.append(f"GK: {get_name(result.bench_order.bench_gk)}")
                if result.bench_order.bench_1:
                    bench_items.append(f"1st: {get_name(result.bench_order.bench_1)}")
                if result.bench_order.bench_2:
                    bench_items.append(f"2nd: {get_name(result.bench_order.bench_2)}")
                if result.bench_order.bench_3:
                    bench_items.append(f"3rd: {get_name(result.bench_order.bench_3)}")
                lines.append(f"  Bench Order: {', '.join(bench_items)}")

        # Captain
        lines.append("\n[CAPTAIN] CAPTAIN RECOMMENDATION:")
        lines.append("-" * 70)
        if result.captain:
            lines.append(f"  Captain: {result.captain}")
        if result.vice_captain:
            lines.append(f"  Vice-Captain: {result.vice_captain}")
        if result.differential_captain:
            lines.append(f"  Differential: {result.differential_captain}")

        # Chip Strategy
        if result.chip_recommendations:
            lines.append("\n[CHIPS] CHIP STRATEGY:")
            lines.append("-" * 70)
            for chip in result.chip_recommendations:
                status = ">>> PLAY" if chip.is_recommended else "   Hold"
                lines.append(f"  {status} {chip.chip_name.upper()}: Score {chip.score:.0f}/100")
                for reason in chip.reasons[:2]:
                    lines.append(f"         - {reason}")

        # Warnings
        if result.warnings:
            lines.append("\n[!] WARNINGS:")
            lines.append("-" * 70)
            for warning in result.warnings:
                lines.append(f"  - {warning}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)
