# data/processor.py
# Data cleaning, streak calculations, and stat formatting.
# The dashboard and newsletter import from here — never from fetcher directly
# for anything that requires computation.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

from data.fetcher import (
    get_batting_leaderboard,
    get_pitching_leaderboard,
    get_standings,
    get_todays_schedule,
    get_recent_hitting_stats,
    get_recent_pitching_stats,
    get_season_hitting_stats,
    get_season_pitching_stats,
)


# ---------------------------------------------------------------------------
# Utility: innings pitched conversion
# ---------------------------------------------------------------------------

def innings_to_float(ip) -> float:
    """
    Convert innings pitched from MLB's "6.2" notation to a true float.

    MLB records partial innings as tenths: "6.2" means 6 and 2/3 innings,
    not 6.2. This converts that to 6.667 so ERA math works correctly.
    """
    try:
        ip = float(ip)
        full = int(ip)
        partial = round(ip - full, 1)
        # .1 = 1 out, .2 = 2 outs — convert to thirds
        return full + (partial * 10 / 3)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Utility: stat formatting
# ---------------------------------------------------------------------------

def format_avg(value) -> str:
    """Format a batting average / OBP / SLG as a 3-decimal string: '.315'"""
    try:
        f = float(value)
        return f"{f:.3f}".lstrip("0") or ".000"
    except (ValueError, TypeError):
        return ".000"


def format_era(value) -> str:
    """Format an ERA or WHIP as a 2-decimal string: '3.14'"""
    try:
        return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return "-.--"


def format_ops(value) -> str:
    """Format OPS as a 3-decimal string: '.925'"""
    return format_avg(value)


def trend_arrow(current, baseline) -> str:
    """
    Return a visual trend arrow comparing current value to a baseline.
    Uses a 5% threshold so minor noise doesn't show an arrow.
    """
    try:
        c, b = float(current), float(baseline)
        if b == 0:
            return "→"
        delta_pct = (c - b) / abs(b)
        if delta_pct > 0.05:
            return "↑"
        elif delta_pct < -0.05:
            return "↓"
        return "→"
    except (ValueError, TypeError):
        return "→"


# ---------------------------------------------------------------------------
# Utility: safe OPS from string columns
# ---------------------------------------------------------------------------

def _to_float_series(series: pd.Series) -> pd.Series:
    """Convert a Series of stat strings (e.g. '.315') to floats, coercing errors to 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


# ---------------------------------------------------------------------------
# Rolling trend charts (for Page 2 — individual player)
# ---------------------------------------------------------------------------

def calculate_rolling_ops(game_log_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Compute a rolling OPS trend from a player's hitting game log.

    Uses an at-bat-weighted rolling window so that 0-AB games (walks-only
    appearances) don't distort the line.

    Args:
        game_log_df: DataFrame from get_player_hitting_game_log()
        window:      Number of days to include in the rolling window

    Returns a DataFrame with columns: date, rolling_ops
    """
    if game_log_df.empty:
        return pd.DataFrame(columns=["date", "rolling_ops"])

    df = game_log_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    df["ops_float"] = _to_float_series(df["ops"])
    df["ab_float"] = _to_float_series(df["at_bats"])

    # Weight OPS by at-bats so multi-AB games count more than 1-AB games
    df["ops_x_ab"] = df["ops_float"] * df["ab_float"]

    # Rolling sum over the window, then divide to get weighted average OPS
    rolling_ab = df["ab_float"].rolling(window=window, min_periods=1).sum()
    rolling_ops_x_ab = df["ops_x_ab"].rolling(window=window, min_periods=1).sum()

    df["rolling_ops"] = (rolling_ops_x_ab / rolling_ab.replace(0, np.nan)).round(3)

    return df[["date", "rolling_ops"]].dropna()


def calculate_rolling_era(game_log_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Compute a rolling ERA trend from a player's pitching game log.

    ERA is recalculated from raw components (earned runs / innings pitched)
    over the rolling window rather than averaging per-game ERA values, which
    would be misleading for short outings.

    Args:
        game_log_df: DataFrame from get_player_pitching_game_log()
        window:      Number of starts to include in the rolling window

    Returns a DataFrame with columns: date, rolling_era
    """
    if game_log_df.empty:
        return pd.DataFrame(columns=["date", "rolling_era"])

    df = game_log_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    df["ip_float"] = df["innings_pitched"].apply(innings_to_float)
    df["er_float"] = _to_float_series(df["earned_runs"])

    rolling_er = df["er_float"].rolling(window=window, min_periods=1).sum()
    rolling_ip = df["ip_float"].rolling(window=window, min_periods=1).sum()

    # ERA = (earned runs / innings pitched) * 9
    df["rolling_era"] = ((rolling_er / rolling_ip.replace(0, np.nan)) * 9).round(2)

    return df[["date", "rolling_era"]].dropna()


# ---------------------------------------------------------------------------
# Hot / cold streak calculations (for Page 3 — streaks)
# ---------------------------------------------------------------------------

def get_hot_cold_hitters(n_days: int = 7, top_n: int = 10) -> dict:
    """
    Find the hottest and coldest hitters over the last n_days using the MLB Stats API.

    Strategy:
    - Fetch each player's stats over the last n_days (byDateRange endpoint)
    - Fetch each player's full-season stats for the baseline
    - Merge on player_id and compute OPS delta (recent OPS minus season OPS)
    - Hot = highest positive delta, Cold = most negative delta
    - Minimum 10 AB in the window to filter out 1-game flukes

    Returns a dict: {"hot": DataFrame, "cold": DataFrame}
    Each DataFrame has columns:
        name, team, ab, recent_ops, season_ops, ops_delta, trend
    """
    empty = pd.DataFrame(columns=["name", "team", "ab", "recent_ops", "season_ops", "ops_delta", "trend"])
    year  = datetime.now().year

    try:
        recent_df = get_recent_hitting_stats(n_days, year)
        season_df = get_season_hitting_stats(year)

        if recent_df.empty or season_df.empty:
            return {"hot": empty, "cold": empty}

        # Require at least 10 AB in the window — filters out bench players with 1-2 PA
        recent_df = recent_df[recent_df["ab"] >= 10].copy()

        recent_df["ops_float"] = _to_float_series(recent_df["ops"])
        season_df["ops_float"] = _to_float_series(season_df["ops"])

        merged = recent_df[["player_id", "player_name", "team", "ab", "ops_float"]].merge(
            season_df[["player_id", "ops_float"]],
            on="player_id",
            suffixes=("_recent", "_season"),
            how="inner",
        )

        merged = merged.rename(columns={
            "player_name":     "name",
            "ops_float_recent":"recent_ops",
            "ops_float_season":"season_ops",
        })

        merged["ops_delta"] = (merged["recent_ops"] - merged["season_ops"]).round(3)
        merged["trend"]     = merged.apply(
            lambda r: trend_arrow(r["recent_ops"], r["season_ops"]), axis=1
        )

        # Split by sign first — a player can only appear in one list.
        # delta > 0  →  hot candidates only
        # delta < 0  →  cold candidates only
        # delta == 0 →  excluded from both
        hot  = merged[merged["ops_delta"] > 0].nlargest(top_n,  "ops_delta").reset_index(drop=True)
        cold = merged[merged["ops_delta"] < 0].nsmallest(top_n, "ops_delta").reset_index(drop=True)
        return {"hot": hot, "cold": cold}

    except Exception as e:
        print(f"[processor] Error calculating hitting streaks: {e}")
        return {"hot": empty, "cold": empty}


def get_hot_cold_pitchers(n_days: int = 7, top_n: int = 5) -> dict:
    """
    Find the hottest and coldest pitchers over the last n_days using the MLB Stats API.

    Strategy:
    - Fetch each pitcher's stats over the last n_days (byDateRange endpoint)
    - Fetch each pitcher's full-season stats for the baseline
    - Merge on player_id and compute ERA delta (recent ERA minus season ERA)
    - Hot = most negative delta (ERA improved), Cold = most positive delta (ERA got worse)
    - Minimum 3 outs recorded (~1 IP) in the window

    Returns a dict: {"hot": DataFrame, "cold": DataFrame}
    Each DataFrame has columns:
        name, team, innings_pitched, recent_era, season_era, era_delta, trend
    """
    empty = pd.DataFrame(columns=["name", "team", "innings_pitched", "recent_era", "season_era", "era_delta", "trend"])
    year  = datetime.now().year

    try:
        recent_df = get_recent_pitching_stats(n_days, year)
        season_df = get_season_pitching_stats(year)

        if recent_df.empty or season_df.empty:
            return {"hot": empty, "cold": empty}

        # Require at least 3 outs (~1 IP) to filter out token mop-up appearances
        recent_df = recent_df[recent_df["outs_pitched"] >= 3].copy()

        recent_df["era_float"] = _to_float_series(recent_df["era"])
        season_df["era_float"] = _to_float_series(season_df["era"])

        merged = recent_df[["player_id", "player_name", "team", "innings_pitched", "era_float"]].merge(
            season_df[["player_id", "era_float"]],
            on="player_id",
            suffixes=("_recent", "_season"),
            how="inner",
        )

        merged = merged.rename(columns={
            "player_name":     "name",
            "era_float_recent":"recent_era",
            "era_float_season":"season_era",
        })

        merged["era_delta"] = (merged["recent_era"] - merged["season_era"]).round(2)

        # For pitchers, lower ERA = hot — invert the trend arrow
        merged["trend"] = merged.apply(
            lambda r: trend_arrow(r["season_era"], r["recent_era"]), axis=1
        )

        # Split by sign — ERA improved (negative delta) = hot, got worse (positive) = cold.
        # delta == 0 excluded from both.
        hot  = merged[merged["era_delta"] < 0].nsmallest(top_n, "era_delta").reset_index(drop=True)
        cold = merged[merged["era_delta"] > 0].nlargest(top_n,  "era_delta").reset_index(drop=True)
        return {"hot": hot, "cold": cold}

    except Exception as e:
        print(f"[processor] Error calculating pitching streaks: {e}")
        return {"hot": empty, "cold": empty}


# ---------------------------------------------------------------------------
# Standings helpers (for Page 1 and the newsletter)
# ---------------------------------------------------------------------------

def split_standings_by_league(standings_df: pd.DataFrame) -> dict:
    """
    Split a flat standings DataFrame into AL and NL division groups.

    Returns a dict keyed by division name, each value a DataFrame of teams
    sorted by winning percentage descending.
    """
    if standings_df.empty:
        return {}

    divisions = {}
    for division, group in standings_df.groupby("division"):
        # Sort by PCT descending (convert string to float for sorting)
        group = group.copy()
        group["pct_float"] = pd.to_numeric(group["PCT"], errors="coerce").fillna(0)
        divisions[division] = group.sort_values("pct_float", ascending=False).drop(
            columns=["pct_float"]
        ).reset_index(drop=True)

    return divisions


def is_playoff_position(rank: int, division_size: int = 5) -> bool:
    """Return True if the rank qualifies for a playoff spot (top 3 in division)."""
    return rank <= 3


# ---------------------------------------------------------------------------
# Newsletter data aggregation
# ---------------------------------------------------------------------------

def get_weekly_summary(year: int = None) -> dict:
    """
    Aggregate the data needed for the weekly newsletter.

    Pulls standings, top performers, and streaks, then packages everything
    into a single dict that gets passed to the Claude API prompt.

    Returns a dict with keys:
        week_ending, standings_snapshot, top_hitters, top_pitchers,
        hot_hitters, cold_hitters, hot_pitchers, cold_pitchers,
        todays_matchups
    """
    year = year or datetime.now().year
    week_ending = date.today().strftime("%B %d, %Y")

    # --- Standings snapshot (top team per division) ---
    standings_df = get_standings(year)
    standings_snapshot = {}
    if not standings_df.empty:
        for division, group in standings_df.groupby("division"):
            top_team = group.sort_values(
                "W", ascending=False
            ).iloc[0]
            standings_snapshot[division] = {
                "leader": top_team["team"],
                "W": int(top_team["W"]),
                "L": int(top_team["L"]),
                "PCT": top_team["PCT"],
                "streak": top_team["streak"],
            }

    # --- Season leaderboards (top 5 per category) ---
    batting_df = get_batting_leaderboard(year)
    top_hitters = []
    if not batting_df.empty:
        cols = [c for c in ["Name", "Team", "HR", "RBI", "AVG", "OPS", "wRC+"] if c in batting_df.columns]
        top_hitters = batting_df[cols].head(5).to_dict(orient="records")

    pitching_df = get_pitching_leaderboard(year)
    top_pitchers = []
    if not pitching_df.empty:
        cols = [c for c in ["Name", "Team", "W", "ERA", "WHIP", "IP", "FIP"] if c in pitching_df.columns]
        top_pitchers = pitching_df.nsmallest(5, "ERA")[cols].to_dict(orient="records") if "ERA" in pitching_df.columns else []

    # --- Hot / cold streaks ---
    hitter_streaks = get_hot_cold_hitters(n_days=7, top_n=5)
    pitcher_streaks = get_hot_cold_pitchers(n_days=7, top_n=3)

    def _df_to_records(df):
        return df.to_dict(orient="records") if not df.empty else []

    # --- Today's matchups ---
    schedule_df = get_todays_schedule()
    matchups = []
    if not schedule_df.empty:
        for _, row in schedule_df.iterrows():
            matchups.append({
                "away": row["away_team"],
                "home": row["home_team"],
                "away_pitcher": row.get("away_probable_pitcher", "TBD"),
                "home_pitcher": row.get("home_probable_pitcher", "TBD"),
            })

    return {
        "week_ending": week_ending,
        "standings_snapshot": standings_snapshot,
        "top_hitters": top_hitters,
        "top_pitchers": top_pitchers,
        "hot_hitters": _df_to_records(hitter_streaks["hot"]),
        "cold_hitters": _df_to_records(hitter_streaks["cold"]),
        "hot_pitchers": _df_to_records(pitcher_streaks["hot"]),
        "cold_pitchers": _df_to_records(pitcher_streaks["cold"]),
        "todays_matchups": matchups,
    }
