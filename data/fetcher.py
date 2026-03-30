# data/fetcher.py
# All data fetching lives here. Zero API keys required.
# Standings use the MLB Stats API (free, no key needed).
# Player stats and leaderboards use pybaseball (FanGraphs / Baseball Reference).
# The dashboard and newsletter import from this file; they never fetch data directly.

import json
import os
import requests
import pandas as pd
from datetime import datetime, date, timedelta

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

try:
    import pybaseball
    pybaseball.cache.enable()   # Use pybaseball's built-in disk cache
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("[fetcher] WARNING: pybaseball not installed. Run: pip install pybaseball")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = "data/cache"

# Cache TTLs in minutes
TTL_STANDINGS    = 60     # Refresh standings every hour
TTL_SEASON_STATS = 1440   # Refresh season stats once per day
TTL_SCHEDULE     = 10     # Refresh today's schedule every 10 minutes
TTL_ROSTER       = 60     # Refresh rosters every hour

# pybaseball.standings() returns one DataFrame per division in this order
DIVISION_NAMES = [
    "AL East", "AL Central", "AL West",
    "NL East", "NL Central", "NL West",
]

# ---------------------------------------------------------------------------
# File-based JSON cache (prevents hammering the source on every page load)
# ---------------------------------------------------------------------------

def _safe_key(key: str) -> str:
    """Strip characters that are invalid in filenames."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in key)


def get_cached(key: str, ttl_minutes: int = 60):
    """
    Return cached JSON data if it exists and is younger than ttl_minutes.
    Returns None on cache miss or expiry.
    """
    path = os.path.join(CACHE_DIR, f"{_safe_key(key)}.json")
    if not os.path.exists(path):
        return None
    age_minutes = (datetime.now().timestamp() - os.path.getmtime(path)) / 60
    if age_minutes > ttl_minutes:
        return None
    with open(path) as f:
        return json.load(f)


def set_cached(key: str, data):
    """Write JSON-serialisable data to the cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{_safe_key(key)}.json"), "w") as f:
        json.dump(data, f)


def _current_year() -> int:
    return datetime.now().year


def _mlb_get(endpoint: str, params: dict = None) -> dict:
    """
    Make a GET request to the MLB Stats API.
    Returns parsed JSON on success, empty dict on any error.
    """
    url = f"{MLB_API_BASE}{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[fetcher] MLB API error {url}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Standings  (MLB Stats API — free, no key required)
# ---------------------------------------------------------------------------

def _fetch_standings_for_year(year: int) -> list:
    """
    Hit the MLB Stats API standings endpoint for one year.
    Returns the raw list of division records, or [] on failure / no data.
    """
    try:
        response = requests.get(
            f"{MLB_API_BASE}/standings",
            params={
                "leagueId": "103,104",
                "season": year,
                "standingsType": "regularSeason",
                "hydrate": "division",
            },
            timeout=10,
        )
        response.raise_for_status()
        records = response.json().get("records", [])
        # Confirm there's real game data — an empty season returns records with 0 wins/losses
        total_games = sum(
            tr.get("wins", 0) + tr.get("losses", 0)
            for rec in records
            for tr in rec.get("teamRecords", [])
        )
        return records if total_games > 0 else []
    except Exception as e:
        print(f"[fetcher] MLB API standings error (season={year}): {e}")
        return []


def get_standings(year: int = None) -> pd.DataFrame:
    """
    Fetch AL and NL standings from the MLB Stats API.

    Tries the requested year first. If the season hasn't started yet
    (zero games played), falls back to the previous year automatically.

    Returns a DataFrame with columns:
        division, team, W, L, PCT, GB, home, away, last_10, streak, season
    The 'season' column lets the dashboard show which year is displayed.
    """
    year = year or _current_year()
    cache_key = f"standings_{year}"

    cached = get_cached(cache_key, TTL_STANDINGS)
    if cached is not None:
        return pd.DataFrame(cached)

    empty_cols = ["division", "team", "W", "L", "PCT", "GB", "home", "away", "last_10", "streak", "season"]

    # Try requested year; fall back one year if no games have been played yet
    records = _fetch_standings_for_year(year)
    actual_year = year
    if not records:
        print(f"[fetcher] No standings data for {year}, falling back to {year - 1}")
        records = _fetch_standings_for_year(year - 1)
        actual_year = year - 1

    if not records:
        return pd.DataFrame(columns=empty_cols)

    rows = []
    for division_record in records:
        division_name = division_record.get("division", {}).get("nameShort", "Unknown")

        for team_record in division_record.get("teamRecords", []):
            splits = {
                s["type"]: s
                for s in team_record.get("records", {}).get("splitRecords", [])
            }
            home     = splits.get("home", {})
            away     = splits.get("away", {})
            last_ten = splits.get("lastTen", {})
            streak   = team_record.get("streak", {}).get("streakCode", "-")

            rows.append({
                "division": division_name,
                "team":     team_record.get("team", {}).get("name", "Unknown"),
                "W":        team_record.get("wins", 0),
                "L":        team_record.get("losses", 0),
                "PCT":      team_record.get("winningPercentage", ".000"),
                "GB":       team_record.get("gamesBack", "-"),
                "home":     f"{home.get('wins', 0)}-{home.get('losses', 0)}",
                "away":     f"{away.get('wins', 0)}-{away.get('losses', 0)}",
                "last_10":  f"{last_ten.get('wins', 0)}-{last_ten.get('losses', 0)}",
                "streak":   streak,
                "season":   actual_year,
            })

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    df = pd.DataFrame(rows)
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


# ---------------------------------------------------------------------------
# Player search  (MLB Stats API)
# ---------------------------------------------------------------------------

def search_players(name: str) -> pd.DataFrame:
    """
    Search for players by name using the MLB Stats API.

    Returns a DataFrame with columns:
        player_id, full_name, position, position_type, team,
        bat_side, pitch_hand, active
    """
    empty = pd.DataFrame(columns=[
        "player_id", "full_name", "position", "position_type",
        "team", "bat_side", "pitch_hand", "active",
    ])

    if not name or len(name.strip()) < 2:
        return empty

    cache_key = f"search_{_safe_key(name.strip().lower())}"
    cached = get_cached(cache_key, TTL_SEASON_STATS)
    if cached is not None:
        return pd.DataFrame(cached)

    data = _mlb_get("/people/search", params={"names": name.strip(), "sportId": 1})

    if not data.get("people"):
        return empty

    rows = []
    for p in data["people"]:
        rows.append({
            "player_id":     p.get("id"),
            "full_name":     p.get("fullName", "Unknown"),
            "position":      p.get("primaryPosition", {}).get("abbreviation", "N/A"),
            "position_type": p.get("primaryPosition", {}).get("type", ""),
            "team":          p.get("currentTeam", {}).get("name", "N/A"),
            "bat_side":      p.get("batSide", {}).get("code", "?"),
            "pitch_hand":    p.get("pitchHand", {}).get("code", "?"),
            "active":        p.get("active", False),
        })

    df = pd.DataFrame(rows)
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


def get_player_info(player_id: int) -> dict:
    """
    Fetch bio and team info for a single player.

    Returns a dict with keys:
        full_name, team, position, position_type, bat_side, pitch_hand,
        jersey_number, age
    """
    cache_key = f"player_info_{player_id}"
    cached = get_cached(cache_key, TTL_ROSTER)
    if cached is not None:
        return cached

    data = _mlb_get(f"/people/{player_id}", params={"hydrate": "currentTeam"})

    if not data.get("people"):
        return {}

    p = data["people"][0]
    info = {
        "full_name":     p.get("fullName", "Unknown"),
        "team":          p.get("currentTeam", {}).get("name", "N/A"),
        "position":      p.get("primaryPosition", {}).get("abbreviation", "N/A"),
        "position_type": p.get("primaryPosition", {}).get("type", ""),
        "bat_side":      p.get("batSide", {}).get("code", "?"),
        "pitch_hand":    p.get("pitchHand", {}).get("code", "?"),
        "jersey_number": p.get("primaryNumber", ""),
        "age":           p.get("currentAge", ""),
    }
    set_cached(cache_key, info)
    return info


# ---------------------------------------------------------------------------
# Player season stats  (MLB Stats API)
# ---------------------------------------------------------------------------

def get_player_hitting_stats(player_id: int, year: int = None) -> pd.DataFrame:
    """
    Fetch season hitting stats for a player from the MLB Stats API.

    Returns a single-row DataFrame with columns:
        avg, obp, slg, ops, hits, doubles, triples, home_runs, rbi,
        stolen_bases, at_bats, games, plate_appearances, walks, strikeouts, babip
    """
    empty = pd.DataFrame(columns=[
        "avg", "obp", "slg", "ops", "hits", "doubles", "triples",
        "home_runs", "rbi", "stolen_bases", "at_bats", "games",
        "plate_appearances", "walks", "strikeouts", "babip",
    ])

    year = year or _current_year()
    cache_key = f"hitting_stats_{player_id}_{year}"

    cached = get_cached(cache_key, TTL_SEASON_STATS)
    if cached is not None:
        return pd.DataFrame(cached)

    data = _mlb_get(
        f"/people/{player_id}/stats",
        params={"stats": "season", "group": "hitting", "season": year},
    )

    if not data.get("stats") or not data["stats"][0].get("splits"):
        return empty

    s = data["stats"][0]["splits"][0]["stat"]
    row = {
        "avg":              s.get("avg", ".000"),
        "obp":              s.get("obp", ".000"),
        "slg":              s.get("slg", ".000"),
        "ops":              s.get("ops", ".000"),
        "hits":             s.get("hits", 0),
        "doubles":          s.get("doubles", 0),
        "triples":          s.get("triples", 0),
        "home_runs":        s.get("homeRuns", 0),
        "rbi":              s.get("rbi", 0),
        "stolen_bases":     s.get("stolenBases", 0),
        "at_bats":          s.get("atBats", 0),
        "games":            s.get("gamesPlayed", 0),
        "plate_appearances":s.get("plateAppearances", 0),
        "walks":            s.get("baseOnBalls", 0),
        "strikeouts":       s.get("strikeOuts", 0),
        "babip":            s.get("babip", ".000"),
    }

    df = pd.DataFrame([row])
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


def get_player_pitching_stats(player_id: int, year: int = None) -> pd.DataFrame:
    """
    Fetch season pitching stats for a player from the MLB Stats API.

    Returns a single-row DataFrame with columns:
        era, whip, innings_pitched, wins, losses, saves, games, games_started,
        strikeouts, walks, hits, home_runs, k_per_9, bb_per_9, hits_per_9, era_str
    """
    empty = pd.DataFrame(columns=[
        "era", "whip", "innings_pitched", "wins", "losses", "saves",
        "games", "games_started", "strikeouts", "walks", "hits",
        "home_runs", "k_per_9", "bb_per_9", "hits_per_9",
    ])

    year = year or _current_year()
    cache_key = f"pitching_stats_{player_id}_{year}"

    cached = get_cached(cache_key, TTL_SEASON_STATS)
    if cached is not None:
        return pd.DataFrame(cached)

    data = _mlb_get(
        f"/people/{player_id}/stats",
        params={"stats": "season", "group": "pitching", "season": year},
    )

    if not data.get("stats") or not data["stats"][0].get("splits"):
        return empty

    s = data["stats"][0]["splits"][0]["stat"]
    row = {
        "era":              s.get("era", "-.--"),
        "whip":             s.get("whip", "-.--"),
        "innings_pitched":  s.get("inningsPitched", "0.0"),
        "wins":             s.get("wins", 0),
        "losses":           s.get("losses", 0),
        "saves":            s.get("saves", 0),
        "games":            s.get("gamesPitched", 0),
        "games_started":    s.get("gamesStarted", 0),
        "strikeouts":       s.get("strikeOuts", 0),
        "walks":            s.get("baseOnBalls", 0),
        "hits":             s.get("hits", 0),
        "home_runs":        s.get("homeRuns", 0),
        "k_per_9":          s.get("strikeoutsPer9Inn", "0.00"),
        "bb_per_9":         s.get("walksPer9Inn", "0.00"),
        "hits_per_9":       s.get("hitsPer9Inn", "0.00"),
    }

    df = pd.DataFrame([row])
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


# ---------------------------------------------------------------------------
# Player game logs  (MLB Stats API — used for season trend charts)
# ---------------------------------------------------------------------------

def get_player_hitting_game_log(player_id: int, year: int = None) -> pd.DataFrame:
    """
    Fetch the game-by-game hitting log for a player from the MLB Stats API.

    Each row is one game. Counting stats (AB, H, HR, RBI) are per-game.
    Rate stats (avg, obp, slg, ops) are cumulative season-to-date after
    each game — perfect for plotting a season trend line.

    Returns a DataFrame with columns:
        date, opponent, at_bats, hits, doubles, triples, home_runs, rbi,
        walks, stolen_bases, avg, obp, slg, ops
    """
    empty = pd.DataFrame(columns=[
        "date", "opponent", "at_bats", "hits", "doubles", "triples",
        "home_runs", "rbi", "walks", "stolen_bases", "avg", "obp", "slg", "ops",
    ])

    year = year or _current_year()
    cache_key = f"hitting_log_{player_id}_{year}"

    cached = get_cached(cache_key, ttl_minutes=60)
    if cached is not None:
        return pd.DataFrame(cached)

    data = _mlb_get(
        f"/people/{player_id}/stats",
        params={"stats": "gameLog", "group": "hitting", "season": year},
    )

    if not data.get("stats") or not data["stats"][0].get("splits"):
        return empty

    rows = []
    for split in data["stats"][0]["splits"]:
        s = split.get("stat", {})
        rows.append({
            "date":          split.get("date", ""),
            "opponent":      split.get("opponent", {}).get("name", ""),
            "at_bats":       s.get("atBats", 0),
            "hits":          s.get("hits", 0),
            "doubles":       s.get("doubles", 0),
            "triples":       s.get("triples", 0),
            "home_runs":     s.get("homeRuns", 0),
            "rbi":           s.get("rbi", 0),
            "walks":         s.get("baseOnBalls", 0),
            "stolen_bases":  s.get("stolenBases", 0),
            "avg":           s.get("avg", ".000"),
            "obp":           s.get("obp", ".000"),
            "slg":           s.get("slg", ".000"),
            "ops":           s.get("ops", ".000"),
        })

    df = pd.DataFrame(rows)
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


def get_player_pitching_game_log(player_id: int, year: int = None) -> pd.DataFrame:
    """
    Fetch the game-by-game pitching log for a player from the MLB Stats API.

    Counting stats (IP, ER, K, BB) are per-game.
    Rate stats (era, whip) are cumulative season-to-date after each outing.

    Returns a DataFrame with columns:
        date, opponent, innings_pitched, earned_runs, strikeouts,
        walks, hits, home_runs, era, whip
    """
    empty = pd.DataFrame(columns=[
        "date", "opponent", "innings_pitched", "earned_runs",
        "strikeouts", "walks", "hits", "home_runs", "era", "whip",
    ])

    year = year or _current_year()
    cache_key = f"pitching_log_{player_id}_{year}"

    cached = get_cached(cache_key, ttl_minutes=60)
    if cached is not None:
        return pd.DataFrame(cached)

    data = _mlb_get(
        f"/people/{player_id}/stats",
        params={"stats": "gameLog", "group": "pitching", "season": year},
    )

    if not data.get("stats") or not data["stats"][0].get("splits"):
        return empty

    rows = []
    for split in data["stats"][0]["splits"]:
        s = split.get("stat", {})
        rows.append({
            "date":             split.get("date", ""),
            "opponent":         split.get("opponent", {}).get("name", ""),
            "innings_pitched":  s.get("inningsPitched", "0.0"),
            "earned_runs":      s.get("earnedRuns", 0),
            "strikeouts":       s.get("strikeOuts", 0),
            "walks":            s.get("baseOnBalls", 0),
            "hits":             s.get("hits", 0),
            "home_runs":        s.get("homeRuns", 0),
            "era":              s.get("era", "-.--"),
            "whip":             s.get("whip", "-.--"),
        })

    df = pd.DataFrame(rows)
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


# ---------------------------------------------------------------------------
# Season leaderboards  (FanGraphs via pybaseball)
# ---------------------------------------------------------------------------

def get_batting_leaderboard(year: int = None) -> pd.DataFrame:
    """
    Fetch the full-season batting leaderboard from FanGraphs via pybaseball.

    Key columns: Name, Team, G, PA, AB, H, HR, RBI, SB, AVG, OBP, SLG, OPS, wRC+, WAR
    """
    year = year or _current_year()
    cache_key = f"batting_leaderboard_{year}"

    cached = get_cached(cache_key, TTL_SEASON_STATS)
    if cached is not None:
        return pd.DataFrame(cached)

    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame(columns=["Name", "Team", "G", "PA", "HR", "RBI", "AVG", "OBP", "SLG", "OPS", "wRC+"])

    try:
        df = pybaseball.batting_stats(year, qual=50)
        set_cached(cache_key, df.to_dict(orient="records"))
        return df
    except Exception as e:
        print(f"[fetcher] batting_stats error: {e}")
        return pd.DataFrame(columns=["Name", "Team", "G", "PA", "HR", "RBI", "AVG", "OBP", "SLG", "OPS", "wRC+"])


def get_pitching_leaderboard(year: int = None) -> pd.DataFrame:
    """
    Fetch the full-season pitching leaderboard from FanGraphs via pybaseball.

    Key columns: Name, Team, G, GS, IP, W, L, SV, ERA, WHIP, K/9, BB/9, FIP, WAR
    """
    year = year or _current_year()
    cache_key = f"pitching_leaderboard_{year}"

    cached = get_cached(cache_key, TTL_SEASON_STATS)
    if cached is not None:
        return pd.DataFrame(cached)

    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame(columns=["Name", "Team", "G", "GS", "IP", "W", "L", "ERA", "WHIP", "FIP"])

    try:
        df = pybaseball.pitching_stats(year, qual=20)
        set_cached(cache_key, df.to_dict(orient="records"))
        return df
    except Exception as e:
        print(f"[fetcher] pitching_stats error: {e}")
        return pd.DataFrame(columns=["Name", "Team", "G", "GS", "IP", "W", "L", "ERA", "WHIP", "FIP"])


def get_statcast_spray_data(player_id: int, year: int = 2024,
                             player_name: str = None) -> pd.DataFrame:
    """
    Fetch Statcast batted ball data for a batter from Baseball Savant via pybaseball.

    Uses playerid_lookup(last, first) to get key_mlbam — the correct player ID
    for statcast_batter().  The MLB Stats API player_id is NOT always equal to
    key_mlbam, so this lookup is required.

    Returns a DataFrame with columns (subset kept):
        game_date, events, hc_x, hc_y, bb_type, launch_speed,
        launch_angle, hit_distance_sc
    Returns an empty DataFrame when pybaseball is unavailable or no data found.
    """
    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame()

    if not player_name:
        print(f"[fetcher] get_statcast_spray_data: player_name required for "
              f"playerid_lookup — returning empty")
        return pd.DataFrame()

    # ── Step 1: resolve key_mlbam via playerid_lookup ─────────────────────
    # Split "Aaron Judge" → last="Judge", first="Aaron"
    # For "Vladimir Guerrero Jr." use parts[1] as last to skip suffix
    parts = player_name.strip().split()
    first = parts[0]
    last  = parts[1] if len(parts) >= 2 else parts[0]

    try:
        lkp = pybaseball.playerid_lookup(last, first)
        print(f"[fetcher] playerid_lookup('{last}', '{first}'): {len(lkp)} row(s)")
        if lkp.empty:
            print(f"[fetcher]   no results — cannot fetch Statcast data")
            return pd.DataFrame()
        # If multiple rows (e.g. father/son), prefer the most recently active
        if "mlb_played_last" in lkp.columns:
            lkp = lkp.sort_values("mlb_played_last", ascending=False)
        key_mlbam = int(lkp["key_mlbam"].iloc[0])
        print(f"[fetcher]   key_mlbam={key_mlbam}  (MLB Stats API player_id={player_id})")
    except Exception as e:
        print(f"[fetcher] playerid_lookup error: {e}")
        return pd.DataFrame()

    # ── Step 2: check cache ────────────────────────────────────────────────
    start     = f"{year}-03-01"
    end       = f"{year}-11-30"
    cache_key = f"statcast_{key_mlbam}_{year}"

    cached = get_cached(cache_key, TTL_SEASON_STATS)
    if cached is not None:
        print(f"[fetcher] statcast cache hit  key_mlbam={key_mlbam}  year={year}  "
              f"rows={len(cached)}")
        return pd.DataFrame(cached)

    # ── Step 3: fetch from Baseball Savant ────────────────────────────────
    try:
        print(f"[fetcher] statcast_batter('{start}', '{end}', player_id={key_mlbam})")
        df = pybaseball.statcast_batter(start, end, player_id=key_mlbam)

        print(f"[fetcher]   rows returned : {len(df)}")
        print(f"[fetcher]   first 3 cols  : {list(df.columns[:3])}")
        if not df.empty:
            hc_x_ok = "hc_x" in df.columns
            hc_y_ok = "hc_y" in df.columns
            print(f"[fetcher]   hc_x present={hc_x_ok}  hc_y present={hc_y_ok}")
            if hc_x_ok:
                print(f"[fetcher]   hc_x non-null: {df['hc_x'].notna().sum()} / {len(df)}")

        if df.empty:
            return df

        keep = [
            "game_date", "events", "hc_x", "hc_y",
            "bb_type", "launch_speed", "launch_angle", "hit_distance_sc",
        ]
        df = df[[c for c in keep if c in df.columns]].copy()
        set_cached(cache_key, df.to_dict(orient="records"))
        return df
    except Exception as e:
        print(f"[fetcher] statcast_batter error  key_mlbam={key_mlbam}  "
              f"{start}→{end}: {e}")
        return pd.DataFrame()


def get_team_batting_stats(year: int = None) -> pd.DataFrame:
    """Fetch team-level batting stats from FanGraphs via pybaseball."""
    year = year or _current_year()
    cache_key = f"team_batting_{year}"

    cached = get_cached(cache_key, TTL_SEASON_STATS)
    if cached is not None:
        return pd.DataFrame(cached)

    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame(columns=["Team", "G", "PA", "HR", "RBI", "AVG", "OBP", "SLG", "OPS"])

    try:
        df = pybaseball.team_batting(year)
        set_cached(cache_key, df.to_dict(orient="records"))
        return df
    except Exception as e:
        print(f"[fetcher] team_batting error: {e}")
        return pd.DataFrame(columns=["Team", "G", "PA", "HR", "RBI", "AVG", "OBP", "SLG", "OPS"])


# ---------------------------------------------------------------------------
# Streak data  (MLB Stats API — byDateRange for recent, season for baseline)
# ---------------------------------------------------------------------------

def _fetch_all_splits(params: dict) -> list:
    """
    Paginate through the /stats endpoint and return every split record.
    The API caps each response at 250; we loop with offset until done.
    """
    all_splits = []
    offset = 0
    limit  = 250

    while True:
        p = {**params, "limit": limit, "offset": offset}
        data = _mlb_get("/stats", params=p)
        if not data.get("stats"):
            break
        block  = data["stats"][0]
        splits = block.get("splits", [])
        all_splits.extend(splits)
        total = block.get("totalSplits", 0)
        offset += limit
        if offset >= total:
            break

    return all_splits


def get_recent_hitting_stats(n_days: int = 7, year: int = None) -> pd.DataFrame:
    """
    Fetch aggregate hitting stats for every player over the last n_days,
    using the MLB Stats API byDateRange stats endpoint.

    Returns a DataFrame with columns:
        player_id, player_name, team, games, ab, hits, home_runs,
        rbi, walks, avg, obp, slg, ops
    """
    year  = year or _current_year()
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=n_days)).strftime("%Y-%m-%d")

    cache_key = f"recent_hitting_{n_days}d_{end}"
    cached = get_cached(cache_key, ttl_minutes=10)
    if cached is not None:
        return pd.DataFrame(cached)

    empty = pd.DataFrame(columns=[
        "player_id", "player_name", "team", "games", "ab", "hits",
        "home_runs", "rbi", "walks", "avg", "obp", "slg", "ops",
    ])

    splits = _fetch_all_splits({
        "stats":      "byDateRange",
        "group":      "hitting",
        "playerPool": "all",
        "season":     year,
        "startDate":  start,
        "endDate":    end,
    })

    if not splits:
        return empty

    rows = []
    for sp in splits:
        s = sp.get("stat", {})
        rows.append({
            "player_id":   sp.get("player", {}).get("id"),
            "player_name": sp.get("player", {}).get("fullName", "Unknown"),
            "team":        sp.get("team",   {}).get("name", "N/A"),
            "games":       s.get("gamesPlayed", 0),
            "ab":          s.get("atBats",       0),
            "hits":        s.get("hits",          0),
            "home_runs":   s.get("homeRuns",      0),
            "rbi":         s.get("rbi",           0),
            "walks":       s.get("baseOnBalls",   0),
            "avg":         s.get("avg",    ".000"),
            "obp":         s.get("obp",    ".000"),
            "slg":         s.get("slg",    ".000"),
            "ops":         s.get("ops",    ".000"),
        })

    df = pd.DataFrame(rows)
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


def get_recent_pitching_stats(n_days: int = 7, year: int = None) -> pd.DataFrame:
    """
    Fetch aggregate pitching stats for every pitcher over the last n_days.

    Returns a DataFrame with columns:
        player_id, player_name, team, games, outs_pitched, innings_pitched,
        earned_runs, strikeouts, walks, hits, era, whip
    """
    year  = year or _current_year()
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=n_days)).strftime("%Y-%m-%d")

    cache_key = f"recent_pitching_{n_days}d_{end}"
    cached = get_cached(cache_key, ttl_minutes=10)
    if cached is not None:
        return pd.DataFrame(cached)

    empty = pd.DataFrame(columns=[
        "player_id", "player_name", "team", "games", "outs_pitched",
        "innings_pitched", "earned_runs", "strikeouts", "walks", "hits",
        "era", "whip",
    ])

    splits = _fetch_all_splits({
        "stats":      "byDateRange",
        "group":      "pitching",
        "playerPool": "all",
        "season":     year,
        "startDate":  start,
        "endDate":    end,
    })

    if not splits:
        return empty

    rows = []
    for sp in splits:
        s = sp.get("stat", {})
        outs = s.get("outsPitched", s.get("outs", 0))
        rows.append({
            "player_id":      sp.get("player", {}).get("id"),
            "player_name":    sp.get("player", {}).get("fullName", "Unknown"),
            "team":           sp.get("team",   {}).get("name", "N/A"),
            "games":          s.get("gamesPitched", 0),
            "outs_pitched":   outs,
            # Convert raw outs to standard X.Y innings notation via the float
            "innings_pitched":s.get("inningsPitched", "0.0"),
            "earned_runs":    s.get("earnedRuns",     0),
            "strikeouts":     s.get("strikeOuts",     0),
            "walks":          s.get("baseOnBalls",    0),
            "hits":           s.get("hits",           0),
            "era":            s.get("era",   "-.--"),
            "whip":           s.get("whip",  "-.--"),
        })

    df = pd.DataFrame(rows)
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


def get_season_hitting_stats(year: int = None) -> pd.DataFrame:
    """
    Fetch full-season hitting stats for all players via MLB Stats API.
    Used as the baseline for hot/cold streak comparisons.

    Returns a DataFrame with columns:
        player_id, player_name, team, ab, ops, avg, obp, slg
    """
    year      = year or _current_year()
    cache_key = f"season_hitting_{year}"

    cached = get_cached(cache_key, TTL_SEASON_STATS)
    if cached is not None:
        return pd.DataFrame(cached)

    empty = pd.DataFrame(columns=["player_id", "player_name", "team", "ab", "ops", "avg", "obp", "slg"])

    splits = _fetch_all_splits({
        "stats":      "season",
        "group":      "hitting",
        "playerPool": "all",
        "season":     year,
    })

    if not splits:
        return empty

    rows = []
    for sp in splits:
        s = sp.get("stat", {})
        rows.append({
            "player_id":   sp.get("player", {}).get("id"),
            "player_name": sp.get("player", {}).get("fullName", "Unknown"),
            "team":        sp.get("team",   {}).get("name", "N/A"),
            "ab":          s.get("atBats",  0),
            "ops":         s.get("ops",    ".000"),
            "avg":         s.get("avg",    ".000"),
            "obp":         s.get("obp",    ".000"),
            "slg":         s.get("slg",    ".000"),
        })

    df = pd.DataFrame(rows)
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


def get_season_pitching_stats(year: int = None) -> pd.DataFrame:
    """
    Fetch full-season pitching stats for all pitchers via MLB Stats API.
    Used as the baseline for hot/cold streak comparisons.

    Returns a DataFrame with columns:
        player_id, player_name, team, innings_pitched, outs_pitched, era, whip
    """
    year      = year or _current_year()
    cache_key = f"season_pitching_{year}"

    cached = get_cached(cache_key, TTL_SEASON_STATS)
    if cached is not None:
        return pd.DataFrame(cached)

    empty = pd.DataFrame(columns=["player_id", "player_name", "team", "innings_pitched", "outs_pitched", "era", "whip"])

    splits = _fetch_all_splits({
        "stats":      "season",
        "group":      "pitching",
        "playerPool": "all",
        "season":     year,
    })

    if not splits:
        return empty

    rows = []
    for sp in splits:
        s = sp.get("stat", {})
        outs = s.get("outsPitched", s.get("outs", 0))
        rows.append({
            "player_id":      sp.get("player", {}).get("id"),
            "player_name":    sp.get("player", {}).get("fullName", "Unknown"),
            "team":           sp.get("team",   {}).get("name", "N/A"),
            "innings_pitched":s.get("inningsPitched", "0.0"),
            "outs_pitched":   outs,
            "era":            s.get("era",  "-.--"),
            "whip":           s.get("whip", "-.--"),
        })

    df = pd.DataFrame(rows)
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


# ---------------------------------------------------------------------------
# Schedule / roster
# ---------------------------------------------------------------------------

def get_schedule(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch games for a date range from the MLB Stats API, including probable starters.

    Hydrates probablePitcher and team so a single request returns everything
    needed to render a matchup card without extra lookups.

    Returns a DataFrame with columns:
        game_id, game_date, status,
        home_team, home_team_id, away_team, away_team_id,
        home_score, away_score,
        home_probable_pitcher_id, home_probable_pitcher,
        away_probable_pitcher_id, away_probable_pitcher
    """
    cache_key = f"schedule_{start_date}_{end_date}"
    cached = get_cached(cache_key, TTL_SCHEDULE)
    if cached is not None:
        return pd.DataFrame(cached)

    empty_cols = [
        "game_id", "game_date", "status",
        "home_team", "home_team_id", "away_team", "away_team_id",
        "home_score", "away_score",
        "home_probable_pitcher_id", "home_probable_pitcher",
        "away_probable_pitcher_id", "away_probable_pitcher",
    ]

    data = _mlb_get("/schedule", params={
        "sportId":   1,
        "startDate": start_date,
        "endDate":   end_date,
        "hydrate":   "probablePitcher,team",
    })

    if not data.get("dates"):
        return pd.DataFrame(columns=empty_cols)

    rows = []
    for date_entry in data["dates"]:
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            home  = teams.get("home", {})
            away  = teams.get("away", {})

            home_pitcher = home.get("probablePitcher") or {}
            away_pitcher = away.get("probablePitcher") or {}

            rows.append({
                "game_id":   game.get("gamePk"),
                "game_date": game.get("gameDate", ""),
                "status":    game.get("status", {}).get("detailedState", "Scheduled"),
                "home_team":    home.get("team", {}).get("name", "TBD"),
                "home_team_id": home.get("team", {}).get("id"),
                "away_team":    away.get("team", {}).get("name", "TBD"),
                "away_team_id": away.get("team", {}).get("id"),
                "home_score":   home.get("score"),
                "away_score":   away.get("score"),
                "home_probable_pitcher_id": home_pitcher.get("id"),
                "home_probable_pitcher":    home_pitcher.get("fullName", "TBD"),
                "away_probable_pitcher_id": away_pitcher.get("id"),
                "away_probable_pitcher":    away_pitcher.get("fullName", "TBD"),
            })

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    df = pd.DataFrame(rows)
    set_cached(cache_key, df.to_dict(orient="records"))
    return df


def get_todays_schedule() -> pd.DataFrame:
    """Returns today's games with probable starters."""
    today = date.today().strftime("%Y-%m-%d")
    return get_schedule(today, today)


def get_team_roster(team_id: int) -> pd.DataFrame:
    """Returns a team's active roster. Stub — wired up in a later step."""
    return pd.DataFrame(columns=["player_id", "full_name", "position", "jersey_number", "status"])
