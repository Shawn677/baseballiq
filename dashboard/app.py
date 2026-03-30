# dashboard/app.py
# The main Streamlit dashboard. Run from the project root with:
#   streamlit run dashboard/app.py

import sys
import os
import time

# Ensure imports from data/ work when running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime, date, timedelta

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data.fetcher import (
    get_standings,
    search_players,
    get_player_info,
    get_player_hitting_stats,
    get_player_pitching_stats,
    get_player_hitting_game_log,
    get_player_pitching_game_log,
    get_schedule,
    get_statcast_spray_data,
)
from data.processor import (
    split_standings_by_league,
    calculate_rolling_ops,
    calculate_rolling_era,
    get_hot_cold_hitters,
    get_hot_cold_pitchers,
)

# ---------------------------------------------------------------------------
# Countdown timer utility
# ---------------------------------------------------------------------------

def _show_countdown(cache_key: str, ttl_minutes: int):
    """
    Render a live JavaScript countdown showing time until the next cache refresh.

    Computes the seconds remaining from the cache file's last-modified timestamp,
    then injects a small JS snippet that ticks down every second inside an iframe.
    Resets to the full TTL on each Streamlit re-run (page interaction).
    """
    from data.fetcher import _safe_key, CACHE_DIR
    path = os.path.join(CACHE_DIR, f"{_safe_key(cache_key)}.json")

    if not os.path.exists(path):
        st.caption("Next update in: —")
        return

    age_seconds  = int(time.time() - os.path.getmtime(path))
    remaining    = max(0, ttl_minutes * 60 - age_seconds)
    init_min     = remaining // 60
    init_sec     = remaining % 60

    # Unique element ID so multiple timers on the same page don't collide
    uid = _safe_key(cache_key)

    html = f"""
    <p style="font-size:0.78em;color:#888;margin:4px 0 0 0;">
        Next update in:&nbsp;<span id="cd-{uid}">{init_min}m {init_sec:02d}s</span>
    </p>
    <script>
    (function() {{
        var s = {remaining};
        var el = document.getElementById('cd-{uid}');
        if (!el) return;
        var t = setInterval(function() {{
            s--;
            if (s <= 0) {{
                el.textContent = 'updating\u2026';
                clearInterval(t);
                return;
            }}
            var m = Math.floor(s / 60);
            var sec = s % 60;
            el.textContent = m + 'm ' + (sec < 10 ? '0' : '') + sec + 's';
        }}, 1000);
    }})();
    </script>
    """
    components.html(html, height=28, scrolling=False)


# ---------------------------------------------------------------------------
# App-wide configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BaseballIQ",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — baseball green theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d2b1a !important;
    border-right: 1px solid #1a5c38;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #d6ede0 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1f5c38 !important;
    margin: 0.5rem 0;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption p {
    color: #6aad88 !important;
    font-size: 0.78rem !important;
}
/* Nav radio — hide the label, space the items */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 0px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: block !important;
    padding: 9px 14px !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    cursor: pointer;
    transition: background 0.15s;
    margin: 1px 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.09) !important;
}

/* ── Main background ── */
.stApp {
    background-color: #f7fcf9;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #f7fcf9;
}

/* ── Page headings ── */
h1 { color: #0d2b1a !important; }
h2 { color: #0d2b1a !important; }
h3 { color: #1a5c38 !important; }

/* ── Horizontal rule ── */
hr { border-color: #cce8d8 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #d0e8d9;
    border-radius: 10px;
    padding: 10px 14px !important;
    box-shadow: 0 1px 4px rgba(13,43,26,0.07);
}
[data-testid="stMetricLabel"] p {
    color: #557a63 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #0d2b1a !important;
    font-weight: 700 !important;
}

/* ── Dataframe header row ── */
[data-testid="stDataFrame"] th {
    background-color: #dff0e8 !important;
    color: #0d2b1a !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stDataFrame"] td {
    font-size: 0.88rem !important;
}

/* ── Tab bar ── */
[data-testid="stTabs"] [role="tab"] {
    color: #1a5c38;
    font-weight: 500;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    border-bottom: 3px solid #1a5c38;
    color: #0d2b1a;
    font-weight: 700;
}

/* ── Select/input boxes ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stDateInput"] > div > div {
    border-color: #b8d9c7 !important;
    border-radius: 8px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #1a5c38 !important;
}

/* ── Info box ── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown("""
<div style="padding: 1.1rem 0.5rem 0.6rem; text-align: center;">
    <div style="font-size: 2.4rem; line-height: 1;">⚾</div>
    <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff !important;
                letter-spacing: -0.01em; margin-top: 0.3rem;">
        BaseballIQ
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊  Standings",
        "👤  Player Stats",
        "🔥  Hot / Cold Streaks",
        "⚾  Pitching Matchups",
        "ℹ️  About",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Data: MLB Stats API · FanGraphs via pybaseball")

# ---------------------------------------------------------------------------
# Helpers shared across pages
# ---------------------------------------------------------------------------

# Division display order — determines which column (AL vs NL) each goes in
AL_DIVISIONS = ["AL East", "AL Central", "AL West"]
NL_DIVISIONS = ["NL East", "NL Central", "NL West"]
ALL_DIVISIONS = AL_DIVISIONS + NL_DIVISIONS


def _color_standings_rows(row: pd.Series) -> list:
    """
    Color-code a standings row based on its position within the division.

    row.name is the integer index (0-based) within the sorted division table:
      - 0–2  (rank 1–3): green background  → playoff position
      - 3–4  (rank 4–5): red background    → out of playoff picture
    """
    if row.name <= 2:
        color = "background-color: #d4edda; color: #155724"   # green
    else:
        color = "background-color: #f8d7da; color: #721c24"   # red
    return [color] * len(row)


def _render_division_table(division_name: str, df: pd.DataFrame):
    """
    Render a single division standings table with a header and color styling.
    Called once per division inside each column.
    """
    st.markdown(f"**{division_name}**")

    if df.empty:
        st.info("No data available.")
        return

    # Rename columns for display
    display_df = df.rename(columns={
        "team": "Team",
        "W": "W",
        "L": "L",
        "PCT": "PCT",
        "GB": "GB",
        "home": "Home",
        "away": "Away",
        "last_10": "L10",
        "streak": "Streak",
    })

    # Only keep the columns we want to show
    cols = ["Team", "W", "L", "PCT", "GB", "Home", "Away", "L10", "Streak"]
    display_df = display_df[[c for c in cols if c in display_df.columns]]

    styled = display_df.style.apply(_color_standings_rows, axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Team":   st.column_config.TextColumn("Team",   width="medium"),
            "W":      st.column_config.NumberColumn("W",    width="small", format="%d"),
            "L":      st.column_config.NumberColumn("L",    width="small", format="%d"),
            "PCT":    st.column_config.TextColumn("PCT",    width="small"),
            "GB":     st.column_config.TextColumn("GB",     width="small"),
            "Home":   st.column_config.TextColumn("Home",   width="small"),
            "Away":   st.column_config.TextColumn("Away",   width="small"),
            "L10":    st.column_config.TextColumn("L10",    width="small"),
            "Streak": st.column_config.TextColumn("Streak", width="small"),
        },
    )


# ---------------------------------------------------------------------------
# Page 1: Standings
# ---------------------------------------------------------------------------

def show_standings():
    # Hero banner — shown only on the default landing page
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d2b1a 0%, #1a5c38 65%, #2d8a56 100%);
        padding: 1.8rem 2.2rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 18px rgba(13,43,26,0.20);
    ">
        <div style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0.35rem;">
            <span style="font-size:2rem; line-height:1;">⚾</span>
            <span style="font-size:1.95rem; font-weight:800; color:#ffffff;
                         letter-spacing:-0.02em; line-height:1;">
                BaseballIQ
            </span>
        </div>
        <p style="color:#a8d9bc; margin:0; font-size:1rem; font-weight:400; line-height:1.4;">
            Live standings, player analytics, hot/cold streaks, and daily pitching matchups — all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.title("Team Standings")

    # --- Controls ---
    col_year, col_division = st.columns([1, 3])

    with col_year:
        current_year = datetime.now().year
        year = st.selectbox(
            "Season",
            options=list(range(current_year, current_year - 5, -1)),
            index=0,
        )

    with col_division:
        division_filter = st.selectbox(
            "Division",
            options=["All Divisions"] + ALL_DIVISIONS,
            index=0,
        )

    st.markdown("---")

    # --- Fetch and process ---
    with st.spinner("Loading standings..."):
        try:
            raw_df = get_standings(year)
        except Exception as e:
            print(f"[standings] get_standings error: {e}")
            st.warning(
                "Unable to load standings right now — please try again in a moment."
            )
            return

    if raw_df.empty:
        st.warning(
            "Standings data is unavailable right now. "
            "This can happen before Opening Day or if the MLB Stats API is unreachable. "
            "Try again in a few minutes."
        )
        return

    # If the API fell back to a prior year, let the user know
    try:
        actual_season = raw_df["season"].iloc[0] if "season" in raw_df.columns else year
    except Exception:
        actual_season = year

    if actual_season != year:
        st.info(
            f"The {year} season hasn't started yet — showing {actual_season} standings."
        )

    try:
        divisions = split_standings_by_league(raw_df)
    except Exception as e:
        print(f"[standings] split_standings_by_league error: {e}")
        st.warning(
            "Unable to process standings data right now — please try again in a moment."
        )
        return

    # --- Legend ---
    legend_col1, legend_col2, _ = st.columns([1, 1, 4])
    with legend_col1:
        st.markdown(
            "<span style='background:#d4edda;color:#155724;padding:2px 8px;"
            "border-radius:4px;font-size:0.85em'>● Playoff position</span>",
            unsafe_allow_html=True,
        )
    with legend_col2:
        st.markdown(
            "<span style='background:#f8d7da;color:#721c24;padding:2px 8px;"
            "border-radius:4px;font-size:0.85em'>● Out of playoffs</span>",
            unsafe_allow_html=True,
        )

    st.markdown("")  # small spacer

    # --- Render tables ---
    try:
        if division_filter == "All Divisions":
            # Side-by-side: AL on the left, NL on the right
            al_col, nl_col = st.columns(2)

            with al_col:
                st.subheader("American League")
                for div in AL_DIVISIONS:
                    if div in divisions:
                        _render_division_table(div, divisions[div])
                        st.markdown("")  # spacer between divisions

            with nl_col:
                st.subheader("National League")
                for div in NL_DIVISIONS:
                    if div in divisions:
                        _render_division_table(div, divisions[div])
                        st.markdown("")

        else:
            # Single division — render full-width
            if division_filter in divisions:
                league = "American League" if division_filter in AL_DIVISIONS else "National League"
                st.subheader(f"{league} · {division_filter}")
                _render_division_table(division_filter, divisions[division_filter])
            else:
                st.info(f"No data found for {division_filter}.")
    except Exception as e:
        print(f"[standings] render error: {e}")
        st.warning(
            "Unable to display standings right now — please try again in a moment."
        )
        return

    try:
        _show_countdown(f"standings_{year}", ttl_minutes=60)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Page 2: Player stats and trends
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Spray chart helper (Page 2 — Player Stats)
# ---------------------------------------------------------------------------

def _draw_spray_chart(statcast_df: pd.DataFrame, player_name: str,
                      year: int = 2025, no_data_msg: str = None):
    """
    Draw a baseball field spray chart from Statcast batted ball data.

    Coordinate transformation applied to hc_x / hc_y:
        hit_x = hc_x - 125.42      (0 = centre, neg = left, pos = right)
        hit_y = 198.27 - hc_y      (0 = home plate, pos = away from plate)

    Scale: ~2.35 Statcast units per foot (400 ft CF ≈ 170 units up).
    Field geometry is derived from that scale and drawn with Plotly
    graph_objects traces so we can control every element independently.

    Guards against missing hc_x/hc_y columns (column schema can vary by
    season or pybaseball version) and renders no_data_msg when data is absent.
    """
    _fallback = no_data_msg or "No Statcast data available for this player."

    # Guard: columns may be absent if Statcast hasn't published location data yet
    if statcast_df.empty or "hc_x" not in statcast_df.columns or "hc_y" not in statcast_df.columns:
        st.info(_fallback)
        return

    df = statcast_df.dropna(subset=["hc_x", "hc_y"]).copy()
    if df.empty:
        st.info("Statcast location data not available for this player.")
        return

    # ── Transform coordinates ─────────────────────────────────────────────
    df["hit_x"] = df["hc_x"] - 125.42
    df["hit_y"] = 198.27 - df["hc_y"]

    # ── Categorise each batted ball ───────────────────────────────────────
    def _cat(e):
        if pd.isna(e):
            return "Out"
        e = str(e).lower()
        if "home_run" in e: return "Home Run"
        if "triple"   in e: return "Triple"
        if "double"   in e: return "Double"
        if "single"   in e: return "Single"
        return "Out"

    df["category"] = df["events"].apply(_cat)

    fig = go.Figure()

    # ── Field outline ──────────────────────────────────────────────────────
    # All measurements in Statcast units; foul corners at (±100, 100),
    # CF wall at (0, 175).  Arc centre solved analytically: (0, 70.8),
    # radius 104.2 — passes exactly through all three boundary points.

    WALL  = dict(color="rgba(255,255,255,0.55)", width=1.5)
    DIRT  = dict(color="rgba(200,170,120,0.30)", width=10)
    GRASS = dict(color="rgba(255,255,255,0.65)", width=1.5)

    # Foul lines
    for sx in (-1, 1):
        fig.add_trace(go.Scatter(
            x=[0, sx * 125], y=[0, 125],
            mode="lines", line=WALL, showlegend=False, hoverinfo="skip",
        ))

    # Outfield wall arc
    theta_arc = np.linspace(np.radians(16.3), np.radians(163.7), 120)
    fig.add_trace(go.Scatter(
        x=104.2 * np.cos(theta_arc),
        y=70.8  + 104.2 * np.sin(theta_arc),
        mode="lines", line=WALL, showlegend=False, hoverinfo="skip",
    ))

    # Infield dirt circle (radius ≈ 40 units, centred at (0, 38))
    theta_dirt = np.linspace(0, np.pi, 80)
    fig.add_trace(go.Scatter(
        x=40 * np.cos(theta_dirt),
        y=38 + 40 * np.sin(theta_dirt),
        mode="lines", line=DIRT, showlegend=False, hoverinfo="skip",
    ))

    # Infield diamond  (90 ft ÷ 2.35 ≈ 38 units per side, 45° → 27 each axis)
    bases = [(0, 0), (27, 27), (0, 54), (-27, 27), (0, 0)]
    fig.add_trace(go.Scatter(
        x=[b[0] for b in bases], y=[b[1] for b in bases],
        mode="lines", line=GRASS, showlegend=False, hoverinfo="skip",
    ))

    # Base labels
    for bx, by, label in [(27, 27, "1B"), (0, 54, "2B"), (-27, 27, "3B")]:
        fig.add_annotation(
            x=bx, y=by, text=label,
            font=dict(color="rgba(255,255,255,0.70)", size=9),
            showarrow=False, yshift=9,
        )

    # Pitcher's mound (60.5 ft = ~26 units)
    theta_m = np.linspace(0, 2 * np.pi, 40)
    fig.add_trace(go.Scatter(
        x=3.5 * np.cos(theta_m),
        y=26 + 3.5 * np.sin(theta_m),
        mode="lines",
        line=dict(color="rgba(200,170,120,0.50)", width=2),
        fill="toself", fillcolor="rgba(200,170,120,0.22)",
        showlegend=False, hoverinfo="skip",
    ))

    # ── Hit dots grouped by category ─────────────────────────────────────
    PALETTE = [
        ("Single",   "#00e676"),
        ("Double",   "#40a0ff"),
        ("Triple",   "#ffab40"),
        ("Home Run", "#ff4444"),
        ("Out",      "rgba(210,210,210,0.40)"),
    ]
    for cat, color in PALETTE:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["hit_x"], y=sub["hit_y"],
            mode="markers",
            name=f"{cat} ({len(sub)})",
            marker=dict(
                color=color, size=7, opacity=0.85,
                line=dict(width=0.4, color="rgba(0,0,0,0.25)"),
            ),
            hovertemplate=f"<b>{cat}</b><extra></extra>",
        ))

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"{player_name} — {year} Spray Chart  ({len(df)} batted balls)",
            font=dict(size=14, color="#0d2b1a"),
            x=0.5, xanchor="center",
        ),
        height=520,
        plot_bgcolor="#1e5c2b",
        paper_bgcolor="#ffffff",
        xaxis=dict(
            range=[-175, 175],
            showgrid=False, zeroline=False, visible=False,
            scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(
            range=[-30, 215],
            showgrid=False, zeroline=False, visible=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="center", x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#cce8d8", borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=75, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def _hitting_metrics(stats_df: pd.DataFrame, log_df: pd.DataFrame,
                     player_name: str = None, player_id: int = None,
                     year: int = None):
    """Render hitting stat cards, the season OPS trend chart, and a spray chart."""

    if stats_df.empty:
        st.info("No hitting stats found for this player and season.")
        return

    s = stats_df.iloc[0]

    # Compute last-15-games OPS from the raw per-game counting stats in the log,
    # so the delta on each metric shows recent form vs full season.
    recent_ops = None
    if not log_df.empty and len(log_df) >= 3:
        recent = log_df.tail(15)
        ab    = recent["at_bats"].sum()
        h     = recent["hits"].sum()
        dbl   = recent["doubles"].sum()
        trpl  = recent["triples"].sum()
        hr    = recent["home_runs"].sum()
        bb    = recent["walks"].sum()
        sng   = h - dbl - trpl - hr
        tb    = sng + 2*dbl + 3*trpl + 4*hr
        obp_r = (h + bb) / (ab + bb) if (ab + bb) > 0 else 0
        slg_r = tb / ab               if ab > 0          else 0
        recent_ops = round(obp_r + slg_r, 3)

    season_ops = float(s.get("ops", 0) or 0)
    ops_delta  = round(recent_ops - season_ops, 3) if recent_ops is not None else None

    # --- Stat cards ---
    st.markdown("##### Season stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AVG",  s.get("avg",  ".000"))
    c2.metric("OBP",  s.get("obp",  ".000"))
    c3.metric("SLG",  s.get("slg",  ".000"))
    c4.metric(
        "OPS",
        s.get("ops", ".000"),
        delta=f"{ops_delta:+.3f} last 15 G" if ops_delta is not None else None,
        delta_color="normal",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("HR",  s.get("home_runs",    0))
    c6.metric("RBI", s.get("rbi",          0))
    c7.metric("SB",  s.get("stolen_bases", 0))
    c8.metric("BABIP", s.get("babip", ".000"))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("G",  s.get("games", 0))
    c10.metric("AB", s.get("at_bats", 0))
    c11.metric("H",  s.get("hits",    0))
    c12.metric("BB", s.get("walks",   0))

    # --- OPS trend chart ---
    st.markdown("##### Season OPS trend")
    if log_df.empty:
        st.info("Game log not available for this player.")
    else:
        trend_df = calculate_rolling_ops(log_df, window=30)
        if trend_df.empty:
            st.info("Not enough games to draw a trend line yet.")
        else:
            fig = px.line(
                trend_df,
                x="date",
                y="rolling_ops",
                labels={"date": "", "rolling_ops": "OPS"},
                title="30-game rolling OPS",
            )
            fig.update_traces(line_color="#1a5c38", line_width=2.5)
            fig.add_hline(
                y=season_ops,
                line_dash="dash",
                line_color="#8ab89e",
                annotation_text=f"Season avg {season_ops:.3f}",
                annotation_position="bottom right",
            )
            fig.update_layout(
                height=320,
                margin=dict(t=40, b=20, l=0, r=0),
                plot_bgcolor="#f7fcf9",
                paper_bgcolor="#ffffff",
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Spray chart (follows the season dropdown) ---
    if player_id is not None:
        spray_year = year or 2024
        st.markdown("---")
        st.markdown(f"##### {spray_year} Spray Chart")
        with st.spinner(f"Loading {spray_year} Statcast data…"):
            statcast_df = get_statcast_spray_data(
                player_id, year=spray_year, player_name=player_name
            )
        _draw_spray_chart(statcast_df, player_name or "Player", year=spray_year)


def _pitching_metrics(stats_df: pd.DataFrame, log_df: pd.DataFrame):
    """Render pitching stat cards and the season ERA trend chart."""

    if stats_df.empty:
        st.info("No pitching stats found for this player and season.")
        return

    s = stats_df.iloc[0]

    # Compute last-3-starts ERA for the metric delta
    recent_era = None
    if not log_df.empty and len(log_df) >= 2:
        from data.processor import innings_to_float
        recent = log_df.tail(3)
        er = recent["earned_runs"].sum()
        ip = recent["innings_pitched"].apply(innings_to_float).sum()
        if ip > 0:
            recent_era = round((er / ip) * 9, 2)

    season_era = float(s.get("era", 0) or 0)
    era_delta  = round(recent_era - season_era, 2) if recent_era is not None else None
    # For ERA, lower is better — invert the colour logic
    era_delta_colour = "inverse" if era_delta is not None else "normal"

    # --- Stat cards ---
    st.markdown("##### Season stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "ERA",
        s.get("era", "-.--"),
        delta=f"{era_delta:+.2f} last 3 GS" if era_delta is not None else None,
        delta_color=era_delta_colour,
    )
    c2.metric("WHIP",  s.get("whip",        "-.--"))
    c3.metric("K/9",   s.get("k_per_9",     "0.00"))
    c4.metric("BB/9",  s.get("bb_per_9",    "0.00"))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("W",   s.get("wins",            0))
    c6.metric("L",   s.get("losses",          0))
    c7.metric("SV",  s.get("saves",           0))
    c8.metric("IP",  s.get("innings_pitched", "0.0"))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("G",   s.get("games",         0))
    c10.metric("GS", s.get("games_started", 0))
    c11.metric("K",  s.get("strikeouts",    0))
    c12.metric("BB", s.get("walks",         0))

    # --- ERA trend chart ---
    st.markdown("##### Season ERA trend")
    if log_df.empty:
        st.info("Game log not available for this player.")
        return

    trend_df = calculate_rolling_era(log_df, window=30)
    if trend_df.empty:
        st.info("Not enough starts to draw a trend line yet.")
        return

    fig = px.line(
        trend_df,
        x="date",
        y="rolling_era",
        labels={"date": "", "rolling_era": "ERA"},
        title="Season ERA trend (cumulative)",
    )
    fig.update_traces(line_color="#c0392b", line_width=2.5)
    fig.add_hline(
        y=season_era,
        line_dash="dash",
        line_color="#e8a89e",
        annotation_text=f"Season ERA {season_era:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        height=320,
        margin=dict(t=40, b=20, l=0, r=0),
        plot_bgcolor="#f7fcf9",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_player_stats():
    st.title("Player Stats & Trends")

    # --- Controls ---
    col_search, col_year = st.columns([3, 1])
    with col_search:
        query = st.text_input(
            "Search player",
            placeholder="e.g. Aaron Judge, Paul Skenes, Shohei Ohtani",
        )
    with col_year:
        current_year = datetime.now().year
        year = st.selectbox(
            "Season",
            list(range(current_year, current_year - 5, -1)),
            key="player_year",
        )

    if not query:
        st.info("Enter a player name above to get started.")
        return

    # --- Search ---
    with st.spinner("Searching..."):
        results = search_players(query)

    if results.empty:
        st.warning(f"No players found matching **{query}**. Try a different spelling.")
        return

    # If multiple results let the user pick; if one, auto-select
    if len(results) == 1:
        selected_row = results.iloc[0]
    else:
        # Build display labels — filter to active players first, then add inactive
        active    = results[results["active"] == True]
        inactive  = results[results["active"] != True]
        ordered   = pd.concat([active, inactive]).reset_index(drop=True)

        labels = ordered.apply(
            lambda r: f"{r['full_name']}  ({r['position']} · {r['team']})",
            axis=1,
        ).tolist()

        choice = st.selectbox("Select player", labels)
        selected_row = ordered.iloc[labels.index(choice)]

    player_id   = int(selected_row["player_id"])
    player_name = selected_row["full_name"]

    # --- Player header ---
    with st.spinner(f"Loading {player_name}..."):
        info        = get_player_info(player_id)
        hit_stats   = get_player_hitting_stats(player_id, year)
        pitch_stats = get_player_pitching_stats(player_id, year)
        hit_log     = get_player_hitting_game_log(player_id, year)
        pitch_log   = get_player_pitching_game_log(player_id, year)

    st.markdown("---")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Player",   info.get("full_name", player_name))
    h2.metric("Team",     info.get("team", "N/A"))
    h3.metric("Position", info.get("position", "N/A"))
    h4.metric("Bats / Throws",
              f"{info.get('bat_side','?')} / {info.get('pitch_hand','?')}")
    st.markdown("---")

    # --- Tabs ---
    # Auto-open the relevant tab based on position
    is_pitcher = info.get("position_type", "") == "Pitcher"
    tab_order  = ["⚾ Hitting", "⚾ Pitching"] if not is_pitcher else ["⚾ Pitching", "⚾ Hitting"]

    tab_hit, tab_pitch = st.tabs(tab_order) if not is_pitcher else st.tabs(tab_order)

    if not is_pitcher:
        with tab_hit:
            _hitting_metrics(hit_stats, hit_log,
                             player_name=player_name, player_id=player_id,
                             year=year)
        with tab_pitch:
            _pitching_metrics(pitch_stats, pitch_log)
    else:
        with tab_hit:
            _pitching_metrics(pitch_stats, pitch_log)
        with tab_pitch:
            _hitting_metrics(hit_stats, hit_log,
                             player_name=player_name, player_id=player_id,
                             year=year)


def _streak_table(df: pd.DataFrame, kind: str, is_hot: bool):
    """
    Render a single hot or cold streak table with colour-coded rows.

    kind = "hitter" or "pitcher"
    """
    if df.empty:
        st.info("Not enough data to rank streaks yet.")
        return

    if kind == "hitter":
        display = df[["name", "team", "ab", "recent_ops", "season_ops", "ops_delta", "trend"]].copy()
        display.columns = ["Player", "Team", "AB", "Recent OPS", "Season OPS", "Δ OPS", ""]
        display["Recent OPS"] = display["Recent OPS"].apply(lambda v: f"{v:.3f}")
        display["Season OPS"] = display["Season OPS"].apply(lambda v: f"{v:.3f}")
        display["Δ OPS"]      = display["Δ OPS"].apply(lambda v: f"{v:+.3f}")
        delta_col = "Δ OPS"
    else:
        display = df[["name", "team", "innings_pitched", "recent_era", "season_era", "era_delta", "trend"]].copy()
        display.columns = ["Player", "Team", "IP", "Recent ERA", "Season ERA", "Δ ERA", ""]
        display["Recent ERA"] = display["Recent ERA"].apply(lambda v: f"{v:.2f}")
        display["Season ERA"] = display["Season ERA"].apply(lambda v: f"{v:.2f}")
        display["Δ ERA"]      = display["Δ ERA"].apply(lambda v: f"{v:+.2f}")
        delta_col = "Δ ERA"

    def _colour_row(row):
        if is_hot:
            bg = "background-color: #d4edda; color: #155724"
        else:
            bg = "background-color: #f8d7da; color: #721c24"
        return [bg] * len(row)

    styled = display.style.apply(_colour_row, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def show_streaks():
    st.title("Hot / Cold Streaks")

    # --- Window selector ---
    col_days, col_spacer = st.columns([1, 3])
    with col_days:
        n_days = st.selectbox("Window", [7, 14, 30], index=0, format_func=lambda d: f"Last {d} days")

    st.markdown("---")

    # --- Load data ---
    with st.spinner("Calculating streaks…"):
        hitters  = get_hot_cold_hitters(n_days=n_days,  top_n=10)
        pitchers = get_hot_cold_pitchers(n_days=n_days, top_n=5)

    both_empty = (
        hitters["hot"].empty and hitters["cold"].empty
        and pitchers["hot"].empty and pitchers["cold"].empty
    )
    if both_empty:
        st.warning(
            "Streak data isn't available yet. This can happen at the very start of the "
            "season when fewer than 10 at-bats have been recorded in the selected window. "
            "Try expanding the window or check back after a few games have been played."
        )
        return

    today = date.today().strftime("%Y-%m-%d")

    # --- Hitters ---
    st.subheader("🔥 Hitters")
    hot_h_col, cold_h_col = st.columns(2)

    with hot_h_col:
        st.markdown("**Hottest hitters**")
        _streak_table(hitters["hot"], "hitter", is_hot=True)

    with cold_h_col:
        st.markdown("**Coldest hitters**")
        _streak_table(hitters["cold"], "hitter", is_hot=False)

    st.caption("Minimum 10 AB in window · players with Δ OPS = 0 excluded from both lists")
    _show_countdown(f"recent_hitting_{n_days}d_{today}", ttl_minutes=10)

    st.markdown("---")

    # --- Pitchers ---
    st.subheader("⚾ Pitchers")
    hot_p_col, cold_p_col = st.columns(2)

    with hot_p_col:
        st.markdown("**Hottest pitchers**")
        _streak_table(pitchers["hot"], "pitcher", is_hot=True)

    with cold_p_col:
        st.markdown("**Coldest pitchers**")
        _streak_table(pitchers["cold"], "pitcher", is_hot=False)

    st.caption("Minimum 1 IP in window · players with Δ ERA = 0 excluded from both lists")
    _show_countdown(f"recent_pitching_{n_days}d_{today}", ttl_minutes=10)


# ---------------------------------------------------------------------------
# Page 4 helpers: Pitching Matchups
# ---------------------------------------------------------------------------

def _parse_game_time(game_date_str: str) -> str:
    """
    Parse an ISO UTC game timestamp to an approximate ET display string.

    Baseball season runs April–October so EDT (UTC-4) is always in effect.
    Returns an empty string if parsing fails so callers can fall back to status.
    """
    try:
        dt = datetime.strptime(game_date_str[:16], "%Y-%m-%dT%H:%M")
        et = dt - timedelta(hours=4)
        hour   = et.hour % 12 or 12
        am_pm  = "PM" if et.hour >= 12 else "AM"
        return f"{hour}:{et.minute:02d} {am_pm} ET"
    except Exception:
        return ""


def _render_pitcher_stats(pitcher_id: int, year: int):
    """
    Render season stat cards for a single pitcher.

    Fetches via get_player_pitching_stats (cached) and lays out two rows
    of four metrics: ERA / WHIP / W–L / IP, then K/9 / BB/9 / GS / K.
    Shows a caption if stats aren't available yet (early season / no starts).
    """
    stats_df = get_player_pitching_stats(pitcher_id, year)
    if stats_df.empty:
        st.caption("No season stats yet.")
        return

    s = stats_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ERA",  s.get("era",  "-.--"))
    c2.metric("WHIP", s.get("whip", "-.--"))
    c3.metric("W–L",  f"{s.get('wins', 0)}–{s.get('losses', 0)}")
    c4.metric("IP",   s.get("innings_pitched", "0.0"))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("K/9",  s.get("k_per_9",       "0.00"))
    c6.metric("BB/9", s.get("bb_per_9",       "0.00"))
    c7.metric("GS",   s.get("games_started",  0))
    c8.metric("K",    s.get("strikeouts",     0))


def _render_matchup_card(game: pd.Series, year: int):
    """
    Render one game's matchup card.

    Header shows "Away @ Home · time" for scheduled games, live score
    for in-progress, or "Final · score" for completed games.
    Body is two columns (away pitcher | vs | home pitcher) each showing
    season stats from _render_pitcher_stats.
    """
    status     = str(game.get("status", "Scheduled"))
    away_team  = game.get("away_team", "TBD")
    home_team  = game.get("home_team", "TBD")

    # --- Game header ---
    if status == "Final":
        a = int(game.get("away_score") or 0)
        h = int(game.get("home_score") or 0)
        wa = "**" if a > h else ""
        wh = "**" if h > a else ""
        st.subheader(f"Final · {wa}{away_team} {a}{wa} – {wh}{home_team} {h}{wh}")
    elif "Progress" in status or "live" in status.lower():
        a = int(game.get("away_score") or 0)
        h = int(game.get("home_score") or 0)
        st.subheader(f"🔴 LIVE · {away_team} {a} – {home_team} {h}")
    else:
        game_time = _parse_game_time(game.get("game_date", ""))
        st.subheader(f"{away_team} @ {home_team}  ·  {game_time or status}")

    # --- Pitcher side-by-side ---
    away_pid  = game.get("away_probable_pitcher_id")
    home_pid  = game.get("home_probable_pitcher_id")
    away_name = game.get("away_probable_pitcher") or "TBD"
    home_name = game.get("home_probable_pitcher") or "TBD"

    col_away, col_vs, col_home = st.columns([10, 1, 10])

    with col_away:
        st.markdown(f"**{away_team}** (away)")
        st.markdown(f"###### {away_name}")
        if away_pid and str(away_pid) not in ("", "nan", "None"):
            _render_pitcher_stats(int(float(away_pid)), year)
        else:
            st.caption("Probable starter: TBD")

    with col_vs:
        st.markdown(
            "<div style='text-align:center;padding-top:1.6em;"
            "font-size:1.1em;color:#888'>vs</div>",
            unsafe_allow_html=True,
        )

    with col_home:
        st.markdown(f"**{home_team}** (home)")
        st.markdown(f"###### {home_name}")
        if home_pid and str(home_pid) not in ("", "nan", "None"):
            _render_pitcher_stats(int(float(home_pid)), year)
        else:
            st.caption("Probable starter: TBD")


def show_matchups():
    st.title("Pitching Matchups")

    # --- Date picker ---
    col_date, _ = st.columns([1, 3])
    with col_date:
        selected_date = st.date_input("Date", value=date.today())

    date_str = selected_date.strftime("%Y-%m-%d")
    year     = selected_date.year

    # --- Fetch schedule ---
    with st.spinner("Loading schedule…"):
        schedule_df = get_schedule(date_str, date_str)

    if schedule_df.empty:
        st.info(
            f"No games found for {selected_date.strftime('%B %d, %Y')}. "
            "The schedule may not be posted yet, or this is an off day."
        )
        _show_countdown(f"schedule_{date_str}_{date_str}", ttl_minutes=10)
        return

    # Games count header
    n         = len(schedule_df)
    day_label = selected_date.strftime("%A, %B %d, %Y")
    st.markdown(f"**{n} game{'s' if n != 1 else ''} · {day_label}**")
    st.markdown("---")

    for _, game in schedule_df.iterrows():
        with st.container():
            _render_matchup_card(game, year)
        st.markdown("---")

    _show_countdown(f"schedule_{date_str}_{date_str}", ttl_minutes=10)


# ---------------------------------------------------------------------------
# Page 5 — About
# ---------------------------------------------------------------------------

def show_about():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d2b1a 0%, #1a5c38 100%);
                border-radius: 14px; padding: 2rem 2.2rem 1.8rem; margin-bottom: 1.6rem;">
        <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">⚾</div>
        <h1 style="color: #ffffff !important; margin: 0 0 0.5rem; font-size: 2rem;">BaseballIQ</h1>
        <p style="color: #a8d9bc; font-size: 1.05rem; margin: 0; max-width: 600px;">
            A free baseball analytics platform built for fans who want more than just box scores.
            Real stats, real trends, real context — no subscription required.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("### What's Included")
        st.markdown("""
<div style="background:#ffffff; border:1px solid #d0e8d9; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem;">
<table style="width:100%; border-collapse:collapse; font-size:0.93rem;">
<tr style="border-bottom:1px solid #e8f4ee;">
  <td style="padding:10px 12px 10px 0; font-weight:700; color:#0d2b1a; white-space:nowrap; width:1%;">📊&nbsp; Standings</td>
  <td style="padding:10px 0; color:#3a5a47;">Live division standings with win-loss records, streaks, and playoff position highlights updated daily.</td>
</tr>
<tr style="border-bottom:1px solid #e8f4ee;">
  <td style="padding:10px 12px 10px 0; font-weight:700; color:#0d2b1a; white-space:nowrap;">👤&nbsp; Player Stats</td>
  <td style="padding:10px 0; color:#3a5a47;">Search any active player for season stats, rolling OPS or ERA trend charts, and a Statcast spray chart showing where they hit the ball.</td>
</tr>
<tr style="border-bottom:1px solid #e8f4ee;">
  <td style="padding:10px 12px 10px 0; font-weight:700; color:#0d2b1a; white-space:nowrap;">🔥&nbsp; Hot / Cold</td>
  <td style="padding:10px 0; color:#3a5a47;">See which hitters and pitchers are surging or struggling right now, ranked by how far their recent performance deviates from their season average.</td>
</tr>
<tr>
  <td style="padding:10px 12px 4px 0; font-weight:700; color:#0d2b1a; white-space:nowrap;">⚾&nbsp; Matchups</td>
  <td style="padding:10px 0 4px; color:#3a5a47;">Today's full schedule with probable starters, venue details, and season stats for both pitchers side by side.</td>
</tr>
</table>
</div>
        """, unsafe_allow_html=True)

        st.markdown("### Data Sources")
        st.markdown("""
<div style="background:#ffffff; border:1px solid #d0e8d9; border-radius:12px; padding:1.2rem 1.5rem;">
<p style="margin:0 0 0.7rem; font-size:0.93rem; color:#3a5a47;">
    <strong style="color:#0d2b1a;">MLB Stats API</strong> — The official free API from MLB Advanced Media.
    Provides standings, schedules, player rosters, game logs, and season statistics in real time.
</p>
<p style="margin:0; font-size:0.93rem; color:#3a5a47;">
    <strong style="color:#0d2b1a;">pybaseball · FanGraphs · Statcast</strong> — The open-source
    <a href="https://github.com/jldbc/pybaseball" target="_blank"
       style="color:#1a5c38; text-decoration:none; font-weight:600;">pybaseball</a>
    library provides access to FanGraphs leaderboards (wRC+, FIP, xFIP) and
    Baseball Savant Statcast data (exit velocity, launch angle, batted-ball locations).
</p>
</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Coming Soon")
        st.markdown("""
<div style="background:#ffffff; border:1px solid #d0e8d9; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem;">
<ul style="margin:0; padding-left:1.2rem; font-size:0.93rem; color:#3a5a47; line-height:2;">
  <li><strong style="color:#0d2b1a;">Team Stats Search</strong> — full season and split breakdowns by team</li>
  <li><strong style="color:#0d2b1a;">Weekly Newsletter</strong> — AI-written recap of the week's biggest storylines, delivered to your inbox</li>
  <li><strong style="color:#0d2b1a;">Matchup Predictor</strong> — pitcher vs. batter history and platoon splits</li>
  <li><strong style="color:#0d2b1a;">More Sports</strong> — NFL, NBA, and NHL analytics coming in future seasons</li>
</ul>
</div>
        """, unsafe_allow_html=True)

        st.markdown("### Built By")
        st.markdown("""
<div style="background:#ffffff; border:1px solid #d0e8d9; border-radius:12px; padding:1.2rem 1.5rem;">
<p style="margin:0; font-size:0.93rem; color:#3a5a47;">
    Built by a baseball fan who loves data.
</p>
<p style="margin:0.7rem 0 0; font-size:0.83rem; color:#7aaa8e;">
    Have a feature request or found a bug?  Feedback is always welcome.
</p>
</div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Router — send each page selection to the right function
# ---------------------------------------------------------------------------

if page == "📊  Standings":
    show_standings()
elif page == "👤  Player Stats":
    show_player_stats()
elif page == "🔥  Hot / Cold Streaks":
    show_streaks()
elif page == "⚾  Pitching Matchups":
    show_matchups()
elif page == "ℹ️  About":
    show_about()
