"""
debug_statcast.py
-----------------
Standalone diagnostic — run from the project root:
    python debug_statcast.py

Prints the raw output of pybaseball.statcast_batter() for Aaron Judge
(MLB ID 592450) so we can see exactly which columns are returned and
whether hc_x / hc_y are present and populated.

This script bypasses the dashboard cache entirely so results are live.
"""

import sys
import os

# Allow importing from data/ if needed, though this script only uses pybaseball
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pybaseball
    pybaseball.cache.enable()
except ImportError:
    print("ERROR: pybaseball is not installed.  Run: pip install pybaseball")
    sys.exit(1)

JUDGE_ID  = 592450
START     = "2025-03-01"
END       = "2025-11-01"

print(f"\n{'='*60}")
print(f"Fetching statcast_batter for player_id={JUDGE_ID}")
print(f"Date range: {START}  →  {END}")
print(f"{'='*60}\n")

df = pybaseball.statcast_batter(START, END, player_id=JUDGE_ID)

print(f"Shape: {df.shape}  ({len(df)} rows × {len(df.columns)} columns)\n")

print(f"All columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:3d}. {col}")

print(f"\n{'='*60}")
print("First 5 rows (selected columns):")
print(f"{'='*60}")

# Show only the most relevant columns to keep output readable
preview_cols = ["game_date", "player_name", "events", "bb_type",
                "hc_x", "hc_y", "launch_speed", "launch_angle", "hit_distance_sc"]
existing = [c for c in preview_cols if c in df.columns]
print(df[existing].head(5).to_string())

print(f"\n{'='*60}")
print("Location column diagnostics:")
print(f"{'='*60}")

for col in ["hc_x", "hc_y"]:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col}: PRESENT — {non_null} / {len(df)} rows are non-null")
        if non_null > 0:
            print(f"         sample values: {df[col].dropna().head(5).tolist()}")
    else:
        print(f"  {col}: *** MISSING ***")

# Also look for any column whose name contains 'hc' in case of aliasing
hc_like = [c for c in df.columns if "hc" in c.lower()]
if hc_like:
    print(f"\nColumns containing 'hc': {hc_like}")
else:
    print("\nNo columns containing 'hc' found.")

print(f"\n{'='*60}\n")
