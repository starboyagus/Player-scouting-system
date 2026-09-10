"""
Domain definitions and categories for football scouting metrics.
Groups raw and per-90 metrics into intuitive scouting dimensions.
"""

from typing import Dict, List

# Metric categories for scouting domain analysis
METRIC_CATEGORIES: Dict[str, List[str]] = {
    "Attacking / Finishing": [
        "goals_per90",
        "xg_per90",
        "shots_per90",
        "shots_on_target_per90",
        "touches_att_pen_per90"
    ],
    "Creation / Playmaking": [
        "assists_per90",
        "xag_per90",
        "key_passes_per90",
        "passes_completed_per90",
        "progressive_passes_per90",
        "through_balls_per90",
        "pass_completion_pct"
    ],
    "Ball Progression / Dribbling": [
        "progressive_carries_per90",
        "successful_takeons_per90",
        "takeon_success_pct"
    ],
    "Defending / Workrate": [
        "tackles_won_per90",
        "interceptions_per90",
        "blocks_per90",
        "clearances_per90",
        "ball_recoveries_per90",
        "aerials_won_pct"
    ]
}

# Standard radar chart core metrics (12 high-impact dimensions for visual profiles)
RADAR_CORE_METRICS: Dict[str, str] = {
    "goals_per90": "Goals / 90",
    "xg_per90": "xG / 90",
    "shots_per90": "Shots / 90",
    "assists_per90": "Assists / 90",
    "xag_per90": "xAG / 90",
    "key_passes_per90": "Key Passes / 90",
    "progressive_passes_per90": "Prog Passes / 90",
    "progressive_carries_per90": "Prog Carries / 90",
    "successful_takeons_per90": "Take-ons / 90",
    "tackles_won_per90": "Tackles / 90",
    "interceptions_per90": "Interceptions / 90",
    "ball_recoveries_per90": "Recoveries / 90"
}

# Position broad groups for tactical filtering
POSITION_GROUPS = {
    "FW": ["Forward", "Striker", "Center Forward"],
    "W": ["Winger", "Left Winger", "Right Winger"],
    "AM": ["Attacking Midfielder", "Number 10"],
    "CM": ["Central Midfielder", "Box-to-Box"],
    "DM": ["Defensive Midfielder", "Anchor"],
    "FB": ["Fullback", "Left Back", "Right Back", "Wingback"],
    "CB": ["Center Back"]
}

def get_all_feature_cols() -> List[str]:
    """Returns a flattened list of all per-90 and rate feature columns."""
    features = []
    for cat_features in METRIC_CATEGORIES.values():
        features.extend(cat_features)
    return list(dict.fromkeys(features))
