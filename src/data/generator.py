"""
Dataset Generator for Premier League 2025/2026 Season.
Generates realistic, domain-accurate player performance statistics across all 20 clubs.
Features real star players with signature tactical profiles, supplemented by statistically
coherent squad players with position-specific distributions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

CLUBS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
    "Leicester City", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham Hotspur",
    "West Ham United", "Wolverhampton Wanderers"
]

# Baseline per-90 distributions by broad tactical role
ROLE_PROFILES = {
    "FW": {
        "goals": (0.45, 0.22), "xg": (0.48, 0.20), "shots": (2.8, 0.9), "shots_on_target": (1.2, 0.5),
        "assists": (0.15, 0.10), "xag": (0.16, 0.10), "key_passes": (1.1, 0.5), "passes_completed": (18.0, 6.0),
        "progressive_passes": (1.8, 0.9), "through_balls": (0.2, 0.15), "pass_completion_pct": (74.0, 5.0),
        "progressive_carries": (2.5, 1.2), "successful_takeons": (1.1, 0.6), "takeon_success_pct": (45.0, 10.0),
        "touches_att_pen": (5.5, 2.0),
        "tackles_won": (0.5, 0.3), "interceptions": (0.2, 0.15), "blocks": (0.7, 0.3),
        "clearances": (0.4, 0.3), "ball_recoveries": (2.2, 0.8), "aerials_won_pct": (42.0, 12.0)
    },
    "W": {
        "goals": (0.30, 0.18), "xg": (0.32, 0.16), "shots": (2.3, 0.8), "shots_on_target": (0.9, 0.4),
        "assists": (0.22, 0.14), "xag": (0.25, 0.12), "key_passes": (1.8, 0.7), "passes_completed": (28.0, 8.0),
        "progressive_passes": (3.4, 1.3), "through_balls": (0.35, 0.2), "pass_completion_pct": (77.0, 5.0),
        "progressive_carries": (5.2, 1.8), "successful_takeons": (2.4, 1.1), "takeon_success_pct": (52.0, 9.0),
        "touches_att_pen": (4.8, 1.8),
        "tackles_won": (1.1, 0.5), "interceptions": (0.5, 0.3), "blocks": (1.1, 0.4),
        "clearances": (0.5, 0.3), "ball_recoveries": (3.8, 1.1), "aerials_won_pct": (34.0, 10.0)
    },
    "AM": {
        "goals": (0.24, 0.14), "xg": (0.26, 0.13), "shots": (2.0, 0.7), "shots_on_target": (0.8, 0.3),
        "assists": (0.28, 0.15), "xag": (0.31, 0.14), "key_passes": (2.4, 0.8), "passes_completed": (42.0, 10.0),
        "progressive_passes": (5.5, 1.8), "through_balls": (0.6, 0.3), "pass_completion_pct": (81.0, 5.0),
        "progressive_carries": (3.8, 1.4), "successful_takeons": (1.6, 0.8), "takeon_success_pct": (53.0, 8.0),
        "touches_att_pen": (3.6, 1.4),
        "tackles_won": (1.3, 0.5), "interceptions": (0.7, 0.35), "blocks": (1.2, 0.4),
        "clearances": (0.6, 0.3), "ball_recoveries": (4.5, 1.2), "aerials_won_pct": (36.0, 10.0)
    },
    "CM": {
        "goals": (0.12, 0.08), "xg": (0.13, 0.08), "shots": (1.3, 0.5), "shots_on_target": (0.4, 0.2),
        "assists": (0.16, 0.10), "xag": (0.17, 0.10), "key_passes": (1.5, 0.6), "passes_completed": (52.0, 12.0),
        "progressive_passes": (5.8, 1.7), "through_balls": (0.3, 0.2), "pass_completion_pct": (85.0, 4.5),
        "progressive_carries": (2.8, 1.1), "successful_takeons": (1.2, 0.6), "takeon_success_pct": (58.0, 8.0),
        "touches_att_pen": (1.8, 0.9),
        "tackles_won": (2.1, 0.7), "interceptions": (1.2, 0.5), "blocks": (1.4, 0.5),
        "clearances": (1.2, 0.5), "ball_recoveries": (6.2, 1.4), "aerials_won_pct": (48.0, 10.0)
    },
    "DM": {
        "goals": (0.06, 0.05), "xg": (0.06, 0.05), "shots": (0.8, 0.4), "shots_on_target": (0.2, 0.15),
        "assists": (0.09, 0.07), "xag": (0.10, 0.07), "key_passes": (0.9, 0.4), "passes_completed": (60.0, 14.0),
        "progressive_passes": (5.2, 1.6), "through_balls": (0.2, 0.15), "pass_completion_pct": (88.0, 4.0),
        "progressive_carries": (1.7, 0.8), "successful_takeons": (0.8, 0.4), "takeon_success_pct": (62.0, 8.0),
        "touches_att_pen": (0.8, 0.5),
        "tackles_won": (2.7, 0.8), "interceptions": (1.8, 0.6), "blocks": (1.8, 0.5),
        "clearances": (1.8, 0.7), "ball_recoveries": (7.4, 1.6), "aerials_won_pct": (55.0, 9.0)
    },
    "FB": {
        "goals": (0.05, 0.04), "xg": (0.06, 0.05), "shots": (0.7, 0.4), "shots_on_target": (0.2, 0.15),
        "assists": (0.15, 0.11), "xag": (0.17, 0.11), "key_passes": (1.4, 0.6), "passes_completed": (48.0, 11.0),
        "progressive_passes": (4.9, 1.5), "through_balls": (0.2, 0.15), "pass_completion_pct": (80.0, 4.5),
        "progressive_carries": (3.2, 1.3), "successful_takeons": (1.3, 0.7), "takeon_success_pct": (55.0, 9.0),
        "touches_att_pen": (1.5, 0.8),
        "tackles_won": (2.2, 0.7), "interceptions": (1.4, 0.5), "blocks": (1.5, 0.5),
        "clearances": (2.2, 0.8), "ball_recoveries": (5.2, 1.2), "aerials_won_pct": (46.0, 10.0)
    },
    "CB": {
        "goals": (0.04, 0.04), "xg": (0.05, 0.04), "shots": (0.5, 0.3), "shots_on_target": (0.15, 0.1),
        "assists": (0.03, 0.03), "xag": (0.03, 0.03), "key_passes": (0.3, 0.2), "passes_completed": (58.0, 14.0),
        "progressive_passes": (3.5, 1.4), "through_balls": (0.08, 0.07), "pass_completion_pct": (89.5, 3.5),
        "progressive_carries": (1.2, 0.7), "successful_takeons": (0.4, 0.3), "takeon_success_pct": (68.0, 10.0),
        "touches_att_pen": (0.7, 0.4),
        "tackles_won": (1.7, 0.5), "interceptions": (1.6, 0.5), "blocks": (1.9, 0.6),
        "clearances": (4.5, 1.4), "ball_recoveries": (5.5, 1.2), "aerials_won_pct": (64.0, 8.0)
    }
}

# Curated Premier League Star Players with realistic 2025/26 profiles
STAR_PLAYERS = [
    {"player": "Erling Haaland", "squad": "Manchester City", "position": "FW", "age": 25, "minutes": 2350,
     "p90": {"goals": 0.92, "xg": 0.88, "shots": 4.1, "shots_on_target": 2.1, "assists": 0.15, "xag": 0.14, "key_passes": 0.9, "passes_completed": 14.2, "progressive_passes": 1.1, "through_balls": 0.1, "pass_completion_pct": 74.5, "progressive_carries": 1.9, "successful_takeons": 0.6, "takeon_success_pct": 46.0, "touches_att_pen": 7.8, "tackles_won": 0.2, "interceptions": 0.1, "blocks": 0.3, "clearances": 0.4, "ball_recoveries": 1.5, "aerials_won_pct": 48.0}},
    {"player": "Mohamed Salah", "squad": "Liverpool", "position": "W", "age": 33, "minutes": 2420,
     "p90": {"goals": 0.68, "xg": 0.62, "shots": 3.4, "shots_on_target": 1.5, "assists": 0.38, "xag": 0.35, "key_passes": 2.5, "passes_completed": 29.5, "progressive_passes": 4.6, "through_balls": 0.6, "pass_completion_pct": 76.8, "progressive_carries": 4.8, "successful_takeons": 1.8, "takeon_success_pct": 48.5, "touches_att_pen": 8.1, "tackles_won": 0.8, "interceptions": 0.3, "blocks": 0.8, "clearances": 0.3, "ball_recoveries": 3.4, "aerials_won_pct": 28.0}},
    {"player": "Bukayo Saka", "squad": "Arsenal", "position": "W", "age": 24, "minutes": 2480,
     "p90": {"goals": 0.48, "xg": 0.44, "shots": 2.7, "shots_on_target": 1.1, "assists": 0.42, "xag": 0.39, "key_passes": 2.8, "passes_completed": 32.0, "progressive_passes": 4.8, "through_balls": 0.5, "pass_completion_pct": 79.5, "progressive_carries": 6.8, "successful_takeons": 2.6, "takeon_success_pct": 54.0, "touches_att_pen": 6.9, "tackles_won": 1.7, "interceptions": 0.6, "blocks": 1.2, "clearances": 0.6, "ball_recoveries": 4.4, "aerials_won_pct": 42.0}},
    {"player": "Cole Palmer", "squad": "Chelsea", "position": "AM", "age": 23, "minutes": 2510,
     "p90": {"goals": 0.52, "xg": 0.49, "shots": 3.2, "shots_on_target": 1.3, "assists": 0.36, "xag": 0.38, "key_passes": 2.9, "passes_completed": 44.5, "progressive_passes": 6.4, "through_balls": 0.9, "pass_completion_pct": 82.3, "progressive_carries": 4.2, "successful_takeons": 1.9, "takeon_success_pct": 56.5, "touches_att_pen": 4.8, "tackles_won": 1.2, "interceptions": 0.6, "blocks": 1.0, "clearances": 0.4, "ball_recoveries": 4.1, "aerials_won_pct": 32.0}},
    {"player": "Martin Odegaard", "squad": "Arsenal", "position": "AM", "age": 27, "minutes": 2380,
     "p90": {"goals": 0.28, "xg": 0.30, "shots": 2.4, "shots_on_target": 0.9, "assists": 0.35, "xag": 0.42, "key_passes": 3.2, "passes_completed": 54.0, "progressive_passes": 8.2, "through_balls": 0.8, "pass_completion_pct": 85.0, "progressive_carries": 4.1, "successful_takeons": 1.7, "takeon_success_pct": 58.0, "touches_att_pen": 4.2, "tackles_won": 1.5, "interceptions": 0.8, "blocks": 1.3, "clearances": 0.4, "ball_recoveries": 5.4, "aerials_won_pct": 35.0}},
    {"player": "Rodri", "squad": "Manchester City", "position": "DM", "age": 29, "minutes": 2200,
     "p90": {"goals": 0.18, "xg": 0.14, "shots": 1.7, "shots_on_target": 0.5, "assists": 0.22, "xag": 0.20, "key_passes": 1.6, "passes_completed": 92.5, "progressive_passes": 9.4, "through_balls": 0.5, "pass_completion_pct": 92.8, "progressive_carries": 2.4, "successful_takeons": 1.1, "takeon_success_pct": 72.0, "touches_att_pen": 1.5, "tackles_won": 2.6, "interceptions": 1.5, "blocks": 1.8, "clearances": 1.9, "ball_recoveries": 8.9, "aerials_won_pct": 68.0}},
    {"player": "Declan Rice", "squad": "Arsenal", "position": "CM", "age": 27, "minutes": 2580,
     "p90": {"goals": 0.19, "xg": 0.15, "shots": 1.4, "shots_on_target": 0.4, "assists": 0.25, "xag": 0.22, "key_passes": 1.7, "passes_completed": 62.0, "progressive_passes": 6.8, "through_balls": 0.3, "pass_completion_pct": 89.2, "progressive_carries": 3.5, "successful_takeons": 1.3, "takeon_success_pct": 64.0, "touches_att_pen": 2.1, "tackles_won": 2.8, "interceptions": 1.9, "blocks": 1.6, "clearances": 1.7, "ball_recoveries": 7.8, "aerials_won_pct": 56.0}},
    {"player": "Trent Alexander-Arnold", "squad": "Liverpool", "position": "FB", "age": 27, "minutes": 2310,
     "p90": {"goals": 0.09, "xg": 0.08, "shots": 1.5, "shots_on_target": 0.4, "assists": 0.36, "xag": 0.38, "key_passes": 2.7, "passes_completed": 68.0, "progressive_passes": 8.8, "through_balls": 0.7, "pass_completion_pct": 78.4, "progressive_carries": 2.8, "successful_takeons": 1.2, "takeon_success_pct": 53.0, "touches_att_pen": 1.4, "tackles_won": 1.9, "interceptions": 1.3, "blocks": 1.4, "clearances": 2.1, "ball_recoveries": 6.5, "aerials_won_pct": 44.0}},
    {"player": "William Saliba", "squad": "Arsenal", "position": "CB", "age": 24, "minutes": 2650,
     "p90": {"goals": 0.06, "xg": 0.04, "shots": 0.4, "shots_on_target": 0.1, "assists": 0.03, "xag": 0.02, "key_passes": 0.3, "passes_completed": 72.0, "progressive_passes": 4.1, "through_balls": 0.05, "pass_completion_pct": 92.5, "progressive_carries": 1.6, "successful_takeons": 0.5, "takeon_success_pct": 75.0, "touches_att_pen": 0.8, "tackles_won": 2.1, "interceptions": 1.8, "blocks": 1.9, "clearances": 4.1, "ball_recoveries": 6.1, "aerials_won_pct": 65.0}},
    {"player": "Virgil van Dijk", "squad": "Liverpool", "position": "CB", "age": 34, "minutes": 2600,
     "p90": {"goals": 0.08, "xg": 0.09, "shots": 0.9, "shots_on_target": 0.3, "assists": 0.04, "xag": 0.03, "key_passes": 0.4, "passes_completed": 76.5, "progressive_passes": 4.8, "through_balls": 0.1, "pass_completion_pct": 91.2, "progressive_carries": 1.2, "successful_takeons": 0.3, "takeon_success_pct": 80.0, "touches_att_pen": 1.1, "tackles_won": 1.4, "interceptions": 1.9, "blocks": 1.8, "clearances": 5.4, "ball_recoveries": 6.3, "aerials_won_pct": 78.5}},
    {"player": "Alexander Isak", "squad": "Newcastle United", "position": "FW", "age": 26, "minutes": 2240,
     "p90": {"goals": 0.72, "xg": 0.68, "shots": 3.3, "shots_on_target": 1.6, "assists": 0.18, "xag": 0.16, "key_passes": 1.3, "passes_completed": 21.0, "progressive_passes": 2.3, "through_balls": 0.25, "pass_completion_pct": 78.0, "progressive_carries": 4.2, "successful_takeons": 2.2, "takeon_success_pct": 54.0, "touches_att_pen": 6.4, "tackles_won": 0.6, "interceptions": 0.2, "blocks": 0.7, "clearances": 0.5, "ball_recoveries": 2.6, "aerials_won_pct": 39.0}},
    {"player": "Bruno Fernandes", "squad": "Manchester United", "position": "AM", "age": 31, "minutes": 2550,
     "p90": {"goals": 0.34, "xg": 0.32, "shots": 2.9, "shots_on_target": 1.0, "assists": 0.33, "xag": 0.41, "key_passes": 3.4, "passes_completed": 48.0, "progressive_passes": 7.6, "through_balls": 0.85, "pass_completion_pct": 76.5, "progressive_carries": 3.1, "successful_takeons": 1.2, "takeon_success_pct": 49.0, "touches_att_pen": 3.8, "tackles_won": 1.8, "interceptions": 0.9, "blocks": 1.4, "clearances": 0.8, "ball_recoveries": 5.2, "aerials_won_pct": 38.0}},
    {"player": "Heung-min Son", "squad": "Tottenham Hotspur", "position": "W", "age": 33, "minutes": 2320,
     "p90": {"goals": 0.55, "xg": 0.46, "shots": 2.8, "shots_on_target": 1.4, "assists": 0.32, "xag": 0.29, "key_passes": 2.2, "passes_completed": 27.0, "progressive_passes": 3.6, "through_balls": 0.4, "pass_completion_pct": 79.0, "progressive_carries": 4.9, "successful_takeons": 1.9, "takeon_success_pct": 51.0, "touches_att_pen": 5.8, "tackles_won": 0.9, "interceptions": 0.4, "blocks": 0.9, "clearances": 0.4, "ball_recoveries": 3.2, "aerials_won_pct": 31.0}},
    {"player": "Kaoru Mitoma", "squad": "Brighton", "position": "W", "age": 28, "minutes": 2180,
     "p90": {"goals": 0.35, "xg": 0.31, "shots": 2.1, "shots_on_target": 0.8, "assists": 0.26, "xag": 0.28, "key_passes": 2.0, "passes_completed": 26.5, "progressive_passes": 3.2, "through_balls": 0.3, "pass_completion_pct": 78.5, "progressive_carries": 7.9, "successful_takeons": 3.4, "takeon_success_pct": 61.0, "touches_att_pen": 6.5, "tackles_won": 1.4, "interceptions": 0.7, "blocks": 1.1, "clearances": 0.4, "ball_recoveries": 4.2, "aerials_won_pct": 33.0}},
    {"player": "Bruno Guimaraes", "squad": "Newcastle United", "position": "CM", "age": 28, "minutes": 2520,
     "p90": {"goals": 0.18, "xg": 0.15, "shots": 1.5, "shots_on_target": 0.5, "assists": 0.24, "xag": 0.21, "key_passes": 1.8, "passes_completed": 58.5, "progressive_passes": 6.9, "through_balls": 0.45, "pass_completion_pct": 86.4, "progressive_carries": 3.6, "successful_takeons": 1.8, "takeon_success_pct": 66.0, "touches_att_pen": 2.3, "tackles_won": 2.9, "interceptions": 1.4, "blocks": 1.5, "clearances": 1.4, "ball_recoveries": 7.3, "aerials_won_pct": 52.0}},
    {"player": "Pedro Porro", "squad": "Tottenham Hotspur", "position": "FB", "age": 26, "minutes": 2400,
     "p90": {"goals": 0.12, "xg": 0.11, "shots": 1.8, "shots_on_target": 0.6, "assists": 0.28, "xag": 0.29, "key_passes": 2.3, "passes_completed": 52.0, "progressive_passes": 6.2, "through_balls": 0.35, "pass_completion_pct": 77.5, "progressive_carries": 4.1, "successful_takeons": 1.7, "takeon_success_pct": 54.0, "touches_att_pen": 2.4, "tackles_won": 2.9, "interceptions": 1.6, "blocks": 1.7, "clearances": 2.4, "ball_recoveries": 5.9, "aerials_won_pct": 47.0}},
    {"player": "Alexis Mac Allister", "squad": "Liverpool", "position": "CM", "age": 27, "minutes": 2390,
     "p90": {"goals": 0.16, "xg": 0.14, "shots": 1.6, "shots_on_target": 0.5, "assists": 0.22, "xag": 0.24, "key_passes": 1.9, "passes_completed": 64.0, "progressive_passes": 6.5, "through_balls": 0.4, "pass_completion_pct": 87.8, "progressive_carries": 2.7, "successful_takeons": 1.3, "takeon_success_pct": 63.0, "touches_att_pen": 2.0, "tackles_won": 2.6, "interceptions": 1.5, "blocks": 1.4, "clearances": 1.3, "ball_recoveries": 7.1, "aerials_won_pct": 49.0}},
    {"player": "Josko Gvardiol", "squad": "Manchester City", "position": "FB", "age": 24, "minutes": 2340,
     "p90": {"goals": 0.14, "xg": 0.12, "shots": 1.3, "shots_on_target": 0.4, "assists": 0.15, "xag": 0.16, "key_passes": 1.4, "passes_completed": 74.0, "progressive_passes": 6.8, "through_balls": 0.25, "pass_completion_pct": 88.5, "progressive_carries": 4.4, "successful_takeons": 1.5, "takeon_success_pct": 65.0, "touches_att_pen": 2.8, "tackles_won": 2.3, "interceptions": 1.6, "blocks": 1.6, "clearances": 2.5, "ball_recoveries": 6.0, "aerials_won_pct": 59.0}},
    {"player": "Ollie Watkins", "squad": "Aston Villa", "position": "FW", "age": 30, "minutes": 2490,
     "p90": {"goals": 0.58, "xg": 0.56, "shots": 3.1, "shots_on_target": 1.4, "assists": 0.32, "xag": 0.28, "key_passes": 1.5, "passes_completed": 17.5, "progressive_passes": 2.1, "through_balls": 0.2, "pass_completion_pct": 73.0, "progressive_carries": 3.6, "successful_takeons": 1.4, "takeon_success_pct": 49.0, "touches_att_pen": 6.2, "tackles_won": 0.7, "interceptions": 0.3, "blocks": 0.8, "clearances": 0.6, "ball_recoveries": 2.5, "aerials_won_pct": 41.0}},
    {"player": "Eberechi Eze", "squad": "Crystal Palace", "position": "AM", "age": 27, "minutes": 2290,
     "p90": {"goals": 0.38, "xg": 0.35, "shots": 3.1, "shots_on_target": 1.2, "assists": 0.26, "xag": 0.31, "key_passes": 2.5, "passes_completed": 36.0, "progressive_passes": 4.8, "through_balls": 0.5, "pass_completion_pct": 82.0, "progressive_carries": 5.6, "successful_takeons": 3.1, "takeon_success_pct": 59.0, "touches_att_pen": 4.5, "tackles_won": 1.4, "interceptions": 0.6, "blocks": 1.0, "clearances": 0.5, "ball_recoveries": 4.6, "aerials_won_pct": 34.0}}
]

def generate_player_dataset(n_squad_players_per_club: int = 22, seed: int = 42) -> pd.DataFrame:
    """
    Generates a full league dataset including star players and synthetic squad players.
    All count statistics are derived realistically from minutes played and per-90 rates.
    """
    np.random.seed(seed)
    records = []

    # 1. Add Star Players
    for star in STAR_PLAYERS:
        mins = star["minutes"]
        p90 = star["p90"]
        rec = {
            "player": star["player"],
            "squad": star["squad"],
            "position": star["position"],
            "position_category": star["position"],
            "age": star["age"],
            "minutes": mins,
            "league": "Premier League",
            "season": "2025-2026"
        }
        # Compute count stats from per-90 rates: count = round(p90 * mins / 90)
        for stat, val in p90.items():
            if stat.endswith("_pct"):
                rec[stat] = round(val, 1)
            else:
                rec[stat] = max(0, int(round((val * mins) / 90.0)))
        records.append(rec)

    # 2. Add Squad Players for each club
    first_names = ["James", "Jack", "Harry", "George", "Oliver", "Lucas", "Noah", "Leo", "Thomas", "Arthur",
                   "Mateo", "Gabriel", "David", "Carlos", "Alex", "Marcus", "Mason", "Liam", "Ben", "Dan"]
    last_names = ["Smith", "Taylor", "Johnson", "Williams", "Brown", "Jones", "Wilson", "Davies", "Evans", "King",
                  "Garcia", "Silva", "Santos", "Fernandez", "Rodriguez", "Gomez", "Lopez", "Diallo", "Traore", "Mendy"]

    pos_pool = ["FW", "FW", "W", "W", "W", "AM", "AM", "CM", "CM", "CM", "DM", "DM", "FB", "FB", "FB", "FB", "CB", "CB", "CB", "CB"]

    player_id = 1
    existing_stars = {s["player"] for s in STAR_PLAYERS}

    for club in CLUBS:
        for idx in range(n_squad_players_per_club):
            name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
            if name in existing_stars:
                name = f"{name} {chr(65 + (player_id % 26))}"
            existing_stars.add(name)

            pos = pos_pool[idx % len(pos_pool)]
            age = int(np.random.randint(18, 35))
            mins = int(np.random.randint(350, 2750))

            prof = ROLE_PROFILES[pos]
            rec = {
                "player": name,
                "squad": club,
                "position": pos,
                "position_category": pos,
                "age": age,
                "minutes": mins,
                "league": "Premier League",
                "season": "2025-2026"
            }

            # Sample each metric with normal distribution, clipped to non-negative
            for metric, (mean, std) in prof.items():
                val = max(0.0, np.random.normal(mean, std))
                if metric.endswith("_pct"):
                    rec[metric] = round(min(100.0, val), 1)
                else:
                    rec[metric] = max(0, int(round((val * mins) / 90.0)))

            records.append(rec)
            player_id += 1

    df = pd.DataFrame(records)
    return df

def save_default_dataset(output_path: str = "data/raw/premier_league_2025_2026.csv") -> Path:
    """Generates and saves the Premier League 2025-2026 dataset to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_player_dataset()
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} players across {df['squad'].nunique()} clubs saved to {out}")
    return out

if __name__ == "__main__":
    save_default_dataset()
