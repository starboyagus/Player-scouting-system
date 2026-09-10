"""
Unit tests for data ingestion, validation, and feature preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.loader import PlayerDataLoader
from src.features.preprocessor import FeaturePreprocessor

@pytest.fixture
def sample_raw_df():
    return pd.DataFrame([
        {
            "player": "Player A", "squad": "Club 1", "position": "FW",
            "position_category": "FW", "age": 24, "minutes": 900,
            "goals": 9, "xg": 8.0, "shots": 30, "shots_on_target": 15,
            "assists": 2, "xag": 2.5, "key_passes": 10, "passes_completed": 150,
            "progressive_passes": 20, "through_balls": 2, "pass_completion_pct": 75.0,
            "progressive_carries": 25, "successful_takeons": 10, "takeon_success_pct": 50.0,
            "touches_att_pen": 50, "tackles_won": 5, "interceptions": 2,
            "blocks": 7, "clearances": 4, "ball_recoveries": 20, "aerials_won_pct": 40.0,
            "league": "Premier League", "season": "2025-2026"
        },
        {
            "player": "Player B", "squad": "Club 2", "position": "CB",
            "position_category": "CB", "age": 28, "minutes": 1800,
            "goals": 1, "xg": 0.8, "shots": 5, "shots_on_target": 2,
            "assists": 0, "xag": 0.2, "key_passes": 2, "passes_completed": 1000,
            "progressive_passes": 50, "through_balls": 1, "pass_completion_pct": 90.0,
            "progressive_carries": 10, "successful_takeons": 2, "takeon_success_pct": 60.0,
            "touches_att_pen": 5, "tackles_won": 30, "interceptions": 25,
            "blocks": 20, "clearances": 60, "ball_recoveries": 90, "aerials_won_pct": 70.0,
            "league": "Premier League", "season": "2025-2026"
        },
        {
            "player": "Player C", "squad": "Club 1", "position": "FW",
            "position_category": "FW", "age": 20, "minutes": 180,  # Below default min_minutes
            "goals": 1, "xg": 0.5, "shots": 3, "shots_on_target": 1,
            "assists": 0, "xag": 0.1, "key_passes": 1, "passes_completed": 20,
            "progressive_passes": 2, "through_balls": 0, "pass_completion_pct": 70.0,
            "progressive_carries": 3, "successful_takeons": 1, "takeon_success_pct": 40.0,
            "touches_att_pen": 5, "tackles_won": 1, "interceptions": 0,
            "blocks": 1, "clearances": 0, "ball_recoveries": 2, "aerials_won_pct": 30.0,
            "league": "Premier League", "season": "2025-2026"
        }
    ])

def test_per_90_calculation(sample_raw_df):
    preprocessor = FeaturePreprocessor()
    df_p90 = preprocessor.compute_per_90_metrics(sample_raw_df, ["goals", "shots"])

    # Player A: 9 goals in 900 minutes -> exactly 0.9 goals/90
    assert "goals_per90" in df_p90.columns
    assert df_p90.loc[0, "goals_per90"] == 0.9
    # 30 shots in 900 minutes -> exactly 3.0 shots/90
    assert df_p90.loc[0, "shots_per90"] == 3.0

def test_zero_minutes_guard():
    preprocessor = FeaturePreprocessor()
    df_zero = pd.DataFrame([{"player": "Ghost", "minutes": 0, "goals": 0}])
    df_res = preprocessor.compute_per_90_metrics(df_zero, ["goals"])
    assert df_res.loc[0, "goals_per90"] == 0.0

def test_percentile_computation(sample_raw_df):
    preprocessor = FeaturePreprocessor()
    df_p90 = preprocessor.compute_per_90_metrics(sample_raw_df, ["goals"])
    df_pctile = preprocessor.compute_percentiles(df_p90, ["goals_per90"], group_by_position=False)

    assert "goals_per90_pctile" in df_pctile.columns
    # Percentiles must be strictly bounded between 0 and 100
    assert df_pctile["goals_per90_pctile"].min() >= 0.0
    assert df_pctile["goals_per90_pctile"].max() <= 100.0

def test_scaling_shape(sample_raw_df):
    preprocessor = FeaturePreprocessor(scaler_type="robust")
    features = ["goals", "shots"]
    df_scaled, X_scaled = preprocessor.fit_transform(sample_raw_df, features)

    assert X_scaled.shape == (3, 2)
    assert "goals_scaled" in df_scaled.columns
    assert "shots_scaled" in df_scaled.columns
