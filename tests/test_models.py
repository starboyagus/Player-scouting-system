"""
Unit tests for K-Means Clustering and Player Similarity Engine.
"""

import pytest
import numpy as np
import pandas as pd
from src.pipeline import ScoutingPipeline
from src.models.cluster import PlayerClusteringModel
from src.models.similarity import PlayerSimilarityEngine

@pytest.fixture(scope="module")
def pipeline():
    return ScoutingPipeline.load_default()

def test_clustering_output(pipeline):
    df = pipeline.df_processed
    assert "cluster" in df.columns
    assert "archetype" in df.columns
    assert "pca_x" in df.columns
    assert "pca_y" in df.columns

    # Clusters should be valid integers within [0, n_clusters-1]
    n_clusters = pipeline.cluster_model.n_clusters
    assert df["cluster"].min() >= 0
    assert df["cluster"].max() < n_clusters

def test_optimal_k_evaluation(pipeline):
    X = pipeline.preprocessor.transform(pipeline.df_processed)
    res_df = pipeline.cluster_model.evaluate_optimal_k(X, k_range=range(3, 6))
    assert "k" in res_df.columns
    assert "inertia" in res_df.columns
    assert "silhouette_score" in res_df.columns
    assert len(res_df) == 3

def test_similarity_search_and_self_exclusion(pipeline):
    sim_engine = pipeline.similarity_engine
    target_name = "Bukayo Saka"
    matches = sim_engine.get_similar_players(target_name, top_n=5)

    assert len(matches) == 5
    # Target player should NOT be returned in their own match list
    assert target_name not in matches["player"].values
    # Similarity score should be between 0 and 100
    assert (matches["similarity_score"] >= 0).all()
    assert (matches["similarity_score"] <= 100).all()
    # Matches should be in descending order of similarity
    assert matches["similarity_score"].is_monotonic_decreasing

def test_similarity_same_position_filter(pipeline):
    sim_engine = pipeline.similarity_engine
    target_name = "William Saliba"  # CB
    matches = sim_engine.get_similar_players(
        target_name,
        top_n=5,
        same_position_only=True
    )
    assert (matches["position"] == "CB").all()

def test_similarity_exclude_same_club(pipeline):
    sim_engine = pipeline.similarity_engine
    target_name = "Mohamed Salah"  # Liverpool
    matches = sim_engine.get_similar_players(
        target_name,
        top_n=5,
        exclude_same_club=True
    )
    assert (matches["squad"] != "Liverpool").all()

def test_category_similarity_breakdown(pipeline):
    sim_engine = pipeline.similarity_engine
    cats = sim_engine.get_category_similarity("Mohamed Salah", "Bukayo Saka")
    assert isinstance(cats, dict)
    for cat_name, score in cats.items():
        assert 0.0 <= score <= 100.0

def test_role_diagnosis_and_match_reasons(pipeline):
    sim_engine = pipeline.similarity_engine
    diag = sim_engine.get_player_role_diagnosis("Bukayo Saka")
    assert "archetype" in diag
    assert "narrative" in diag
    assert len(diag["top_traits"]) > 0

    reasons = sim_engine.get_candidate_match_reasons("Bukayo Saka", "Mohamed Salah")
    assert isinstance(reasons, list)
    assert len(reasons) > 0

    verdict = sim_engine.get_scout_recruitment_verdict("Bukayo Saka", "Mohamed Salah")
    assert "age_verdict" in verdict
    assert "trade_offs" in verdict
