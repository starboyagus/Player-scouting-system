"""
End-to-End Scouting Pipeline Orchestrator.
Connects data ingestion, feature engineering, K-Means clustering, and similarity scoring
into a unified, cached interface ready for application and dashboard consumption.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from .data.loader import PlayerDataLoader, load_config
from .features.preprocessor import FeaturePreprocessor
from .features.metrics import get_all_feature_cols
from .models.cluster import PlayerClusteringModel
from .models.similarity import PlayerSimilarityEngine

class ScoutingPipeline:
    """
    Unified pipeline that orchestrates the entire player scouting workflow.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.loader = PlayerDataLoader(self.config)
        self.preprocessor = FeaturePreprocessor(
            scaler_type=self.config.get("preprocessing", {}).get("scaler", "robust")
        )
        self.cluster_model = PlayerClusteringModel(
            n_clusters=self.config.get("clustering", {}).get("n_clusters", 6),
            random_state=self.config.get("clustering", {}).get("random_state", 42)
        )
        self.similarity_engine: Optional[PlayerSimilarityEngine] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []

    def run(
        self,
        league: Optional[str] = None,
        season: Optional[str] = None,
        min_minutes: Optional[int] = None,
        save_processed: bool = True
    ) -> "ScoutingPipeline":
        """
        Executes the full pipeline:
        1. Load raw data with minutes threshold
        2. Compute per-90 metrics & within-position percentiles
        3. Fit scaler and scale feature matrix
        4. Fit K-Means clustering & 2D PCA projection
        5. Initialize Similarity Engine
        """
        # 1. Ingest Data
        df_raw = self.loader.load_league_data(
            league=league,
            season=season,
            min_minutes=min_minutes
        )

        # 2. Compute per-90 rates
        per_90_source = self.config["features"]["per_90_metrics"]
        rate_source = self.config["features"]["rate_metrics"]
        df_p90 = self.preprocessor.compute_per_90_metrics(df_raw, per_90_source, rate_source)

        # Feature columns list
        self.feature_cols = get_all_feature_cols()

        # 3. Compute percentiles (within position groups)
        df_pctile = self.preprocessor.compute_percentiles(
            df_p90,
            self.feature_cols,
            group_by_position=True
        )

        # 4. Scale features
        df_scaled, X_scaled = self.preprocessor.fit_transform(df_pctile, self.feature_cols)

        # 5. Fit Clustering & PCA
        df_clustered = self.cluster_model.fit_predict(df_scaled, X_scaled, self.feature_cols)

        # 6. Initialize Similarity Engine
        self.df_processed = df_clustered
        self.similarity_engine = PlayerSimilarityEngine(
            df=self.df_processed,
            feature_cols=self.feature_cols
        )

        # 7. Optionally save processed artifact
        if save_processed:
            proc_dir = Path(self.config["data"]["processed_dir"])
            proc_dir.mkdir(parents=True, exist_ok=True)
            out_file = proc_dir / f"processed_{self.config['data']['current_league']}_{self.config['data']['current_season'].replace('-', '_')}.csv"
            self.df_processed.to_csv(out_file, index=False)
            print(f"Processed dataset saved to {out_file}")

        return self

    @classmethod
    def load_default(cls) -> "ScoutingPipeline":
        """Convenience factory method to instantiate and run default pipeline."""
        pipeline = cls()
        return pipeline.run()

if __name__ == "__main__":
    pipeline = ScoutingPipeline.load_default()
    print("\n--- Pipeline Run Summary ---")
    print(f"Total analyzed players: {len(pipeline.df_processed)}")
    print(f"Discovered Archetypes:\n{pipeline.cluster_model.cluster_names}")
    print("\nTest Twin Search: Mohamed Salah")
    matches = pipeline.similarity_engine.get_similar_players("Mohamed Salah", top_n=5)
    print(matches[["player", "squad", "position", "similarity_score", "archetype"]])
