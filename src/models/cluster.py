"""
Tactical Clustering Model for Football Players.
Uses K-Means clustering to discover tactical roles and archetypes from multi-dimensional performance data.
Evaluates cluster quality via Inertia (Elbow method) and Silhouette scores, and provides PCA projections.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Archetype naming heuristics based on dominant tactical features
ARCHETYPE_FALLBACK_LABELS = {
    0: "Direct Goal Threat / Poacher",
    1: "Creative Playmaker / Chance Creator",
    2: "Dynamic Ball-Carrying Winger",
    3: "Box-to-Box Engine / Progressor",
    4: "Defensive Anchor / Ball Winner",
    5: "Ball-Playing Central Defender"
}

class PlayerClusteringModel:
    """
    K-Means clustering engine for grouping players into tactical archetypes.
    Includes PCA dimensionality reduction for visual inspection.
    """

    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self.pca: Optional[PCA] = None
        self.cluster_names: Dict[int, str] = {}
        self.feature_cols: List[str] = []

    def evaluate_optimal_k(
        self,
        X: np.ndarray,
        k_range: range = range(3, 11)
    ) -> pd.DataFrame:
        """
        Computes Inertia and Silhouette scores across a range of k values
        to assist in selecting the optimal number of clusters.
        """
        results = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            inertia = km.inertia_
            sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
            results.append({
                "k": k,
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil, 4)
            })
        return pd.DataFrame(results)

    def fit_predict(
        self,
        df: pd.DataFrame,
        X_scaled: np.ndarray,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Fits KMeans and 2D PCA on scaled feature matrix.
        Appends 'cluster', 'archetype', 'pca_x', and 'pca_y' columns to the dataframe.
        """
        self.feature_cols = feature_cols
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=15
        )
        labels = self.kmeans.fit_predict(X_scaled)

        # Fit 2D PCA for spatial visualization
        self.pca = PCA(n_components=2, random_state=self.random_state)
        pca_coords = self.pca.fit_transform(X_scaled)

        df_out = df.copy()
        df_out["cluster"] = labels
        df_out["pca_x"] = np.round(pca_coords[:, 0], 3)
        df_out["pca_y"] = np.round(pca_coords[:, 1], 3)

        # Generate descriptive tactical labels for each cluster
        self.cluster_names = self._generate_archetype_labels(df_out)
        df_out["archetype"] = df_out["cluster"].map(self.cluster_names)

        return df_out

    def _generate_archetype_labels(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        Assigns human-interpretable tactical names to clusters dynamically based on
        dominant position and top percentile attributes of the cluster centroid.
        """
        names = {}
        for cluster_id in range(self.n_clusters):
            c_players = df[df["cluster"] == cluster_id]
            if c_players.empty:
                names[cluster_id] = f"Tactical Cluster {cluster_id}"
                continue

            top_pos = c_players["position"].mode()[0] if not c_players["position"].empty else "All"

            # Profile dominant traits
            mean_goals = c_players.get("goals_per90", pd.Series([0])).mean()
            mean_shots = c_players.get("shots_per90", pd.Series([0])).mean()
            mean_prog_carries = c_players.get("progressive_carries_per90", pd.Series([0])).mean()
            mean_prog_passes = c_players.get("progressive_passes_per90", pd.Series([0])).mean()
            mean_tackles = c_players.get("tackles_won_per90", pd.Series([0])).mean()
            mean_clearances = c_players.get("clearances_per90", pd.Series([0])).mean()
            mean_key_passes = c_players.get("key_passes_per90", pd.Series([0])).mean()

            if top_pos in ["FW", "Striker"] or mean_goals > 0.45 or mean_shots > 2.8:
                name = "Direct Goal Threat / Box Striker"
            elif top_pos in ["W"] or mean_prog_carries > 4.5:
                name = "Attacking Winger / Ball Carrier"
            elif top_pos in ["AM"] or (mean_key_passes > 2.0 and mean_prog_passes > 5.0):
                name = "Advanced Playmaker / Chance Creator"
            elif top_pos in ["CM"]:
                name = "Box-to-Box Midfielder / Engine"
            elif top_pos in ["DM"] or (mean_tackles > 2.3 and mean_prog_passes > 5.0):
                name = "Defensive Anchor / Deep Distributor"
            elif top_pos in ["FB"]:
                name = "Overlapping Wingback / Wide Progressor"
            elif top_pos in ["CB"] or mean_clearances > 3.5:
                name = "Defensive Stopper / Central Shield"
            else:
                name = f"Tactical Profile ({top_pos})"

            names[cluster_id] = f"{name} [{top_pos}]"

        return names

    def get_cluster_centroids(self) -> pd.DataFrame:
        """Returns cluster centroids in original scaled feature space."""
        if self.kmeans is None:
            raise RuntimeError("Model must be fitted first.")
        return pd.DataFrame(self.kmeans.cluster_centers_, columns=self.feature_cols)
