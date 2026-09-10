"""
Feature Preprocessor for Player Scouting.
Performs per-90 rate normalization, percentile conversions, and statistical scaling
(RobustScaler / StandardScaler) tailored for soccer performance metrics.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from .metrics import get_all_feature_cols

class FeaturePreprocessor:
    """
    Transforms raw player counting statistics into normalized per-90 metrics,
    calculates percentile rankings, and prepares scaled feature matrices for ML models.
    """

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type.lower()
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'robust' or 'standard'.")

        self.feature_cols: List[str] = []
        self.is_fitted: bool = False

    def compute_per_90_metrics(
        self,
        df: pd.DataFrame,
        per_90_source_cols: List[str],
        rate_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculates per-90 rates: Stat_90 = (Stat / Minutes) * 90.
        Adds new columns with '_per90' suffix and retains percentage columns.
        """
        df_out = df.copy()

        if "minutes" not in df_out.columns:
            raise KeyError("DataFrame must contain 'minutes' column for per-90 calculation.")

        # Guard against 0 or negative minutes
        safe_minutes = df_out["minutes"].replace(0, np.nan)

        for col in per_90_source_cols:
            if col in df_out.columns:
                target_col = f"{col}_per90"
                df_out[target_col] = ((df_out[col] / safe_minutes) * 90.0).round(3)
                # Fill any NaNs created by 0 minutes with 0
                df_out[target_col] = df_out[target_col].fillna(0.0)

        # Ensure rate columns exist and are clean
        if rate_cols:
            for col in rate_cols:
                if col in df_out.columns:
                    df_out[col] = df_out[col].fillna(df_out[col].median())

        return df_out

    def compute_percentiles(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        group_by_position: bool = True
    ) -> pd.DataFrame:
        """
        Computes percentile rank (0-100) for each feature.
        If group_by_position is True, calculates percentiles relative to peers in the same position group.
        """
        df_out = df.copy()

        for col in feature_cols:
            if col not in df_out.columns:
                continue
            pct_col_name = f"{col}_pctile"

            if group_by_position and "position" in df_out.columns:
                df_out[pct_col_name] = (
                    df_out.groupby("position")[col]
                    .rank(pct=True, method="average")
                    * 100.0
                ).round(1)
            else:
                df_out[pct_col_name] = (
                    df_out[col].rank(pct=True, method="average") * 100.0
                ).round(1)

            # Fill any NaNs with 50th percentile (median)
            df_out[pct_col_name] = df_out[pct_col_name].fillna(50.0)

        return df_out

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fits the scaler on the feature columns and transforms the feature matrix.
        Returns the updated DataFrame and the scaled numpy feature matrix.
        """
        self.feature_cols = feature_cols or get_all_feature_cols()
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Features missing from DataFrame: {missing}")

        X = df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True

        # Attach scaled features back for convenience
        df_scaled = df.copy()
        for i, col in enumerate(self.feature_cols):
            df_scaled[f"{col}_scaled"] = X_scaled[:, i]

        return df_scaled, X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms new data using the already-fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before calling transform().")
        X = df[self.feature_cols].values
        return self.scaler.transform(X)
