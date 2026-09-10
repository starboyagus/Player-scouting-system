"""
Data Ingestion and Validation Layer.
Handles loading raw player statistics, validating schemas, and applying minutes filtering.
Designed to be league-agnostic to support multi-league expansion.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")

def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class PlayerDataLoader:
    """
    Robust data loader for football player scouting statistics.
    Enforces schema integrity and filters for statistical significance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.raw_dir = Path(self.config["data"]["raw_dir"])
        self.min_minutes_default = self.config["data"].get("min_minutes", 450)

    def load_league_data(
        self,
        league: Optional[str] = None,
        season: Optional[str] = None,
        file_path: Optional[str] = None,
        min_minutes: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Loads player data for a given league and season or from a specific CSV path.
        Applies minimum minutes played threshold to eliminate low-sample noise.
        """
        if file_path:
            target_path = Path(file_path)
        else:
            league = league or self.config["data"]["current_league"]
            season = season or self.config["data"]["current_season"]
            filename = f"{league.lower().replace(' ', '_')}_{season.replace('-', '_')}.csv"
            target_path = self.raw_dir / filename

        if not target_path.exists():
            raise FileNotFoundError(
                f"Data file does not exist at {target_path}. "
                "Ensure data generator or CSV file is in place."
            )

        df = pd.read_csv(target_path)
        self.validate_schema(df)

        # Apply minutes threshold
        threshold = min_minutes if min_minutes is not None else self.min_minutes_default
        initial_count = len(df)
        df_filtered = df[df["minutes"] >= threshold].copy()
        filtered_count = len(df_filtered)

        print(f"Loaded {initial_count} players from {target_path.name}. "
              f"Retained {filtered_count} players with >= {threshold} minutes.")

        return df_filtered

    def validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validates that required ID columns and metric columns exist in DataFrame.
        Raises ValueError with details if columns are missing.
        """
        id_cols = self.config["features"]["id_columns"]
        per_90_metrics = self.config["features"]["per_90_metrics"]
        rate_metrics = self.config["features"]["rate_metrics"]

        required_cols = set(id_cols + per_90_metrics + rate_metrics)
        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(list(missing))}")

    def list_available_leagues(self) -> List[str]:
        """Lists all league CSV datasets available in raw directory."""
        if not self.raw_dir.exists():
            return []
        return [f.stem for f in self.raw_dir.glob("*.csv")]
