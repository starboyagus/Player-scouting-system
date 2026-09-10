"""
Player Similarity Engine for Football Scouting.
Finds statistical player twins using Cosine Similarity and Euclidean Distance.
Supports multi-criteria filtering (position, age, club, minutes) and category-level breakdowns.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from ..features.metrics import METRIC_CATEGORIES

class PlayerSimilarityEngine:
    """
    Engine for finding closest statistical matches to any target player.
    Provides customizable filters for real-world recruitment workflows.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        scaled_feature_cols: Optional[List[str]] = None
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.scaled_feature_cols = scaled_feature_cols or [f"{c}_scaled" for c in feature_cols]

        # Verify scaled feature columns exist in DataFrame
        for col in self.scaled_feature_cols:
            if col not in self.df.columns:
                raise KeyError(f"Scaled feature column '{col}' not found in DataFrame.")

        self.X_scaled = self.df[self.scaled_feature_cols].values
        # Precompute cosine similarity matrix for fast lookups
        self._cosine_sim_matrix = cosine_similarity(self.X_scaled)

    def find_player(self, query: str) -> Optional[pd.Series]:
        """Searches for a player by exact or partial name (case-insensitive)."""
        query_lower = query.strip().lower()
        exact = self.df[self.df["player"].str.lower() == query_lower]
        if not exact.empty:
            return exact.iloc[0]

        partial = self.df[self.df["player"].str.lower().str.contains(query_lower, na=False)]
        if not partial.empty:
            return partial.iloc[0]
        return None

    def get_all_player_names(self) -> List[str]:
        """Returns sorted list of all unique player names in the dataset."""
        return sorted(self.df["player"].unique().tolist())

    def get_similar_players(
        self,
        player_name: str,
        top_n: int = 5,
        metric: str = "cosine",
        same_position_only: bool = False,
        position_filter: Optional[List[str]] = None,
        exclude_same_club: bool = False,
        max_age: Optional[int] = None,
        min_minutes: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Finds the top N most similar players to the target player.

        Args:
            player_name: Name of target player to find matches for.
            top_n: Number of similar players to return.
            metric: 'cosine' or 'euclidean'.
            same_position_only: If True, only search within player's exact position.
            position_filter: List of positions to restrict matches to (e.g. ['FW', 'W']).
            exclude_same_club: If True, exclude players currently playing for the same club.
            max_age: Maximum age filter (e.g., scouting younger replacements).
            min_minutes: Minimum minutes played threshold.

        Returns:
            DataFrame of top N matching players with similarity scores and key metadata.
        """
        target = self.find_player(player_name)
        if target is None:
            raise ValueError(f"Player '{player_name}' not found in dataset.")

        target_idx = target.name
        target_vector = self.X_scaled[target_idx].reshape(1, -1)

        # Compute raw scores
        if metric == "cosine":
            sim_scores = self._cosine_sim_matrix[target_idx]
            # Convert [-1, 1] cosine score to percentage [0, 100]
            match_percentages = np.clip((sim_scores + 1.0) / 2.0 * 100.0, 0.0, 100.0)
        elif metric == "euclidean":
            dists = euclidean_distances(target_vector, self.X_scaled)[0]
            # Normalized distance to percentage: 100 / (1 + dist)
            match_percentages = 100.0 / (1.0 + (dists / np.sqrt(len(self.feature_cols))))
        else:
            raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine' or 'euclidean'.")

        # Build candidate DataFrame
        candidates = self.df.copy()
        candidates["similarity_score"] = np.round(match_percentages, 1)

        # Apply filters
        # 1. Exclude the target player themselves
        candidates = candidates[candidates.index != target_idx]

        # 2. Same position only
        if same_position_only:
            candidates = candidates[candidates["position"] == target["position"]]

        # 3. Specific position filter list
        if position_filter:
            candidates = candidates[candidates["position"].isin(position_filter)]

        # 4. Exclude same club
        if exclude_same_club:
            candidates = candidates[candidates["squad"] != target["squad"]]

        # 5. Age filter
        if max_age is not None:
            candidates = candidates[candidates["age"] <= max_age]

        # 6. Minutes filter
        if min_minutes is not None:
            candidates = candidates[candidates["minutes"] >= min_minutes]

        # Sort descending by similarity score
        results = candidates.sort_values(by="similarity_score", ascending=False).head(top_n)

        display_cols = [
            "player", "squad", "position", "age", "minutes",
            "similarity_score", "archetype", "cluster"
        ]
        available_cols = [c for c in display_cols if c in results.columns]

        return results[available_cols].reset_index(drop=True)

    def get_category_similarity(
        self,
        player_a_name: str,
        player_b_name: str
    ) -> Dict[str, float]:
        """
        Computes sub-similarity percentages across distinct tactical categories:
        Attacking, Creation, Progression, and Defending.
        """
        p_a = self.find_player(player_a_name)
        p_b = self.find_player(player_b_name)

        if p_a is None or p_b is None:
            raise ValueError("Both players must exist in dataset.")

        cat_scores = {}
        for category, raw_cols in METRIC_CATEGORIES.items():
            # Find scaled columns belonging to this category
            valid_scaled_cols = [f"{c}_scaled" for c in raw_cols if f"{c}_scaled" in self.df.columns]
            if not valid_scaled_cols:
                continue

            vec_a = self.df.loc[p_a.name, valid_scaled_cols].values.reshape(1, -1)
            vec_b = self.df.loc[p_b.name, valid_scaled_cols].values.reshape(1, -1)

            cos_sim = cosine_similarity(vec_a, vec_b)[0][0]
            pct = np.clip((cos_sim + 1.0) / 2.0 * 100.0, 0.0, 100.0)
            cat_scores[category] = round(float(pct), 1)

        return cat_scores

    def get_head_to_head_comparison(
        self,
        player_a_name: str,
        player_b_name: str,
        metric_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Returns a side-by-side percentile comparison table between two players.
        Formats strictly as percentage with 2 decimals using comma decimal separator (e.g. '00,00%').
        """
        p_a = self.find_player(player_a_name)
        p_b = self.find_player(player_b_name)
        if p_a is None or p_b is None:
            raise ValueError("Both players must exist.")

        metrics = metric_cols or self.feature_cols
        records = []
        for m in metrics:
            pctile_a = p_a.get(f"{m}_pctile", np.nan)
            pctile_b = p_b.get(f"{m}_pctile", np.nan)

            if pd.isna(pctile_a):
                pctile_a = 50.0
            if pd.isna(pctile_b):
                pctile_b = 50.0

            diff = pctile_b - pctile_a

            val_a_str = f"{float(pctile_a):.2f}%".replace(".", ",")
            val_b_str = f"{float(pctile_b):.2f}%".replace(".", ",")

            if abs(diff) < 0.005:
                diff_str = "00,00%"
            else:
                sign = "+" if diff > 0 else ""
                diff_str = f"{sign}{float(diff):.2f}%".replace(".", ",")

            records.append({
                "Metric": m.replace("_per90", " /90").replace("_", " ").title(),
                f"{p_a['player']} (%)": val_a_str,
                f"{p_b['player']} (%)": val_b_str,
                "Difference (%)": diff_str
            })

        return pd.DataFrame(records)

    def get_player_role_diagnosis(self, player_name: str) -> Dict[str, Any]:
        """
        Diagnoses why a tactical role/archetype fits the player best,
        highlighting their standout percentiles and statistical drivers.
        """
        player = self.find_player(player_name)
        if player is None:
            raise ValueError(f"Player '{player_name}' not found.")

        archetype = player.get("archetype", "Balanced Profile")

        # Collect all percentiles for this player
        traits = []
        for m in self.feature_cols:
            pctile_col = f"{m}_pctile"
            val = player.get(m, 0.0)
            pctile = player.get(pctile_col, 50.0)
            clean_name = m.replace("_per90", " /90").replace("_", " ").title()

            traits.append({
                "metric_key": m,
                "name": clean_name,
                "value": round(float(val), 2),
                "percentile": round(float(pctile), 1)
            })

        # Sort by percentile descending
        traits.sort(key=lambda x: x["percentile"], reverse=True)
        top_traits = traits[:5]

        # Domain narrative based on top traits
        top_keys = [t["metric_key"] for t in top_traits[:3]]
        explanation_parts = []
        for t in top_traits[:3]:
            explanation_parts.append(f"{t['name']} ({t['percentile']}th %ile, {t['value']})")

        narrative = (
            f"Classified as **{archetype}** due to standout performance in "
            f"{', '.join(explanation_parts)}."
        )

        return {
            "archetype": archetype,
            "narrative": narrative,
            "top_traits": top_traits
        }

    def get_candidate_match_reasons(
        self,
        target_name: str,
        candidate_name: str,
        top_k: int = 3
    ) -> List[str]:
        """
        Identifies the primary statistical drivers explaining WHY this candidate
        was selected as a close stylistic match for the target.
        """
        target = self.find_player(target_name)
        cand = self.find_player(candidate_name)
        if target is None or cand is None:
            return []

        differences = []
        for m in self.feature_cols:
            scaled_col = f"{m}_scaled"
            if scaled_col in self.df.columns:
                target_val = target.get(m, 0.0)
                cand_val = cand.get(m, 0.0)
                target_scaled = target.get(scaled_col, 0.0)
                cand_scaled = cand.get(scaled_col, 0.0)

                # Distance in normalized space
                scaled_diff = abs(target_scaled - cand_scaled)
                # Weight by how important/high this metric is for target
                target_pctile = target.get(f"{m}_pctile", 50.0)

                differences.append({
                    "metric": m.replace("_per90", " /90").replace("_", " ").title(),
                    "scaled_diff": scaled_diff,
                    "target_pctile": target_pctile,
                    "target_val": round(target_val, 2),
                    "cand_val": round(cand_val, 2)
                })

        # Find metrics where both players are closest in scaled space, prioritizing target's strengths
        differences.sort(key=lambda x: (x["scaled_diff"] * 0.7 - (x["target_pctile"] / 100.0) * 0.3))

        reasons = []
        for item in differences[:top_k]:
            reasons.append(
                f"**{item['metric']}**: Near-identical output ({item['cand_val']} vs {item['target_val']})"
            )

        return reasons

    def get_scout_recruitment_verdict(
        self,
        target_name: str,
        candidate_name: str
    ) -> Dict[str, Any]:
        """
        Synthesizes a structured scouting recruitment verdict highlighting age delta,
        stylistic alignment, and tactical trade-offs.
        """
        target = self.find_player(target_name)
        cand = self.find_player(candidate_name)
        if target is None or cand is None:
            return {}

        age_diff = int(cand["age"]) - int(target["age"])
        if age_diff < -2:
            age_verdict = f"Youth Upside: {abs(age_diff)} years younger with high development ceiling."
        elif age_diff > 2:
            age_verdict = f"Veteran Experience: {age_diff} years older with immediate leadership value."
        else:
            age_verdict = f"Direct Age Peer: Same competitive window ({cand['age']} vs {target['age']})."

        # Compare attacking vs defending delta
        t_goals = target.get("goals_per90", 0.0)
        c_goals = cand.get("goals_per90", 0.0)
        t_tackles = target.get("tackles_won_per90", 0.0)
        c_tackles = cand.get("tackles_won_per90", 0.0)

        trade_offs = []
        if c_goals > t_goals + 0.1:
            trade_offs.append(f"+ Higher goal scoring rate (+{round(c_goals - t_goals, 2)} goals/90)")
        elif c_goals < t_goals - 0.1:
            trade_offs.append(f"- Lower goal conversion (-{round(t_goals - c_goals, 2)} goals/90)")

        if c_tackles > t_tackles + 0.3:
            trade_offs.append(f"+ Increased defensive workrate (+{round(c_tackles - t_tackles, 2)} tackles/90)")
        elif c_tackles < t_tackles - 0.3:
            trade_offs.append(f"- Less defensive engagement (-{round(t_tackles - c_tackles, 2)} tackles/90)")

        if not trade_offs:
            trade_offs.append("Balanced output across attacking and defensive metrics.")

        return {
            "age_verdict": age_verdict,
            "trade_offs": trade_offs
        }
