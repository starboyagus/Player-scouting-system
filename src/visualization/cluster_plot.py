"""
Matplotlib Cluster Visualization for Tactical Archetypes.
Visualizes player tactical roles projected onto 2D PCA space, highlighting target and match players.
"""

from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_tactical_clusters(
    df: pd.DataFrame,
    target_player_name: Optional[str] = None,
    matched_player_name: Optional[str] = None,
    title: str = "Tactical Landscape (PCA Projection of Playing Styles)"
) -> plt.Figure:
    """
    Renders 2D PCA scatter plot showing player clusters, with optional highlight markers.
    """
    bg_color = "#11141c"
    grid_color = "#222838"
    text_color = "#e2e8f0"

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Color palette for clusters
    palette = ["#38bdf8", "#a855f7", "#ec4899", "#f59e0b", "#10b981", "#6366f1", "#14b8a6", "#f43f5e"]
    clusters = sorted(df["cluster"].unique())

    # Plot base player points by cluster
    for i, c_id in enumerate(clusters):
        c_df = df[df["cluster"] == c_id]
        archetype = c_df["archetype"].iloc[0] if "archetype" in c_df.columns else f"Cluster {c_id}"
        color = palette[i % len(palette)]

        ax.scatter(
            c_df["pca_x"],
            c_df["pca_y"],
            c=color,
            alpha=0.35,
            s=45,
            label=archetype,
            edgecolors="none"
        )

    # Highlight Target Player if specified
    if target_player_name:
        target_row = df[df["player"].str.lower() == target_player_name.lower()]
        if not target_row.empty:
            tx = target_row["pca_x"].iloc[0]
            ty = target_row["pca_y"].iloc[0]
            t_name = target_row["player"].iloc[0]

            ax.scatter(
                tx, ty,
                c="#00f0ff",
                s=200,
                marker="*",
                edgecolors="#ffffff",
                linewidth=1.5,
                zorder=10,
                label=f"Target: {t_name}"
            )
            ax.annotate(
                t_name,
                (tx, ty),
                xytext=(8, 8),
                textcoords="offset points",
                color="#00f0ff",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=bg_color, ec="#00f0ff", lw=1.0)
            )

    # Highlight Matched Player if specified
    if matched_player_name:
        match_row = df[df["player"].str.lower() == matched_player_name.lower()]
        if not match_row.empty:
            mx = match_row["pca_x"].iloc[0]
            my = match_row["pca_y"].iloc[0]
            m_name = match_row["player"].iloc[0]

            ax.scatter(
                mx, my,
                c="#ff3366",
                s=160,
                marker="D",
                edgecolors="#ffffff",
                linewidth=1.5,
                zorder=9,
                label=f"Match: {m_name}"
            )
            ax.annotate(
                m_name,
                (mx, my),
                xytext=(8, -14),
                textcoords="offset points",
                color="#ff3366",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=bg_color, ec="#ff3366", lw=1.0)
            )

    ax.set_xlabel("Principal Component 1 (Offensive Threat & Direct Action)", color=text_color, size=10)
    ax.set_ylabel("Principal Component 2 (Progression & Defensive Solidity)", color=text_color, size=10)
    ax.set_title(title, color=text_color, size=13, weight="bold", pad=15)

    ax.tick_params(colors=text_color, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    ax.grid(True, color=grid_color, linestyle=":", alpha=0.6)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        facecolor=bg_color,
        edgecolor=grid_color,
        fontsize=8.5,
        labelcolor=text_color
    )

    plt.tight_layout()
    return fig
