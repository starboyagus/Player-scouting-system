"""
Matplotlib Radar / Spider Chart Generator for Player Profiles.
Generates publication-quality polar charts comparing player percentile rankings.
Designed with a modern sports analytics aesthetic.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..features.metrics import RADAR_CORE_METRICS

def plot_player_radar(
    player_a_data: pd.Series,
    player_b_data: Optional[pd.Series] = None,
    metrics_dict: Optional[Dict[str, str]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Renders a professional radar chart comparing one or two players on percentile scales (0-100).

    Args:
        player_a_data: pd.Series containing percentile columns for player A (Target).
        player_b_data: Optional pd.Series containing percentile columns for player B (Comparison).
        metrics_dict: Mapping of {raw_or_p90_metric_name: Display_Label}.
                      Defaults to RADAR_CORE_METRICS.
        title: Custom chart title.

    Returns:
        matplotlib.figure.Figure ready for st.pyplot() or saving to disk.
    """
    metrics_map = metrics_dict or RADAR_CORE_METRICS
    labels = list(metrics_map.values())
    metric_keys = list(metrics_map.keys())
    n_vars = len(labels)

    # Compute angles for each metric axis (polar coordinates)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    # Close the polygon loop
    angles += angles[:1]

    # Extract percentiles (fallback to 50 if missing)
    def extract_values(player_series: pd.Series) -> List[float]:
        vals = []
        for k in metric_keys:
            pctile_col = f"{k}_pctile"
            if pctile_col in player_series:
                vals.append(float(player_series[pctile_col]))
            elif k in player_series:
                # If percentile column missing, clip raw value to 0-100 if applicable
                vals.append(float(np.clip(player_series[k], 0, 100)))
            else:
                vals.append(50.0)
        vals += vals[:1]  # close loop
        return vals

    vals_a = extract_values(player_a_data)

    # Theme colors
    bg_color = "#11141c"
    grid_color = "#2a3142"
    text_color = "#e2e8f0"
    color_a = "#00f0ff"  # Neon cyan for Player A
    color_b = "#ff3366"  # Vivid coral/red for Player B

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Rotate so first axis is at top (pi/2) and clockwise
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw grid concentric circles at 20, 40, 60, 80, 100 percentiles
    ax.set_rlabel_position(0)
    plt.yticks(
        [25, 50, 75, 100],
        ["25th", "50th", "75th", "100th"],
        color="#718096",
        size=8
    )
    plt.ylim(0, 100)

    # Styling grid lines
    ax.grid(color=grid_color, linestyle="--", linewidth=0.8, alpha=0.8)
    ax.spines["polar"].set_color(grid_color)

    # Set metric category labels on radial spokes
    plt.xticks(angles[:-1], labels, color=text_color, size=10, weight="bold")

    # Plot Player A (Target)
    name_a = f"{player_a_data.get('player', 'Target')} ({player_a_data.get('squad', '')})"
    ax.plot(angles, vals_a, color=color_a, linewidth=2.4, linestyle="solid", label=name_a)
    ax.fill(angles, vals_a, color=color_a, alpha=0.25)

    # Plot Player B (Comparison Match) if provided
    if player_b_data is not None:
        vals_b = extract_values(player_b_data)
        name_b = f"{player_b_data.get('player', 'Match')} ({player_b_data.get('squad', '')})"
        ax.plot(angles, vals_b, color=color_b, linewidth=2.4, linestyle="solid", label=name_b)
        ax.fill(angles, vals_b, color=color_b, alpha=0.22)

    # Chart Title and Legend
    if title:
        plt.title(title, size=14, color=text_color, weight="bold", pad=28)
    else:
        chart_title = f"Scouting Radar: {name_a}"
        if player_b_data is not None:
            chart_title += f"\nvs {name_b}"
        plt.title(chart_title, size=13, color=text_color, weight="bold", pad=28)

    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.25, 1.12),
        facecolor=bg_color,
        edgecolor=grid_color,
        fontsize=9,
        labelcolor=text_color
    )

    plt.tight_layout()
    return fig
