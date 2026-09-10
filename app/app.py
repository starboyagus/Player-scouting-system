"""
ProScout Analytics - Football Player Scouting & Similarity Dashboard.
Interactive Streamlit application featuring top-bar navigation, tactical cluster exploration,
player directory filtering, and CSV dataset ingestion.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io

from src.pipeline import ScoutingPipeline
from src.visualization.cluster_plot import plot_tactical_clusters
from src.visualization.radar import plot_player_radar
from src.features.metrics import POSITION_GROUPS

# ---------------------------------------------------------
# Page Configuration (Wide Layout, No Sidebar)
# ---------------------------------------------------------
st.set_page_config(
    page_title="ProScout | Football Scouting System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling (Dark Analytics Theme)
st.markdown("""
<style>
    /* Hide default sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 96%;
    }
    .top-header {
        background: linear-gradient(90deg, #11141c 0%, #1a2030 100%);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        border: 1px solid #2d3748;
        margin-bottom: 1.2rem;
    }
    .title-text {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f7fafc;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .subtitle-text {
        font-size: 0.95rem;
        color: #a0aec0;
        margin-top: 0.2rem;
    }
    .metric-card {
        background-color: #171b26;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .archetype-badge {
        display: inline-block;
        background: #2b6cb0;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .instructions-card {
        background: #141824;
        border: 1px solid #2b354f;
        border-radius: 12px;
        padding: 1.8rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Pipeline Caching & Session State
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Scouting Pipeline...")
def get_default_pipeline() -> ScoutingPipeline:
    return ScoutingPipeline.load_default()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = get_default_pipeline()

if "selected_player" not in st.session_state:
    st.session_state.selected_player = "-- Select a Player --"

pipeline: ScoutingPipeline = st.session_state.pipeline
df_processed = pipeline.df_processed
similarity_engine = pipeline.similarity_engine
all_players = similarity_engine.get_all_player_names()

# ---------------------------------------------------------
# Top Bar: Header & System Branding
# ---------------------------------------------------------
st.markdown("""
<div class="top-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="title-text">⚽ PROSCOUT | Player Scouting & Similarity System</h1>
            <p class="subtitle-text">Premier League 2025/26 • Tactical Archetype Discovery (K-Means) • Statistical Twin Finder (Cosine / Euclidean)</p>
        </div>
        <div style="text-align: right;">
            <span style="background: #22543d; color: #9ae6b4; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                ● Active League: Premier League 25/26
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Top Bar: Navigation & Search Controls (No Sidebar)
# ---------------------------------------------------------
col_search, col_filter_exp, col_slider_exp = st.columns([2.5, 1.8, 1.8])

# 1. Player Search Dropdown
with col_search:
    player_options = ["-- Select a Player --"] + all_players
    # Sync with session state if set externally
    current_index = 0
    if st.session_state.selected_player in player_options:
        current_index = player_options.index(st.session_state.selected_player)

    selected_player = st.selectbox(
        "🔍 Search or Select Target Player",
        options=player_options,
        index=current_index,
        help="Type or select any player from the 2025/26 season to find their statistical twin."
    )
    st.session_state.selected_player = selected_player

# 2. Filtering Dropdown (Expander)
with col_filter_exp:
    with st.expander("⚙️ Tactical & Scouting Filters", expanded=False):
        sim_metric = st.selectbox(
            "Similarity Metric",
            options=["cosine", "euclidean"],
            format_func=lambda x: "Cosine Similarity (Tactical Shape)" if x == "cosine" else "Euclidean Distance (Output Volume)",
            help="Cosine measures stylistic profile alignment. Euclidean measures exact per-90 volume proximity."
        )
        pos_filter_option = st.selectbox(
            "Position Constraint",
            options=["Any Position", "Same Position Only", "Forwards (FW/W)", "Midfielders (AM/CM/DM)", "Defenders (FB/CB)"]
        )
        exclude_same_club = st.checkbox(
            "Exclude Current Club",
            value=True,
            help="Crucial for transfer scouting: excludes players from the target player's own team."
        )

# 3. Sliders Dropdown (Expander)
with col_slider_exp:
    with st.expander("🎚️ Thresholds & Limits", expanded=False):
        top_n = st.slider("Number of Similar Matches (Top N)", min_value=3, max_value=15, value=5)
        max_age = st.slider("Maximum Age (Transfer Targets)", min_value=18, max_value=36, value=35)
        min_mins = st.slider("Minimum Minutes Played", min_value=300, max_value=2500, value=450, step=50)

# Resolve position filter choices
same_position_only = (pos_filter_option == "Same Position Only")
pos_list_filter = None
if pos_filter_option == "Forwards (FW/W)":
    pos_list_filter = ["FW", "W"]
elif pos_filter_option == "Midfielders (AM/CM/DM)":
    pos_list_filter = ["AM", "CM", "DM"]
elif pos_filter_option == "Defenders (FB/CB)":
    pos_list_filter = ["FB", "CB"]

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab_scout, tab_clusters, tab_directory, tab_import = st.tabs([
    "🎯 Scouting & Player Twins",
    "🗺️ Tactical Landscape (Clusters)",
    "📋 Player Directory & Filter",
    "📤 Import Statistics (CSV)"
])

# =========================================================
# TAB 1: SCOUTING & PLAYER TWINS
# =========================================================
with tab_scout:
    if selected_player == "-- Select a Player --":
        # Welcome Page explaining system and how to use it
        st.markdown("""
        <div class="instructions-card">
            <h2 style="color: #63b3ed; margin-top: 0;">👋 Welcome to ProScout Analytics</h2>
            <p style="font-size: 1.05rem; color: #cbd5e0; line-height: 1.6;">
                This system uses <strong>K-Means Unsupervised Clustering</strong> and <strong>Multidimensional Vector Similarity</strong>
                to find <em>statistical player twins</em> — footballers who replicate the tactical style and performance characteristics
                of target players.
            </p>
            <hr style="border-color: #2d3748; margin: 1.5rem 0;">
            <h4 style="color: #e2e8f0;">🚀 How to Get Started:</h4>
            <ol style="color: #a0aec0; line-height: 1.9; font-size: 0.95rem;">
                <li>Use the <strong>top search bar</strong> above to pick any Premier League 2025/26 player.</li>
                <li>Adjust the <strong>Tactical Filters</strong> (e.g., restrict to same position, or filter out players from the same club).</li>
                <li>Tune the <strong>Thresholds</strong> to scout younger replacement candidates or filter by minutes.</li>
                <li>Explore the <strong>Tactical Landscape</strong> tab to see all player clusters in 2D PCA space.</li>
                <li>Need to scout other leagues? Drop in your CSV in the <strong>Import Statistics</strong> tab!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🌟 Quick Start: Try One of These Top Profiles")
        quick_cols = st.columns(5)
        sample_stars = [
            ("Mohamed Salah", "Liverpool", "W"),
            ("Erling Haaland", "Man City", "FW"),
            ("Bukayo Saka", "Arsenal", "W"),
            ("Cole Palmer", "Chelsea", "AM"),
            ("William Saliba", "Arsenal", "CB")
        ]
        for col, (star_name, club, pos) in zip(quick_cols, sample_stars):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-weight: 700; color: #f7fafc; font-size: 1.05rem;">{star_name}</div>
                    <div style="color: #718096; font-size: 0.85rem;">{club} • {pos}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Scout {star_name.split()[-1]}", key=f"quick_{star_name}", use_container_width=True):
                    st.session_state.selected_player = star_name
                    st.rerun()

    else:
        # Target Player has been selected!
        target_player_data = similarity_engine.find_player(selected_player)

        if target_player_data is None:
            st.error(f"Player '{selected_player}' not found.")
        else:
            # 1. Target Player Profile Header
            p_name = target_player_data["player"]
            p_squad = target_player_data["squad"]
            p_pos = target_player_data["position"]
            p_age = target_player_data["age"]
            p_mins = target_player_data["minutes"]
            p_archetype = target_player_data.get("archetype", "Tactical Profile")

            c_info1, c_info2, c_info3, c_info4, c_info5 = st.columns(5)
            c_info1.metric("Player", p_name)
            c_info2.metric("Club", p_squad)
            c_info3.metric("Position", p_pos)
            c_info4.metric("Age", f"{p_age} yrs")
            c_info5.metric("Minutes Played", f"{p_mins:,} min")

            # 1. Role Diagnosis & Best Role Explanation
            role_diag = similarity_engine.get_player_role_diagnosis(p_name)
            
            with st.container(border=True):
                col_r1, col_r2 = st.columns([3, 1])
                with col_r1:
                    st.caption("OPTIMAL TACTICAL ROLE")
                    st.markdown(f"<h3 style='margin: 0; color: #38bdf8;'>{p_archetype}</h3>", unsafe_allow_html=True)
                with col_r2:
                    st.markdown("<div style='text-align: right; padding-top: 8px;'><span style='background: #1e293b; padding: 6px 12px; border-radius: 6px; border: 1px solid #334155; font-size: 0.85rem; color: #cbd5e1;'>⭐ Role Fit: <strong>Top Match</strong></span></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top: 10px; color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;'>{role_diag['narrative']}</div>", unsafe_allow_html=True)

            # Highlight statistics explaining WHY that role is the best
            st.markdown("##### 📌 Key Statistical Drivers for this Role (Standout Percentiles)")
            driver_cols = st.columns(len(role_diag["top_traits"]))
            for col, trait in zip(driver_cols, role_diag["top_traits"]):
                with col:
                    pct = trait["percentile"]
                    pct_fmt = f"{pct:.2f}%".replace(".", ",")
                    badge_color = "#10b981" if pct >= 80 else "#38bdf8" if pct >= 65 else "#f59e0b"
                    with st.container(border=True):
                        st.caption(trait["name"])
                        st.markdown(f"<div style='font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin: 2px 0 6px 0;'>{trait['value']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<span style='background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}55; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>{pct_fmt}ile</span>", unsafe_allow_html=True)

            st.markdown("<hr style='border-color: #2d3748; margin: 20px 0;'>", unsafe_allow_html=True)

            # 2. Similarity Search & Candidate Targets
            try:
                matches = similarity_engine.get_similar_players(
                    player_name=p_name,
                    top_n=top_n,
                    metric=sim_metric,
                    same_position_only=same_position_only,
                    position_filter=pos_list_filter,
                    exclude_same_club=exclude_same_club,
                    max_age=max_age,
                    min_minutes=min_mins
                )

                st.subheader(f"👥 Closest Statistical Targets for {p_name}")
                st.caption(f"Ranked by {sim_metric.title()} similarity across tactical dimensions. Select a candidate below to launch direct comparison.")

                if matches.empty:
                    st.warning("No players matched the specified filter criteria. Try relaxing age or position restrictions in the top bar.")
                else:
                    # Interactive Candidate Selector
                    match_names = matches["player"].tolist()
                    default_compare = match_names[0] if match_names else None

                    # Candidate Cards with Explanations of WHY each was selected
                    for idx, row in matches.iterrows():
                        c_name = row["player"]
                        c_squad = row["squad"]
                        c_pos = row["position"]
                        c_age = row["age"]
                        c_score = row["similarity_score"]

                        # Extract main stats on why this candidate was matched
                        reasons = similarity_engine.get_candidate_match_reasons(p_name, c_name, top_k=3)
                        reasons_html = " • ".join(reasons)

                        with st.container(border=True):
                            c_card1, c_card2 = st.columns([4, 1])
                            with c_card1:
                                st.markdown(f"**#{idx+1} {c_name}** &nbsp;•&nbsp; <span style='color: #94a3b8;'>{c_squad} • {c_pos} • {c_age} yrs</span>", unsafe_allow_html=True)
                                st.caption(f"🎯 **Key Match Congruence**: {reasons_html}")
                            with c_card2:
                                st.markdown(f"<div style='text-align: right; padding-top: 4px;'><span style='background: #0284c7; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 700;'>{c_score}% Match</span></div>", unsafe_allow_html=True)

                    st.markdown("<hr style='border-color: #2d3748; margin: 25px 0;'>", unsafe_allow_html=True)

                    # 3. Deep-Dive Head-to-Head Comparison
                    st.markdown("### ⚔️ Head-to-Head Comparison & Recruitment Deep Dive")
                    
                    compare_player_name = st.selectbox(
                        "Select Target Candidate to Compare:",
                        options=match_names,
                        index=0,
                        help="Choose which candidate to compare directly against your target player on the radar and metric tables."
                    )

                    target_data = similarity_engine.find_player(p_name)
                    compare_data = similarity_engine.find_player(compare_player_name)

                    if target_data is not None and compare_data is not None:
                        col_radar, col_scout = st.columns([1.5, 1.1])

                        # A. Dual Radar Chart
                        with col_radar:
                            st.markdown("##### 🕸️ Tactical Percentile Radar")
                            st.caption(f"Comparing **{p_name}** (Cyan) vs **{compare_player_name}** (Coral) on 0-100% scales.")
                            fig_radar = plot_player_radar(
                                player_a_data=target_data,
                                player_b_data=compare_data
                            )
                            st.pyplot(fig_radar, use_container_width=True)
                            plt.close(fig_radar)

                        # B. Category Similarity Bars & Recruitment Verdict
                        with col_scout:
                            st.markdown("##### 📊 Tactical Category Congruence")
                            cat_scores = similarity_engine.get_category_similarity(p_name, compare_player_name)
                            
                            for cat_title, score in cat_scores.items():
                                col_c1, col_c2 = st.columns([3, 1])
                                col_c1.markdown(f"<span style='font-size: 0.88rem; color: #e2e8f0;'>{cat_title}</span>", unsafe_allow_html=True)
                                col_c2.markdown(f"<span style='font-weight: 700; color: #38bdf8; font-size: 0.88rem;'>{score}%</span>", unsafe_allow_html=True)
                                st.progress(score / 100.0)

                            st.markdown("<br>", unsafe_allow_html=True)
                            verdict = similarity_engine.get_scout_recruitment_verdict(p_name, compare_player_name)

                            with st.container(border=True):
                                st.markdown("##### 📋 Scout Recruitment Verdict")
                                st.markdown(f"**📅 Age Profile**: {verdict.get('age_verdict', '')}")
                                st.markdown("**⚖️ Tactical Trade-offs**:")
                                for trade_off in verdict.get('trade_offs', []):
                                    st.markdown(f"- {trade_off}")

                        # C. Detailed Metric Difference Table (Percentiles Only, 2 Decimals "00,00%")
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("##### 📑 Detailed Metric Breakdown & Statistical Differences (Percentiles Only)")
                        st.caption("Displays relative percentile rankings (00,00% to 100,00%) and candidate difference.")
                        
                        h2h_df = similarity_engine.get_head_to_head_comparison(p_name, compare_player_name)
                        
                        # Style delta column with badges
                        def style_diff(val):
                            if isinstance(val, str):
                                if val.startswith("+"):
                                    return "color: #34d399; font-weight: 700;"
                                elif val.startswith("-"):
                                    return "color: #f87171; font-weight: 700;"
                            return ""

                        styled_h2h = h2h_df.style.map(style_diff, subset=["Difference (%)"])
                        st.dataframe(styled_h2h, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error computing similarity: {e}")

# =========================================================
# TAB 2: TACTICAL LANDSCAPE (CLUSTERS & PCA)
# =========================================================
with tab_clusters:
    st.markdown("### 🗺️ Tactical Landscape (Unsupervised K-Means Archetypes)")
    st.caption("Dimensionality reduction using 2D Principal Component Analysis (PCA). Points situated closer together share similar playing styles.")

    target_name_param = selected_player if selected_player != "-- Select a Player --" else None
    top_match_param = None
    if target_name_param:
        try:
            m_quick = similarity_engine.get_similar_players(target_name_param, top_n=1)
            if not m_quick.empty:
                top_match_param = m_quick.iloc[0]["player"]
        except Exception:
            pass

    c_plot, c_archetypes = st.columns([2.8, 1.2])

    with c_plot:
        fig = plot_tactical_clusters(
            df_processed,
            target_player_name=target_name_param,
            matched_player_name=top_match_param
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c_archetypes:
        st.markdown("#### 🏷️ Discovered Archetypes")
        archetype_counts = df_processed["archetype"].value_counts()
        for arch_name, count in archetype_counts.items():
            st.markdown(f"""
            <div style="background: #171c26; border-left: 3px solid #38bdf8; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px;">
                <div style="font-weight: 600; font-size: 0.9rem; color: #f7fafc;">{arch_name}</div>
                <div style="font-size: 0.8rem; color: #a0aec0;">{count} players in dataset</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 3: PLAYER DIRECTORY & FILTER
# =========================================================
with tab_directory:
    st.markdown("### 📋 League Player Directory")
    st.caption("Search, filter, and inspect player statistics across all clubs and positions.")

    c_dir1, c_dir2, c_dir3 = st.columns([2, 1.5, 1.5])
    with c_dir1:
        text_query = st.text_input("Filter by Player Name or Club", "")
    with c_dir2:
        pos_choices = ["All Positions"] + sorted(df_processed["position"].unique().tolist())
        sel_dir_pos = st.selectbox("Position", pos_choices)
    with c_dir3:
        arch_choices = ["All Archetypes"] + sorted(df_processed["archetype"].dropna().unique().tolist())
        sel_dir_arch = st.selectbox("Tactical Archetype", arch_choices)

    filtered_dir_df = df_processed.copy()
    if text_query:
        mask = (
            filtered_dir_df["player"].str.contains(text_query, case=False, na=False) |
            filtered_dir_df["squad"].str.contains(text_query, case=False, na=False)
        )
        filtered_dir_df = filtered_dir_df[mask]

    if sel_dir_pos != "All Positions":
        filtered_dir_df = filtered_dir_df[filtered_dir_df["position"] == sel_dir_pos]

    if sel_dir_arch != "All Archetypes":
        filtered_dir_df = filtered_dir_df[filtered_dir_df["archetype"] == sel_dir_arch]

    show_cols = [
        "player", "squad", "position", "age", "minutes", "archetype",
        "goals_per90", "xg_per90", "assists_per90", "xag_per90",
        "progressive_passes_per90", "progressive_carries_per90",
        "tackles_won_per90", "interceptions_per90"
    ]
    avail_cols = [c for c in show_cols if c in filtered_dir_df.columns]

    st.markdown(f"**Showing {len(filtered_dir_df)} players:**")
    st.dataframe(
        filtered_dir_df[avail_cols].sort_values(by="minutes", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# TAB 4: IMPORT STATISTICS (CSV)
# =========================================================
with tab_import:
    st.markdown("### 📤 Import Custom Statistics (Multi-League Expansion)")
    st.markdown("""
    Easily expand ProScout to **La Liga, Serie A, Bundesliga, MLS**, or import actual FBref/Kaggle CSV exports.
    The system automatically validates columns against `config/config.yaml` and recalculates clustering and similarity models.
    """)

    uploaded_file = st.file_uploader("Upload Player Statistics CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully: {len(uploaded_df)} rows detected.")

            # Run schema validation
            try:
                pipeline.loader.validate_schema(uploaded_df)
                st.info("✅ Schema Validation Passed: All required metrics and ID columns are present!")

                if st.button("Apply Dataset & Re-run Scouting Pipeline", type="primary"):
                    with st.spinner("Processing new dataset, calculating per-90 metrics, and fitting K-Means..."):
                        # Save temp raw CSV
                        target_path = Path("data/raw") / f"custom_{uploaded_file.name}"
                        uploaded_df.to_csv(target_path, index=False)
                        # Re-run pipeline with custom file
                        new_pipeline = ScoutingPipeline(pipeline.config)
                        new_pipeline.loader.raw_dir = Path("data/raw")
                        new_pipeline.run(file_path=str(target_path))

                        st.session_state.pipeline = new_pipeline
                        st.session_state.selected_player = "-- Select a Player --"
                        st.success("🎉 Pipeline updated with custom dataset! Switch to the Scouting tab to explore.")
                        st.rerun()

            except ValueError as ve:
                st.error(f"❌ Schema Validation Failed: {ve}")
                st.markdown("Please ensure your CSV contains all required columns defined in `config/config.yaml`.")

        except Exception as ex:
            st.error(f"Error reading CSV file: {ex}")

    with st.expander("ℹ️ Expected CSV Format & Column Checklist"):
        st.write("Your CSV must contain the following columns:")
        st.code("""
# ID Columns:
player, squad, position, age, minutes, league, season

# Counting Metrics (Total Season Volume):
goals, xg, shots, shots_on_target, assists, xag, key_passes, passes_completed,
progressive_passes, through_balls, progressive_carries, successful_takeons,
touches_att_pen, tackles_won, interceptions, blocks, clearances, ball_recoveries

# Rate Metrics (%):
pass_completion_pct, takeon_success_pct, aerials_won_pct
        """)
