# ⚽ Football Player Scouting & Similarity System
> **Multi-League Scalable**  
> *A Machine Learning & Systems Engineering Portfolio Project*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)]()

---

## 📌 Project Overview & Engineering Motivation

In modern football recruitment, scouting departments face two fundamental challenges:
1. **The Playing Time Distortion**: Comparing raw counting stats (e.g., total goals, tackles) misleads analysis because playing time varies wildly between starters and rotation players.
2. **Identifying Stylistic Replacement Targets**: When a club needs to replace a key player (e.g., an aging winger or a sold playmaker) or find undervalued talent in other leagues, they need to identify players who replicate the exact **tactical profile**, not just raw output.

This project delivers an end-to-end, modular data science and machine learning system that:
- Normalizes performance data via **Per-90 minute rates** and **within-position percentile rankings**.
- Groups players into tactical archetypes using **K-Means Clustering** with **PCA dimensionality reduction**.
- Computes statistical player twins using **Cosine Similarity** and **Weighted Euclidean Distance**.
- Visualizes player comparison profiles using custom **Matplotlib Polar Radar Charts**.
- Connects to an interactive **Streamlit** dashboard.

---

## 🏗️ Systems Architecture & Directory Layout

The codebase follows professional software engineering separation of concerns:

```
Player Scouting System/
├── config/
│   └── config.yaml           # Single source of truth: metrics, thresholds, hyperparameters
├── data/
│   ├── raw/                        # Raw league CSV datasets
│   └── processed/              # Preprocessed & clustered feature matrices
├── src/
│   ├── data/
│   │   ├── generator.py        # Realistic 25/26 data generator with star & squad distributions
│   │   └── loader.py           # Schema validator and minutes-threshold filter
│   ├── features/
│   │   ├── metrics.py          # Domain taxonomy (Attacking, Creation, Progression, Defending)
│   │   └── preprocessor.py     # Per-90 rate engine, percentile rankings & RobustScaler
│   ├── models/
│   │   ├── cluster.py          # K-Means clustering, Silhouette/Elbow analysis, PCA projection
│   │   └── similarity.py       # Multi-criteria twin search engine (Cosine & Euclidean)
│   ├── visualization/
│   │   ├── radar.py            # Matplotlib percentile radar/spider chart generator
│   │   └── cluster_plot.py     # 2D PCA tactical landscape scatter plot
│   └── pipeline.py             # Unified end-to-end orchestrator service
├── tests/
│   ├── test_data.py            # Unit tests for per-90 math, null guards, and scaling
│   └── test_models.py          # Unit tests for clustering, similarity, and filters
├── requirements.txt
└── README.md
```

---

## 🔬 Mathematical & Statistical Methodology

### 1. Per-90 Normalization
To eliminate the distortion caused by differing playing time, all volume metrics are normalized per standard 90-minute match:
$$\text{Stat}_{90} = \left(\frac{\text{Stat}}{\text{Minutes}}\right) \times 90$$

A minimum playing time filter ($\text{Minutes} \ge 450$) is enforced to avoid small-sample variance.

### 2. Robust Feature Scaling
Unlike standard normalization ($Z = \frac{x - \mu}{\sigma}$), which is vulnerable to extreme statistical outliers (e.g., world-class goalscorers):
$$\tilde{x} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$
where $\text{IQR} = Q_3 - Q_1$. This ensures distance metrics reflect true tactical distributions.

### 3. Tactical Archetype Discovery (K-Means Clustering)
Players are grouped into $k$ tactical archetypes by minimizing the within-cluster sum of squares (WCSS):
$$\arg\min_S \sum_{i=1}^k \sum_{x \in S_i} \|x - \mu_i\|^2$$
Cluster numbers are evaluated via the **Elbow Method** (Inertia) and **Silhouette Analysis**:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

### 4. Player Similarity Metric (Cosine Similarity)
To find players who share the same tactical profile shape regardless of absolute team possession volume:
$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^n u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n v_i^2}}$$

Converted into an intuitive match percentage:
$$\text{Match Percentage} = \max\left(0, \min\left(100, \frac{\cos(\theta) + 1}{2} \times 100\right)\right)$$

---

## Getting Started

### 1. Installation

Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

Execute the end-to-end pipeline:
```bash
python -m src.pipeline
```

### 3. Run Automated Tests

```bash
python -m pytest -v
```

### 4. Run the application

```bash 
python -m streamlit run app/app.py
```

---

## 📈 Scalability to Other Leagues

The system is designed to be multi-league ready:
1. Drop any league CSV into `data/raw/` (e.g., `la_liga_2025_2026.csv`, `serie_a_2025_2026.csv`).
2. Run `pipeline.run(league="la_liga", season="2025-2026")`.
3. The schema validator will verify compatibility and prepare cross-league twin recommendations.
