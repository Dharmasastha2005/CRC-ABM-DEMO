# CRC Tumour ABM — Interactive Research Demo

An interactive agent-based model of colorectal cancer radiotherapy response, built as a Streamlit research demo.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## What it models

A 30×30 spatial lattice where each voxel (20 µm) holds one cell agent. The simulation integrates:

- **Oxygen diffusion** — explicit finite-difference PDE on a 30×30 grid
- **LQ radiobiology** — Linear-Quadratic cell kill with Alper–Howard–Flanders OER correction for hypoxia
- **Metabolic adaptation** — aerobic ATP (normoxia) vs. glycolytic fallback (hypoxia threshold pO₂ < 0.3)
- **CTL immune dynamics** — recruitment to decaying cells, diffusion, and hypoxic exclusion (toggleable)
- **Fractionated RT protocols** — configurable dose per fraction and number of fractions

## Parameters you can tune

| Parameter | Range | Effect |
|-----------|-------|--------|
| Fraction dose | 1–10 Gy | Direct LQ kill per fraction |
| N fractions | 1–6 | Total course length |
| Alpha (α) | 0.05–0.50 | Linear kill component |
| Beta (β) | 0.005–0.06 | Quadratic kill component |
| CTL kill probability | 0.05–0.50 | Immune pressure on tumour |
| Random seed | 0–9999 | Reproducible stochastic runs |

## Run locally

```bash
git clone https://github.com/your-username/crc-abm-demo
cd crc-abm-demo
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account → select this repo → set `app.py` as the entry point
4. Click Deploy — you get a public URL in ~2 minutes

## Project context

This demo accompanies a computational biology project on CRC radiotherapy modelling, including:
- GNN/GAT analysis of cardiovascular disease-gene networks (BIOXPLORE 2026, IEEE)
- ODE tumour–immune escape/elimination modelling
- Global sensitivity analysis (LHS) of radiosensitivity parameters
- Cross-validation against experimental confluence data (Frontiers dataset)

Full analysis notebooks: `CRC_Model_Clean.ipynb`

---

*Grid: 30×30 · Voxel: 20 µm · Duration: 145 hours · Built with Streamlit + NumPy*
