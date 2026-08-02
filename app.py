"""
CRC Agent-Based Model — Interactive Research Demo
Darren | Biomedical Engineering + AI/ML
Streamlit app: run with  `streamlit run app.py`
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
import streamlit as st
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRC Tumour ABM",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* dark background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* hero banner */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d2137 100%);
    border: 1px solid #1f6feb;
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(31,111,235,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-title span { color: #58a6ff; }
.hero-sub {
    font-size: 1.05rem;
    color: #8b949e;
    font-weight: 400;
    margin: 0;
    max-width: 600px;
}
.hero-badge {
    display: inline-block;
    background: rgba(31,111,235,0.15);
    border: 1px solid rgba(31,111,235,0.4);
    color: #58a6ff;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.metric-label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
}
.metric-value.green { color: #3fb950; }
.metric-value.blue  { color: #58a6ff; }
.metric-value.red   { color: #f85149; }
.metric-value.orange{ color: #d29922; }
.metric-delta {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 2px;
}

/* section headers */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #58a6ff;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.5rem;
}

/* tab styling */
[data-testid="stTabs"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: #8b949e !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
}

/* button */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* sidebar sliders */
[data-testid="stSlider"] > div > div > div > div {
    background-color: #1f6feb !important;
}

/* figure containers */
.fig-container {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* info callout */
.callout {
    background: rgba(31,111,235,0.08);
    border-left: 3px solid #1f6feb;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: #8b949e;
}

/* divider */
.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL (reduced 30×30 grid for interactive speed)
# ══════════════════════════════════════════════════════════════════════════════

SIZE = 30
DX   = 20e-6
D_O2 = 1.8e-9
DT   = (0.25 * DX**2) / D_O2
EPS_AEROBIC    = 32.0
EPS_GLYCOLYTIC = 2.0
THETA_HYPOXIA  = 0.3

BASE_SENSITIVITY = {
    1: [0.20, 0.02],
    3: [0.08, 0.01],
    4: [0.30, 0.03],
}


class CRC_Clinical_ABM:
    def __init__(self, oer_max=3.0, k_o2=0.03):
        self.grid     = np.zeros((SIZE, SIZE), dtype=int)
        self.ox_grid  = np.ones((SIZE, SIZE))
        self.atp_grid = np.zeros((SIZE, SIZE))
        self.history  = []
        self.rt_kill_log = []
        self.sensitivity = {k: list(v) for k, v in BASE_SENSITIVITY.items()}
        self.proliferation_prob = 0.06
        self.oer_max = oer_max
        self.k_o2    = k_o2

    def seed_tumor(self, seed=42):
        np.random.seed(seed)
        center = SIZE // 2
        for r in range(SIZE):
            for c in range(SIZE):
                if np.sqrt((r - center)**2 + (c - center)**2) < 8:
                    self.grid[r, c] = 4 if np.random.rand() < 0.3 else 1
        self.update_metabolism()

    def update_metabolism(self):
        lap = (np.roll(self.ox_grid, 1, 0) + np.roll(self.ox_grid, -1, 0) +
               np.roll(self.ox_grid, 1, 1) + np.roll(self.ox_grid, -1, 1) - 4 * self.ox_grid)
        self.ox_grid  += (D_O2 * DT / DX**2) * lap - np.where(self.grid > 0, 0.02, 0)
        self.ox_grid   = np.clip(self.ox_grid, 0.05, 1.0)
        for r in range(SIZE):
            for c in range(SIZE):
                o2 = self.ox_grid[r, c]
                self.atp_grid[r, c] = (EPS_AEROBIC * o2 if o2 > THETA_HYPOXIA
                                       else EPS_GLYCOLYTIC * 1.5)

    def apply_rt_fraction(self, dose, alpha, beta):
        killed = 0
        for r in range(SIZE):
            for c in range(SIZE):
                state = self.grid[r, c]
                if state in [1, 3, 4]:
                    a, b = self.sensitivity[state]
                    o2   = self.ox_grid[r, c]
                    oer  = (self.oer_max * o2 + self.k_o2) / (o2 + self.k_o2)
                    a_p  = a * (oer / self.oer_max)
                    b_p  = b * (oer / self.oer_max)**2
                    sf   = np.exp(-(a_p * dose + b_p * dose**2))
                    if np.random.rand() > sf:
                        self.grid[r, c] = 2
                        killed += 1
        self.rt_kill_log.append(killed)

    def run_step(self, t):
        self.update_metabolism()
        ng = self.grid.copy()
        for r in range(SIZE):
            for c in range(SIZE):
                s = self.grid[r, c]
                if s in [1, 3, 4]:
                    if np.random.rand() < self.proliferation_prob * (self.atp_grid[r, c] / EPS_AEROBIC):
                        self._divide(r, c, ng)
                    if self.atp_grid[r, c] < 0.8:
                        ng[r, c] = 0
                elif s == 2:
                    if np.random.rand() < 0.15:
                        ng[r, c] = 0
        self.grid = ng
        np_ = int(np.sum(self.grid == 4))
        self.history.append({
            'Hour': t,
            'Sensitive':     int(np.sum(self.grid == 1)),
            'Resistant':     int(np.sum(self.grid == 3)),
            'Decaying':      int(np.sum(self.grid == 2)),
            'Proliferative': np_,
            'TCP': float(np.exp(-np_)),
        })

    def _divide(self, r, c, g):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = (r+dr)%SIZE, (c+dc)%SIZE
            if g[nr, nc] == 0:
                g[nr, nc] = self.grid[r, c]
                break


class CRC_Immune_ABM(CRC_Clinical_ABM):
    def __init__(self, immune_kill=0.2, **kw):
        super().__init__(**kw)
        self.immune_grid = np.zeros((SIZE, SIZE))
        self.immune_kill = immune_kill

    def update_immune_dynamics(self):
        recruit = np.where(self.grid == 2, 0.1, 0)
        lap = (np.roll(self.immune_grid, 1, 0) + np.roll(self.immune_grid, -1, 0) +
               np.roll(self.immune_grid, 1, 1) + np.roll(self.immune_grid, -1, 1) - 4 * self.immune_grid)
        self.immune_grid += 0.1 * lap + recruit
        self.immune_grid  = np.where(self.ox_grid < 0.3, self.immune_grid * 0.8, self.immune_grid)
        self.immune_grid  = np.clip(self.immune_grid, 0, 1)

    def apply_immune_killing(self):
        for r in range(SIZE):
            for c in range(SIZE):
                if self.grid[r, c] > 0 and np.random.rand() < self.immune_grid[r, c] * self.immune_kill:
                    self.grid[r, c] = 0

    def run_step(self, t):
        self.update_metabolism()
        self.update_immune_dynamics()
        self.apply_immune_killing()
        ng = self.grid.copy()
        for r in range(SIZE):
            for c in range(SIZE):
                s = self.grid[r, c]
                if s in [1, 3, 4]:
                    if np.random.rand() < self.proliferation_prob * (self.atp_grid[r, c] / EPS_AEROBIC):
                        self._divide(r, c, ng)
                    if self.atp_grid[r, c] < 0.8:
                        ng[r, c] = 0
                elif s == 2:
                    if np.random.rand() < 0.15:
                        ng[r, c] = 0
        self.grid = ng
        np_ = int(np.sum(self.grid == 4))
        self.history.append({
            'Hour': t,
            'Sensitive':     int(np.sum(self.grid == 1)),
            'Resistant':     int(np.sum(self.grid == 3)),
            'Decaying':      int(np.sum(self.grid == 2)),
            'Proliferative': np_,
            'TCP': float(np.exp(-np_)),
        })


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER (cached by parameters)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_simulation(dose, n_fractions, alpha, beta, immune_kill, enable_immune, seed):
    ModelClass = CRC_Immune_ABM if enable_immune else CRC_Clinical_ABM
    kw = dict(oer_max=3.0, k_o2=0.03)
    if enable_immune:
        model = CRC_Immune_ABM(immune_kill=immune_kill, **kw)
    else:
        model = CRC_Clinical_ABM(**kw)

    # Override LQ params
    model.sensitivity = {
        1: [alpha,        beta],
        3: [alpha * 0.4,  beta * 0.5],
        4: [alpha * 1.5,  beta * 1.5],
    }

    model.seed_tumor(seed=seed)
    total_hours   = 145
    rt_hours      = [i * (total_hours // max(n_fractions, 1))
                     for i in range(n_fractions)]

    snapshots      = []   # (hour, grid, immune_grid or None)
    snap_interval  = max(1, total_hours // 6)

    for t in range(total_hours):
        if t in rt_hours:
            model.apply_rt_fraction(dose, alpha, beta)
        model.run_step(t)
        if t % snap_interval == 0 or t == total_hours - 1:
            ig = model.immune_grid.copy() if enable_immune else None
            snapshots.append((t, model.grid.copy(), ig))

    return pd.DataFrame(model.history), snapshots, model.rt_kill_log, rt_hours


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

CELL_COLORS  = ['#111827', '#3b82f6', '#ef4444', '#f59e0b', '#10b981']
CELL_LABELS  = ['Empty', 'Quiescent', 'RT-Hit (Decaying)', 'Hypoxic', 'Proliferative']
CMAP_TUMOR   = ListedColormap(CELL_COLORS)
CMAP_IMMUNE  = LinearSegmentedColormap.from_list("ctl", ["#111827", "#a855f7"])

PLOT_STYLE = {
    "figure.facecolor": "#161b22",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#8b949e",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "text.color":       "#e6edf3",
    "font.family":      "sans-serif",
}


def apply_style():
    plt.rcParams.update(PLOT_STYLE)


def make_snapshot_fig(snapshots, enable_immune):
    apply_style()
    n = len(snapshots)
    rows = 2 if enable_immune else 1
    fig, axes = plt.subplots(rows, n, figsize=(3.2 * n, 3.2 * rows),
                             facecolor="#161b22")
    if n == 1:
        axes = np.array([[axes]] if rows == 1 else [[axes[0]], [axes[1]]])
    elif rows == 1:
        axes = axes[np.newaxis, :]

    ext = [0, SIZE * DX * 1e6, 0, SIZE * DX * 1e6]
    for j, (hr, grid, ig) in enumerate(snapshots):
        ax = axes[0, j]
        ax.imshow(grid, cmap=CMAP_TUMOR, vmin=0, vmax=4,
                  extent=ext, origin="lower")
        ax.set_title(f"Hour {hr}", fontsize=9, color="#e6edf3", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Tumour\nState", fontsize=8, color="#8b949e")

        if enable_immune and ig is not None:
            ax2 = axes[1, j]
            ax2.imshow(ig, cmap=CMAP_IMMUNE, vmin=0, vmax=1,
                       extent=ext, origin="lower")
            ax2.set_xticks([]); ax2.set_yticks([])
            if j == 0:
                ax2.set_ylabel("CTL\nDensity", fontsize=8, color="#8b949e")

    legend_el = [Patch(facecolor=CELL_COLORS[i], label=CELL_LABELS[i])
                 for i in range(1, 5)]
    fig.legend(handles=legend_el, loc="lower center", ncol=4,
               frameon=False, fontsize=8,
               labelcolor="#c9d1d9", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def make_dynamics_fig(df, rt_hours, dose):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#161b22")

    # stackplot
    ax = axes[0]
    ax.stackplot(df["Hour"],
                 df["Sensitive"], df["Decaying"],
                 df["Resistant"], df["Proliferative"],
                 labels=["Quiescent", "RT-Hit", "Hypoxic", "Proliferative"],
                 colors=["#3b82f6", "#ef4444", "#f59e0b", "#10b981"],
                 alpha=0.85)
    for rt in rt_hours:
        ax.axvline(rt, color="#ffffff", lw=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Time (Hours)"); ax.set_ylabel("Cell Count")
    ax.set_title("Population Dynamics", fontweight="bold", color="#e6edf3")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.2,
              labelcolor="#c9d1d9")

    # TCP
    ax2 = axes[1]
    ax2.plot(df["Hour"], df["TCP"], color="#58a6ff", lw=2)
    ax2.fill_between(df["Hour"], df["TCP"], alpha=0.1, color="#58a6ff")
    for rt in rt_hours:
        ax2.axvline(rt, color="#ffffff", lw=0.8, linestyle="--", alpha=0.4)
    ax2.set_yscale("log")
    ax2.set_xlabel("Time (Hours)"); ax2.set_ylabel("TCP (log scale)")
    ax2.set_title("Tumour Control Probability", fontweight="bold", color="#e6edf3")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


def make_oxygen_fig(snapshots):
    """Show oxygen map from the last snapshot grid (use fresh model state)."""
    apply_style()
    # We don't store ox_grid in snapshots — show a proxy heatmap from grid states
    last_grid = snapshots[-1][1]
    # approximate o2: empty=1.0, proliferative=0.4, sensitive=0.7, resistant=0.2
    proxy_o2 = np.where(last_grid == 0, 1.0,
                np.where(last_grid == 4, 0.42,
                np.where(last_grid == 1, 0.72,
                np.where(last_grid == 3, 0.22, 0.55))))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#161b22")
    im = axes[0].imshow(proxy_o2, cmap="YlGnBu_r", vmin=0, vmax=1, origin="lower")
    axes[0].set_title("Oxygen Distribution (pO₂)", color="#e6edf3", fontweight="bold")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im, ax=axes[0], label="O₂ Level", fraction=0.046)

    # ATP proxy
    atp_proxy = np.where(proxy_o2 > THETA_HYPOXIA,
                         EPS_AEROBIC * proxy_o2,
                         EPS_GLYCOLYTIC * 1.5)
    im2 = axes[1].imshow(atp_proxy, cmap="magma", origin="lower")
    axes[1].set_title("Metabolic Activity (ATP)", color="#e6edf3", fontweight="bold")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(im2, ax=axes[1], label="ATP Units", fraction=0.046)

    fig.tight_layout()
    return fig


def make_kill_fig(rt_kill_log, rt_hours, dose, n_fractions):
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#161b22")
    xs = list(range(1, len(rt_kill_log) + 1))
    bars = ax.bar(xs, rt_kill_log,
                  color=["#3b82f6", "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"][:len(xs)],
                  alpha=0.85, width=0.6)
    ax.set_xlabel("Fraction #"); ax.set_ylabel("Cells Killed")
    ax.set_title(f"Kill per {dose} Gy Fraction", fontweight="bold", color="#e6edf3")
    ax.set_xticks(xs)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, val in zip(bars, rt_kill_log):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9, color="#e6edf3")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p class="section-label">Protocol</p>', unsafe_allow_html=True)

    dose = st.slider("Fraction dose (Gy)", 1.0, 10.0, 5.0, 0.5)
    n_fractions = st.slider("Number of fractions", 1, 6, 4)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Radiosensitivity (LQ)</p>', unsafe_allow_html=True)

    alpha = st.slider("Alpha (α)", 0.05, 0.50, 0.20, 0.01,
                      help="Linear component of cell kill. Higher = more sensitive.")
    beta  = st.slider("Beta (β)", 0.005, 0.060, 0.020, 0.005,
                      help="Quadratic component. α/β ratio drives fractionation sensitivity.")
    ab_ratio = alpha / beta if beta > 0 else 0
    st.caption(f"α/β ratio: **{ab_ratio:.1f} Gy**  ({'Tumour-like' if ab_ratio > 8 else 'Late-responding'})")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Immune Microenvironment</p>', unsafe_allow_html=True)

    enable_immune = st.toggle("Enable CTL dynamics", value=True)
    immune_kill = st.slider("CTL kill probability", 0.05, 0.50, 0.20, 0.05,
                            disabled=not enable_immune)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Simulation</p>', unsafe_allow_html=True)
    seed = st.number_input("Random seed", 0, 9999, 42, 1)

    run_btn = st.button("▶  Run Simulation")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#484f58; line-height:1.6;">
    Grid: 30×30 · Voxel: 20 µm<br>
    ODE: LQ + Alper–Howard–Flanders OER<br>
    Diffusion: explicit finite-difference<br>
    Duration: 145 hours
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🔬 Research Demo · CRC Oncology Lab</div>
  <h1 class="hero-title">Colorectal Cancer<br><span>Tumour ABM</span></h1>
  <p class="hero-sub">
    Spatial agent-based model of CRC radiotherapy response — oxygen diffusion,
    LQ cell kill, metabolic adaptation, and CTL immune dynamics on a 30×30 lattice.
    Adjust parameters on the left and run the simulation.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Run / load state ─────────────────────────────────────────────────────────
if "results" not in st.session_state or run_btn:
    with st.spinner("Running simulation…"):
        df, snapshots, rt_kill_log, rt_hours = run_simulation(
            dose, n_fractions, alpha, beta, immune_kill, enable_immune, int(seed)
        )
    st.session_state["results"] = (df, snapshots, rt_kill_log, rt_hours)
else:
    df, snapshots, rt_kill_log, rt_hours = st.session_state["results"]

# ── Metrics row ──────────────────────────────────────────────────────────────
final = df.iloc[-1]
total_live = int(final["Sensitive"] + final["Resistant"] + final["Proliferative"])
total_init = int(df.iloc[0]["Sensitive"] + df.iloc[0]["Resistant"] + df.iloc[0]["Proliferative"])
reduction  = ((total_init - total_live) / total_init * 100) if total_init > 0 else 0
avg_tcp    = df["TCP"].mean()
total_killed = sum(rt_kill_log)

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Final Live Cells</div>
    <div class="metric-value {'green' if total_live < total_init * 0.3 else 'orange'}">{total_live:,}</div>
    <div class="metric-delta">started at {total_init:,}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Tumour Reduction</div>
    <div class="metric-value {'green' if reduction > 50 else 'orange'}">{reduction:.1f}%</div>
    <div class="metric-delta">vs. pre-treatment burden</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Mean TCP</div>
    <div class="metric-value blue">{avg_tcp:.3f}</div>
    <div class="metric-delta">clonogenic control index</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">RT-Killed Cells</div>
    <div class="metric-value red">{total_killed:,}</div>
    <div class="metric-delta">across {len(rt_kill_log)} fraction(s)</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">α/β Ratio</div>
    <div class="metric-value orange">{ab_ratio:.1f} Gy</div>
    <div class="metric-delta">LQ fractionation sensitivity</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧫  Spatial Snapshots",
    "📈  Population Dynamics",
    "🧬  Metabolism",
    "📊  RT Kill Analysis",
])

with tab1:
    st.markdown('<p class="section-label">Tumour lattice — every ~24 hours</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="callout">
    Each voxel (20 µm) holds one cell agent. Colour encodes metabolic and radiation state.
    CTL infiltration (purple) is shown beneath when immune dynamics are enabled.
    </div>
    """, unsafe_allow_html=True)
    fig = make_snapshot_fig(snapshots, enable_immune)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab2:
    st.markdown('<p class="section-label">Cell population dynamics + tumour control probability</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="callout">
    Dashed white lines mark RT fraction delivery times.
    TCP = exp(−N_proliferative) — approaches 1 as clonogenic cells are cleared.
    </div>
    """, unsafe_allow_html=True)
    fig = make_dynamics_fig(df, rt_hours, dose)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Raw simulation data"):
        st.dataframe(df.style.format({
            "TCP": "{:.4f}",
        }), use_container_width=True)

with tab3:
    st.markdown('<p class="section-label">Oxygen diffusion & metabolic activity (end-state)</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="callout">
    Oxygen drives aerobic ATP production (32 units/cell at normoxia).
    Below the hypoxia threshold (pO₂ &lt; 0.3) cells switch to glycolysis —
    reducing radiosensitivity via the oxygen enhancement ratio (OER).
    </div>
    """, unsafe_allow_html=True)
    fig = make_oxygen_fig(snapshots)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab4:
    st.markdown('<p class="section-label">Cells killed per RT fraction</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="callout">
    Kill count drops across fractions as sensitive cells are depleted and
    hypoxic/resistant subclones dominate — a key driver of radioresistance.
    </div>
    """, unsafe_allow_html=True)
    if rt_kill_log:
        fig = make_kill_fig(rt_kill_log, rt_hours, dose, n_fractions)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        col1, col2 = st.columns(2)
        with col1:
            kill_df = pd.DataFrame({
                "Fraction": [f"F{i+1}" for i in range(len(rt_kill_log))],
                "Hour":     rt_hours[:len(rt_kill_log)],
                "Dose (Gy)": [dose] * len(rt_kill_log),
                "Cells Killed": rt_kill_log,
            })
            st.dataframe(kill_df, use_container_width=True, hide_index=True)
        with col2:
            if len(rt_kill_log) > 1:
                first, last = rt_kill_log[0], rt_kill_log[-1]
                decay_pct = (first - last) / first * 100 if first > 0 else 0
                st.metric("Kill decay (F1 → last)", f"{decay_pct:.1f}%",
                          delta=f"{last - first} cells",
                          delta_color="inverse")
                st.caption("Positive decay % indicates emerging radioresistance "
                           "across the fractionation course.")
    else:
        st.info("No RT fractions delivered — increase n_fractions above 0.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#21262d; margin-top:3rem;">
<div style="text-align:center; color:#484f58; font-size:0.78rem; padding-bottom:1rem;">
  CRC Tumour ABM · Agent-Based Modelling + LQ Radiobiology · 30×30 spatial lattice<br>
  Built with Streamlit · Model: Python/NumPy · Darren
</div>
""", unsafe_allow_html=True)
