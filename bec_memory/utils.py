"""
bec_memory.utils
================

Plotting helpers and utility functions for the BEC Memory Model package.

Design principle: all plotting functions accept a matplotlib Axes object
and return it, allowing callers to compose multi-panel figures freely.
No figure-level state is managed here — only axes-level drawing.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "bec_blue": "#1a78c2",
    "condensate": "#00b4d8",
    "thermal": "#f77f00",
    "ebbinghaus": "#d62828",
    "power_law": "#2d6a4f",
    "bogoliubov": "#7b2d8b",
    "false_memory": "#e63946",
    "background": "#0d1117",
    "grid": "#21262d",
    "text": "#e6edf3",
}


def apply_dark_style(fig: Figure, ax_list: Optional[list] = None) -> None:
    """Apply a clean dark-mode style to figure and axes.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.
    ax_list : list of Axes, optional
        Axes to style.  If None, styles ``fig.get_axes()``.
    """
    bg = PALETTE["background"]
    grid_c = PALETTE["grid"]
    text_c = PALETTE["text"]

    fig.set_facecolor(bg)
    axes = ax_list if ax_list is not None else fig.get_axes()
    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=text_c, labelsize=9)
        ax.xaxis.label.set_color(text_c)
        ax.yaxis.label.set_color(text_c)
        if ax.get_title():
            ax.title.set_color(text_c)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_c)
        ax.grid(True, color=grid_c, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)


# ── Core plotting helpers ─────────────────────────────────────────────────────

def plot_condensate_fraction(
    ax: Axes,
    T_c: float = 1.0,
    n_points: int = 300,
    label: str = "BEC model",
    color: Optional[str] = None,
    annotate: bool = True,
) -> Axes:
    """Plot condensate fraction n₀/N as a function of T/T_c.

    Parameters
    ----------
    ax : Axes
        Target axes.
    T_c : float
        Critical temperature (sets x-axis scale).
    n_points : int
        Number of curve points.
    label : str
        Legend label.
    color : str, optional
        Line colour (defaults to PALETTE["condensate"]).
    annotate : bool
        If True, add T_c annotation and phase labels.

    Returns
    -------
    Axes
        The modified axes (for chaining).
    """
    c = color or PALETTE["condensate"]
    T_arr = np.linspace(0.0, 1.5 * T_c, n_points)
    frac = np.maximum(0.0, 1.0 - (T_arr / T_c) ** 3)

    ax.fill_between(T_arr / T_c, frac, alpha=0.15, color=c)
    ax.plot(T_arr / T_c, frac, color=c, linewidth=2.0, label=label)
    ax.axvline(1.0, color=PALETTE["thermal"], linestyle="--", linewidth=1.2,
               alpha=0.8, label=r"$T = T_c$")

    if annotate:
        ax.text(0.3, 0.5, "BEC Phase\n(Memory Encoded)",
                transform=ax.transData, fontsize=8,
                color=PALETTE["condensate"], alpha=0.8,
                ha="center", va="center")
        ax.text(1.2, 0.1, "Thermal Phase\n(Memory Lost)",
                transform=ax.transData, fontsize=8,
                color=PALETTE["thermal"], alpha=0.8,
                ha="center", va="center")

    ax.set_xlabel(r"$T / T_c$  (Noise / Attention Threshold)", fontsize=10)
    ax.set_ylabel(r"$n_0 / N$  (Memory Strength)", fontsize=10)
    ax.set_title("Condensate Fraction — Memory Strength vs. Noise", fontsize=11)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)
    return ax


def plot_forgetting_curves(
    ax: Axes,
    times: np.ndarray,
    bec_retention: np.ndarray,
    ebb_retention: Optional[np.ndarray] = None,
    power_retention: Optional[np.ndarray] = None,
    empirical_times: Optional[np.ndarray] = None,
    empirical_data: Optional[np.ndarray] = None,
) -> Axes:
    """Plot BEC and comparison forgetting curves.

    Parameters
    ----------
    ax : Axes
        Target axes.
    times : np.ndarray
        Time axis.
    bec_retention : np.ndarray
        BEC model retention values.
    ebb_retention : np.ndarray, optional
        Ebbinghaus model values (for comparison).
    power_retention : np.ndarray, optional
        Power-law model values.
    empirical_times : np.ndarray, optional
        Empirical data time points.
    empirical_data : np.ndarray, optional
        Empirical retention measurements.

    Returns
    -------
    Axes
    """
    ax.fill_between(times, bec_retention, alpha=0.12,
                    color=PALETTE["bec_blue"])
    ax.plot(times, bec_retention, color=PALETTE["bec_blue"], linewidth=2.2,
            label="BEC model", zorder=5)

    if ebb_retention is not None:
        ax.plot(times, ebb_retention, color=PALETTE["ebbinghaus"],
                linewidth=1.6, linestyle="--", label="Ebbinghaus (1885)")

    if power_retention is not None:
        ax.plot(times, power_retention, color=PALETTE["power_law"],
                linewidth=1.6, linestyle=":", label="Power-law (Wixted 1991)")

    if empirical_times is not None and empirical_data is not None:
        ax.scatter(empirical_times, empirical_data, color="white",
                   edgecolors=PALETTE["bec_blue"], s=40, zorder=10,
                   label="Empirical data")

    ax.set_xlabel("Time since learning (days)", fontsize=10)
    ax.set_ylabel("Retention fraction", fontsize=10)
    ax.set_title("BEC Memory Forgetting Curve", fontsize=11)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)
    return ax


def plot_bogoliubov_spectrum(
    ax: Axes,
    k: np.ndarray,
    energy: np.ndarray,
    false_prob: np.ndarray,
    k_transition: float,
) -> Axes:
    """Plot Bogoliubov dispersion and false-memory probability.

    Parameters
    ----------
    ax : Axes
        Target axes (left y-axis for energy, right for probability).
    k : np.ndarray
        Quasi-momentum.
    energy : np.ndarray
        Bogoliubov energy E_k.
    false_prob : np.ndarray
        False-memory probability P(k).
    k_transition : float
        k = 1/ξ transition from semantic to perceptual modes.

    Returns
    -------
    Axes
    """
    ax.plot(k, energy, color=PALETTE["bogoliubov"], linewidth=2.0,
            label=r"$E_k$ (quasi-particle energy)", zorder=5)
    ax.axvline(k_transition, color="white", linestyle="--",
               linewidth=1.0, alpha=0.5, label=r"$k = 1/\xi$ (transition)")

    ax_r = ax.twinx()
    ax_r.plot(k, false_prob, color=PALETTE["false_memory"], linewidth=1.5,
              linestyle="-.", label="False-memory P(k)")
    ax_r.set_ylabel("False-memory probability", color=PALETTE["false_memory"],
                    fontsize=9)
    ax_r.tick_params(axis="y", colors=PALETTE["false_memory"])
    ax_r.set_ylim(-0.05, 1.1)

    ax.set_xlabel(r"Quasi-momentum $k$ (memory specificity)", fontsize=10)
    ax.set_ylabel(r"Quasi-particle energy $E_k$", fontsize=10)
    ax.set_title("Bogoliubov Spectrum — False Memory Model", fontsize=11)

    # Add region labels
    if k_transition > 0:
        ax.text(k_transition / 2, energy.max() * 0.8, "Semantic\nfalsities",
                fontsize=8, ha="center", color=PALETTE["bogoliubov"], alpha=0.8)
        ax.text(k_transition + (k.max() - k_transition) / 2, energy.max() * 0.8,
                "Perceptual\nfalsities", fontsize=8, ha="center",
                color=PALETTE["false_memory"], alpha=0.8)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    return ax


def plot_phase_diagram(
    ax: Axes,
    model,
    T_range: Tuple[float, float] = (0.0, 1.5),
    N_range: Tuple[float, float] = (1e3, 1e5),
    n_grid: int = 80,
) -> Axes:
    """Plot the BEC-memory phase diagram in (T, N) space.

    Parameters
    ----------
    ax : Axes
        Target axes.
    model : BECMemoryModel
        Model instance (for T_c scaling).
    T_range : tuple
        (T_min, T_max) range for x-axis.
    N_range : tuple
        (N_min, N_max) range for y-axis (log scale).
    n_grid : int
        Grid resolution.

    Returns
    -------
    Axes
    """
    T_arr = np.linspace(*T_range, n_grid)
    N_arr = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), n_grid)
    T_c = model.config.T_c

    # Memory strength across grid
    T_grid, N_grid = np.meshgrid(T_arr, N_arr)
    frac = np.maximum(0.0, 1.0 - (T_grid / T_c) ** 3)

    im = ax.contourf(T_arr, N_arr, frac, levels=30,
                     cmap="Blues", alpha=0.85)
    plt.colorbar(im, ax=ax, label="Memory strength (condensate fraction)")
    ax.contour(T_arr, N_arr, frac, levels=[0.0], colors=PALETTE["thermal"],
               linewidths=2.0)

    ax.set_xlabel(r"Noise level $T$ (distraction)", fontsize=10)
    ax.set_ylabel(r"Neurons engaged $N$", fontsize=10)
    ax.set_yscale("log")
    ax.set_title("Memory Phase Diagram: BEC analogy", fontsize=11)
    ax.text(0.7 * T_c, N_range[0] * 5, "FORGOTTEN", color=PALETTE["thermal"],
            fontsize=10, fontweight="bold", alpha=0.8)
    ax.text(0.2 * T_c, N_range[1] * 0.4, "ENCODED", color=PALETTE["condensate"],
            fontsize=10, fontweight="bold", alpha=0.8)
    return ax


# ── Figure factory ────────────────────────────────────────────────────────────

def make_summary_figure(
    model,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Figure:
    """Generate the canonical 4-panel BEC Memory Model summary figure.

    Panels
    ------
    1. Condensate fraction vs T/T_c
    2. BEC forgetting curve vs Ebbinghaus
    3. Bogoliubov quasi-particle spectrum
    4. Memory phase diagram

    Parameters
    ----------
    model : BECMemoryModel
        Model instance to visualise.
    save_path : str, optional
        If given, save figure to this path.
    dpi : int
        Figure resolution.

    Returns
    -------
    Figure
    """
    from bec_memory.forgetting import ForgettingCurve
    from bec_memory.false_memory import BogoliubovSpectrum

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Bose-Einstein Condensate Memory Model\n"
        "Cornell (1995) · Ketterle (1995) · BEC-Memory Analogy",
        fontsize=13, fontweight="bold", color=PALETTE["text"], y=0.99,
    )
    apply_dark_style(fig, axes.flatten().tolist())

    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1 — condensate fraction
    plot_condensate_fraction(ax1, T_c=model.config.T_c)

    # Panel 2 — forgetting curves
    t, bec_ret = model.forgetting_curve(t_max=30.0)
    ebb_ret = ForgettingCurve.ebbinghaus_model(np.where(t > 0, t, 1e-5))
    plot_forgetting_curves(ax2, t, bec_ret, ebb_retention=ebb_ret)

    # Panel 3 — Bogoliubov spectrum
    n0 = model.condensate_fraction(model.config.T_c * 0.3) * model.config.N_neurons
    spec = BogoliubovSpectrum(n0=max(n0, 1.0), g=model.config.g_coupling)
    analysis = spec.spectrum_analysis(k_max=8.0, T=model.config.T_c * 0.3)
    plot_bogoliubov_spectrum(
        ax3,
        analysis["k"],
        analysis["energy"],
        analysis["false_probability"],
        analysis["k_transition"],
    )

    # Panel 4 — phase diagram
    plot_phase_diagram(ax4, model)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=PALETTE["background"])

    return fig
