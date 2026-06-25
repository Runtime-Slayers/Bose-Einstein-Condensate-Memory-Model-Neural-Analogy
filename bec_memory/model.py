"""
bec_memory.model
================

Core BECMemoryModel class implementing the Bose-Einstein Condensate
analogy for human memory dynamics.

The fundamental correspondence used throughout this module:

    BEC Physics                  ↔   Memory Science
    ─────────────────────────────────────────────────
    Atoms at high T (thermal)    ↔   Unstructured neural activity
    Cooling below T_c            ↔   Learning / encoding
    BEC formation (condensation) ↔   Memory trace formation
    Ground-state occupation n₀   ↔   Long-term memory strength
    Temperature T                ↔   Neural noise / distraction level
    Critical temperature T_c     ↔   Attention threshold for learning
    Particle number N            ↔   Number of activated neurons
    Chemical potential μ         ↔   Motivation / salience
    Thermal excitations          ↔   Forgetting / interference
    Quasi-particles (Bogoliubov) ↔   False memories / distortions
    Stimulated emission          ↔   Cue-triggered retrieval
    Superfluidity                ↔   Effortless recall (automaticity)
    Vortices                     ↔   Traumatic memory loops

Key equations
-------------
1. Condensate fraction (harmonic trap, Bagnato 1987):
       n₀/N = max(0, 1 − (T/T_c)^3)

2. BEC critical temperature (harmonic trap, N atoms):
       T_c = (ħω / k_B) × (N / ζ(3))^(1/3)

3. BEC-derived forgetting curve:
       M(t) = M₀ × max(0, 1 − (T(t)/T_c)^3)
   where T(t) = T₀ + α·t  (attention drifts with time)
   This recovers the Ebbinghaus exponential for T(t) ≈ T_c.

References
----------
Bagnato, Pritchard & Kleppner (1987) PRA 35:4354
Pitaevskii & Stringari (2003) Bose–Einstein Condensation, Oxford
Ebbinghaus (1885) Über das Gedächtnis, Duncker & Humblot
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import zeta as riemann_zeta

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants (SI)
# ──────────────────────────────────────────────────────────────────────────────
K_B = 1.380649e-23   # Boltzmann constant  [J K⁻¹]
HBAR = 1.054571817e-34  # Reduced Planck     [J s]
ZETA3 = float(riemann_zeta(3))  # ζ(3) ≈ 1.20206


@dataclass
class ModelConfig:
    """Configuration parameters for BECMemoryModel.

    Parameters
    ----------
    N_neurons : int
        Total number of neurons available for memory encoding.
    T_c : float
        Critical "temperature" (dimensionless attention threshold, > 0).
    g_coupling : float
        Self-reinforcement coupling constant (rehearsal strength).
    cognitive_mass : float
        Effective cognitive inertia (resistance to change).
    noise_amplitude : float
        Baseline neural noise level.
    trap_freq_Hz : float
        Effective harmonic trap frequency (maps to neural coherence bandwidth).
    """
    N_neurons: int = 10_000
    T_c: float = 1.0
    g_coupling: float = 0.01
    cognitive_mass: float = 1.0
    noise_amplitude: float = 0.1
    trap_freq_Hz: float = 100.0


class BECMemoryModel:
    """Bose-Einstein Condensate analogy model for human memory.

    This class provides a mathematical framework that maps BEC physics
    onto memory dynamics.  It is NOT a claim that the brain is quantum —
    rather, it uses the well-understood mathematics of Bose-Einstein
    condensation as a novel modelling language for memory phenomena.

    Parameters
    ----------
    config : ModelConfig, optional
        Configuration object.  Defaults to ``ModelConfig()`` if not supplied.

    Examples
    --------
    >>> model = BECMemoryModel()
    >>> model.condensate_fraction(0.5)   # ~0.875
    0.875
    >>> t, m = model.forgetting_curve(t_max=30.0)
    >>> t.shape, m.shape
    ((300,), (300,))
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config if config is not None else ModelConfig()

    # ── Core BEC equations ────────────────────────────────────────────────────

    def condensate_fraction(self, T: float) -> float:
        """Fraction of neurons in the coherent memory trace.

        Uses the harmonic-trap ideal-Bose-gas result (Bagnato 1987):

            n₀/N = max(0, 1 − (T/T_c)^3)

        Parameters
        ----------
        T : float
            Effective noise temperature (same units as T_c).

        Returns
        -------
        float
            Fraction ∈ [0, 1].  Returns 0.0 for T ≥ T_c.
        """
        T_c = self.config.T_c
        if T >= T_c:
            return 0.0
        return float(max(0.0, 1.0 - (T / T_c) ** 3))

    def memory_strength(self, T: float, M0: float = 1.0) -> float:
        """Absolute memory strength at noise level T.

        Parameters
        ----------
        T : float
            Effective noise temperature.
        M0 : float
            Initial (maximum) memory strength.

        Returns
        -------
        float
            Memory strength M ∈ [0, M0].
        """
        return M0 * self.condensate_fraction(T)

    def critical_temperature(
        self,
        N: Optional[int] = None,
        omega: Optional[float] = None,
    ) -> float:
        """Compute critical temperature for BEC in a harmonic trap.

        Uses the thermodynamic-limit formula (Pitaevskii & Stringari 2003):

            T_c = (ħω / k_B) × (N / ζ(3))^(1/3)

        Parameters
        ----------
        N : int, optional
            Number of bosons (neurons).  Defaults to ``config.N_neurons``.
        omega : float, optional
            Angular trap frequency [rad s⁻¹].  Defaults to
            ``2π × config.trap_freq_Hz``.

        Returns
        -------
        float
            Critical temperature T_c [K] (physical BEC scale).

        Notes
        -----
        The returned value is in Kelvin — useful for direct comparison with
        published BEC experiments (Cornell 1995: T_c ≈ 170 nK).
        """
        if N is None:
            N = self.config.N_neurons
        if omega is None:
            omega = 2.0 * np.pi * self.config.trap_freq_Hz
        return (HBAR * omega / K_B) * (N / ZETA3) ** (1.0 / 3.0)

    # ── Time-domain dynamics ──────────────────────────────────────────────────

    def noise_temperature(
        self,
        t: np.ndarray,
        T0: float = 0.1,
        alpha: float = 0.03,
    ) -> np.ndarray:
        """Model attention drift as linearly increasing noise temperature.

        T(t) = T₀ + α·t

        Parameters
        ----------
        t : array-like
            Time points [arbitrary units, e.g. days].
        T0 : float
            Initial noise level (just after learning).
        alpha : float
            Rate of attention / focus decay.

        Returns
        -------
        np.ndarray
            Noise temperature at each time point.
        """
        t = np.asarray(t, dtype=float)
        return T0 + alpha * t

    def forgetting_curve(
        self,
        t_max: float = 30.0,
        n_points: int = 300,
        T0: float = 0.1,
        alpha: float = 0.03,
        M0: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the BEC-derived forgetting curve.

        M(t) = M₀ × max(0, 1 − (T(t)/T_c)^3)
        with T(t) = T₀ + α·t

        This recovers Ebbinghaus-like exponential decay for T(t) near T_c.

        Parameters
        ----------
        t_max : float
            End time [days].
        n_points : int
            Number of time points.
        T0 : float
            Initial noise level.
        alpha : float
            Noise drift rate.
        M0 : float
            Initial memory strength.

        Returns
        -------
        times : np.ndarray, shape (n_points,)
        retention : np.ndarray, shape (n_points,)
        """
        times = np.linspace(0.0, t_max, n_points)
        T_arr = self.noise_temperature(times, T0=T0, alpha=alpha)
        retention = np.array([self.memory_strength(T, M0) for T in T_arr])
        return times, retention

    # ── Fitting utilities ─────────────────────────────────────────────────────

    def fit_forgetting_curve(
        self,
        times: np.ndarray,
        retention: np.ndarray,
    ) -> dict:
        """Fit both BEC and Ebbinghaus models to empirical retention data.

        Parameters
        ----------
        times : np.ndarray
            Observed time points.
        retention : np.ndarray
            Observed retention fractions ∈ [0, 1].

        Returns
        -------
        dict with keys:
            ``bec_params``   — (T0, alpha, Tc_fit, M0)
            ``bec_r2``       — coefficient of determination for BEC fit
            ``ebb_params``   — (k, c) for Ebbinghaus M(t)=k/(log(t)+c)
            ``ebb_r2``       — coefficient of determination for Ebbinghaus
        """
        from bec_memory.forgetting import ForgettingCurve
        return ForgettingCurve.fit_both(times, retention)

    # ── Retrieval dynamics ────────────────────────────────────────────────────

    def retrieval_probability(
        self,
        T: float,
        cue_strength: float = 0.5,
    ) -> float:
        """Probability of successful memory retrieval given a cue.

        Models retrieval as stimulated emission from the condensate:
        retrieval probability scales with condensate fraction and cue strength.

        Parameters
        ----------
        T : float
            Current noise temperature.
        cue_strength : float
            Cue quality ∈ [0, 1].

        Returns
        -------
        float
            Retrieval probability ∈ [0, 1].
        """
        n0_frac = self.condensate_fraction(T)
        # Stimulated emission: P ∝ n₀ × cue_strength (amplification)
        return float(np.clip(n0_frac * (1.0 + cue_strength) / 2.0, 0.0, 1.0))

    # ── Representation ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the model configuration."""
        cfg = self.config
        T_c_phys = self.critical_temperature()
        lines = [
            "BECMemoryModel Summary",
            "=" * 40,
            f"  N neurons      : {cfg.N_neurons:,}",
            f"  T_c (scaled)   : {cfg.T_c:.3f}",
            f"  T_c (physical) : {T_c_phys * 1e9:.1f} nK",
            f"  g coupling     : {cfg.g_coupling:.4f}",
            f"  Noise amp.     : {cfg.noise_amplitude:.4f}",
            f"  Trap freq.     : {cfg.trap_freq_Hz:.1f} Hz",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BECMemoryModel(N={self.config.N_neurons}, "
            f"T_c={self.config.T_c}, g={self.config.g_coupling})"
        )
