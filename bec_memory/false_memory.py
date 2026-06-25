"""
bec_memory.false_memory
=======================

Bogoliubov quasi-particle model for false memory formation.

In a BEC, small perturbations around the ground state create
quasi-particles described by the Bogoliubov dispersion relation.
These quasi-particles represent deviations from perfect coherence —
analogous to distortions and false memories in recall.

Bogoliubov dispersion relation
-------------------------------
    E_k = √(ε_k(ε_k + 2·g·n₀))

where:
    ε_k = ℏ²k²/(2m)   (free-particle energy)
    g   = interaction coupling constant
    n₀  = condensate density (memory strength)
    k   = quasi-momentum (memory specificity)

Memory analogy
--------------
    Low k  (long-wavelength quasi-particles) → Semantic false memories
           (the gist is recalled but details are wrong)
    High k (short-wavelength quasi-particles) → Perceptual false memories
           (specific sensory details are distorted)
    n_k = 1/(exp(E_k/k_B T) − 1)             → Bose-Einstein occupation
           (probability of false memory of type k at noise level T)

References
----------
Bogoliubov, N. N. (1947). J. Phys. USSR 11:23.
Pitaevskii & Stringari (2003). BEC. Oxford.
Roediger, H. L., & McDermott, K. B. (1995). J. Exp. Psych.: LMC 21:803.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# Physical-scale constants (dimensionless default)
HBAR_DEFAULT = 1.0
MASS_DEFAULT = 1.0
K_B_DEFAULT = 1.0


class BogoliubovSpectrum:
    """Bogoliubov quasi-particle spectrum for false memory analysis.

    Parameters
    ----------
    n0 : float
        Condensate density (∝ memory strength), n₀ ≥ 0.
    g : float
        Interaction coupling (∝ self-reinforcement strength), g > 0.
    hbar : float
        Effective ℏ (dimensionless, default 1.0).
    mass : float
        Cognitive mass (dimensionless, default 1.0).

    Examples
    --------
    >>> spec = BogoliubovSpectrum(n0=0.8, g=0.01)
    >>> k = np.linspace(0, 5, 100)
    >>> E = spec.energy(k)
    >>> E.shape
    (100,)
    """

    def __init__(
        self,
        n0: float = 0.8,
        g: float = 0.01,
        hbar: float = HBAR_DEFAULT,
        mass: float = MASS_DEFAULT,
    ) -> None:
        if n0 < 0:
            raise ValueError(f"n0 must be ≥ 0, got {n0}")
        if g <= 0:
            raise ValueError(f"g must be > 0, got {g}")
        self.n0 = float(n0)
        self.g = float(g)
        self.hbar = float(hbar)
        self.mass = float(mass)

    # ── Dispersion relation ───────────────────────────────────────────────────

    def free_particle_energy(self, k: np.ndarray) -> np.ndarray:
        """Free-particle (kinetic) energy ε_k = ℏ²k²/(2m).

        Parameters
        ----------
        k : array-like
            Quasi-momentum array.

        Returns
        -------
        np.ndarray
            ε_k values.
        """
        k = np.asarray(k, dtype=float)
        return self.hbar ** 2 * k ** 2 / (2.0 * self.mass)

    def energy(self, k: np.ndarray) -> np.ndarray:
        """Bogoliubov quasi-particle energy.

        E_k = √(ε_k · (ε_k + 2·g·n₀))

        Parameters
        ----------
        k : array-like
            Quasi-momentum values.

        Returns
        -------
        np.ndarray
            Quasi-particle energies E_k ≥ 0.
        """
        k = np.asarray(k, dtype=float)
        eps = self.free_particle_energy(k)
        return np.sqrt(np.maximum(0.0, eps * (eps + 2.0 * self.g * self.n0)))

    def sound_speed(self) -> float:
        """Speed of sound c_s in the condensate.

        c_s = √(g·n₀/m)

        Maps to: speed of memory retrieval propagation.

        Returns
        -------
        float
            Sound speed (dimensionless).
        """
        return float(np.sqrt(self.g * self.n0 / self.mass))

    def healing_length(self) -> float:
        """Healing length ξ of the condensate.

        ξ = ℏ / √(2·m·g·n₀)

        Maps to: spatial coherence scale of memory trace.
        For k < 1/ξ → phonon-like (semantic) excitations.
        For k > 1/ξ → particle-like (perceptual) excitations.

        Returns
        -------
        float
            Healing length ξ.
        """
        if self.n0 == 0 or self.g == 0:
            return np.inf
        return float(self.hbar / np.sqrt(2.0 * self.mass * self.g * self.n0))

    # ── False-memory statistics ───────────────────────────────────────────────

    def occupation_number(
        self,
        k: np.ndarray,
        T: float,
        k_B: float = K_B_DEFAULT,
    ) -> np.ndarray:
        """Bose-Einstein occupation of quasi-particle mode k at temperature T.

        n_k = 1 / (exp(E_k / k_B·T) − 1)

        Maps to: expected number of false memories of type k at noise level T.

        Parameters
        ----------
        k : array-like
            Quasi-momentum values.
        T : float
            Noise temperature (T > 0).
        k_B : float
            Boltzmann constant (dimensionless default: 1.0).

        Returns
        -------
        np.ndarray
            Occupation numbers (can be large for low-energy modes).
        """
        k = np.asarray(k, dtype=float)
        E = self.energy(k)
        if T <= 0:
            return np.zeros_like(k)
        exponent = E / (k_B * T)
        # Clip to avoid overflow; modes with E≈0 get large occupation
        with np.errstate(over="ignore", invalid="ignore"):
            occ = np.where(exponent < 500, 1.0 / (np.expm1(exponent + 1e-300)), 0.0)
        return np.maximum(0.0, occ)

    def false_memory_probability(
        self,
        k: np.ndarray,
        T: float,
        k_B: float = K_B_DEFAULT,
    ) -> np.ndarray:
        """Probability of a false memory of type k at noise level T.

        Uses logistic transform of occupation number:
            P_false(k) = tanh(n_k / (1 + n_k))

        This maps the occupation (∈ [0, ∞)) to a probability (∈ [0, 1]).

        Parameters
        ----------
        k : array-like
            Quasi-momentum (spatial frequency of memory distortion).
        T : float
            Noise temperature.
        k_B : float
            Boltzmann constant.

        Returns
        -------
        np.ndarray
            False-memory probability ∈ [0, 1].
        """
        n_k = self.occupation_number(k, T, k_B)
        return np.tanh(n_k / (1.0 + n_k))

    def classify_false_memories(
        self,
        k: np.ndarray,
    ) -> np.ndarray:
        """Classify quasi-particle modes by false-memory type.

        Returns
        -------
        np.ndarray of str
            'semantic' for k < 1/ξ (long-wavelength)
            'perceptual' for k ≥ 1/ξ (short-wavelength)
        """
        xi = self.healing_length()
        k = np.asarray(k, dtype=float)
        return np.where(k < 1.0 / xi, "semantic", "perceptual")

    # ── Full spectrum analysis ────────────────────────────────────────────────

    def spectrum_analysis(
        self,
        k_max: float = 10.0,
        n_points: int = 200,
        T: float = 0.3,
    ) -> dict:
        """Compute full Bogoliubov spectrum and false-memory statistics.

        Parameters
        ----------
        k_max : float
            Maximum quasi-momentum to analyse.
        n_points : int
            Number of k-points.
        T : float
            Noise temperature.

        Returns
        -------
        dict with keys:
            k, energy, occupation, false_prob, healing_length,
            sound_speed, xi_inv (1/ξ), semantic_fraction
        """
        k = np.linspace(0.0, k_max, n_points)
        E = self.energy(k)
        n_k = self.occupation_number(k, T)
        P = self.false_memory_probability(k, T)
        xi = self.healing_length()
        xi_inv = 1.0 / xi if xi != np.inf else 0.0
        sem_frac = float(np.mean(k < xi_inv)) if xi_inv > 0 else 1.0

        return {
            "k": k,
            "energy": E,
            "occupation": n_k,
            "false_probability": P,
            "healing_length": xi,
            "sound_speed": self.sound_speed(),
            "k_transition": xi_inv,
            "semantic_fraction": sem_frac,
        }

    def __repr__(self) -> str:
        return (
            f"BogoliubovSpectrum(n0={self.n0:.3f}, g={self.g:.4f}, "
            f"c_s={self.sound_speed():.4f}, ξ={self.healing_length():.4f})"
        )
