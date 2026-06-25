"""
bec_memory.gpe
==============

Gross-Pitaevskii equation (GPE) solver for the BEC memory wavefunction.

The GPE describes the time evolution of the macroscopic condensate
wavefunction ψ(x, t):

    iℏ ∂ψ/∂t = [−ℏ²∇²/(2m) + V_ext(x) + g|ψ|²] ψ

Memory analogy mapping
----------------------
    ψ(x, t)   → Memory wavefunction (amplitude of memory trace)
    V_ext(x)  → External stimulus (lecture, sensory input)
    g|ψ|²     → Self-reinforcement (rehearsal strengthens memory)
    m         → Cognitive mass (resistance to change)
    ℏ∂ψ/∂t    → Rate of memory trace evolution

The GPE is solved on a 1D spatial grid using split-step Fourier methods,
which are the industry standard for this equation (Bao & Cai 2013).

References
----------
Gross (1961) Nuovo Cimento 20:454
Pitaevskii (1961) JETP 13:451
Bao & Cai (2013) Kinetic and Related Models 6:1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# NumPy 2.0 renamed trapz → trapezoid; support both.
_trapz = getattr(np, "trapezoid", None)
if _trapz is None:
    _trapz = getattr(np, "trapz")


@dataclass
class GPEConfig:
    """Configuration for the GPE solver.

    Parameters
    ----------
    grid_points : int
        Number of spatial grid points (power of 2 recommended).
    x_max : float
        Half-width of the spatial domain [dimensionless units].
    dt : float
        Time step [dimensionless units].
    hbar : float
        Effective ℏ [dimensionless].
    mass : float
        Cognitive mass m [dimensionless].
    g_coupling : float
        Nonlinear coupling constant g (self-reinforcement).
    """
    grid_points: int = 256
    x_max: float = 10.0
    dt: float = 0.01
    hbar: float = 1.0
    mass: float = 1.0
    g_coupling: float = 0.01


class GPESolver:
    """Split-step Fourier solver for the Gross-Pitaevskii equation.

    Maps BEC wavefunction evolution onto memory trace dynamics.

    Parameters
    ----------
    config : GPEConfig, optional
        Solver configuration.

    Examples
    --------
    >>> solver = GPESolver()
    >>> solver.initialize_gaussian(width=1.0)
    >>> psi, t = solver.evolve(steps=100)
    >>> psi.shape
    (256,)
    """

    def __init__(self, config: GPEConfig | None = None) -> None:
        self.config = config if config is not None else GPEConfig()
        cfg = self.config

        # Spatial grid
        self.x = np.linspace(-cfg.x_max, cfg.x_max, cfg.grid_points)
        self.dx = self.x[1] - self.x[0]

        # Momentum grid (for kinetic energy operator)
        self.k = np.fft.fftfreq(cfg.grid_points, d=self.dx) * 2.0 * np.pi

        # Harmonic trap potential V = ½ x²
        self.V_trap = 0.5 * self.x ** 2

        # Kinetic energy phase factor (constant across steps)
        self._kinetic_phase = np.exp(
            -1j * cfg.hbar * self.k ** 2 / (2.0 * cfg.mass) * cfg.dt
        )

        # Wavefunction (to be initialized)
        self.psi: np.ndarray = np.zeros(cfg.grid_points, dtype=complex)
        self.time: float = 0.0

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize_gaussian(
        self,
        width: float = 1.0,
        center: float = 0.0,
        normalize: bool = True,
    ) -> None:
        """Initialize ψ as a Gaussian ground-state approximation.

        Parameters
        ----------
        width : float
            Gaussian width σ.
        center : float
            Center of the Gaussian.
        normalize : bool
            If True, normalize ⟨ψ|ψ⟩ = 1.
        """
        psi = np.exp(-0.5 * ((self.x - center) / width) ** 2, dtype=float).astype(complex)
        if normalize:
            norm = np.sqrt(_trapz(np.abs(psi) ** 2, self.x))
            psi /= norm
        self.psi = psi
        self.time = 0.0

    def initialize_thomas_fermi(self) -> None:
        """Initialize using the Thomas-Fermi approximation.

        Valid for large N (strong interactions).  Sets ψ ∝ √(max(0, μ − V)).
        """
        cfg = self.config
        mu = 1.0  # chemical potential (set to 1 in dimensionless units)
        n = np.maximum(0.0, (mu - self.V_trap) / cfg.g_coupling)
        psi = np.sqrt(n).astype(complex)
        norm = np.sqrt(_trapz(np.abs(psi) ** 2, self.x))
        if norm > 0:
            psi /= norm
        self.psi = psi
        self.time = 0.0

    # ── Evolution ─────────────────────────────────────────────────────────────

    def evolve(
        self,
        steps: int = 100,
        external_stimulus: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, float]:
        """Evolve the wavefunction forward in time using split-step FFT.

        Split-step operator splitting (Strang splitting):
            ψ(t+dt) ≈ e^{-iV·dt/2ℏ} · FFT⁻¹[e^{-iK·dt/ℏ} · FFT[e^{-iV·dt/2ℏ} ψ(t)]]

        Parameters
        ----------
        steps : int
            Number of time steps to evolve.
        external_stimulus : np.ndarray, optional
            Additional potential V_ext(x) representing external input.

        Returns
        -------
        psi : np.ndarray
            Wavefunction after evolution.
        time : float
            Current simulation time.
        """
        cfg = self.config
        V = self.V_trap.copy()
        if external_stimulus is not None:
            if external_stimulus.shape != V.shape:
                raise ValueError(
                    f"external_stimulus shape {external_stimulus.shape} "
                    f"must match grid shape {V.shape}"
                )
            V = V + external_stimulus

        psi = self.psi.copy()
        dt = cfg.dt
        hbar = cfg.hbar
        g = cfg.g_coupling

        for _ in range(steps):
            # Half-step: potential + nonlinear
            potential_phase_half = np.exp(
                -1j * (V + g * np.abs(psi) ** 2) * dt / (2.0 * hbar)
            )
            psi = potential_phase_half * psi

            # Full-step: kinetic (in Fourier space)
            psi_k = np.fft.fft(psi)
            psi_k = self._kinetic_phase * psi_k
            psi = np.fft.ifft(psi_k)

            # Half-step: potential + nonlinear
            potential_phase_half = np.exp(
                -1j * (V + g * np.abs(psi) ** 2) * dt / (2.0 * hbar)
            )
            psi = potential_phase_half * psi

        self.psi = psi
        self.time += steps * dt
        return self.psi.copy(), self.time

    # ── Observables ───────────────────────────────────────────────────────────

    def density(self) -> np.ndarray:
        """Return the probability density |ψ(x)|².

        Maps to: spatial distribution of memory trace strength.
        """
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        """Return ⟨ψ|ψ⟩ (should remain ≈ 1.0 under unitary evolution)."""
        return float(_trapz(np.abs(self.psi) ** 2, self.x))

    def chemical_potential(self) -> float:
        """Estimate the chemical potential μ via energy expectation.

        Maps to: motivation / memory salience.
        """
        cfg = self.config
        psi = self.psi
        # Kinetic energy via finite differences
        d2psi = np.gradient(np.gradient(psi, self.x), self.x)
        KE = -cfg.hbar ** 2 / (2.0 * cfg.mass) * np.real(
            _trapz(np.conj(psi) * d2psi, self.x)
        )
        # Potential energy
        PE = float(np.real(_trapz(np.conj(psi) * self.V_trap * psi, self.x)))
        # Interaction energy
        IE = 0.5 * cfg.g_coupling * float(_trapz(np.abs(psi) ** 4, self.x))
        return KE + PE + IE

    def ground_state_energy(self) -> float:
        """Compute total energy E = ⟨ψ|H|ψ⟩.

        Maps to: baseline stability of long-term memory.
        """
        return self.chemical_potential()

    # ── Imaginary-time relaxation ─────────────────────────────────────────────

    def find_ground_state(
        self,
        steps: int = 500,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """Find GPE ground state via imaginary-time evolution.

        Replaces t → −iτ, which causes excited states to decay
        exponentially, leaving the ground state.

        Parameters
        ----------
        steps : int
            Maximum imaginary-time steps.
        tol : float
            Convergence tolerance on energy change.

        Returns
        -------
        np.ndarray
            Ground-state wavefunction ψ₀.
        """
        cfg = self.config
        psi = self.psi.copy()
        if np.all(psi == 0):
            self.initialize_gaussian()
            psi = self.psi.copy()

        dt = cfg.dt
        hbar = cfg.hbar
        g = cfg.g_coupling

        # Imaginary-time kinetic propagator
        k_it = np.exp(-hbar * self.k ** 2 / (2.0 * cfg.mass) * dt)
        prev_energy = np.inf

        for step in range(steps):
            # Half-step potential
            psi = np.exp(-(self.V_trap + g * np.abs(psi) ** 2) * dt / (2.0 * hbar)) * psi

            # Kinetic step
            psi_k = np.fft.fft(psi)
            psi_k = k_it * psi_k
            psi = np.fft.ifft(psi_k)

            # Half-step potential
            psi = np.exp(-(self.V_trap + g * np.abs(psi) ** 2) * dt / (2.0 * hbar)) * psi

            # Renormalize
            norm = np.sqrt(_trapz(np.abs(psi) ** 2, self.x))
            if norm > 0:
                psi /= norm

            # Check convergence every 50 steps
            if step % 50 == 0:
                d2psi = np.gradient(np.gradient(psi, self.x), self.x)
                KE = float(np.real(-hbar ** 2 / (2 * cfg.mass) * _trapz(np.conj(psi) * d2psi, self.x)))
                PE = float(np.real(_trapz(np.conj(psi) * self.V_trap * psi, self.x)))
                IE = 0.5 * g * float(_trapz(np.abs(psi) ** 4, self.x))
                energy = KE + PE + IE
                if abs(energy - prev_energy) < tol:
                    break
                prev_energy = energy

        self.psi = psi
        return self.psi.copy()

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"GPESolver(grid={cfg.grid_points}, x_max={cfg.x_max}, "
            f"dt={cfg.dt}, g={cfg.g_coupling})"
        )
