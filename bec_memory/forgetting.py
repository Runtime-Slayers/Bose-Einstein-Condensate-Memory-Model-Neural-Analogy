"""
bec_memory.forgetting
=====================

Forgetting curve models: BEC-derived and classical Ebbinghaus comparison.

The central result of this module is that the BEC condensate fraction
formula — when noise temperature is allowed to drift linearly with time —
naturally recovers Ebbinghaus-like exponential forgetting.

BEC forgetting model
--------------------
    M(t) = M₀ × max(0, 1 − (T(t) / T_c)³)
    T(t) = T₀ + α·t   (attention drift)

For T(t) ≈ T_c (near the forgetting threshold):
    1 − (T(t)/T_c)³ ≈ 3(T_c − T(t))/T_c ~ exp(−t/τ)

Classical Ebbinghaus model (1885)
----------------------------------
    M(t) = k / (log(t) + c)

Power-law forgetting (Wixted & Ebbesen 1991)
--------------------------------------------
    M(t) = a · t^{−b}

References
----------
Ebbinghaus, H. (1885). Über das Gedächtnis. Duncker & Humblot.
Wixted, J. T., & Ebbesen, E. B. (1991). Psychological Science 2:409–415.
Rubin, D. C., & Wenzel, A. E. (1996). Psychological Review 103:734–760.
Bagnato, Pritchard & Kleppner (1987). PRA 35:4354.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


class ForgettingCurve:
    """Collection of forgetting-curve models and fitting utilities.

    All models return retention fraction ∈ [0, 1] for times t > 0.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0.01, 30, 100)
    >>> M = ForgettingCurve.bec_model(t, T0=0.1, alpha=0.03, Tc=1.0, M0=1.0)
    >>> M.shape
    (100,)
    """

    # ── BEC-derived model ─────────────────────────────────────────────────────

    @staticmethod
    def bec_model(
        t: np.ndarray,
        T0: float = 0.1,
        alpha: float = 0.03,
        Tc: float = 1.0,
        M0: float = 1.0,
    ) -> np.ndarray:
        """BEC condensate-fraction forgetting curve.

        M(t) = M₀ × max(0, 1 − ((T₀ + α·t) / T_c)³)

        Parameters
        ----------
        t : array-like
            Time points (days, trials, etc.).
        T0 : float
            Initial noise temperature (just after learning).
        alpha : float
            Rate at which noise increases with time.
        Tc : float
            Critical temperature (attention threshold).
        M0 : float
            Maximum memory strength (at t=0).

        Returns
        -------
        np.ndarray
            Retention fraction at each time point.
        """
        t = np.asarray(t, dtype=float)
        T_t = T0 + alpha * t
        frac = np.maximum(0.0, 1.0 - (T_t / Tc) ** 3)
        return M0 * frac

    # ── Classical Ebbinghaus model ────────────────────────────────────────────

    @staticmethod
    def ebbinghaus_model(
        t: np.ndarray,
        k: float = 1.84,
        c: float = 1.25,
    ) -> np.ndarray:
        """Ebbinghaus (1885) logarithmic forgetting curve.

        M(t) = k / (log(t) + c)

        Parameters
        ----------
        t : array-like
            Time since learning (t > 0).
        k : float
            Scale factor.
        c : float
            Offset parameter.

        Returns
        -------
        np.ndarray
            Retention fraction.
        """
        t = np.asarray(t, dtype=float)
        return np.clip(k / (np.log(np.maximum(t, 1e-10)) + c), 0.0, 1.0)

    # ── Power-law model (Wixted & Ebbesen 1991) ───────────────────────────────

    @staticmethod
    def power_law_model(
        t: np.ndarray,
        a: float = 1.0,
        b: float = 0.5,
    ) -> np.ndarray:
        """Power-law forgetting (Wixted & Ebbesen 1991).

        M(t) = a · t^{−b}

        Parameters
        ----------
        t : array-like
            Time since learning (t > 0).
        a : float
            Scale factor.
        b : float
            Forgetting exponent.

        Returns
        -------
        np.ndarray
            Retention fraction.
        """
        t = np.asarray(t, dtype=float)
        return np.clip(a * np.power(np.maximum(t, 1e-10), -b), 0.0, 1.0)

    # ── Exponential model ─────────────────────────────────────────────────────

    @staticmethod
    def exponential_model(
        t: np.ndarray,
        M0: float = 1.0,
        tau: float = 10.0,
    ) -> np.ndarray:
        """Simple exponential forgetting.

        M(t) = M₀ · exp(−t / τ)

        Parameters
        ----------
        t : array-like
            Time since learning.
        M0 : float
            Initial strength.
        tau : float
            Time constant [same units as t].

        Returns
        -------
        np.ndarray
            Retention fraction.
        """
        t = np.asarray(t, dtype=float)
        return M0 * np.exp(-t / tau)

    # ── Fitting ───────────────────────────────────────────────────────────────

    @staticmethod
    def fit_bec(
        times: np.ndarray,
        retention: np.ndarray,
    ) -> Tuple[dict, float]:
        """Fit BEC forgetting model to data.

        Parameters
        ----------
        times : np.ndarray
            Observed time points (t > 0).
        retention : np.ndarray
            Observed retention fractions ∈ [0, 1].

        Returns
        -------
        params : dict
            Fitted parameters: T0, alpha, Tc, M0.
        r2 : float
            Coefficient of determination.
        """
        def _bec(t, T0, alpha, Tc, M0):
            return ForgettingCurve.bec_model(t, T0, alpha, Tc, M0)

        try:
            popt, _ = curve_fit(
                _bec, times, retention,
                p0=[0.1, 0.03, 1.0, 1.0],
                bounds=([0, 0, 1e-6, 0], [1, 1, 10, 2]),
                maxfev=10_000,
            )
            T0, alpha, Tc, M0 = popt
            pred = ForgettingCurve.bec_model(times, T0, alpha, Tc, M0)
            r2 = _r_squared(retention, pred)
            return {"T0": T0, "alpha": alpha, "Tc": Tc, "M0": M0}, r2
        except RuntimeError:
            return {"T0": np.nan, "alpha": np.nan, "Tc": np.nan, "M0": np.nan}, 0.0

    @staticmethod
    def fit_ebbinghaus(
        times: np.ndarray,
        retention: np.ndarray,
    ) -> Tuple[dict, float]:
        """Fit Ebbinghaus model to data.

        Parameters
        ----------
        times : np.ndarray
            Observed time points (t > 0).
        retention : np.ndarray
            Observed retention fractions ∈ [0, 1].

        Returns
        -------
        params : dict
            Fitted parameters: k, c.
        r2 : float
            Coefficient of determination.
        """
        def _ebb(t, k, c):
            return ForgettingCurve.ebbinghaus_model(t, k, c)

        try:
            popt, _ = curve_fit(
                _ebb, times, retention,
                p0=[1.84, 1.25],
                maxfev=10_000,
            )
            k, c = popt
            pred = ForgettingCurve.ebbinghaus_model(times, k, c)
            r2 = _r_squared(retention, pred)
            return {"k": k, "c": c}, r2
        except RuntimeError:
            return {"k": np.nan, "c": np.nan}, 0.0

    @staticmethod
    def fit_both(
        times: np.ndarray,
        retention: np.ndarray,
    ) -> Dict:
        """Fit BEC and Ebbinghaus models and return comparison.

        Parameters
        ----------
        times : np.ndarray
        retention : np.ndarray

        Returns
        -------
        dict with keys:
            bec_params, bec_r2, ebb_params, ebb_r2, better_model
        """
        bec_params, bec_r2 = ForgettingCurve.fit_bec(times, retention)
        ebb_params, ebb_r2 = ForgettingCurve.fit_ebbinghaus(times, retention)
        better = "bec" if bec_r2 >= ebb_r2 else "ebbinghaus"
        return {
            "bec_params": bec_params,
            "bec_r2": bec_r2,
            "ebb_params": ebb_params,
            "ebb_r2": ebb_r2,
            "better_model": better,
        }
