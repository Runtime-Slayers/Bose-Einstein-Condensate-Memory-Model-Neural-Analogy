"""
Tests for bec_memory.forgetting — ForgettingCurve models and fitting.
"""
import pytest
import numpy as np
from bec_memory.forgetting import ForgettingCurve, _r_squared


# ── _r_squared ────────────────────────────────────────────────────────────────

class TestRSquared:
    def test_perfect_fit_gives_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert _r_squared(y, y) == pytest.approx(1.0)

    def test_mean_prediction_gives_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y, np.mean(y))
        assert _r_squared(y, y_pred) == pytest.approx(0.0)

    def test_negative_for_bad_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # reversed
        assert _r_squared(y, y_pred) < 0.0


# ── BEC model ─────────────────────────────────────────────────────────────────

class TestBECModel:
    def test_output_shape(self):
        t = np.linspace(0.01, 30, 100)
        M = ForgettingCurve.bec_model(t)
        assert M.shape == (100,)

    def test_values_in_range(self):
        t = np.linspace(0.01, 30, 200)
        M = ForgettingCurve.bec_model(t, T0=0.1, alpha=0.03, Tc=1.0, M0=1.0)
        assert np.all(M >= 0.0)
        assert np.all(M <= 1.0 + 1e-9)

    def test_monotonically_nonincreasing(self):
        t = np.linspace(0, 20, 300)
        M = ForgettingCurve.bec_model(t, T0=0.1, alpha=0.05)
        diffs = np.diff(M)
        assert np.all(diffs <= 1e-12)

    def test_high_alpha_faster_forgetting(self):
        t = np.array([5.0, 10.0, 20.0])
        M_slow = ForgettingCurve.bec_model(t, alpha=0.01)
        M_fast = ForgettingCurve.bec_model(t, alpha=0.1)
        assert np.all(M_fast <= M_slow)

    def test_zero_T_and_alpha_full_retention(self):
        """If noise never increases, memory stays at initial condensate level."""
        t = np.linspace(0, 30, 50)
        M = ForgettingCurve.bec_model(t, T0=0.0, alpha=0.0, Tc=1.0, M0=1.0)
        assert np.allclose(M, 1.0, atol=1e-12)

    def test_M0_scales_result(self):
        t = np.array([1.0, 5.0, 10.0])
        M1 = ForgettingCurve.bec_model(t, M0=1.0)
        M2 = ForgettingCurve.bec_model(t, M0=0.5)
        assert np.allclose(M2, 0.5 * M1, atol=1e-12)


# ── Ebbinghaus model ──────────────────────────────────────────────────────────

class TestEbbinghausModel:
    def test_output_shape(self):
        t = np.linspace(0.1, 30, 50)
        M = ForgettingCurve.ebbinghaus_model(t)
        assert M.shape == (50,)

    def test_values_in_range(self):
        t = np.linspace(0.1, 30, 200)
        M = ForgettingCurve.ebbinghaus_model(t)
        assert np.all(M >= 0.0)
        assert np.all(M <= 1.0 + 1e-9)

    def test_decreasing_with_time(self):
        # Use t > 1 so log(t) > 0 and Ebbinghaus is properly defined
        t = np.linspace(1.0, 30, 100)
        M = ForgettingCurve.ebbinghaus_model(t)
        # Should be broadly decreasing for t > 1
        assert M[-1] <= M[0]

    def test_handles_near_zero_time(self):
        """Should not raise for very small t."""
        M = ForgettingCurve.ebbinghaus_model(np.array([1e-6]))
        assert np.isfinite(M).all()


# ── Power-law model ───────────────────────────────────────────────────────────

class TestPowerLawModel:
    def test_output_shape(self):
        t = np.linspace(0.1, 30, 60)
        M = ForgettingCurve.power_law_model(t)
        assert M.shape == (60,)

    def test_values_in_range(self):
        t = np.linspace(0.1, 30, 200)
        M = ForgettingCurve.power_law_model(t, a=0.8, b=0.5)
        assert np.all(M >= 0.0)
        assert np.all(M <= 1.0 + 1e-9)

    def test_larger_b_faster_forgetting(self):
        t = np.array([2.0, 5.0, 10.0])
        M_slow = ForgettingCurve.power_law_model(t, a=1.0, b=0.2)
        M_fast = ForgettingCurve.power_law_model(t, a=1.0, b=0.8)
        assert np.all(M_fast <= M_slow)


# ── Exponential model ─────────────────────────────────────────────────────────

class TestExponentialModel:
    def test_output_shape(self):
        t = np.linspace(0, 30, 80)
        M = ForgettingCurve.exponential_model(t)
        assert M.shape == (80,)

    def test_starts_at_M0(self):
        M = ForgettingCurve.exponential_model(np.array([0.0]), M0=0.9)
        assert M[0] == pytest.approx(0.9)

    def test_half_life_correct(self):
        """M(τ) = M0 / e  → half-life at t = τ·ln(2)."""
        tau = 5.0
        t_half = tau * np.log(2)
        M = ForgettingCurve.exponential_model(np.array([t_half]), M0=1.0, tau=tau)
        assert M[0] == pytest.approx(0.5, rel=1e-6)


# ── Fitting routines ──────────────────────────────────────────────────────────

class TestFitting:
    @pytest.fixture
    def synthetic_bec_data(self):
        """Generate noiseless BEC retention data for fitting tests."""
        t = np.linspace(0.1, 25, 80)
        M = ForgettingCurve.bec_model(t, T0=0.1, alpha=0.04, Tc=1.0, M0=0.95)
        return t, M

    def test_fit_bec_returns_dict_and_r2(self, synthetic_bec_data):
        t, M = synthetic_bec_data
        params, r2 = ForgettingCurve.fit_bec(t, M)
        assert isinstance(params, dict)
        assert "T0" in params
        assert isinstance(r2, float)

    def test_fit_bec_good_r2_on_synthetic(self, synthetic_bec_data):
        """Fitting noiseless BEC data should give R² > 0.99."""
        t, M = synthetic_bec_data
        params, r2 = ForgettingCurve.fit_bec(t, M)
        assert r2 > 0.99, f"Expected R²>0.99, got {r2:.4f}"

    def test_fit_ebbinghaus_returns_dict_and_r2(self, synthetic_bec_data):
        t, M = synthetic_bec_data
        params, r2 = ForgettingCurve.fit_ebbinghaus(t, M)
        assert isinstance(params, dict)
        assert "k" in params and "c" in params

    def test_fit_both_returns_better_model_key(self, synthetic_bec_data):
        t, M = synthetic_bec_data
        result = ForgettingCurve.fit_both(t, M)
        assert "better_model" in result
        assert result["better_model"] in ("bec", "ebbinghaus")
        # BEC data should be better explained by BEC model
        assert result["better_model"] == "bec"

    def test_fit_both_keys_complete(self, synthetic_bec_data):
        t, M = synthetic_bec_data
        result = ForgettingCurve.fit_both(t, M)
        for key in ("bec_params", "bec_r2", "ebb_params", "ebb_r2",
                    "better_model"):
            assert key in result
