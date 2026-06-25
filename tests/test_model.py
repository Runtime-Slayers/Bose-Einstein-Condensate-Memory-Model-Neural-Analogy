"""
Tests for bec_memory.model — BECMemoryModel core functionality.
"""
import math
import pytest
import numpy as np
from bec_memory.model import BECMemoryModel, ModelConfig, K_B, HBAR, ZETA3


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def default_model():
    return BECMemoryModel()


@pytest.fixture
def custom_model():
    cfg = ModelConfig(N_neurons=5_000, T_c=2.0, g_coupling=0.05,
                      trap_freq_Hz=200.0)
    return BECMemoryModel(cfg)


# ── condensate_fraction ───────────────────────────────────────────────────────

class TestCondensateFraction:
    def test_zero_temperature_gives_full_condensate(self, default_model):
        """At T=0 all neurons should be in the coherent state."""
        assert default_model.condensate_fraction(0.0) == pytest.approx(1.0)

    def test_at_critical_temperature_gives_zero(self, default_model):
        """At T = T_c the condensate fraction must be exactly 0."""
        T_c = default_model.config.T_c
        assert default_model.condensate_fraction(T_c) == pytest.approx(0.0)

    def test_above_critical_temperature_gives_zero(self, default_model):
        """Above T_c there is no condensate."""
        T_c = default_model.config.T_c
        assert default_model.condensate_fraction(T_c * 1.5) == 0.0
        assert default_model.condensate_fraction(T_c * 10.0) == 0.0

    def test_formula_n0_over_N(self, default_model):
        """n0/N = 1 - (T/T_c)^3  (harmonic-trap ideal BEC, Bagnato 1987)."""
        T_c = default_model.config.T_c
        for T_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            T = T_frac * T_c
            expected = 1.0 - T_frac ** 3
            assert default_model.condensate_fraction(T) == pytest.approx(
                expected, rel=1e-9
            )

    def test_fraction_is_between_zero_and_one(self, default_model):
        """Condensate fraction must always be in [0, 1]."""
        T_c = default_model.config.T_c
        for T in np.linspace(0, 2 * T_c, 50):
            frac = default_model.condensate_fraction(T)
            assert 0.0 <= frac <= 1.0

    def test_monotonically_decreasing_with_temperature(self, default_model):
        """Higher noise → lower condensate fraction."""
        T_c = default_model.config.T_c
        T_arr = np.linspace(0.0, T_c * 0.999, 100)
        fracs = [default_model.condensate_fraction(T) for T in T_arr]
        # Each element should be >= the next
        assert all(fracs[i] >= fracs[i + 1] for i in range(len(fracs) - 1))

    def test_custom_T_c_scaling(self, custom_model):
        """Custom T_c should scale properly."""
        T_c = custom_model.config.T_c  # 2.0
        # At T = 0.5 * T_c = 1.0:  frac = 1 - 0.5^3 = 0.875
        assert custom_model.condensate_fraction(1.0) == pytest.approx(0.875)


# ── memory_strength ───────────────────────────────────────────────────────────

class TestMemoryStrength:
    def test_full_strength_at_zero_noise(self, default_model):
        assert default_model.memory_strength(0.0, M0=1.0) == pytest.approx(1.0)

    def test_zero_strength_above_Tc(self, default_model):
        T_c = default_model.config.T_c
        assert default_model.memory_strength(T_c, M0=1.0) == pytest.approx(0.0)
        assert default_model.memory_strength(T_c * 2, M0=1.0) == 0.0

    def test_scales_with_M0(self, default_model):
        T = 0.5 * default_model.config.T_c
        for M0 in [0.5, 1.0, 2.0]:
            expected = M0 * (1.0 - 0.5 ** 3)
            assert default_model.memory_strength(T, M0=M0) == pytest.approx(
                expected, rel=1e-9
            )


# ── critical_temperature ──────────────────────────────────────────────────────

class TestCriticalTemperature:
    def test_Tc_positive(self, default_model):
        assert default_model.critical_temperature() > 0.0

    def test_Tc_scales_with_N_one_third(self, default_model):
        """T_c ∝ N^(1/3) for harmonic trap (Pitaevskii & Stringari 2003)."""
        Tc1 = default_model.critical_temperature(N=1_000)
        Tc8 = default_model.critical_temperature(N=8_000)
        ratio = Tc8 / Tc1
        assert ratio == pytest.approx(2.0, rel=1e-6)  # (8000/1000)^(1/3) = 2

    def test_Cornell_Wieman_order_of_magnitude(self, default_model):
        """Cornell-Wieman Rb-87: N=2000, omega=2π×180 Hz → T_c ≈ 170 nK."""
        omega = 2.0 * math.pi * 180.0
        Tc_K = default_model.critical_temperature(N=2_000, omega=omega)
        Tc_nK = Tc_K * 1e9
        # Published T_c ≈ 170 nK; within factor 3 (trap not perfectly harmonic)
        assert 50.0 < Tc_nK < 500.0

    def test_scales_with_omega(self, default_model):
        """T_c ∝ ω."""
        Tc1 = default_model.critical_temperature(N=1_000,
                                                  omega=2 * math.pi * 100)
        Tc2 = default_model.critical_temperature(N=1_000,
                                                  omega=2 * math.pi * 200)
        assert Tc2 / Tc1 == pytest.approx(2.0, rel=1e-6)


# ── forgetting_curve ──────────────────────────────────────────────────────────

class TestForgettingCurve:
    def test_output_shapes(self, default_model):
        t, M = default_model.forgetting_curve(n_points=100)
        assert t.shape == (100,)
        assert M.shape == (100,)

    def test_starts_at_M0(self, default_model):
        """At t=0, retention should equal M0 (with low T0)."""
        t, M = default_model.forgetting_curve(t_max=30.0, T0=0.01, alpha=0.0)
        assert M[0] > 0.99  # essentially full retention

    def test_monotonically_decreasing(self, default_model):
        """Memory should not increase over time (no rehearsal modelled)."""
        t, M = default_model.forgetting_curve(T0=0.1, alpha=0.05)
        for i in range(len(M) - 1):
            assert M[i] >= M[i + 1] - 1e-12  # allow float tolerance

    def test_retention_in_valid_range(self, default_model):
        _, M = default_model.forgetting_curve()
        assert np.all(M >= 0.0)
        assert np.all(M <= 1.0 + 1e-10)

    def test_no_alpha_means_constant_retention(self, default_model):
        """If alpha=0, noise is constant → retention stays constant."""
        T0 = 0.3
        t, M = default_model.forgetting_curve(T0=T0, alpha=0.0, n_points=50)
        expected = default_model.condensate_fraction(T0)
        assert np.allclose(M, expected, atol=1e-12)


# ── retrieval_probability ─────────────────────────────────────────────────────

class TestRetrievalProbability:
    def test_high_noise_low_retrieval(self, default_model):
        T_c = default_model.config.T_c
        p = default_model.retrieval_probability(T=T_c * 1.5)
        assert p == pytest.approx(0.0)

    def test_low_noise_reasonable_retrieval(self, default_model):
        p = default_model.retrieval_probability(T=0.05, cue_strength=0.9)
        assert p > 0.5

    def test_strong_cue_better_than_weak(self, default_model):
        T = 0.4
        p_weak = default_model.retrieval_probability(T, cue_strength=0.1)
        p_strong = default_model.retrieval_probability(T, cue_strength=0.9)
        assert p_strong >= p_weak

    def test_probability_in_range(self, default_model):
        for T in [0.0, 0.3, 0.7, 1.0, 1.5]:
            for cue in [0.0, 0.5, 1.0]:
                p = default_model.retrieval_probability(T, cue_strength=cue)
                assert 0.0 <= p <= 1.0


# ── summary / repr ────────────────────────────────────────────────────────────

def test_summary_is_string(default_model):
    s = default_model.summary()
    assert isinstance(s, str)
    assert "BECMemoryModel" in s
    assert "T_c" in s


def test_repr(default_model):
    r = repr(default_model)
    assert "BECMemoryModel" in r
    assert "T_c" in r
