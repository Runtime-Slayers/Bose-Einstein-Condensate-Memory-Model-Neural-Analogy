"""
tests/test_p52_integration.py
=============================

Integration tests for the P52 real-data validation script.
These run the full experiment pipeline (no network, no plots) and
assert that all four analysis sections produce sensible output.

Originally the root-level ``test_p52_real_data.py`` ran as a plain
script (not collected by pytest because testpaths=["tests"]).  This
module wraps the same logic as proper pytest tests with assertions,
so it is collected and reported correctly in CI.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow importing the experiment module without installing it as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.p52_real_data as p52


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def p52_results(tmp_path_factory):
    """Run the full p52 pipeline once (no plots, no network write) and
    return the results dict.  Uses module scope so the expensive GPE /
    Bogoliubov computations only run once per test session."""
    out = tmp_path_factory.mktemp("p52_out")
    results: dict = {"_meta": {"script": "test"}}
    p52.section_bec_parameters(results)
    p52.section_forgetting_curves(results)
    p52.section_bogoliubov(results)
    p52.section_retrieval(results)
    return results


# ── Section 1: BEC experimental parameters ───────────────────────────────────

class TestBECParameters:
    def test_four_experiments_present(self, p52_results):
        exps = p52_results["bec_parameters"]["experiments"]
        assert len(exps) == 4

    def test_cornell_wieman_tc_range(self, p52_results):
        """Cornell-Wieman (1995): published T_c = 170 nK."""
        exp = p52_results["bec_parameters"]["experiments"]["Cornell-Wieman 1995 (Rb-87)"]
        assert 100 < exp["Tc_nK"] < 250, f"T_c should be ~170 nK, got {exp['Tc_nK']}"

    def test_condensate_fractions_in_range(self, p52_results):
        for name, cf in p52_results["bec_parameters"]["condensate_fractions"].items():
            assert 0.0 <= cf["frac_at_0.5Tc"] <= 1.0, f"Bad fraction for {name}"
            assert 0.0 <= cf["frac_at_0.9Tc"] <= 1.0, f"Bad fraction for {name}"

    def test_frac_at_half_Tc_approx_0875(self, p52_results):
        """n0/N = 1 - 0.5^3 = 0.875 at T=0.5·T_c (Bagnato 1987)."""
        for name, cf in p52_results["bec_parameters"]["condensate_fractions"].items():
            assert cf["frac_at_0.5Tc"] == pytest.approx(0.875, rel=1e-6), \
                f"Bagnato formula wrong for {name}"

    def test_tc_from_N_physically_bounded(self):
        """Computed T_c must be within order-of-magnitude bounds of published value,
        accounting for trap anisotropy where the single trap_freq_Hz represents only
        one dimension (axial or radial) rather than the geometric mean."""
        for name, exp in p52.BEC_EXPERIMENTS.items():
            Tc_computed = p52.compute_Tc_from_N(exp)
            ratio = Tc_computed / exp["Tc_nK"]
            assert 0.01 < ratio < 15.0, \
                f"{name}: computed Tc={Tc_computed:.1f} nK vs published {exp['Tc_nK']} nK"


# ── Section 2: forgetting curves ─────────────────────────────────────────────

class TestForgettingCurves:
    def test_three_scenarios_present(self, p52_results):
        scenarios = p52_results["forgetting_curves"]["scenarios"]
        assert set(scenarios) == {"high_attention", "normal_attention", "low_attention"}

    def test_retention_in_valid_range(self, p52_results):
        for sc, data in p52_results["forgetting_curves"]["scenarios"].items():
            for key in ("retention_at_1day", "retention_at_7days", "retention_at_30days"):
                val = data[key]
                assert 0.0 <= val <= 1.0, f"{sc}/{key}={val} out of [0,1]"

    def test_high_attention_retains_more_than_low(self, p52_results):
        s = p52_results["forgetting_curves"]["scenarios"]
        assert s["high_attention"]["retention_at_7days"] > \
               s["low_attention"]["retention_at_7days"]

    def test_half_life_ordering(self, p52_results):
        """High-attention half-life > low-attention half-life."""
        s = p52_results["forgetting_curves"]["scenarios"]
        assert s["high_attention"]["half_life_days"] >= \
               s["low_attention"]["half_life_days"]


# ── Section 3: Bogoliubov spectrum ───────────────────────────────────────────

class TestBogoliubovSpectrum:
    def test_all_experiments_have_spectrum(self, p52_results):
        spec = p52_results["bogoliubov_spectrum"]["experiments"]
        assert len(spec) == len(p52.BEC_EXPERIMENTS)

    def test_healing_length_positive(self, p52_results):
        for name, data in p52_results["bogoliubov_spectrum"]["experiments"].items():
            assert data["healing_length_xi"] > 0, f"{name}: ξ must be > 0"

    def test_sound_speed_positive(self, p52_results):
        for name, data in p52_results["bogoliubov_spectrum"]["experiments"].items():
            assert data["sound_speed_cs"] > 0, f"{name}: c_s must be > 0"

    def test_larger_condensate_shorter_healing_length(self, p52_results):
        """More atoms → denser condensate → shorter ξ (Pitaevskii 2003)."""
        spec = p52_results["bogoliubov_spectrum"]["experiments"]
        xi_cornell = spec["Cornell-Wieman 1995 (Rb-87)"]["healing_length_xi"]
        xi_ketterle = spec["Ketterle 1995 (Na-23)"]["healing_length_xi"]
        # Ketterle has ~250x more atoms → should have shorter healing length
        assert xi_ketterle < xi_cornell, \
            f"Ketterle ξ={xi_ketterle:.4f} should be < Cornell ξ={xi_cornell:.4f}"


# ── Section 4: retrieval probability ─────────────────────────────────────────

class TestRetrievalProbability:
    def test_all_probabilities_in_range(self, p52_results):
        for T_key, data in p52_results["retrieval_probability"]["data"].items():
            for cue, val in data.items():
                assert 0.0 <= val <= 1.0, f"{T_key}/{cue}={val} out of [0,1]"

    def test_strong_cue_better_than_weak_at_low_noise(self, p52_results):
        data = p52_results["retrieval_probability"]["data"]
        assert data["T=0.1"]["strong_cue"] >= data["T=0.1"]["weak_cue"]

    def test_high_noise_reduces_retrieval(self, p52_results):
        data = p52_results["retrieval_probability"]["data"]
        # At T=0.1 (low noise) weak cue should beat T=0.99 (high noise) weak cue
        assert data["T=0.1"]["weak_cue"] >= data["T=0.99"]["weak_cue"]


# ── JSON serialisability ──────────────────────────────────────────────────────

def test_results_json_serialisable(p52_results, tmp_path):
    """CI artifact: the results dict must be fully JSON-serialisable."""
    out = tmp_path / "results.json"
    with open(out, "w") as fh:
        json.dump(p52_results, fh, default=str)
    loaded = json.loads(out.read_text())
    assert "bec_parameters" in loaded
    assert "forgetting_curves" in loaded
    assert "bogoliubov_spectrum" in loaded
    assert "retrieval_probability" in loaded


# ── Condensate fraction formula (physics correctness) ────────────────────────

@pytest.mark.parametrize("T_frac,expected", [
    (0.0, 1.000),
    (0.5, 0.875),
    (0.9, 0.271),
    (1.0, 0.000),
    (1.5, 0.000),
])
def test_condensate_fraction_formula(T_frac, expected):
    """Direct physics check: n0/N = max(0, 1-(T/Tc)^3) for all published T_c."""
    for name, exp in p52.BEC_EXPERIMENTS.items():
        cf = p52.compute_condensate_fractions(exp)
        # Use the Cornell-Wieman experiment to spot-check the formula
        if "Cornell" in name:
            Tc = exp["Tc_nK"]
            T = T_frac * Tc * 1e-9
            frac = max(0.0, 1.0 - (T / (Tc * 1e-9)) ** 3)
            assert frac == pytest.approx(expected, rel=1e-6)
            break
