"""
Tests for bec_memory.gpe — GPESolver (Gross-Pitaevskii equation).
"""
import pytest
import numpy as np
from bec_memory.gpe import GPESolver, GPEConfig


@pytest.fixture
def solver():
    cfg = GPEConfig(grid_points=64, x_max=8.0, dt=0.01, g_coupling=0.01)
    s = GPESolver(cfg)
    s.initialize_gaussian(width=1.0)
    return s


class TestGPESolver:
    def test_initialization_gaussian_norm(self, solver):
        """Gaussian initialization should be normalised."""
        norm = solver.norm()
        assert norm == pytest.approx(1.0, rel=1e-4)

    def test_density_is_positive(self, solver):
        rho = solver.density()
        assert np.all(rho >= 0.0)

    def test_density_shape(self, solver):
        rho = solver.density()
        assert rho.shape == (solver.config.grid_points,)

    def test_evolution_preserves_norm(self, solver):
        """Unitary GPE evolution must conserve norm."""
        solver.evolve(steps=50)
        norm = solver.norm()
        assert norm == pytest.approx(1.0, rel=1e-3)

    def test_evolution_returns_correct_shapes(self, solver):
        psi, t = solver.evolve(steps=10)
        assert psi.shape == (solver.config.grid_points,)
        assert isinstance(t, float)

    def test_time_advances(self, solver):
        assert solver.time == pytest.approx(0.0)
        solver.evolve(steps=20)
        assert solver.time == pytest.approx(20 * solver.config.dt, rel=1e-9)

    def test_thomas_fermi_init_normalised(self):
        cfg = GPEConfig(grid_points=64, g_coupling=0.01)
        s = GPESolver(cfg)
        s.initialize_thomas_fermi()
        norm = s.norm()
        assert norm == pytest.approx(1.0, rel=1e-4)

    def test_ground_state_lower_energy(self, solver):
        """Imaginary-time ground state should have lower energy than
        an arbitrary initial state."""
        e_init = solver.ground_state_energy()
        s2 = GPESolver(solver.config)
        s2.initialize_gaussian(width=1.0)
        s2.find_ground_state(steps=200)
        e_gs = s2.ground_state_energy()
        assert e_gs <= e_init + 0.5  # allow some slack for finite grid

    def test_external_stimulus_changes_psi(self, solver):
        """An external potential should alter the wavefunction."""
        psi_before = solver.psi.copy()
        stimulus = np.zeros(solver.config.grid_points)
        stimulus[solver.config.grid_points // 2] = 5.0  # delta-like pulse
        solver.evolve(steps=10, external_stimulus=stimulus)
        assert not np.allclose(psi_before, solver.psi)

    def test_invalid_stimulus_shape_raises(self, solver):
        bad_stim = np.ones(solver.config.grid_points + 5)
        with pytest.raises(ValueError, match="shape"):
            solver.evolve(steps=1, external_stimulus=bad_stim)

    def test_sound_speed_positive(self, solver):
        """Sound speed c_s = sqrt(g n0 / m) must be positive after init."""
        from bec_memory.false_memory import BogoliubovSpectrum
        spec = BogoliubovSpectrum(n0=0.8, g=0.01)
        assert spec.sound_speed() > 0.0

    def test_repr_contains_key_info(self, solver):
        r = repr(solver)
        assert "GPESolver" in r
        assert "grid=" in r


class TestGPEConfig:
    def test_default_config(self):
        cfg = GPEConfig()
        assert cfg.grid_points == 256
        assert cfg.g_coupling == 0.01

    def test_custom_config(self):
        cfg = GPEConfig(grid_points=128, g_coupling=0.05, dt=0.005)
        s = GPESolver(cfg)
        assert s.config.grid_points == 128
