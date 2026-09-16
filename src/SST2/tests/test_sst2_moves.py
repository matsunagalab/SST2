#!/usr/bin/env python3
# coding: utf-8

"""Rung moves of the SST2 driver, checked on a stub without any MD."""

import numpy as np
import pytest
from openmm import unit

from SST2.sst2 import SST2

KJ = unit.kilojoules_per_mole
KT = (unit.BOLTZMANN_CONSTANT_kB * 300 * unit.kelvin * unit.AVOGADRO_CONSTANT_NA).value_in_unit(KJ)


class _Rest2Stub:
    def __init__(self):
        self.scales = []

    def scale_nonbonded_torsion(self, scale):
        self.scales.append(scale)


def _stub(temperatures, weights, move, seed=1, current=0):
    st = SST2.__new__(SST2)
    st.rest2 = _Rest2Stub()
    st.move = move
    st._rng = np.random.default_rng(seed)
    # numpy Generator has .random(); SST2 only uses that method
    st.temperatures = [t * unit.kelvin for t in temperatures]
    st.temp_ref_index = 0
    st.lambdas = [temperatures[0] / t for t in temperatures]
    st._e_fracs = [0.5, 1.0]
    st._weights = list(weights)
    st._updateWeights = False
    st.currentTemperature = current
    return st


def _balanced_weights(st, E_frac, E_pw, offsets_kT):
    """Weights that cancel the energy term so that rung j has log-weight
    offsets_kT[j] (in kT) relative to rung 0."""
    st._weights = [0.0] * len(st.lambdas)
    st.currentTemperature = 0
    log_w = st._log_weights(E_frac, E_pw)  # energy part only, kT units
    return [(-log_w[j] + offsets_kT[j]) * KT for j in range(len(st.lambdas))]


def _exact_gibbs(st, E_frac, E_pw):
    log_w = st._log_weights(E_frac, E_pw)
    p = np.exp(log_w - log_w.max())
    return p / p.sum()


def test_log_weights_formula():
    st = _stub([300, 400, 600], [0.0, 10.0, 25.0], "gibbs", current=1)
    E_frac = {0.5: -100 * KJ, 1.0: -400 * KJ}
    E_pw = -800 * KJ
    log_w = st._log_weights(E_frac, E_pw)
    assert log_w[1] == 0.0
    lam_i = st.lambdas[1]
    for j in (0, 2):
        lam_j = st.lambdas[j]
        expected = (
            (lam_i ** 0.5 - lam_j ** 0.5) * (-100)
            + (lam_i ** 1.0 - lam_j ** 1.0) * (-400)
            + (lam_i ** 0.5 - lam_j ** 0.5) * (-800)
            + (st.weights[j] - st.weights[1])
        ) / KT
        assert log_w[j] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("current", [0, 1, 2])
def test_gibbs_move_samples_conditional(current):
    """The empirical move frequencies match the exact conditional
    distribution, independently of the starting rung."""
    E_frac = {0.5: -100 * KJ, 1.0: -400 * KJ}
    E_pw = -800 * KJ
    st = _stub([300, 330, 360], [0.0, 0.0, 0.0], "gibbs", seed=3, current=current)
    st._weights = _balanced_weights(st, E_frac, E_pw, [0.0, 1.0, -0.5])
    st.currentTemperature = current
    p_exact = _exact_gibbs(st, E_frac, E_pw)
    assert 0.05 < p_exact.max() < 0.95, "pick a case where every rung is reachable"
    n = 20000
    counts = np.zeros(3)
    for _ in range(n):
        st.currentTemperature = current
        j = st._attemptTemperatureChange(E_frac, E_pw)
        counts[j] += 1
    freq = counts / n
    # 4 sigma binomial tolerance
    for j in range(3):
        sigma = np.sqrt(p_exact[j] * (1 - p_exact[j]) / n)
        assert abs(freq[j] - p_exact[j]) < 4 * sigma + 1e-9, (j, freq, p_exact)


def test_neighbor_move_detailed_balance():
    """For fixed energies the neighbour move is a Metropolis chain on the
    rung index; its transition matrix must satisfy detailed balance with
    respect to the same conditional distribution."""
    E_frac = {0.5: -100 * KJ, 1.0: -400 * KJ}
    E_pw = -800 * KJ
    st = _stub([300, 330, 360], [0.0, 0.0, 0.0], "neighbor", seed=5)
    st._weights = _balanced_weights(st, E_frac, E_pw, [0.0, 1.0, -0.5])
    pi = _exact_gibbs(st, E_frac, E_pw)
    n = 40000
    T = np.zeros((3, 3))
    for i in range(3):
        for _ in range(n):
            st.currentTemperature = i
            j = st._attemptTemperatureChange(E_frac, E_pw)
            T[i, j] += 1
    T /= n
    # exact Metropolis matrix with symmetric 1/2 proposals and rejection at the ends
    for i in range(3):
        for j in (i - 1, i + 1):
            if 0 <= j < 3:
                a = min(1.0, pi[j] / pi[i])
                assert T[i, j] == pytest.approx(0.5 * a, abs=4 * np.sqrt(0.25 / n) + 1e-9), (i, j)
    for i in range(3):
        for j in range(3):
            if i != j and T[i, j] > 0:
                assert pi[i] * T[i, j] == pytest.approx(pi[j] * T[j, i], rel=0.15)


def test_move_applies_lambda_to_rest2():
    E_frac = {0.5: -100 * KJ, 1.0: -400 * KJ}
    E_pw = -800 * KJ
    st = _stub([300, 330, 360], [0.0, 0.0, 0.0], "gibbs", seed=2)
    st._weights = _balanced_weights(st, E_frac, E_pw, [0.0, 0.0, 0.0])
    for _ in range(200):
        before = st.currentTemperature
        j = st._attemptTemperatureChange(E_frac, E_pw)
        if j != before:
            assert st.rest2.scales[-1] == st.lambdas[j]
            assert st.currentTemperature == j
    assert len(st.rest2.scales) > 0
