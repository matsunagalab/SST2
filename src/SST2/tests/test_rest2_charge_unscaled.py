#!/usr/bin/env python3
# coding: utf-8

"""
REST2 with ``charge_unscaled_index``: part of the solute keeps its charges
(only its Lennard-Jones epsilon and torsions are scaled).

The solute is a residue range inside the protein chain; the side-chain atoms
of its first half keep their charges, everything else in the range is fully
scaled.  What must hold for any lambda:

* particle parameters: charge x sqrt(lambda) on the charge-scaled solute atoms,
  unchanged on the charge-unscaled ones, epsilon x lambda on every solute atom;
* every "unscaled" quantity reported by compute_all_energies() is independent
  of lambda (the acceptance rule assumes it);
* closure: E(lambda) - E(1) of the Context equals the same difference rebuilt
  from the decomposition;
* lambda = 1 restores the original energy.
"""

import numpy as np
import pytest
import openmm
from openmm import unit
import openmm.app as app
import pdb_numpy

import SST2.tools as tools
import SST2.rest2 as rest2
from .datafiles import PDB_PROT_PEP_SOL

KJ = unit.kilojoules_per_mole
BACKBONE = {"N", "H", "CA", "HA", "C", "O", "HA2", "HA3"}


def _setup():
    coor = pdb_numpy.Coor(PDB_PROT_PEP_SOL)
    solute = coor.get_index_select("chain A and resid > 10 and resid < 21")
    pdb = app.PDBFile(PDB_PROT_PEP_SOL)
    atoms = list(pdb.topology.atoms())
    first_half = {r for r in range(11, 16)}
    charge_unscaled = [
        i for i in solute
        if atoms[i].name not in BACKBONE and int(atoms[i].residue.id) in first_half
    ]
    assert 0 < len(charge_unscaled) < len(solute)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
    )
    return pdb, forcefield, system, solute, charge_unscaled


def _rest2(pdb, forcefield, system, solute, charge_unscaled, platform):
    dt = 2 * unit.femtosecond
    temperature = 300 * unit.kelvin
    return rest2.REST2(
        system=system,
        pdb=pdb,
        forcefield=forcefield,
        solute_index=solute,
        integrator=openmm.LangevinMiddleIntegrator(temperature, 1 / unit.picoseconds, dt),
        temperature=temperature,
        dt=dt,
        platform_name=platform,
        subsystem="copy",
        charge_unscaled_index=charge_unscaled,
    )


def test_charge_unscaled_parameters_and_decomposition():
    pdb, forcefield, system, solute, charge_unscaled = _setup()
    platform = tools.get_fastest_platform_name()
    nb0 = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    q0 = {i: nb0.getParticleParameters(i)[0] for i in solute}
    eps0 = {i: nb0.getParticleParameters(i)[2] for i in solute}

    reference = tools.setup_simulation(
        system=system, position=pdb.positions, topology=pdb.topology,
        integrator=openmm.VerletIntegrator(0.001), temperature=300 * unit.kelvin,
        platform_name=platform,
    )
    E_ref_total = reference.context.getState(getEnergy=True).getPotentialEnergy()

    test = _rest2(pdb, forcefield, system, solute, charge_unscaled, platform)
    E1 = test.simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert abs((E1 - E_ref_total).value_in_unit(KJ)) < 1e-3 * abs(E_ref_total.value_in_unit(KJ))

    unscaled = set(charge_unscaled)
    ref = None
    for scale in [1.0, 0.7, 0.5, 0.35, 1.0]:
        test.scale_nonbonded_torsion(scale)

        nb = test.system_forces["NonbondedForce"]
        for i in solute:
            q, _, eps = nb.getParticleParameters(i)
            want_q = q0[i] if i in unscaled else q0[i] * np.sqrt(scale)
            assert q.value_in_unit(unit.elementary_charge) == pytest.approx(
                want_q.value_in_unit(unit.elementary_charge), abs=1e-6)
            assert eps.value_in_unit(KJ) == pytest.approx(eps0[i].value_in_unit(KJ) * scale, abs=1e-9)

        E_frac, E_pp2, E_ww, E_pw = test.compute_all_energies()
        E_state = test.simulation.context.getState(getEnergy=True).getPotentialEnergy()
        E_rebuilt = E_pp2 + E_ww + np.sqrt(scale) * E_pw
        for frac, E_t in E_frac.items():
            E_rebuilt += scale ** frac * E_t
        if ref is None:
            ref = dict(E_frac=dict(E_frac), E_pp2=E_pp2, E_pw=E_pw, E_ww=E_ww,
                       E_state=E_state, E_rebuilt=E_rebuilt)
            continue
        # unscaled components do not depend on lambda (absolute, kJ/mol)
        for frac, E_t in E_frac.items():
            assert abs((E_t - ref["E_frac"][frac]).value_in_unit(KJ)) < 0.5, (
                f"E_frac[{frac}] drifts at lambda={scale}: {E_t} vs {ref['E_frac'][frac]}")
        assert abs((E_pw - ref["E_pw"]).value_in_unit(KJ)) < 0.5, (
            f"E_pw drifts at lambda={scale}: {E_pw} vs {ref['E_pw']}")
        # closure on the lambda-dependent part (absolute, kJ/mol)
        d_state = (E_state - ref["E_state"]).value_in_unit(KJ)
        d_rebuilt = (E_rebuilt - ref["E_rebuilt"]).value_in_unit(KJ)
        print(f"lambda={scale}: dE context {d_state:.3f}  rebuilt {d_rebuilt:.3f}")
        assert d_state == pytest.approx(d_rebuilt, abs=1.0), f"closure broken at lambda={scale}"

    E_back = test.simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert abs((E_back - E1).value_in_unit(KJ)) < 1e-2


def test_charge_unscaled_must_be_subset():
    pdb, forcefield, system, solute, _ = _setup()
    outside = [i for i in range(system.getNumParticles()) if i not in set(solute)][:3]
    with pytest.raises(ValueError):
        _rest2(pdb, forcefield, system, solute, outside, tools.get_fastest_platform_name())
