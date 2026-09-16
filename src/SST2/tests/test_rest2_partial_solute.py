#!/usr/bin/env python3
# coding: utf-8

"""
Partial-solute (gREST-style) checks for REST2.

The upstream tests always take a whole chain as the solute, so the
fractional torsion buckets (k/4 with k < 4) and the boundary 1-4
exceptions are never exercised.  These tests take a residue range in
the middle of a chain so that bonded terms and 1-4 pairs cross the
solute/solvent boundary on both sides.

What must hold for any lambda:

* every quantity that compute_all_energies() reports as "unscaled"
  must be independent of lambda (the acceptance rule assumes it);
* the potential rebuilt from the decomposition must equal the Context
  energy (closure);
* restoring lambda = 1 must restore the original energies.

What must hold for the intended REST2 potential (gREST, k/l rule):

* a 1-4 exception with exactly one solute atom is scaled by sqrt(lambda)
  in both chargeProd and epsilon.
"""

import os
import pytest
import numpy as np
import openmm
from openmm import unit
import openmm.app as app
import pdb_numpy

import SST2.tools as tools
import SST2.rest2 as rest2
from .datafiles import PDB_PROT_PEP_SOL

KJ = unit.kilojoules_per_mole


def _named(forces, prefix):
    return sum(
        (f["energy"] for f in forces.values() if f["name"].startswith(prefix)),
        0 * KJ,
    )


def _boundary_exceptions(system, solute_index):
    """Indices of NonbondedForce exceptions with exactly one solute atom and
    a non-zero interaction (i.e. 1-4 pairs, not 1-2/1-3 exclusions)."""
    solute = set(solute_index)
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    out = []
    for i in range(nb.getNumExceptions()):
        p1, p2, q, sig, eps = nb.getExceptionParameters(i)
        if (p1 in solute) + (p2 in solute) != 1:
            continue
        if q._value == 0.0 and eps._value == 0.0:
            continue
        out.append(i)
    return out


@pytest.mark.parametrize(
    "selection",
    [
        "chain B and resid > 1 and resid < 5",   # 3 residues inside the peptide
        "chain A and resid > 10 and resid < 21",  # 10 residues inside the protein
    ],
)
def test_partial_solute_lambda_invariance(tmp_path, selection):
    coor = pdb_numpy.Coor(PDB_PROT_PEP_SOL)
    solute_indices = coor.get_index_select(selection)
    assert 0 < len(solute_indices) < 500

    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    dt = 2 * unit.femtosecond
    temperature = 300 * unit.kelvin
    friction = 1 / unit.picoseconds
    platform = tools.get_fastest_platform_name()

    pdb = app.PDBFile(PDB_PROT_PEP_SOL)
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
    )
    n_torsions_orig = [
        f for f in system.getForces() if isinstance(f, openmm.PeriodicTorsionForce)
    ][0].getNumTorsions()
    boundary_exc = _boundary_exceptions(system, solute_indices)
    assert len(boundary_exc) > 0, "selection must cut through 1-4 pairs"

    simulation = tools.setup_simulation(
        system=system,
        position=pdb.positions,
        topology=pdb.topology,
        integrator=openmm.LangevinMiddleIntegrator(temperature, friction, dt),
        temperature=temperature,
        platform_name=platform,
    )
    E_ref_total = tools.get_specific_forces(system, simulation, "Total")[0]

    test = rest2.REST2(
        system=system,
        pdb=pdb,
        forcefield=forcefield,
        solute_index=solute_indices,
        integrator=openmm.LangevinMiddleIntegrator(temperature, friction, dt),
        temperature=temperature,
        dt=dt,
        platform_name=platform,
    )

    # ---- bucket bookkeeping -------------------------------------------
    torsion_counts = {
        f.getName(): f.getNumTorsions()
        for f in test.system.getForces()
        if isinstance(f, openmm.CustomTorsionForce)
    }
    print("torsion buckets:", torsion_counts)
    assert sum(torsion_counts.values()) == n_torsions_orig
    for k in (1, 2, 3, 4):
        assert torsion_counts.get(f"Torsion_solute_scaled_{k}/4", 0) > 0, (
            f"bucket {k}/4 empty: the selection does not cross the boundary"
        )
    assert set(test.fractional_terms) >= {0.25, 0.5, 0.75, 1.0}

    # ---- lambda = 1 must reproduce the plain system -------------------
    E_rest2_total = tools.get_specific_forces(test.system, test.simulation, "Total")[0]
    assert pytest.approx(E_rest2_total / E_ref_total, 1e-4) == 1.0

    # ---- lambda scan ----------------------------------------------------
    tol = 1e-4
    ref = None
    for scale in [1.0, 0.5, 0.25, 1.0]:
        test.scale_nonbonded_torsion(scale)
        E_frac, E_pp2, E_ww, E_pw = test.compute_all_energies()
        E_state = test.simulation.context.getState(getEnergy=True).getPotentialEnergy()

        # closure: potential rebuilt from the decomposition == Context energy
        E_rebuilt = E_pp2 + E_ww + np.sqrt(scale) * E_pw
        for frac, E_t in E_frac.items():
            E_rebuilt += scale ** frac * E_t
        # E_ww as reported at HEAD (89a4de6) omits the solvent NB energy, so
        # closure can only be checked up to that constant: use the lambda=1
        # reference to define it.
        if ref is None:
            ref = dict(E_frac=dict(E_frac), E_pp2=E_pp2, E_pw=E_pw,
                       offset=E_state - E_rebuilt)
            continue

        print(f"lambda={scale}: E_pw unscaled {E_pw} (ref {ref['E_pw']})")
        # unscaled quantities must not depend on lambda
        for frac, E_t in E_frac.items():
            assert pytest.approx(E_t / ref["E_frac"][frac], tol) == 1.0, (
                f"E_frac[{frac}] drifts with lambda={scale}"
            )
        assert pytest.approx(E_pp2 / ref["E_pp2"], tol) == 1.0
        assert pytest.approx(E_pw / ref["E_pw"], tol) == 1.0, (
            f"unscaled solute-solvent energy drifts with lambda={scale}: "
            f"{E_pw} vs {ref['E_pw']}"
        )
        assert pytest.approx(
            (E_rebuilt + ref["offset"]) / E_state, tol
        ) == 1.0, f"closure broken at lambda={scale}"

    # ---- round trip -----------------------------------------------------
    E_back = tools.get_specific_forces(test.system, test.simulation, "Total")[0]
    assert pytest.approx(E_back / E_ref_total, 1e-6) == 1.0


def test_partial_solute_boundary_14_scaled():
    """A 1-4 exception with one solute atom must carry sqrt(lambda) on both
    chargeProd and epsilon (gREST k/l rule with k=1, l=2)."""
    selection = "chain B and resid > 1 and resid < 5"
    coor = pdb_numpy.Coor(PDB_PROT_PEP_SOL)
    solute_indices = coor.get_index_select(selection)

    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    dt = 2 * unit.femtosecond
    temperature = 300 * unit.kelvin
    friction = 1 / unit.picoseconds
    platform = tools.get_fastest_platform_name()

    pdb = app.PDBFile(PDB_PROT_PEP_SOL)
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
    )
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    boundary_exc = _boundary_exceptions(system, solute_indices)
    params_1 = {i: nb.getExceptionParameters(i) for i in boundary_exc}
    print(f"{len(boundary_exc)} boundary 1-4 exceptions")

    test = rest2.REST2(
        system=system,
        pdb=pdb,
        forcefield=forcefield,
        solute_index=solute_indices,
        integrator=openmm.LangevinMiddleIntegrator(temperature, friction, dt),
        temperature=temperature,
        dt=dt,
        platform_name=platform,
    )
    scale = 0.25
    test.scale_nonbonded_torsion(scale)
    nb = test.system_forces["NonbondedForce"]
    for i in boundary_exc:
        p1, p2, q, sig, eps = nb.getExceptionParameters(i)
        _, _, q1, _, eps1 = params_1[i]
        assert q._value == pytest.approx(q1._value * np.sqrt(scale), rel=1e-6, abs=1e-12), (
            f"boundary exception {i} ({p1},{p2}) chargeProd not scaled by sqrt(lambda)"
        )
        assert eps._value == pytest.approx(eps1._value * np.sqrt(scale), rel=1e-6, abs=1e-12), (
            f"boundary exception {i} ({p1},{p2}) epsilon not scaled by sqrt(lambda)"
        )


def test_partial_solute_subsystems_keep_original_bonds():
    """The solute-only and solvent-only sub-topologies must contain exactly
    the bonds of the full topology restricted to their atoms.  Rebuilding
    them from a PDB file used to add a peptide bond between the two
    residues flanking the solute (they become adjacent in the chain)."""
    selection = "chain B and resid > 1 and resid < 5"
    coor = pdb_numpy.Coor(PDB_PROT_PEP_SOL)
    solute_indices = coor.get_index_select(selection)
    solute = set(solute_indices)

    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    dt = 2 * unit.femtosecond
    temperature = 300 * unit.kelvin
    friction = 1 / unit.picoseconds
    pdb = app.PDBFile(PDB_PROT_PEP_SOL)
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
    )
    test = rest2.REST2(
        system=system,
        pdb=pdb,
        forcefield=forcefield,
        solute_index=solute_indices,
        integrator=openmm.LangevinMiddleIntegrator(temperature, friction, dt),
        temperature=temperature,
        dt=dt,
        platform_name=tools.get_fastest_platform_name(),
    )
    n_solute_bonds = sum(
        1 for a, b in pdb.topology.bonds() if a.index in solute and b.index in solute
    )
    n_solvent_bonds = sum(
        1 for a, b in pdb.topology.bonds() if a.index not in solute and b.index not in solute
    )
    assert test.simulation_solute.topology.getNumBonds() == n_solute_bonds
    assert test.simulation_solvent.topology.getNumBonds() == n_solvent_bonds

    # no bond may join two residues that are not consecutive in the full topology
    for top in (test.simulation_solute.topology, test.simulation_solvent.topology):
        for a, b in top.bonds():
            if a.residue is not b.residue and a.residue.chain is b.residue.chain:
                assert abs(int(a.residue.id) - int(b.residue.id)) == 1, (
                    f"spurious bond {a.residue.name}{a.residue.id}:{a.name}-"
                    f"{b.residue.name}{b.residue.id}:{b.name}"
                )
