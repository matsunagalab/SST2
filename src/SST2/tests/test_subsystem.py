#!/usr/bin/env python3
# coding: utf-8

"""Copy-based sub-Systems (SST2.subsystem) against the ForceField route."""

import copy
import pytest
import numpy as np
import openmm
from openmm import unit
import openmm.app as app
import pdb_numpy

import SST2.tools as tools
import SST2.rest2 as rest2
from SST2.subsystem import build_subsystem
from .datafiles import PDB_PROT_PEP_SOL

KJ = unit.kilojoules_per_mole
SELECTION = "chain B and resid > 1 and resid < 5"


def _system_and_pdb():
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    pdb = app.PDBFile(PDB_PROT_PEP_SOL)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.amu,
    )
    return ff, pdb, system


def _energies_by_name(system, positions, box):
    sysc = copy.deepcopy(system)
    out = {}
    for i, f in enumerate(sysc.getForces()):
        f.setForceGroup(i)
    ctx = openmm.Context(sysc, openmm.VerletIntegrator(1.0),
                         openmm.Platform.getPlatformByName("CPU"))
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions)
    for i, f in enumerate(sysc.getForces()):
        e = ctx.getState(getEnergy=True, groups={i}).getPotentialEnergy().value_in_unit(KJ)
        out[f.getName()] = out.get(f.getName(), 0.0) + e
    return out


def _rest2(route, ff, pdb, system):
    T = 300 * unit.kelvin
    dt = 2 * unit.femtosecond
    coor = pdb_numpy.Coor(PDB_PROT_PEP_SOL)
    sol = coor.get_index_select(SELECTION)
    return rest2.REST2(
        system=system,
        pdb=pdb,
        forcefield=ff if route == "forcefield" else None,
        solute_index=sol,
        integrator=openmm.LangevinMiddleIntegrator(T, 1 / unit.picoseconds, dt),
        temperature=T,
        dt=dt,
        platform_name=tools.get_fastest_platform_name(),
        subsystem=route,
    )


def test_build_subsystem_matches_zeroed_parent():
    """The NB energy of the copied solute sub-System equals the parent's NB
    energy with every solvent charge / epsilon zeroed (same PME settings)."""
    ff, pdb, system = _system_and_pdb()
    coor = pdb_numpy.Coor(PDB_PROT_PEP_SOL)
    sol = coor.get_index_select(SELECTION)
    solute = set(sol)
    box = system.getDefaultPeriodicBoxVectors()

    sub, remap = build_subsystem(system, sol)
    assert sub.getNumParticles() == len(sol)
    assert [sub.getParticleMass(remap[i]) for i in sol] == [system.getParticleMass(i) for i in sol]
    n_constraints = sum(
        1 for i in range(system.getNumConstraints())
        if all(p in solute for p in system.getConstraintParameters(i)[:2])
    )
    assert sub.getNumConstraints() == n_constraints
    assert not any(isinstance(f, openmm.MonteCarloBarostat) for f in sub.getForces())

    zeroed = copy.deepcopy(system)
    nb = [f for f in zeroed.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    for i in range(nb.getNumParticles()):
        if i not in solute:
            q, s, e = nb.getParticleParameters(i)
            nb.setParticleParameters(i, 0.0, s, 0.0)
    for k in range(nb.getNumExceptions()):
        p1, p2, q, s, e = nb.getExceptionParameters(k)
        if p1 not in solute or p2 not in solute:
            nb.setExceptionParameters(k, p1, p2, 0.0, s, 0.0)

    sub_pos = [pdb.positions[i] for i in sorted(sol)]
    e_sub = _energies_by_name(sub, sub_pos, box)
    e_par = _energies_by_name(zeroed, pdb.positions, box)
    for name in ("NonbondedForce",):
        assert e_sub[name] == pytest.approx(e_par[name], abs=0.05), name


def test_copy_route_matches_forcefield_route():
    """Both routes give the same sub-System energies term by term and the
    same REST2 decomposition."""
    ff, pdb, system = _system_and_pdb()
    a = _rest2("forcefield", ff, pdb, copy.deepcopy(system))
    b = _rest2("copy", ff, pdb, copy.deepcopy(system))

    for label, sa, sb in (
        ("solute", a.simulation_solute, b.simulation_solute),
        ("solvent", a.simulation_solvent, b.simulation_solvent),
    ):
        fa = {f["name"]: f["energy"].value_in_unit(KJ)
              for f in tools.get_forces(sa.system, sa).values()}
        fb = {f["name"]: f["energy"].value_in_unit(KJ)
              for f in tools.get_forces(sb.system, sb).values()}
        for name in fa:
            if name in ("CMMotionRemover", "Total"):
                continue
            assert name in fb, f"{label}: {name} missing in copy route"
            assert fb[name] == pytest.approx(fa[name], rel=1e-6, abs=0.02), f"{label}: {name}"

    for scale in (1.0, 0.5):
        a.scale_nonbonded_torsion(scale)
        b.scale_nonbonded_torsion(scale)
        Ea = a.compute_all_energies()
        Eb = b.compute_all_energies()
        for frac in Ea[0]:
            assert Eb[0][frac].value_in_unit(KJ) == pytest.approx(Ea[0][frac].value_in_unit(KJ), rel=1e-6, abs=0.02)
        assert Eb[1].value_in_unit(KJ) == pytest.approx(Ea[1].value_in_unit(KJ), rel=1e-6, abs=0.02)
        assert Eb[3].value_in_unit(KJ) == pytest.approx(Ea[3].value_in_unit(KJ), rel=1e-6, abs=0.05)


def test_copy_route_from_serialized_system_without_forcefield():
    """The MDClaw case: a System deserialized from XML, no ForceField."""
    ff, pdb, system = _system_and_pdb()
    # a barostat in the parent must not reach the sub-Systems
    with_baro = copy.deepcopy(system)
    with_baro.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
    sub, _ = build_subsystem(with_baro, range(10))
    assert not any(isinstance(f, openmm.MonteCarloBarostat) for f in sub.getForces())

    # MDClaw hands over a barostat-free system.xml; REST2 adds its own barostat
    xml = openmm.XmlSerializer.serialize(system)
    system2 = openmm.XmlSerializer.deserialize(xml)
    t = _rest2("copy", None, pdb, system2)
    assert not any(isinstance(f, openmm.MonteCarloBarostat) for f in t.system_solute.getForces())
    ref = None
    for scale in (1.0, 0.5, 0.25):
        t.scale_nonbonded_torsion(scale)
        E_frac, E_pp2, E_ww, E_pw = t.compute_all_energies()
        if ref is None:
            ref = (dict(E_frac), E_pp2, E_pw)
            continue
        for frac, E_t in E_frac.items():
            assert pytest.approx(E_t / ref[0][frac], 1e-4) == 1.0
        assert pytest.approx(E_pp2 / ref[1], 1e-4) == 1.0
        assert pytest.approx(E_pw / ref[2], 1e-4) == 1.0


def test_forcefield_route_still_requires_forcefield():
    ff, pdb, system = _system_and_pdb()
    with pytest.raises(ValueError):
        _rest2("forcefield", None, pdb, system)
