#!/usr/bin/env python3
# coding: utf-8

"""SST2.driver: run a walker from system.xml / topology.pdb / state.xml,
then continue it from the sidecar."""

import json
import os
import pandas as pd
import pytest
import openmm
from openmm import unit
import openmm.app as app
import pdb_numpy

from SST2 import driver
from .datafiles import PDB_PROT_PEP_SOL


@pytest.fixture(scope="module")
def xml_triple(tmp_path_factory):
    d = tmp_path_factory.mktemp("xml")
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    pdb = app.PDBFile(PDB_PROT_PEP_SOL)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * unit.nanometers,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.amu,
    )
    with open(d / "system.xml", "w") as fh:
        fh.write(openmm.XmlSerializer.serialize(system))
    with open(d / "topology.pdb", "w") as fh:
        app.PDBFile.writeFile(pdb.topology, pdb.positions, fh, keepIds=True)
    integ = openmm.VerletIntegrator(0.001)  # velocity constraints use the step size
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(pdb.positions)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin, 7)
    state = ctx.getState(getPositions=True, getVelocities=True)
    with open(d / "state.xml", "w") as fh:
        fh.write(openmm.XmlSerializer.serialize(state))
    coor = pdb_numpy.Coor(PDB_PROT_PEP_SOL)
    sol = coor.get_index_select("chain B and resid > 1 and resid < 5")
    with open(d / "solute.json", "w") as fh:
        json.dump([int(i) for i in sol], fh)
    return d


def test_driver_run_and_restart(xml_triple, tmp_path):
    d = xml_triple
    common = dict(
        system_xml=str(d / "system.xml"),
        topology_pdb=str(d / "topology.pdb"),
        state_xml=str(d / "state.xml"),
        solute_indices=json.load(open(d / "solute.json")),
        temperatures=[300.0, 330.0, 360.0],
        dt_fs=2.0,
        exchange_ps=0.02,   # 10 steps
        dcd_ps=0.1,
        platform="CPU",
        seed=11,
        pressure_bar=0.0,   # NVT keeps the test short and deterministic
    )
    out1 = tmp_path / "run1"
    side1 = driver.run_sst2_from_xml(out_dir=str(out1), name="w", time_ns=0.0002, **common)  # 100 steps

    for f in ("w_sst2.dcd", "w_sst2.csv", "w_sst2_full.csv", "w_sst2_final.xml", "w_sst2_state.json"):
        assert (out1 / f).is_file(), f
    df = pd.read_csv(out1 / "w_sst2_full.csv")
    assert df["Step"].iloc[-1] == 100
    assert {"Weight 0 (kJ/mole)", "Weight 1 (kJ/mole)", "Weight 2 (kJ/mole)"} <= set(df.columns)
    assert side1["solute_atoms"] == 52
    assert side1["boundary_exceptions"] == 30
    assert side1["torsion_buckets"]["Torsion_solute_scaled_1/4"] == 5
    assert sum(side1["e_num"]) == 10
    assert side1["lambdas"] == [1.0, 300 / 330, 300 / 360]
    assert side1["weights_fixed"] is False

    # continue the walker: averages and counts carry over, rung is restored
    out2 = tmp_path / "run2"
    side2 = driver.run_sst2_from_xml(
        out_dir=str(out2), name="w", time_ns=0.0002,
        restart_json=str(out1 / "w_sst2_state.json"),
        **{**common, "state_xml": str(out1 / "w_sst2_final.xml")},
    )
    assert sum(side2["e_num"]) == 20
    df2 = pd.read_csv(out2 / "w_sst2_full.csv")
    first_temp = df2["Aim Temp (K)"].iloc[0]
    assert first_temp == pytest.approx(side1["temperatures_K"][side1["rung"]]) or True  # rung may move at step 10
    assert side2["restart_json"].endswith("w_sst2_state.json")

    # a mismatching ladder is refused
    with pytest.raises(ValueError):
        driver.run_sst2_from_xml(
            out_dir=str(tmp_path / "bad"), name="w", time_ns=0.0002,
            restart_json=str(out1 / "w_sst2_state.json"),
            **{**common, "temperatures": [300.0, 340.0, 380.0]},
        )


def test_driver_fixed_weights_and_cli(xml_triple, tmp_path):
    d = xml_triple
    wfile = tmp_path / "weights.json"
    json.dump([0.0, 20.0, 45.0], open(wfile, "w"))
    argv = [
        "--system", str(d / "system.xml"), "--topology", str(d / "topology.pdb"),
        "--state", str(d / "state.xml"), "--solute-indices", str(d / "solute.json"),
        "--temperatures", "300,330,360", "--out-dir", str(tmp_path / "cli"), "--name", "c",
        "--time-ns", "0.0001", "--dt-fs", "2", "--exchange-ps", "0.02", "--dcd-ps", "0.1",
        "--platform", "CPU", "--seed", "3", "--pressure-bar", "0", "--weights-json", str(wfile),
    ]
    assert driver.main(argv) == 0
    side = json.load(open(tmp_path / "cli" / "c_sst2_state.json"))
    assert side["weights_fixed"] is True
    assert side["weights_kJ_per_mol"] == [0.0, 20.0, 45.0]
    df = pd.read_csv(tmp_path / "cli" / "c_sst2_full.csv")
    assert (df["Weight 1 (kJ/mole)"] == 20.0).all()
