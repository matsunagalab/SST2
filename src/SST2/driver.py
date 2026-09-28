#!/usr/bin/env python3
# coding: utf-8

"""
Run SST2 from a serialized OpenMM System.

Entry point for workflow engines (MDClaw) that hand over a prepared
system as ``system.xml`` + ``topology.pdb`` + ``state.xml`` and a list
of solute atom indices.  No ForceField object is needed: the sub-Systems
for the energy decomposition are copied out of the System
(``REST2(subsystem="copy")``).

Every run writes, under ``out_dir`` with prefix ``name``:

* ``<name>_sst2.dcd`` / ``<name>_sst2.csv``   trajectory and OpenMM log
* ``<name>_sst2_full.csv``                     SST2 report: step, rung
  temperature, per-fraction energies, E_pw, effective weights
* ``<name>_sst2_final.xml``                    final OpenMM State
* ``<name>_sst2_state.json``                   driver sidecar: rung,
  ladder, weights, running averages and counts, seed, provenance.  Pass
  it back as ``restart_json`` to continue the same walker in a new run.
"""

import json
import logging
import os
import sys
import types

import numpy as np
import openmm
import openmm.app as app
from openmm import unit

from SST2 import __version__ as SST2_VERSION
from SST2.rest2 import REST2, run_rest2
from SST2.sst2 import SST2

logger = logging.getLogger(__name__)

KJ = unit.kilojoules_per_mole


def load_xml_inputs(system_xml, topology_pdb, state_xml=None):
    """Return (system, topology, positions, velocities, box) from the XML
    triple.  Positions and box come from ``state_xml`` when given."""
    with open(system_xml) as fh:
        system = openmm.XmlSerializer.deserialize(fh.read())
    pdb = app.PDBFile(topology_pdb)
    topology, positions = pdb.topology, pdb.positions
    velocities = None
    box = topology.getPeriodicBoxVectors()
    if state_xml is not None and os.path.isfile(state_xml):
        with open(state_xml) as fh:
            state = openmm.XmlSerializer.deserialize(fh.read())
        positions = state.getPositions()
        try:
            velocities = state.getVelocities()
        except Exception:  # noqa: BLE001
            velocities = None
        try:
            box = state.getPeriodicBoxVectors()
        except Exception:  # noqa: BLE001
            pass
    if system.getNumParticles() != topology.getNumAtoms():
        raise ValueError(
            f"system.xml has {system.getNumParticles()} particles, "
            f"topology.pdb has {topology.getNumAtoms()} atoms"
        )
    if box is not None:
        system.setDefaultPeriodicBoxVectors(*box)
        topology.setPeriodicBoxVectors(box)
    return system, topology, positions, velocities, box


def _platform_name(platform):
    if platform in (None, "auto", "fastest"):
        from SST2.tools import get_fastest_platform_name
        return get_fastest_platform_name()
    return platform


def _quantity_list(values):
    return [v.value_in_unit(KJ) if unit.is_quantity(v) else float(v) for v in values]


def sidecar_from_sst2(sst2, extra=None):
    """Serializable snapshot of the walker state (rung, ladder, weights,
    running averages)."""
    d = {
        "sst2_version": SST2_VERSION,
        "rung": int(sst2.currentTemperature),
        "temperatures_K": [t.value_in_unit(unit.kelvin) for t in sst2.temperatures],
        "ref_temperature_K": sst2.refTemperature.value_in_unit(unit.kelvin),
        "lambdas": [float(l) for l in sst2.lambdas],
        "fractional_terms": [float(f) for f in sst2._e_fracs],
        "weights_kJ_per_mol": [float(w) for w in sst2.weights],
        "weights_fixed": not sst2._updateWeights,
        "e_num": [int(n) for n in sst2._e_num],
        "e_frac_avg_kJ_per_mol": {
            str(frac): [row[k].value_in_unit(KJ) for row in sst2._e_frac_avg]
            for k, frac in enumerate(sst2._e_fracs)
        },
        "e_solute_solv_avg_kJ_per_mol": _quantity_list(sst2._e_solute_solv_avg),
        "move": sst2.move,
        "step": int(sst2.simulation.currentStep),
    }
    if extra:
        d.update(extra)
    return d


def restore_averages(sst2, sidecar):
    """Load running averages, counts and the current rung from a sidecar
    written by :func:`sidecar_from_sst2` (same ladder required)."""
    temps = [round(t, 6) for t in sidecar["temperatures_K"]]
    mine = [round(t.value_in_unit(unit.kelvin), 6) for t in sst2.temperatures]
    if temps != mine:
        raise ValueError(f"ladder in restart sidecar {temps} differs from {mine}")
    fracs = [float(f) for f in sidecar["fractional_terms"]]
    if fracs != [float(f) for f in sst2._e_fracs]:
        raise ValueError(
            f"fractional terms in sidecar {fracs} differ from {list(sst2._e_fracs)}"
        )
    sst2._e_num = [int(n) for n in sidecar["e_num"]]
    for k, frac in enumerate(sst2._e_fracs):
        col = sidecar["e_frac_avg_kJ_per_mol"][str(frac)]
        for m in range(len(sst2.temperatures)):
            sst2._e_frac_avg[m][k] = col[m] * KJ
    sst2._e_solute_solv_avg = [v * KJ for v in sidecar["e_solute_solv_avg_kJ_per_mol"]]
    rung = int(sidecar["rung"])
    sst2.currentTemperature = rung
    sst2.rest2.scale_nonbonded_torsion(sst2.lambdas[rung])


def run_sst2_from_xml(
    *,
    system_xml,
    topology_pdb,
    solute_indices,
    temperatures,
    out_dir,
    name="run",
    state_xml=None,
    ref_temperature=None,
    time_ns=1.0,
    dt_fs=2.0,
    exchange_ps=2.0,
    report_ps=None,
    dcd_ps=10.0,
    log_ps=None,
    platform="auto",
    seed=None,
    move="gibbs",
    pressure_bar=1.0,
    friction_ps=1.0,
    nonbonded_scale=True,
    exclude_pro_omegas=False,
    weights=None,
    restart_json=None,
    extra_provenance=None,
    charge_unscaled_indices=None,
):
    """Run one SST2 walker.  Returns the sidecar dict (also written to
    ``<out_dir>/<name>_sst2_state.json``)."""
    os.makedirs(out_dir, exist_ok=True)
    generic = os.path.join(out_dir, name)

    system, topology, positions, velocities, box = load_xml_inputs(
        system_xml, topology_pdb, state_xml
    )
    solute_indices = sorted(set(int(i) for i in solute_indices))
    if not solute_indices:
        raise ValueError("solute_indices is empty")
    if solute_indices[-1] >= system.getNumParticles():
        raise ValueError("solute index out of range")
    logger.info(f"- {len(solute_indices)} solute atoms of {system.getNumParticles()}")
    charge_unscaled_indices = sorted(set(int(i) for i in (charge_unscaled_indices or [])))
    if charge_unscaled_indices and not set(charge_unscaled_indices) <= set(solute_indices):
        raise ValueError("charge_unscaled_indices must be a subset of solute_indices")

    temperatures = [float(t) for t in temperatures]
    if ref_temperature is None:
        ref_temperature = temperatures[0]
    ref_temperature = float(ref_temperature)
    if ref_temperature not in temperatures:
        raise ValueError(f"ref_temperature {ref_temperature} is not in the ladder {temperatures}")

    dt = dt_fs * unit.femtosecond
    steps_per_ps = 1000.0 / dt_fs
    tot_steps = int(round(time_ns * 1000.0 * steps_per_ps))
    exchange_steps = max(1, int(round(exchange_ps * steps_per_ps)))
    report_steps = exchange_steps if report_ps is None else max(1, int(round(report_ps * steps_per_ps)))
    dcd_steps = max(1, int(round(dcd_ps * steps_per_ps)))
    log_steps = report_steps if log_ps is None else max(1, int(round(log_ps * steps_per_ps)))

    pdb_like = types.SimpleNamespace(topology=topology, positions=positions)
    integrator = openmm.LangevinMiddleIntegrator(
        ref_temperature * unit.kelvin, friction_ps / unit.picosecond, dt
    )
    if seed is not None:
        integrator.setRandomNumberSeed(int(seed))
    platform_name = _platform_name(platform)

    rest2 = REST2(
        system=system,
        pdb=pdb_like,
        forcefield=None,
        solute_index=solute_indices,
        integrator=integrator,
        nonbonded_scale=nonbonded_scale,
        platform_name=platform_name,
        temperature=ref_temperature * unit.kelvin,
        pressure=None if not pressure_bar else pressure_bar * unit.bar,
        dt=dt,
        friction=friction_ps / unit.picosecond,
        nonbondedMethod=app.PME,
        exclude_Pro_omegas=exclude_pro_omegas,
        subsystem="copy",
        charge_unscaled_index=charge_unscaled_indices,
    )
    if box is not None:
        rest2.simulation.context.setPeriodicBoxVectors(*box)
    if velocities is not None:
        rest2.simulation.context.setVelocities(velocities)
    else:
        rest2.simulation.context.setVelocitiesToTemperature(
            ref_temperature * unit.kelvin, int(seed) if seed is not None else 0
        )

    rest2.simulation.reporters = []
    rest2.simulation.currentStep = 0
    sst2 = SST2(
        rest2,
        temperatures=temperatures,
        refTemperature=ref_temperature,
        weights=weights,
        tempChangeInterval=exchange_steps,
        reportInterval=report_steps,
        reportFile=f"{generic}_sst2_full.csv",
        seed=seed,
        move=move,
    )
    if restart_json is not None:
        with open(restart_json) as fh:
            restore_averages(sst2, json.load(fh))
        logger.info(f"- Restarted averages and rung {sst2.currentTemperature} from {restart_json}")

    logger.info(
        f"- SST2: {tot_steps} steps, exchange every {exchange_steps}, "
        f"report every {report_steps}, dcd every {dcd_steps}"
    )
    run_rest2(
        rest2,
        f"{generic}_sst2",
        tot_steps=tot_steps,
        dt=dt,
        save_step_dcd=dcd_steps,
        save_step_log=log_steps,
        save_step_rest2=report_steps,
        add_REST2_reporter=False,
        remove_reporters=False,
        overwrite=True,
    )
    # run_rest2 -> simulate() writes <generic>_sst2_final.xml at the end

    sidecar = sidecar_from_sst2(
        sst2,
        extra={
            "name": name,
            "system_xml": os.path.abspath(system_xml),
            "topology_pdb": os.path.abspath(topology_pdb),
            "state_xml": os.path.abspath(state_xml) if state_xml else None,
            "solute_atoms": len(solute_indices),
            "charge_unscaled_atoms": len(charge_unscaled_indices),
            "boundary_exceptions": len(getattr(rest2, "init_nb_exept_boundary_index", [])),
            "torsion_buckets": {
                f.getName(): f.getNumTorsions()
                for f in rest2.system.getForces()
                if isinstance(f, openmm.CustomTorsionForce)
            },
            "nonbonded_scale": bool(nonbonded_scale),
            "exclude_pro_omegas": bool(exclude_pro_omegas),
            "pressure_bar": pressure_bar,
            "dt_fs": dt_fs,
            "exchange_ps": exchange_ps,
            "time_ns": time_ns,
            "platform": platform_name,
            "seed": seed,
            "restart_json": os.path.abspath(restart_json) if restart_json else None,
            "final_state_xml": os.path.abspath(f"{generic}_sst2_final.xml"),
            "report_csv": os.path.abspath(f"{generic}_sst2_full.csv"),
            "trajectory_dcd": os.path.abspath(f"{generic}_sst2.dcd"),
            **(extra_provenance or {}),
        },
    )
    with open(f"{generic}_sst2_state.json", "w") as fh:
        json.dump(sidecar, fh, indent=2)
    return sidecar


def parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Run one SST2 walker from system.xml + topology.pdb (+ state.xml)."
    )
    p.add_argument("--system", required=True, dest="system_xml")
    p.add_argument("--topology", required=True, dest="topology_pdb")
    p.add_argument("--state", dest="state_xml", default=None)
    p.add_argument(
        "--solute-indices", required=True,
        help="JSON file holding a list of 0-based atom indices",
    )
    p.add_argument(
        "--temperatures", required=True,
        help="comma-separated rung temperatures in K, increasing",
    )
    p.add_argument("--ref-temp", type=float, default=None, dest="ref_temperature")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default="run")
    p.add_argument("--time-ns", type=float, default=1.0)
    p.add_argument("--dt-fs", type=float, default=2.0)
    p.add_argument("--exchange-ps", type=float, default=2.0)
    p.add_argument("--report-ps", type=float, default=None)
    p.add_argument("--dcd-ps", type=float, default=10.0)
    p.add_argument("--log-ps", type=float, default=None)
    p.add_argument("--platform", default="auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--move", choices=["gibbs", "neighbor"], default="gibbs")
    p.add_argument("--pressure-bar", type=float, default=1.0, help="0 for NVT")
    p.add_argument("--friction-ps", type=float, default=1.0)
    p.add_argument("--only-dihed", action="store_true", help="scale torsions only (gREST dihedral)")
    p.add_argument("--exclude-pro-omega", action="store_true")
    p.add_argument(
        "--charge-unscaled-indices", default=None,
        help="JSON list of solute atom indices whose charges are not scaled (LJ and torsions are)",
    )
    p.add_argument("--weights-json", default=None, help="JSON list of fixed rung free energies (kJ/mol)")
    p.add_argument("--restart-json", default=None, help="sidecar of a previous run of this walker")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    with open(args.solute_indices) as fh:
        solute = json.load(fh)
    charge_unscaled = None
    if args.charge_unscaled_indices:
        with open(args.charge_unscaled_indices) as fh:
            charge_unscaled = json.load(fh)
    weights = None
    if args.weights_json:
        with open(args.weights_json) as fh:
            weights = json.load(fh)
    sidecar = run_sst2_from_xml(
        system_xml=args.system_xml,
        topology_pdb=args.topology_pdb,
        state_xml=args.state_xml,
        solute_indices=solute,
        temperatures=[float(t) for t in args.temperatures.split(",")],
        ref_temperature=args.ref_temperature,
        out_dir=args.out_dir,
        name=args.name,
        time_ns=args.time_ns,
        dt_fs=args.dt_fs,
        exchange_ps=args.exchange_ps,
        report_ps=args.report_ps,
        dcd_ps=args.dcd_ps,
        log_ps=args.log_ps,
        platform=args.platform,
        seed=args.seed,
        move=args.move,
        pressure_bar=args.pressure_bar,
        friction_ps=args.friction_ps,
        nonbonded_scale=not args.only_dihed,
        exclude_pro_omegas=args.exclude_pro_omega,
        weights=weights,
        restart_json=args.restart_json,
        charge_unscaled_indices=charge_unscaled,
    )
    print(json.dumps({k: sidecar[k] for k in ("name", "rung", "step", "weights_kJ_per_mol", "e_num")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
