import argparse
import os
import sys
import logging
import pandas as pd
from tqdm import tqdm

from openmm.app import PDBFile, PDBxFile
from openmm import LangevinMiddleIntegrator, unit, app

import pdb_numpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

import SST2
from SST2.rest2 import REST2, run_rest2
from SST2.sst2 import run_sst2
import SST2.tools as tools
import SST2.analysis.trajectory as trajectory

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add sys.sdout as handler
logger.addHandler(logging.StreamHandler(sys.stdout))


def parser_input():
    # Parse arguments :
    parser = argparse.ArgumentParser(
        description="Simulate a peptide starting from a linear conformation."
    )
    parser.add_argument(
        "-n",
        action="store",
        dest="name",
        help="Output file name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-dir",
        action="store",
        dest="out_dir",
        help="Output directory for intermediate files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-ref_temp",
        action="store",
        dest="ref_temp",
        help="Base temperature, default=300(K)",
        type=float,
        default=300,
    )
    parser.add_argument(
        "-hmr",
        action="store",
        dest="hmr",
        help="Hydrogen mass repartition, default=3.0 a.m.u.",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "-friction",
        action="store",
        dest="friction",
        help="Langevin Integrator friction coefficient default=1.0 (ps-1)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-exclude_Pro_omega",
        action="store_true",
        dest="exclude_Pro_omega",
        help="Exclude Proline omega dihedral scale angles",
    )
    parser.add_argument(
        "-select",
        action="store",
        dest="solute_sel",
        help='Solute selection, default="chain A"',
        default="chain A",
    )
    parser.add_argument(
        "-nonbonded_RF",
        action="store_true",
        dest="nonbonded_RF",
        help="Use Reaction Field to compute the electrostatic part of nonbonded interactions",
    )
    parser.add_argument(
        "-ff",
        action="store",
        dest="ff",
        help="force field, default=amber14",
        default="amber14sb",
    )
    parser.add_argument(
        "-water_ff",
        action="store",
        dest="water_ff",
        help="force field, default=tip3p",
        default="tip3p",
    )
    parser.add_argument(
        "-only_dihed",
        action="store_true",
        dest="only_dihed",
        help="As in gREST paper, only scale dihedral terms (no nonbonded scaling)"
    )
    parser.add_argument(
        "-v",
        action="store_true",
        dest="verbose",
        help="Verbose mode")

    return parser


if __name__ == "__main__":

    my_parser = parser_input()
    args = my_parser.parse_args()
    logger.info(args)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode activated")
        SST2.show_log()

    OUT_PATH = args.out_dir
    name = args.name

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    coor_file = os.path.join(OUT_PATH, name + "_equi_water.cif")

    if coor_file.endswith(".pdb"):
        input_coor = PDBFile(coor_file)
    elif coor_file.endswith(".cif"):
        input_coor = PDBxFile(coor_file)
    else:
        raise ValueError("Input file must be a PDB or CIF file.")
    
    forcefield = tools.get_forcefield(args.ff, args.water_ff)

    coor_init = pdb_numpy.Coor(coor_file)
    solute_indices = coor_init.get_index_select(args.solute_sel)
    all_indices = [int(i.index) for i in input_coor.topology.atoms()]
    assert len(solute_indices) > 0, "No solute atoms selected"

    logger.info(f"- Defining solute part containing {len(solute_indices)} atoms")
    logger.info(
        f"- Defining solvent part containing {len(all_indices) - len(solute_indices)} atoms"
    )

    if not args.nonbonded_RF:
        logger.info("Using PME for nonbonded interactions")
        nonbondedMethod = app.PME
    else:
        logger.info("Using CutoffPeriodic for nonbonded interactions")
        nonbondedMethod = app.CutoffPeriodic

    temperature = args.ref_temp * unit.kelvin
    dt = 4 * unit.femtosecond
    friction = args.friction / unit.picosecond

    system = tools.create_sim_system(
        input_coor,
        temp=temperature,
        forcefield=forcefield,
        h_mass=args.hmr,
        base_force_group=1,
    )

    integrator = LangevinMiddleIntegrator(temperature, friction, dt)

    sys_rest2 = REST2(
        system=system,
        pdb=input_coor,
        forcefield=forcefield,
        solute_index=solute_indices,
        integrator=integrator,
        dt=dt,
        temperature=temperature,
        exclude_Pro_omegas=args.exclude_Pro_omega,
        nonbondedMethod=nonbondedMethod,
        nonbonded_scale = not args.only_dihed,
    )

    sys_rest2.simulation.loadState(os.path.join(OUT_PATH, name + "_equi_water.xml"))

    forces_all = tools.get_forces(sys_rest2.system, sys_rest2.simulation)

    force_names = [forces_all[forces_num]['name'] for forces_num in forces_all]
    force_values = []
    for force_num, force in forces_all.items():
        logger.info(f"Force group {force['name']}: {force['energy']} kJ/mol/nm")
        force_values.append([])

    # Read trajectory
    md = trajectory.read_traj(
        input_coor,
        f'{OUT_PATH}/{name}_sst2')


    for frame in tqdm(md.trajectory, desc="Processing frames"):
        box_vector = frame.triclinic_dimensions
        tot_positions = frame.positions
        sys_rest2.simulation.context.setPeriodicBoxVectors(*box_vector)
        sys_rest2.simulation.context.setPositions(tot_positions)
        forces_all = tools.get_forces(sys_rest2.system, sys_rest2.simulation)
        for i, force in enumerate(forces_all):
            force_values[i].append(forces_all[force]['energy'].value_in_unit(unit.kilojoule_per_mole))

    data_dict = {}
    for i, ener_name in enumerate(force_names):
        data_dict[ener_name] = force_values[i]
    
    ener_df = pd.DataFrame(data_dict)
    time_ax_name = r"$Time\;(\mu s)$"
    ener_df[time_ax_name] = ener_df.index * dt.value_in_unit(unit.picoseconds) /100

    ener_df.to_csv(os.path.join(OUT_PATH, name + "_sst2_recomputed_energy.csv"), index=False)