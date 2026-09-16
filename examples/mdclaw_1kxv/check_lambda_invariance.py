"""lambda-invariance / closure check of REST2(subsystem='copy') on the real 1KXV MDClaw system."""
import json, sys, time, types, numpy as np, openmm, openmm.app as app
from openmm import unit
from SST2.rest2 import REST2
from SST2 import driver
import logging; logging.basicConfig(level=logging.WARNING)
KJ=unit.kilojoules_per_mole
A="/data1/rkp00079/rku00161/runs/glm-5.3-flash-skill-full/attempts/031_nanobody_1kxv/cli_skill_sif__pi__rikyu-glm-5.3-flash__r1/workspace/study/jobs/main/nodes"
sel=sys.argv[1]; platform=sys.argv[2] if len(sys.argv)>2 else "CPU"
system, top, pos, vel, box = driver.load_xml_inputs(f"{A}/topo_001/artifacts/system.system.xml", f"{A}/topo_001/artifacts/system.topology.pdb", f"{A}/prod_001/artifacts/state.xml")
sol=json.load(open(f"/tmp/claude-100160/-home-rku00161-mdclaw/f15e81bf-05e9-4ede-b94f-ee099a549acd/scratchpad/kxv/{sel}.json"))
t0=time.time()
r=REST2(system=system, pdb=types.SimpleNamespace(topology=top, positions=pos), forcefield=None, solute_index=sol,
        integrator=openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 4*unit.femtosecond),
        platform_name=platform, temperature=300*unit.kelvin, pressure=None, dt=4*unit.femtosecond, subsystem="copy")
r.simulation.context.setPeriodicBoxVectors(*box)
print(f"[{sel}] {len(sol)} solute atoms, boundary exceptions {len(r.init_nb_exept_boundary_index)}, REST2 build {time.time()-t0:.1f}s")
print("  buckets:", {f.getName(): f.getNumTorsions() for f in r.system.getForces() if isinstance(f, openmm.CustomTorsionForce)})
E0=r.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
ref=None
for lam in (1.0, 0.5, 0.25, 1.0):
    r.scale_nonbonded_torsion(lam)
    E_frac,E_pp2,E_ww,E_pw=r.compute_all_energies()
    E_state=r.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    rebuilt=(E_pp2+E_ww+np.sqrt(lam)*E_pw).value_in_unit(KJ)+sum((lam**f)*E.value_in_unit(KJ) for f,E in E_frac.items())
    row={"lam":lam,"E_pw":E_pw.value_in_unit(KJ),"E_pp2":E_pp2.value_in_unit(KJ),**{f"E{f}":E.value_in_unit(KJ) for f,E in E_frac.items()},"E_state":E_state,"rebuilt":rebuilt}
    if ref is None: ref=dict(row); ref["offset"]=E_state-rebuilt
    drift=max(abs(row[k]-ref[k])/max(1e-9,abs(ref[k])) for k in row if k.startswith("E") and k not in ("E_state",))
    print(f"  lambda={lam:5.2f} E_pw={row['E_pw']:11.3f} E_pp2={row['E_pp2']:9.3f} " + " ".join(f"{k}={row[k]:8.3f}" for k in row if k[1:2].isdigit()) + f" | max rel drift of unscaled terms {drift:.2e} | closure {(rebuilt+ref['offset'])/E_state-1:+.2e}")
print(f"  round trip: E(lambda=1 again) - E0 = {r.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)-E0:+.4f} kJ/mol")
