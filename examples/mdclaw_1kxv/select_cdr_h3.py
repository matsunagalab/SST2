import json, re, numpy as np, openmm, openmm.app as app
from openmm import unit
A="/data1/rkp00079/rku00161/runs/glm-5.3-flash-skill-full/attempts/031_nanobody_1kxv/cli_skill_sif__pi__rikyu-glm-5.3-flash__r1/workspace/study/jobs/main/nodes"
sys_xml=f"{A}/topo_001/artifacts/system.system.xml"; top_pdb=f"{A}/topo_001/artifacts/system.topology.pdb"; state_xml=f"{A}/prod_001/artifacts/state.xml"
system=openmm.XmlSerializer.deserialize(open(sys_xml).read())
pdb=app.PDBFile(top_pdb)
print("particles", system.getNumParticles(), "atoms", pdb.topology.getNumAtoms())
print("forces", [ (f.getName(), type(f).__name__) for f in system.getForces()])
hm=[system.getParticleMass(a.index).value_in_unit(unit.amu) for a in pdb.topology.atoms() if a.element and a.element.symbol=="H"][:1]
print("H mass", hm, "box", system.getDefaultPeriodicBoxVectors()[0])
st=openmm.XmlSerializer.deserialize(open(state_xml).read())
print("state particles", len(st.getPositions()), "box", st.getPeriodicBoxVectors()[0], "time", st.getTime())
three={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','CYX':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','HIE':'H','HID':'H','HIP':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
for ch in pdb.topology.chains():
    res=[r for r in ch.residues() if r.name in three]
    if len(res)<50: continue
    seq="".join(three[r.name] for r in res)
    print("chain", ch.id, "nres", len(res), "first id", res[0].id)
    print(seq)
    # H3: after the 'C' of the Cys92 (motif C[AT][AR]?) ... to before W of WG.G
    m=re.search(r'WG.G', seq)
    # find last Cys before WGxG
    # Kabat Cys92: the first Cys of the C-terminal part of the domain (the 'YYC' motif);
    # a VHH may carry an extra Cys inside H3 (H3-CDR1 disulfide), so do not take the last one
    cys=[i for i,c in enumerate(seq[:m.start()]) if c=='C' and i>len(seq)*0.6][0]
    h3_start=cys+3          # Kabat 95 (C92, A93, R94)
    h3_end=m.start()-1      # Kabat 102
    h3=res[h3_start:h3_end+1]
    print("Cys92 at", res[cys].id, "WGxG at", res[m.start()].id, "-> H3 residues", [(r.name, r.id) for r in h3])
    idx=[a.index for r in h3 for a in r.atoms()]
    nb=[f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    q=sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in idx)
    print("H3 atoms", len(idx), "net charge", round(q,3), "total system charge", round(sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in range(nb.getNumParticles())),3))
    json.dump(idx, open("/tmp/claude-100160/-home-rku00161-mdclaw/f15e81bf-05e9-4ede-b94f-ee099a549acd/scratchpad/kxv/solute_h3.json","w"))
    # H3 + shell: residues with any heavy atom within 5 A of an H3 heavy atom
    pos=np.array(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    h3_heavy=[a.index for r in h3 for a in r.atoms() if a.element.symbol!="H"]
    shell=[]
    for r in res:
        if r in h3: continue
        heavy=[a.index for a in r.atoms() if a.element.symbol!="H"]
        d=np.min(np.linalg.norm(pos[heavy][:,None,:]-pos[h3_heavy][None,:,:],axis=2))
        if d<5.0: shell.append(r)
    print("shell residues (<5 A):", [(r.name,r.id) for r in shell])
    ss_partners=[]
    for b in pdb.topology.bonds():
        if b[0].element.symbol=="S" and b[1].element.symbol=="S":
            for x,y in ((b[0],b[1]),(b[1],b[0])):
                if x.residue in h3 and y.residue not in h3 and y.residue not in shell:
                    ss_partners.append(y.residue)
    print("disulfide partners added:", [(r.name,r.id) for r in ss_partners])
    shell=shell+ss_partners
    idx2=sorted(idx+[a.index for r in shell for a in r.atoms()])
    q2=sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in idx2)
    print("H3+shell atoms", len(idx2), "net charge", round(q2,3))
    json.dump(idx2, open("/tmp/claude-100160/-home-rku00161-mdclaw/f15e81bf-05e9-4ede-b94f-ee099a549acd/scratchpad/kxv/solute_h3_shell.json","w"))
    # disulfides touching H3?
    for b in pdb.topology.bonds():
        if b[0].element.symbol=="S" and b[1].element.symbol=="S":
            print("SS bond:", b[0].residue.name, b[0].residue.id, "-", b[1].residue.name, b[1].residue.id)
