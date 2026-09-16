# SST2 on an MDClaw-built VHH (1KXV)

Scripts used on 2026-09-16 to validate `REST2(subsystem="copy")` and
`bin/sst2_from_xml.py` on a System built by MDClaw (`build_amber_system`,
HMR, 53,716 atoms).  Paths point at a campaign run on the RIKYU host; edit
them for another system.

* `select_cdr_h3.py`: finds CDR-H3 from the sequence (Kabat Cys92 = first
  Cys of the C-terminal part of the domain, WGxG motif) and writes two
  solute selections: H3 only and H3 + residues within 5 A + disulfide
  partners.  Do not take the last Cys before WGxG: VHHs may carry an
  H3-to-CDR1 disulfide (1KXV: Cys106-Cys30).
* `check_lambda_invariance.py <selection> [platform]`: builds REST2 on the
  XML triple and checks that the unscaled terms of
  `compute_all_energies()` do not depend on lambda, that the rebuilt
  potential closes on the Context energy, and that lambda=1 is restored.
* `solute_h3.json`, `solute_h3_shell.json`: the selections (0-based atom
  indices) for the 1KXV topology.

Driver run used for the GPU smoke test:

```bash
python bin/sst2_from_xml.py --system system.system.xml --topology system.topology.pdb \
  --state state.xml --solute-indices solute_h3.json --temperatures 300,357,424,505,600 \
  --out-dir run_h3 --name kxv_h3 --time-ns 0.02 --dt-fs 4 --exchange-ps 2 --dcd-ps 10 \
  --platform CUDA --seed 1 -v
```
