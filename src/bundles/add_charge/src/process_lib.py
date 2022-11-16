# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

#
# include pre-parameterized/supplement.lib to get ADP, etc. charges
#

# As per the AmberTools 20 docs, recommended assignments are:
#
# protein: ff14SB
#   amino12.lib aminoct12.lib aminont12.lib
# DNA: OL15
#   DNA.OL15.lib
# RNA: OL3
#   RNA.lib
# lipid: lipid17
#   lipid17.lib
# water: tip3p
#   (atomic_ions.lib) solvents.lib; HOH,WAT = TP3
#
import sys
resname_remap = { 'NHE': 'NH2' }
for na in "ACGU":
    for suffix in ('', '3', '5', 'N'):
        resname_remap[na+suffix] = "R"+na+suffix
heavy_synonyms = {'op1': ['o1p'], 'op2': ['o2p'], 'op3': ['o3p'], 'c7': ['c5m']}
cterm_heavy_synonyms = {'o': ['ot1', '1oct'], 'oxt': ['ot2', '2oct']}
def get_heavy_synonyms(resid, heavy):
    if resid.endswith('MET') and heavy == 'sd':
        # so MSE also works
        return ['se']
    if len(resid) == 4 and resid[0] == 'C':
        return cterm_heavy_synonyms.get(heavy, [])
    return heavy_synonyms.get(heavy, [])

heavy_data = {}
hyd_data = {}
for lib in sys.argv[1:]:
    f = open(lib, "r")
    seen = set()
    state = "preamble"
    waiting = None
    for line in f:
        if line.startswith("!entry."):
            entry_type = line.split()[0]
            entry_info = entry_type.split('.')
            if entry_info[-1] == "atoms":
                resid = entry_info[1]
                resid = resname_remap.get(resid, resid)
                heavy = None
                state = "processing"
            else:
                state = "other entry"
            continue
        if state != "processing":
            continue
        if len(resid) > 4:
            continue
        fields = line.split()
        atom_name, atom_type = [eval(quoted) for quoted in fields[:2]]
        charge = float(fields[-1])
        atom_name = atom_name.replace('*', "'").lower()
        resids = [resid]
        final_resids = [resid]
        if resid == "TP3":
            final_resids.extend(["HOH", "WAT"])
        for final_resid in final_resids:
            if atom_name[0] == 'h':
                if heavy is None:
                    if waiting is not None:
                        raise AssertionError("Hydrogen not followed by heavy atom")
                    waiting = (final_resids, charge, atom_type)
                else:
                    for name in [heavy] + get_heavy_synonyms(final_resid, heavy):
                        hyd_data[(final_resid, name)] = (charge, atom_type)
            else:
                for name in [atom_name] + get_heavy_synonyms(final_resid, atom_name):
                    heavy_data[(final_resid, name)] = (charge, atom_type)
                heavy = atom_name
                if waiting:
                    wait_finals, wait_charge, wait_type = waiting
                    for wf in wait_finals:
                        for wname in [heavy] + get_heavy_synonyms(final_resid, heavy):
                            hyd_data[(wf, wname)] = (wait_charge, wait_type)
                    waiting = None
from pprint import pformat
print("heavy_charge_type_data =", pformat(heavy_data, indent=2))
print("hyd_charge_type_data =", pformat(hyd_data, indent=2))
