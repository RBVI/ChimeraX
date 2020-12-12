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
# will likely have to add HOH and ADP, etc. charges afterward
#
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
rna_remap = {}
for na in "ACGU":
    for suffix in ('', '3', '5', 'N'):
        rna_remap[na+suffix] = "R"+na+suffix
heavy_synonyms = {'op1': ['o1p'], 'op2': ['o2p'], 'op3': ['o3p'],
    'c7': ['c5m'], 'o': ['ot1'], 'oxt': ['ot2']}
heavy_data = {}
hyd_data = {}
for lib in sys.argv[1:]:
    f = open(lib, "r")
    seen = set()
    state = "preamble"
    for line in f:
        if line.startswith("!entry."):
            entry_type = line.split()[0]
            entry_info = entry_type.split('.')
            if entry_info[-1] == "atoms":
                resid = entry_info[1]
                resid = rna_remap.get(resid, resid)
                heavy = None
                state = "processing"
            else:
                state = "other entry"
            continue
        if state != "processing":
            continue
        fields = line.split()
        atom_name, atom_type = [eval(quoted) for quoted in fields[:2]]
        charge = fields[-1]
        atom_name = atom_name.replace('*', "'").lower()
        resids = [resid]
        if resid == "TP3":
            resids.extend(["HOH", "WAT"])
        for resid in resids:
            if atom_name[0] == 'h':
                hyd_data[(resid, heavy)] = (charge, atom_type)
            else:
                for name in [atom_name] + heavy_synonyms.get(atom_name, []):
                    heavy_data[(resid, name)] = (charge, atom_type)
                heavy = atom_name
from pprint import pformat
print("heavy_charge_data =", pformat(heavy_data, indent=2))
print("hyd_charge_data =", pformat(hyd_data, indent=2))
