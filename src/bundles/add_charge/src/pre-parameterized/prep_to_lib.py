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
# generates a .lib file usable by process_lib.py
#
#

import os
with open("supplement.lib", 'w') as out:
    for fname in os.listdir('data'):
        if not fname.endswith('.prep'):
            continue
        res_name = fname[:-5]
        state = "pre OMIT"
        with open('data' + os.sep + fname, 'r') as prep:
            print("!entry.%s.unit.atoms" % res_name, file=out)
            for line in prep:
                if state == "pre OMIT":
                    if "OMIT" not in line:
                        continue
                    state = "pre atoms"
                    continue
                if state == "pre atoms":
                    if not line.strip().startswith("1"):
                        continue
                    state = "atoms"
                line = line.strip()
                if not line:
                    break
                _1, name, gaff, _3, _4, _5, _6, _7, _8, _9, charge = line.split()
                if name == "DUMM":
                    continue
                name = name.replace('*', "'")
                name = {
                    "C'N1": "C1D", "C'N2": "C2D", "O'N2": "O2D", "C'N3": "C3D", "O'N3": "O3D", "C'N4": "C4D",
                    "O'N4": "O4D", "C'N5": "C5D", "O'N5": "O5D", "OPN1": "O1N", "OPN2": "O2N", "O3P": "O3",
                    "OPA1": "O1A", "OPA2": "O2A", "O'A5": "O5B", "C'A5": "C5B", "C'A4": "C4B", "O'A4": "O4B",
                    "C'A3": "C3B", "O'A3": "O3B", "C'A2": "C2B", "O'A2": "O2B", "C'A1": "C1B", "P'A2": "P2B",
                    "OA22": "O1X", "OA23": "O2X", "OA24": "O3X",
                }.get(name, name)
                print("%s %s 0 0 1 2 -1 %s" % (repr(name), repr(gaff), charge), file=out)
