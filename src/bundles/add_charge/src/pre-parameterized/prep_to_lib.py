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
    if res_name not in ['ADP']:
        continue
    state = "pre OMIT"
    with open('data' + os.sep + fname, 'r') as prep:
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
            name = name.replace('*', "'").lower()
            name = {
                
            }.get(name, name)
