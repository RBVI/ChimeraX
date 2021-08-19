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

"""Process Chimera fragment definition files into form usable by ChimeraX"""

frag_src_dir = "/Users/pett/src/chimera/libs/BuildStructure/fragments"

cat_lookup = {
    "RING5": "5-member rings",
    "RING6": "6-member rings",
    "RING65": "fused 6+5 member rings",
    "RING66": "fused 6+6 member rings",
    "RING665": "fused 6+6+5 member rings",
    "RING666": "fused 6+6+6 member rings",
}

import os
for frag_file in sorted(os.listdir(frag_src_dir)):
    if not frag_file.endswith(".py") or frag_file.startswith("mk") or frag_file == "__init__.py":
        continue
    with open("fragments/" + frag_file, "w") as outf:
        print("name = %s" % repr(frag_file[:-3]), file=outf)
        with open(frag_src_dir + '/' + frag_file, 'r') as inf:
            state = "category"
            for line in inf:
                if state == "category":
                    category = cat_lookup[line.split()[-1]]
                    print('<Provider name="%s" category="%s"/>' % (frag_file[:-3], category))
                    print("category = %s" % repr(category), file=outf)
                    state = "class"
                    print("atoms = [", file=outf)
                    continue
                if state == "class":
                    state = "atoms"
                    continue
                if state == "atoms":
                    if line.strip()[0] == ']':
                        state = "bonds"
                        print("]\nbonds = [", file=outf)
                        continue
                    print(line.rstrip(), file=outf)
                if state == "bonds":
                    if line.strip()[0] == ']':
                        print("]", file=outf)
                        break
                    print(line.rstrip(), file=outf)
