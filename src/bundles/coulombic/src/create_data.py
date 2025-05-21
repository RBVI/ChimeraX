# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# args are expected to be a series of parameter files, oldest to newest, so that newer ones overwrite the
# old but residues missing from newer ones are still found.  The last 2 parameter files should be for
# N-terminal amino acids and C-terminal amino acids, respectively.
#
# For AmberTools20, the files would be: all_amino03.in, amino12.in, nucleic10.in, aminont12.in,
#       and aminoct12.in.

starting_residues = {}
ending_residues = {}
other_residues = {}

import sys
for file_name in sys.argv[1:]:
    if file_name == sys.argv[-2]:
        default_residues = starting_residues
    elif file_name == sys.argv[-1]:
        default_residues = ending_residues
    else:
        default_residues = other_residues
    with open(file_name, 'rt') as f:
        res_name = None
        for line in f:
            fields = line.strip().split()
            if res_name is None:
                if len(fields) == 3 and fields[1] == "INT":
                    res_name = fields[0]
                    if res_name[-1] == '5':
                        residues = starting_residues
                        res_name = res_name[:-1]
                    elif res_name[-1] == '3':
                        residues = ending_residues
                        res_name = res_name[:-1]
                    elif res_name == "ACE":
                        residues = starting_residues
                    elif res_name == "NHE":
                        residues = ending_residues
                        res_name = "NH2"
                    else:
                        residues = default_residues
                    residues[res_name] = res_dict = {}
            else:
                if len(fields) == 11:
                    if fields[2] != "DU":
                        res_dict[fields[1]] = eval(fields[-1])
                elif len(fields) == 1 and fields[0] == "DONE":
                    res_name = None

from pprint import pformat
with open("data.py", "wt") as f:
    print("starting_residues =", pformat(starting_residues, indent=4), file=f)
    print("ending_residues =", pformat(ending_residues, indent=4), file=f)
    print("other_residues =", pformat(other_residues, indent=4), file=f)

print("%d known starting residues:" % (len(starting_residues)), sorted(list(starting_residues.keys())))
print("%d known ending residues:" % (len(ending_residues)), sorted(list(ending_residues.keys())))
print("%d known other residues:" % (len(other_residues)), sorted(list(other_residues.keys())))
