# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
writes a PIR file
"""

LINELEN = 60

def save(session, alignment, stream):
    nucleic = getattr(alignment.seqs[0], "nucleic", None)
    if nucleic is None:
        nucleic = True
        for res in alignment.seqs[0]:
            if res.isalpha() and res not in "ACGTUXacgtux":
                nucleic = False
                break

    for seq in alignment.seqs:
        pir_type = getattr(seq, "pir_type", None)
        if pir_type is None:
            if nucleic:
                pir_type = "DL"
            else:
                pir_type = "P1"
        print(">%2s;%s" % (pir_type, seq.name), file=stream)
        description = getattr(seq, "description", seq.name)
        print(description, file=stream)
        for i in range(0, len(seq), LINELEN):
            print(seq[i:i+LINELEN], file=stream)
        print("*", file=stream)
