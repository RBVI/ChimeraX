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
reads a PIR file
"""

from chimerax.atomic import Sequence
from ..parse import FormatSyntaxError, make_readable

def read(session, f):
    want = 'init'
    sequences = []
    for line in f.readlines():
        line = line.strip()
        if want == 'init':
            if len(line) < 4:
                continue
            if line[0] != '>' or line[3] != ';':
                continue
            sequences.append(Sequence(name=make_readable(line[4:])))
            pir_type = line[1:3]
            if pir_type in ("P1", "F1"):
                sequences[-1].nucleic = True
            else:
                sequences[-1].nucleic = False
            sequences[-1].pir_type = pir_type
            want = 'description'
        elif want == 'description':
            sequences[-1].description = line
            sequences[-1].pir_description = line
            want = 'sequence'
        elif want == 'sequence':
            if not line:
                continue
            if line[-1] == '*':
                want = 'init'
                line = line[:-1]
            sequences[-1].extend("".join([c for c in line if not c.isspace()]))
    f.close()
    if want != 'init':
        raise FormatSyntaxError("Could not find end of sequence '%s'" % sequences[-1].name)
    return sequences, {}, {}
