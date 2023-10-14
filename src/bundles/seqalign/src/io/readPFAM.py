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
    # skip header crap
    in_header = True
    line_num = 0
    sequences = []
    for line in f.readlines():
        line = line.strip()
        line_num += 1
        if not line:
            continue
        fields = line.split()
        if in_header:
            if len(fields[0]) == 2:
                continue
            if fields[0].startswith('#='):
                # some Pfam seed alignments have undocumented #=RF header
                continue
            in_header = False
        if len(fields) != 2:
            raise FormatSyntaxError(
                "Sequence line %d not of form 'seq-name seq-letters'" % line_num)
        seq = Sequence(name=make_readable(fields[0]))
        seq.extend(fields[1])
        sequences.append(seq)
    f.close()
    return sequences, {}, {}
