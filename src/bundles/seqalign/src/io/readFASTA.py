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

"""
reads an aligned FASTA file
"""

def read(session, f):
    from chimerax.atomic import Sequence
    from ..parse import FormatSyntaxError, make_readable
    in_sequence = False
    sequences = []
    for line in f.readlines():
        if in_sequence:
            if not line or line.isspace():
                in_sequence = False
                continue
            if line[0] == '>':
                in_sequence = False
                # fall through
            else:
                sequences[-1].extend(line.strip())
        if not in_sequence:
            if line[0] == '>':
                if sequences and len(sequences[-1]) == 0:
                    raise FormatSyntaxError("No sequence found for %s"
                        % sequences[-1].name)
                in_sequence = True
                sequences.append(Sequence(name=make_readable(line[1:])))
    return sequences, {}, {}
