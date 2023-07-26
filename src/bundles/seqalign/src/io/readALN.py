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

"""
reads a ClustalW ALN format file
"""

from chimerax.atomic import Sequence
from ..parse import FormatSyntaxError, make_readable

def read(session, f):
    in_header = True
    sequences = []
    line_num = 0
    for line in f.readlines():
        line_num += 1
        if in_header:
            if line.startswith("CLUSTAL"):
                in_header = False
                first_block = True
            else:
                if line.strip() != "":
                    raise FormatSyntaxError("First non-blank line does not start with 'CLUSTAL'")
            continue
        if not line or line[0].isspace():
            if sequences:
                first_block = False
                expect = 0
            continue
        try:
            seq_name, seq_block, num_residues = line.split()
        except ValueError:
            try:
                seq_name, seq_block = line.strip().split()
            except ValueError:
                raise FormatSyntaxError("Line %d is not sequence name followed by sequence "
                    "contents and optional ungapped length" % line_num)
        if first_block:
            sequences.append(Sequence(name=make_readable(seq_name)))
            sequences[-1].append(seq_block)
            continue
        try:
            seq = sequences[expect]
        except IndexError:
            raise FormatSyntaxError("Sequence on line %d not in initial sequence block" % line_num)
        expect += 1
        seq.append(seq_block)
    f.close()
    return sequences, {}, {}
