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
writes a FASTA file
"""

LINELEN = 60

def save(session, alignment, stream):
    for seq in alignment.seqs:
        print(">%s" % seq.name, file=stream)
        for i in range(0, len(seq), LINELEN):
            print(seq[i:i+LINELEN], file=stream)
        print("", file=stream)
