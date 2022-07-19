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
