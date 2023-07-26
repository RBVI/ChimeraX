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

LINELEN = 50

def save(session, alignment, stream):
    print("CLUSTAL W ALN saved from UCSF ChimeraX", file=stream)
    print("", file=stream)

    max_name = max([len(seq.name) for seq in alignment.seqs])
    name_format = "%%-%ds" % (max_name+5)

    from chimerax.atomic import Sequence
    aln_len = len(alignment.seqs[0])
    for start in range(0, aln_len, LINELEN):
        end = min(aln_len, start + LINELEN)

        for seq in alignment.seqs:
            name = seq.name.replace(' ', '_')
            temp_seq = Sequence()
            temp_seq.extend(seq[start:end])
            if len(temp_seq.ungapped()) == 0:
                print(name_format % name, seq[start:end], file=stream)
            else:
                temp_seq = Sequence()
                temp_seq.extend(seq[:end])
                print(name_format % name, seq[start:end], len(temp_seq.ungapped()), file=stream)
        from .. import clustal_strong_groups, clustal_weak_groups
        conservation = []
        for pos in range(start, end):
            # completely conserved?
            first = alignment.seqs[0][pos].upper()
            if first.isupper():
                for seq in alignment.seqs[1:]:
                    if seq[pos].upper() != first:
                        break
                else:
                    # conserved
                    conservation.append('*')
                    continue

            # "strongly"/"weakly" conserved?
            conserved = False
            for groups, character in [(clustal_strong_groups, ':'), (clustal_weak_groups, '.')]:
                for group in groups:
                    for seq in alignment.seqs:
                        if seq[pos].upper() not in group:
                            break
                    else:
                        # conserved
                        conserved = True
                        break
                if conserved:
                    conservation.append(character)
                    break

            if not conserved:
                # remainder
                conservation.append(' ')
        print(name_format % " ", "".join(conservation), file=stream)
        print("", file=stream)

