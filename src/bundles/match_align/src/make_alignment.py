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

def match_to_align(session, chains, dist_cutoff, column_criteria, gap_char, circular, status_prefix=""):
    from chimerax.core.errors import UserError
    col_all = column_criteria == "all"
    if len(chains) == 2 and not circular:
        # Do pair alignment in Python layer since it's fast and needs Needleman-Wunsch, which
        # would require a lot of work to make callable from the C++ layer
        chain1, chain2 = chains
        chain_index = {}
        principal_atoms = []
        for i, r in enumerate(chain1.residues):
            if not r:
                continue
            pa = r.principal_atom
            if pa is None:
                session.logger.warning("Cannot determine principal atom for residue %s" % r)
                continue
            principal_atoms.append(pa)
            chain_index[pa] = i
        from chimerax.atom_search import AtomSearchTree
        tree = AtomSearchTree(principal_atoms, sep_val=dist_cutoff)

        # initialize score array
        from numpy import zeros
        scores = zeros((len(chain1), len(chain2)), float)
        scores -= 1.0

        # find matches and update score array
        from chimerax.geometry import distance
        for i2, r2 in enumerate(chain2.residues):
            if not r2:
                continue
            pa2 = r2.principal_atom
            if not pa2:
                session.logger.warning("Cannot determine principal atom for residue %s" % r2)
                continue
            matches = tree.search(pa2, dist_cutoff)
            for pa1 in matches:
                scores[chain_index[pa1]][i2] = dist_cutoff - distance(pa1.scene_coord, pa2.scene_coord)

        # use Needleman-Wunsch to establish alignment
        from chimerax.alignment_algs import NeedlemanWunsch
        score, seqs = NeedlemanWunsch.nw(chain1, chain2, score_matrix=scores, gap_char=gap_char, score_gap=0,
            score_gap_open=0, return_seqs=True)
        smallest = min(len(chain1), len(chain2))
        min_dots = max(len(chain1), len(chain2)) - smallest
        extra_dots = len(seqs[0]) - smallest - min_dots
        num_matches = smallest - extra_dots
        session.logger.status("%s%d residue pairs aligned" % (status_prefix, num_matches), log=True)

        if num_matches == 0:
            raise UserError("Cannot generate alignment because no residues within cutoff distance")
        return seqs

    from ._msa3d import multi_align
    return multi_align([int(x) for x in chains.pointers], dist_cutoff, col_all, gap_char, circular,
        "", session.logger, UserError)
