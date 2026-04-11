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

from .settings import defaults
from chimerax.core.errors import UserError

def make_alignment(session, chains, *, circular=defaults['circular'],
        column_criteria=defaults["column_criteria"], dist_cutoff=defaults["dist_cutoff"],
        gap_char=defaults["gap_char"], ident=None, iteration_limit=defaults['iteration_limit'],
        min_stretch=defaults['min_stretch'], ref_chain=None, show_alignment=True):
    if len(chains) < 2:
        raise UserError("Must specifiy at least two chains as basis for alignment")
    if ref_chain is None:
        ref_chain = chains[0]
    elif ref_chain not in chains:
        raise UserError("Reference chain must be involved in alignment")
    if len(chains.structures.unique()) != len(chains):
        raise UserError("Specify only one chain per model")

    cutoff_fmt = "%.1f" if int(dist_cutoff) == dist_cutoff else "%g"
    cutoff_text = cutoff_fmt % dist_cutoff
    session.logger.info("Match\N{RIGHTWARDS ARROW}Align cutoff: %s, in column if within cutoff of: %s"
        % (cutoff_text, column_criteria))

    from .make_alignment import match_to_align
    # C++ layer cannot instantiate StructureSeqs, so send in copies that will be modified
    from copy import copy
    ordered = sorted(chains, key=lambda seq: seq.structure.id)
    aligned = [copy(chain) for chain in ordered]
    for aseq in aligned:
        aseq.name = aseq.structure.name + " chain " + aseq.chain_id
    match_to_align(session, aligned, dist_cutoff, column_criteria, gap_char, circular)
    full_cols = fully_populated(aligned)
    session.logger.info("%d fully populated columns" % len(full_cols))
    if ident is None:
        id_num = 1
        while True:
            ident = "MA-%d" % id_num
            for aln in session.alignments.alignments:
                if aln.ident == ident:
                    break
            else:
                # no conflicts
                break
            id_num += 1
    # iteration limt of None means until convergence
    if iteration_limit != 0:
        best = full_cols
        iteration = 1
        if not ref_chain:
            ref_chain = aligned[0]
        from chimerax.std_commands import align
        while True:
            ref_seq = [s for s in aligned if s.structure == ref_chain.structure][0]
            # cull columns based on stretch-length criteria
            stretch = []
            culled = []
            for col in full_cols:
                if not stretch or stretch[-1]+1 == col:
                    stretch.append(col)
                    continue
                if len(stretch) >= min_stretch:
                    culled.extend(stretch)
                stretch = [col]
            if len(stretch) >= min_stretch:
                culled.extend(stretch)
            if min_stretch > 1:
                session.logger.info("%d fully populated columns in at least %d column stetches"
                    % (len(culled), min_stretch))
            if len(culled) < 3:
                session.logger.info("Fewer than 3 fully populated columns; stopping iteration")
                break

            # match
            ref_atoms = column_atoms(ref_seq, culled)
            for seq in aligned:
                if seq.structure == ref_seq.structure:
                    continue
                seq_atoms = column_atoms(seq, culled)
                session.logger.info("Matching %s onto %s" % (seq.name, ref_seq.name))
                prev_aligned = aligned
                aligned = [copy(chain) for chain in ordered]
                for prev, cur in zip(prev_aligned, aligned):
                    cur.name = prev.name
                align.align(session, seq_atoms, ref_atoms)
            match_to_align(session, aligned, dist_cutoff, column_criteria, gap_char, circular)
            full_cols = fully_populated(aligned)
            session.logger.info("Iteration %d: %d fully populated columns" % (iteration, len(full_cols)))
            if len(full_cols) > len(best):
                best = full_cols
            else:
                break
            if iteration_limit and iteration >= iteration_limit:
                break
            iteration += 1

    if full_cols:
        # show pairwise RMSD matrix in fully populated columns
        coords = {}
        for seq in aligned:
            coords[seq] = column_atoms(seq, full_cols).scene_coords
        dsq_sum = 0
        rmsd_matrix = {}
        from math import sqrt
        import numpy
        for i, s1 in enumerate(aligned):
            rmsd_matrix[(s1, s1)] = 0.0
            for s2 in aligned[i+1:]:
                diff = coords[s1] - coords[s2]
                v = sqrt(numpy.sum(diff * diff) / len(coords[s1]))
                dsq_sum += v * v
                rmsd_matrix[(s1,s2)] = rmsd_matrix[(s2,s1)] = v
        overall_rmsd = sqrt(2 * dsq_sum / (len(aligned) * (len(aligned)-1)))
        session.logger.info("Overall RMSD: %.3f" % overall_rmsd)
        table_texts = []
        from chimerax.core.logger import html_table_params
        table_texts.append('<table %s>' % html_table_params)
        table_texts.append(' <thead>')
        table_texts.append('  <tr>')
        table_texts.append('   <th colspan="%d">Pairwise RMSDs across all fully populated columns</th>'
            % (len(aligned)+1))
        table_texts.append('  </tr>')
        table_texts.append(' </thead>')
        table_texts.append(' <tbody>')
        table_texts.append('  <tr>')
        table_texts.append('   ' + ' '.join(['<td style="text-align:center">%s</td>' % item
            for item in (["Model"]+[seq.structure.id_string for seq in aligned])]))
        table_texts.append('  </tr>')
        for s1 in aligned:
            table_texts.append('  <tr>')
            table_texts.append('   ' + ' '.join(['<td style="text-align:center">%s</td>' % item
                for item in ([s1.structure.id_string]+["%.3f" % rmsd_matrix[(s1,s2)] for s2 in aligned])]))
            table_texts.append('  </tr>')
        table_texts.append(' </tbody>')
        table_texts.append('</table>')
        session.logger.info('\n'.join(table_texts), is_html=True)

    #TODO: lots
    alignment = session.alignments.new_alignment(aligned, ident,
        name="Match\N{RIGHTWARDS ARROW}Align", auto_associate=False, viewer=show_alignment)
    for orig, aligned in zip(ordered, aligned):
        alignment.associate(orig, seq=aligned, silent=True)
    for hdr in alignment.headers:
        if hdr.ident == "rmsd":
            hdr.shown = True
            break
    for viewer in alignment.viewers:
        if hasattr(viewer, 'new_region'):
            if full_cols:
                viewer.new_region(columns=full_cols, region_type="matched")
            else:
                viewer.status("No fully populated columns in alignment", color="blue")
    return alignment

def column_atoms(seq, columns):
    seq_columns = [seq.gapped_to_ungapped(i) for i in columns]
    num_residues = seq.num_residues
    residues = seq.residues
    from chimerax.atomic import Atoms
    if seq.circular:
        return Atoms([r.principal_atom for r in [residues[i % num_residues] for i in seq_columns]])
    return Atoms([r.principal_atom for r in [residues[i] for i in seq_columns]])

def fully_populated(seqs):
    full_cols = []
    seq_chars = [seq.characters for seq in seqs]
    for col in range(len(seq_chars[0])):
        for chars in seq_chars:
            if not chars[col].isalpha():
                # not fully populated
                break
        else:
            # fully populated
            full_cols.append(col)
    return full_cols

def register_command(cmd_name, logger):
    from chimerax.core.commands import CmdDesc, register, NonNegativeFloatArg, EnumOf, CharacterArg, \
        BoolArg, Or, NoneArg, NonNegativeIntArg, PositiveIntArg, StringArg
    from chimerax.atomic import UniqueChainsArg, ChainArg
    desc = CmdDesc(
        required = [('chains', UniqueChainsArg)],
        keyword = [
            ('circular', BoolArg),
            ('column_criteria', EnumOf(['any', 'all'])),
            ('dist_cutoff', NonNegativeFloatArg),
            ('gap_char', CharacterArg),
            ('ident', StringArg),
            ('iteration_limit', Or(NonNegativeIntArg, NoneArg)),
            ('min_stretch', PositiveIntArg),
            ('ref_chain', ChainArg),
            ('show_alignment', BoolArg),
        ],
        synopsis = 'Create sequence alignment from structural superposition'
    )
    register(cmd_name, desc, make_alignment, logger=logger)
