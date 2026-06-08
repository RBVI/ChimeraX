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

max_iterations_default = 0 if not defaults['iterate'] else defaults['max_iterations']

def make_alignment(session, chains, *, circular=defaults['circular'],
        column_criterion=defaults["column_criterion"], cutoff_distance=defaults["cutoff_distance"],
        gap_char=defaults["gap_char"], alignment_id=None, max_iterations=max_iterations_default,
        min_stretch=defaults['min_stretch'], ref_chain=None, show_alignment=True):
    """Create a sequence alignment based on a 3D structure superposition

       Returns the alignment, the overall RMSD, the Structural Distance Measure score (DOI:
       10.1007/BF02102452), and the Q-score (DOI: 10.1107/S0907444904026460).  The last three values
       only apply to fully populated columns, and will be None if there are no fully populated columns.

       'chains' are the chains that should compose the alignment.

       'circular' is whether to consider circular permutations of the chains.

       'column_criterion' should be "all" or "any".  "any" means that residues will be put in the same
       column is they are within 'cutoff_distance' of any other residue in that column.  "all" means
       that residues in the same column must be within 'cutoff_distance' of all other residues in that
       column.

       'gap_char' is the character to use to fill in gaps in the computed alignment.

       'alignment_id' is the identifier to give to the computed alignment (for use in sequence-related
       commands). If None, use a generated identifier of the form "MA-N" where N is a number that
       does not conflict with any other alignments.

       'max_iterations' controls how many iterations to do to find the final alignment.  An iteration
       takes the fully populated (i.e. no gap characters) columns from the previous alignment, superimposes
       the structures using those columns, then recomputes an alignment.  Iteration stops when the
       recomputed alignment has no more fully populated columns than the previous alignment, or if it has
       less than three fully populated columns.  An max_iterations that is a positive integer will do no
       more than that many iterations.  An max_iterations of zero does no iterations.  An max_iterations
       of None iterates until convergence.

       'min_stretch' is the minimum number of consecutive fully populated columns needed for those columns
       to be considered in the iteration calculation.  Effectively, fully populated columns that are not
       in a stretch of at least this length are treated as not being fully populated for iteration
       purposes.

       'ref_chain' is the chain that superpositions go onto during the iteration process.  If None, then
       the first chain in 'chains'.

       'show_alignment' controls whether the computed alignment is displayed in a viewer.  Typically
       always True unless being called from a computation-oriented script.
    """
    if circular:
        raise NotImplementedError("Circular permuation support not yet implemented")
    if len(chains) < 2:
        raise UserError("Must specifiy at least two chains as basis for alignment")
    if ref_chain is None:
        ref_chain = chains[0]
    elif ref_chain not in chains:
        raise UserError("Reference chain must be involved in alignment")
    if len(chains.structures.unique()) != len(chains):
        raise UserError("Specify only one chain per model")

    cutoff_fmt = "%.1f" if int(cutoff_distance) == cutoff_distance else "%g"
    cutoff_text = cutoff_fmt % cutoff_distance
    session.logger.info("Match\N{RIGHTWARDS ARROW}Align cutoff: %s, in column if within cutoff of: %s"
        % (cutoff_text, column_criterion))

    ordered = sorted(chains, key=lambda seq: seq.structure.id)
    from .make_alignment import match_to_align
    aligned = match_to_align(session, ordered, cutoff_distance, column_criterion, gap_char, circular)
    full_cols = fully_populated(aligned)
    session.logger.info("%d fully populated columns" % len(full_cols))
    if alignment_id is None:
        id_num = 1
        while True:
            alignment_id = "MA-%d" % id_num
            for aln in session.alignments.alignments:
                if aln.ident == alignment_id:
                    break
            else:
                # no conflicts
                break
            id_num += 1
    # iteration limit of None means until convergence
    if max_iterations != 0:
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
                align.align(session, seq_atoms, ref_atoms)
            aligned = match_to_align(session, aligned, cutoff_distance, column_criterion, gap_char, circular)
            full_cols = fully_populated(aligned)
            session.logger.info("Iteration %d: %d fully populated columns" % (iteration, len(full_cols)))
            if len(full_cols) > len(best):
                best = full_cols
            else:
                break
            if max_iterations and iteration >= max_iterations:
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
        overall_rmsd = sqrt(2 * dsq_sum / (len(aligned) * (len(aligned)-1)))
        session.logger.info("Overall RMSD: %.3f" % overall_rmsd)
        seq_lens = [len([r for r in seq.residues if r and r.principal_atom]) for seq in aligned]
        session.logger.info("Sequence lengths: " + ' '.join(["%d" % sl for sl in seq_lens]))
        num_aligned = len(full_cols)

        # Compute/report SDM as per:
        # 	Comparison of sequence-based and structure-based phylogenetic
        #		trees of homologous proteins: Inferences on protein evolution
        #	Balaji S, Srinivasan N.
        #	J Biosci. 2007 Jan;32(1):83-96.
        rel_rmsd = overall_rmsd / cutoff_distance
        srms = 1.0 - rel_rmsd
        pfte = num_aligned / min(seq_lens)
        w1 = 1.0 - (pfte + srms) / 2.0
        w2 = (pfte + srms) / 2.0
        from math import log
        sdm = -100.0 * log(w1*pfte + w2*srms)
        session.logger.info("SDM (cutoff %s): %.3f" % (cutoff_text, sdm))

        # compute/report Q score as per:
        #	Secondary structure matching (SSM), a new tool for fast
        #		protein structure alignment in three dimensions
        #	Krissinel E, Henrick K.
        #	Acta Crystallogr D Biol Crystallogr. 2004 Dec;
        #		60(Pt 12 Pt 1):2256-68.
        rel_rmsd = overall_rmsd / 3.0
        seq_len_mul = 1.0
        for seq_len in seq_lens:
            seq_len_mul *= seq_len
        q = (num_aligned ** len(aligned)) / ((1.0 + rel_rmsd * rel_rmsd) * seq_len_mul)
        session.logger.info("Q-score: %.3f" % q)

    alignment = session.alignments.new_alignment(aligned, alignment_id,
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
    if full_cols:
        return alignment, overall_rmsd, sdm, q
    return alignment, None, None, None

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
            ('column_criterion', EnumOf(['any', 'all'])),
            ('cutoff_distance', NonNegativeFloatArg),
            ('gap_char', CharacterArg),
            ('alignment_id', StringArg),
            ('max_iterations', Or(NonNegativeIntArg, NoneArg)),
            ('min_stretch', PositiveIntArg),
            ('ref_chain', ChainArg),
            ('show_alignment', BoolArg),
        ],
        synopsis = 'Create sequence alignment from structural superposition'
    )
    register(cmd_name, desc, make_alignment, logger=logger)
