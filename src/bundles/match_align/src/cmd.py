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
    alignment = session.alignments.new_alignment(aligned, ident,
        name="Match\N{RIGHTWARDS ARROW}Align", auto_associate=False, viewer=show_alignment)
    #TODO: lots
    for orig, aligned in zip(ordered, aligned):
        alignment.associate(orig, seq=aligned)
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
