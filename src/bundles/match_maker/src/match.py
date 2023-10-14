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

CP_SPECIFIC_SPECIFIC = "ss"
CP_SPECIFIC_BEST = "bs"
CP_BEST_BEST = "bb"

from chimerax.alignment_algs import NEEDLEMAN_WUNSCH as AA_NEEDLEMAN_WUNSCH, \
    SMITH_WATERMAN as AA_SMITH_WATERMAN

from .settings import defaults
default_ss_matrix = defaults['ss_scores']

from chimerax.core.errors import UserError, LimitationError

# called recursively, so any changes to calling signature need to happen
# in recursive call too...
def align(session, ref, match, matrix_name, algorithm, gap_open, gap_extend, dssp_cache,
                    ss_matrix=defaults["ss_scores"],
                    ss_fraction=defaults["ss_mixture"],
                    gap_open_helix=defaults["helix_open"],
                    gap_open_strand=defaults["strand_open"],
                    gap_open_other=defaults["other_open"],
                    compute_ss=defaults["compute_ss"],
                    keep_computed_ss=defaults['overwrite_ss']):
    from chimerax import sim_matrices
    similarity_matrix = sim_matrices.matrix(matrix_name, session.logger)
    ssf = ss_fraction
    ssm = ss_matrix
    if ssf is not None and ssf is not False and compute_ss:
        need_compute = []
        if ref.structure not in dssp_cache:
            for r in ref.residues:
                if r and len(r.atoms) > 1:
                    # not CA only
                    need_compute.append(ref.structure)
                    dssp_cache[ref.structure] = (ref.structure.residues.ss_ids,
                        ref.structure.residues.ss_types)
                    break
        if match.structure not in dssp_cache:
            for r in match.residues:
                if r and len(r.atoms) > 1:
                    # not CA only
                    need_compute.append(match.structure)
                    dssp_cache[match.structure] = (match.structure.residues.ss_ids,
                        match.structure.residues.ss_types)
                    break
        if need_compute:
            from chimerax import dssp
            for s in need_compute:
                # keep_computed_ss is None in a recursive call
                if not keep_computed_ss and keep_computed_ss is not None:
                    s.ss_change_notify = False
                dssp.compute_ss(s)
    if algorithm == "nw":
        from chimerax.alignment_algs import NeedlemanWunsch
        score, seqs = NeedlemanWunsch.nw(ref, match,
            score_gap=-gap_extend, score_gap_open=0-gap_open,
            similarity_matrix=similarity_matrix, return_seqs=True,
            ss_matrix=ss_matrix, ss_fraction=ss_fraction,
            gap_open_helix=-gap_open_helix,
            gap_open_strand=-gap_open_strand,
            gap_open_other=-gap_open_other)
        gapped_ref, gapped_match = seqs
    elif algorithm =="sw":
        def ss_let(r):
            if not r:
                return ' '
            if r.is_helix:
                return 'H'
            elif r.is_strand:
                return 'S'
            return 'O'
        if ssf is False or ssf is None:
            ssf = 0.0
            ssm = None
        if ssm:
            # account for missing structure (blank SS letter)
            ssm = ssm.copy()
            for let in "HSO ":
                ssm[(let, ' ')] = 0.0
                ssm[(' ', let)] = 0.0
        from chimerax.alignment_algs import SmithWaterman
        score, alignment = SmithWaterman.align(ref.characters, match.characters,
            similarity_matrix, float(gap_open), float(gap_extend),
            gap_char=".", ss_matrix=ssm, ss_fraction=ssf,
            gap_open_helix=float(gap_open_helix),
            gap_open_strand=float(gap_open_strand),
            gap_open_other=float(gap_open_other),
            ss1="".join([ss_let(r) for r in ref.residues]),
            ss2="".join([ss_let(r) for r in match.residues]))
        from chimerax.atomic import StructureSeq, Sequence
        gapped_ref = StructureSeq(structure=ref.structure, chain_id=ref.chain_id)
        gapped_ref.name = ref.structure.name
        gapped_match = StructureSeq(structure=match.structure, chain_id=match.chain_id)
        gapped_match.name = match.structure.name
        # Smith-Waterman may not be entirety of sequences...
        for orig, gapped, sw in [
                (ref, gapped_ref, Sequence(characters=alignment[0])),
                (match, gapped_match, Sequence(characters=alignment[1]))]:
            ungapped = sw.ungapped()
            for i in range(len(orig) - len(ungapped) + 1):
                if ungapped == orig[i:i+len(ungapped)]:
                    break
            else:
                raise ValueError("Smith-Waterman result not"
                    " a subsequence of original sequence")
            gapped.bulk_set(orig.residues[i:i+len(ungapped)], sw.characters)
    else:
        raise ValueError("Unknown sequence alignment algorithm: %s" % algorithm)

    # If the structures are disjoint snippets of the same longer SEQRES,
    # they may be able to be structurally aligned but the SEQRES records
    # will keep them apart.  Try to detect this situation and work around
    # by snipping off sequence ends.
    sr_disjoint = False
    if ref.from_seqres and match.from_seqres:
        struct_match = 0
        for i in range(len(gapped_ref)):
            uri = gapped_ref.gapped_to_ungapped(i)
            if uri is None:
                continue
            umi = gapped_match.gapped_to_ungapped(i)
            if umi is None:
                continue
            if gapped_ref.residues[uri] and gapped_match.residues[umi]:
                struct_match += 1
                if struct_match >= 3:
                    break
        if struct_match < 3:
            seq_match = 0
            for s1, s2 in zip(gapped_ref[:], gapped_match[:]):
                if s1.isalpha() and s2.isalpha():
                    seq_match += 1
                    if seq_match > 3:
                        break
            if seq_match > 3:
                need = 3 - struct_match
                if (ref.residues[:need].count(None) == 3
                or ref.residues[-need:].count(None) == 3) \
                and (match.residues[:need].count(None) == 3
                or match.residues[-need:].count(None) == 3):
                    sr_disjoint = True
    if sr_disjoint:
        from copy import copy
        clipped_ref = copy(ref)
        clipped_match = copy(match)
        for seq in (clipped_ref, clipped_match):
            num_none = 0
            for r in seq.residues:
                if r:
                    break
                num_none += 1
            if num_none:
                seq.bulk_set(seq.residues[num_none:], seq[num_none:])

            num_none = 0
            for r in reversed(seq.residues):
                if r:
                    break
                num_none += 1
            if num_none:
                seq.bulk_set(seq.residues[:-num_none], seq[:-num_none])
        return align(session, clipped_ref, clipped_match, matrix_name, algorithm, gap_open,
            gap_extend, dssp_cache, ss_matrix=ss_matrix, ss_fraction=ss_fraction,
            gap_open_helix=gap_open_helix, gap_open_strand=gap_open_strand,
            gap_open_other=gap_open_other, compute_ss=False, keep_computed_ss=None)
    for orig, aligned in [(ref, gapped_ref), (match, gapped_match)]:
        if hasattr(orig, '_dm_rebuild_info'):
            aligned._dm_rebuild_info = orig._dm_rebuild_info
            _dm_cleanup.append(aligned)
    return score, gapped_ref, gapped_match

def match(session, chain_pairing, match_items, matrix, alg, gap_open, gap_extend, *, cutoff_distance=None,
        show_alignment=defaults['show_alignment'], align=align, domain_residues=(None, None), bring=None,
        verbose=defaults['verbose_logging'], always_raise_errors=False, report_matrix=False,
        **align_kw):
    """Superimpose structures based on sequence alignment

       Returns a list of dictionaries, one per chain pairing.  The dictionaries are:
       {
         "full ref atoms": chimerax.atomic.Atoms
         "full match atoms": chimerax.atomic.Atoms
         "final ref atoms": chimerax.atomic.Atoms
         "final match atoms": chimerax.atomic.Atoms
         "final RMSD": float
         "full RMSD": float
         "transformation matrix": chimerax.geometry.Place
         "aligned ref seq": chimerax.atomic.StructureSeq
         "aligned match seq": chimerax.atomic.StructureSeq
       }
       "full" is before iteration pruning and "final" is afterward.

       'chain_pairing' is the method of pairing chains to match:

       CP_SPECIFIC_SPECIFIC --
       Each reference chain is paired with a specified match chain
       ('match_items' is sequence of (ref_chain, match_chain) tuples)

       CP_SPECIFIC_BEST --
       Single reference chain is paired with best seq-aligning
       chain from one or more structures
       ('match_items' is (reference_chain, [match_structures]))

       CP_BEST_BEST --
       Best seq-aligning pair of chains from reference structure and
       match structure(s) is used
       ('match_items' is (ref_structure, [match_structures]))

       'matrix' is name of similarity matrix

       'alg' is the alignment algorithm: AA_NEEDLEMAN_WUNSCH or AA_SMITH_WATERMAN

       'gap_open' and 'gap_extend' are the gap open/extend penalties used
       for the initial sequence alignment

       'cutoff_distance' is the cutoff used for iterative superposition -- iteration stops
       when all remaining distances are below the cutoff.  If None, no iteration.

       'show_alignment' controls whether the sequence alignment is also shown in a
       sequence viewer.

       'align' allows specification of the actual function align/score one chain to
       another.  See the align() function above.

       'domain_residues' allows matching to be restricted to a subset of the chain(s).
       If given, should be (ref_Residues_collection, match_Residues_collection)

       'bring' specifies other structures that should be transformed along with the
       match structure (so, there must be only one match structure in such a case).

       'verbose', if True, produces additional output to the log, If None, nothing
       will be logged.

       If 'always_raise_errors' is True, then an iteration that goes to too few
       matched atoms will immediately raise an error instead of noting the
       failure in the log and continuing on to other pairings.
    """
    dssp_cache = {}
    alg = alg.lower()
    if alg == "nw" or alg.startswith("needle"):
        alg = "nw"
        alg_name = "Needleman-Wunsch"
    elif alg =="sw" or alg.startswith("smith"):
        alg = "sw"
        alg_name = "Smith-Waterman"
    else:
        raise ValueError("Unknown sequence alignment algorithm: %s" % alg)
    pairings = {}
    small_mol_err_msg = "Reference and/or match model contains no nucleic or"\
        " amino acid chains.\nUse the command-line 'align' command" \
        " to superimpose small molecules/ligands."
    logged_matrix = matrix
    rd_res, md_res = domain_residues
    from chimerax.sim_matrices import matrix_compatible, compatible_matrix_names, protein_matrix
    try:
        if chain_pairing == CP_SPECIFIC_SPECIFIC:
            # specific chain(s) in each

            # various sanity checks
            #
            # (1) can't have same chain matched to multiple refs
            # (2) reference structure can't be a match structure
            match_chains = {}
            match_mols = {}
            ref_mols = {}
            final_matrix_name = {}
            for ref, match in match_items:
                final_matrix_name[match] = matrix
                ref_compatible = matrix_compatible(ref, matrix, session.logger)
                match_compatible = matrix_compatible(match, matrix, session.logger)
                if not ref_compatible and not match_compatible:
                    compatible_names = compatible_matrix_names(match, session.logger)
                    if len(compatible_names) == 1:
                        logged_matrix = final_matrix_name[match] = compatible_names[0]
                        session.logger.info("Using %s matrix instead of %s to match %s to %s"
                            % (logged_matrix, matrix, match, ref))
                        ref_compatible = match_compatible = True

                if not ref_compatible:
                    raise UserError("Reference chain (%s) not compatible with %s similarity"
                        " matrix" % (ref.full_name, matrix))
                if not match_compatible:
                    raise UserError("Match chain (%s) not compatible with %s similarity"
                        " matrix" % (match.full_name, matrix))
                if match in match_chains:
                    raise UserError("Cannot match the same chain to multiple reference chains")
                match_chains[match] = ref
                if match.structure in ref_mols \
                or ref.structure in match_mols \
                or match.structure == ref.structure:
                    raise UserError("Cannot have same molecule"
                        " model provide both reference and match chains")
                match_mols[match.structure] = ref
                ref_mols[ref.structure] = match

            if not match_chains:
                raise UserError("Must select at least one reference"
                                    " chain.\n")

            for match, ref in match_chains.items():
                matrix_name = final_matrix_name[match]
                match, ref = [check_domain_matching([ch], dr)[0] for ch, dr in
                    ((match, md_res), (ref, rd_res))]
                score, s1, s2 = align(session, ref, match, matrix_name, alg,
                            gap_open, gap_extend, dssp_cache, **align_kw)
                pairings.setdefault(s2.structure, []).append((score, s1, s2))

        elif chain_pairing == CP_SPECIFIC_BEST:
            # specific chain in reference;
            # best seq-aligning chain in match model(s)
            ref, matches = match_items
            if not ref or not matches:
                raise UserError("Must select at least one reference and match item.\n")
            if not matrix_compatible(ref, matrix, session.logger):
                compatible_names = compatible_matrix_names(ref, session.logger)
                if len(compatible_names) == 1:
                    session.logger.info("Using %s matrix instead of %s to match to %s"
                        % (compatible_names[0], matrix, ref))
                    logged_matrix = matrix = compatible_names[0]
                else:
                    raise UserError("Reference chain (%s) not compatible"
                                " with %s similarity matrix" % (ref.full_name, matrix))
            ref = check_domain_matching([ref], rd_res)[0]
            for match in matches:
                best_score = None
                seqs = [s for s in match.chains if matrix_compatible(s, matrix, session.logger)]
                if not seqs and match.chains:
                    raise UserError("No chains in match structure"
                        " %s compatible with %s similarity"
                        " matrix" % (match, matrix))
                seqs = check_domain_matching(seqs, md_res)
                for seq in seqs:
                    score, s1, s2 = align(session, ref, seq, matrix, alg,
                            gap_open, gap_extend, dssp_cache, **align_kw)
                    if best_score is None or score > best_score:
                        best_score = score
                        pairing = (score, s1, s2)
                if best_score is None:
                    raise LimitationError(small_mol_err_msg)
                pairings[match]= [pairing]

        elif chain_pairing == CP_BEST_BEST:
            # best seq-aligning pair of chains between
            # reference and match structure(s)
            ref, matches = match_items
            if not ref or not matches:
                raise UserError("Must select at least one reference"
                    " and match item in different models.\n")
            # check chain/matrix compatibilty: for our own sanity only allow one matrix for all
            # matching, not one per ref/match pairing
            ref_data = []
            matches_data = []
            cross_compatible = set()
            for domain_rseq in [s for s in check_domain_matching(ref.chains, rd_res)]:
                compatible_names = compatible_matrix_names(domain_rseq, session.logger)
                ref_data.append(domain_rseq)
                cross_compatible.update(compatible_names)
            for match in matches:
                match_data = []
                match_compatible = set()
                for domain_mseq in [s for s in check_domain_matching(match.chains, md_res)]:
                    compatible_names = compatible_matrix_names(domain_mseq, session.logger)
                    match_data.append(domain_mseq)
                    match_compatible.update(compatible_names)
                matches_data.append((match, match_data))
                cross_compatible &= match_compatible
            if not cross_compatible:
                raise UserError(
                    "No matrix compatible with both reference structure and all match structures")
            if matrix not in cross_compatible:
                if len(cross_compatible) == 1:
                    compatible_matrix = compatible_names[0]
                    session.logger.info("Using %s matrix instead of %s for matching"
                        % (compatible_matrix, matrix))
                    logged_matrix = matrix = compatible_matrix
                else:
                    raise UserError("Chains in reference structure and match structures not both compatible"
                        "with %s similarity matrix" % matrix)

            for match, match_data in matches_data:
                best_score = None
                for mseq in match_data:
                    for rseq in ref_data:
                        score, s1, s2 = align(session, rseq, mseq,
                            matrix, alg, gap_open, gap_extend, dssp_cache, **align_kw)
                        if best_score is None or score > best_score:
                            best_score = score
                            pairing = (score,s1,s2)
                if best_score is None:
                    raise LimitationError(small_mol_err_msg)
                pairings[match]= [pairing]
        else:
            raise ValueError("No such chain-pairing method")
    finally:
        if not align_kw.get('keep_computed_ss', defaults['overwrite_ss']):
            for s, ss_info in dssp_cache.items():
                ss_ids, ss_types = ss_info
                s.residues.ss_ids = ss_ids
                s.residues.ss_types = ss_types
                s.ss_change_notify = True

    logger = session.logger
    ret_vals = []
    logged_params = False
    for match_mol, pairs in pairings.items():
        ref_atoms = []
        match_atoms = []
        region_info = {}
        if verbose:
            seq_pairings = []
        for score, s1, s2 in pairs:
            try:
                ss_matrix = align_kw['ss_matrix']
            except KeyError:
                ss_matrix = default_ss_matrix
            try:
                ss_fraction = align_kw['ss_fraction']
            except KeyError:
                ss_fraction = defaults["ss_mixture"]

            if not logged_params and verbose is not None:
                if ss_fraction is None or ss_fraction is False:
                    ss_rows="""
                        <tr>
                            <td colspan="2" align="center">No secondary-structure guidance used</td>
                        </tr>
                        <tr>
                            <td>Gap open</td>
                            <td>%g</td>
                        </tr>
                        <tr>
                            <td>Gap extend</td>
                            <td>%g</td>
                        </tr>
                    """ % (gap_open, gap_extend)
                else:
                    if 'gap_open_helix' in align_kw:
                        gh = align_kw['gap_open_helix']
                    else:
                        gh = defaults["helix_open"]
                    if 'gap_open_strand' in align_kw:
                        gs = align_kw['gap_open_strand']
                    else:
                        gs = defaults["strand_open"]
                    if 'gap_open_other' in align_kw:
                        go = align_kw['gap_open_other']
                    else:
                        go = defaults["other_open"]
                    ss_rows="""
                        <tr>
                            <td>SS fraction</td>
                            <td>%g</td>
                        </tr>
                        <tr>
                            <td>Gap open (HH/SS/other)</td>
                            <td>%g/%g/%g</td>
                        </tr>
                        <tr>
                            <td>Gap extend</td>
                            <td>%g</td>
                        </tr>
                        <tr>
                            <td>SS matrix</td>
                            <td>
                                <table>
                                    <tr>
                                        <th></th> <th>H</th> <th>S</th> <th>O</th>
                                    </tr>
                                    <tr>
                                        <th>H</th> <td align="right">%g</td> <td align="right">%g</td> <td align="right">%g</td>
                                    </tr>
                                    <tr>
                                        <th>S</th> <td></td> <td align="right">%g</td> <td align="right">%g</td>
                                    </tr>
                                    <tr>
                                        <th>O</th> <td></td> <td></td> <td align="right">%g</td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                    """ % (ss_fraction, gh, gs, go, gap_extend,
                        ss_matrix[('H','H')], ss_matrix[('H','S')], ss_matrix[('H','O')],
                        ss_matrix[('S','S')], ss_matrix[('S','O')], ss_matrix[('O','O')])
                if cutoff_distance is None:
                    iterate_row = """<tr> <td colspan="2" align="center">No iteration</td> </tr>"""
                else:
                    iterate_row = """<tr> <td>Iteration cutoff</td> <td>%g</td></tr>""" % cutoff_distance
                from chimerax.core.logger import html_table_params
                param_table = """
                    <table %s>
                        <tr>
                            <th colspan="2">Parameters</th>
                        </tr>
                        <tr>
                            <td>Chain pairing</td>
                            <td>%s</td>
                        </tr>
                        <tr>
                            <td>Alignment algorithm</td>
                            <td>%s</td>
                        </tr>
                        <tr>
                            <td>Similarity matrix</td>
                            <td>%s</td>
                        </tr>
                        %s
                        %s
                    </table>
                """ % (html_table_params, chain_pairing, alg_name, logged_matrix, ss_rows, iterate_row)
                logger.info(param_table, is_html=True)
                logged_params = True
            logger.status("Matchmaker %s (#%s) with %s (#%s),"
                " sequence alignment score = %g" % (
                s1.name, s1.structure.id_string, s2.name,
                s2.structure.id_string, score), log=(verbose is not None))
            skip = set()
            if show_alignment:
                for s in [s1,s2]:
                    if hasattr(s, '_dm_rebuild_info'):
                        residues = s.residues
                        characters = list(s.characters)
                        for i, c, r in s._dm_rebuild_info:
                            g = s.ungapped_to_gapped(i)
                            characters[g] = c
                            residues[i] = r
                            skip.add(r)
                        s.bulk_set(residues, characters)
                alignment = session.alignments.new_alignment([s1,s2], None, auto_associate=None,
                    name="MatchMaker alignment")
                alignment.auto_associate = True
                for hdr in alignment.headers:
                    hdr.shown = hdr.ident == "rmsd"
            residues1 = s1.residues
            residues2 = s2.residues
            for i in range(len(s1)):
                if s1[i] == "." or s2[i] == ".":
                    continue
                ref_res = residues1[s1.gapped_to_ungapped(i)]
                if not ref_res:
                    continue
                ref_atom = ref_res.principal_atom
                if not ref_atom:
                    continue
                match_res = residues2[s2.gapped_to_ungapped(i)]
                if not match_res:
                    continue
                match_atom = match_res.principal_atom
                if not match_atom:
                    continue
                if ref_res in skip or match_res in skip:
                    continue
                if ref_atom.name != match_atom.name:
                    # nucleic P-only trace vs. full nucleic
                    if ref_atom.name != "P":
                        ref_atom = ref_atom.residue.find_atom("P")
                        if not ref_atom:
                            continue
                    else:
                        match_atom = match_atom.residue.find_atom("P")
                        if not match_atom:
                            continue
                ref_atoms.append(ref_atom)
                match_atoms.append(match_atom)
                if show_alignment and cutoff_distance is not None:
                    for viewer in  alignment.viewers:
                        region_info[ref_atom] = (viewer, i)

            if verbose:
                seq_pairings.append((s1, s2))
        from chimerax.std_commands import align
        if len(match_atoms) < 3:
            msg = "Fewer than 3 residues aligned; cannot match %s with %s" % (s1.name, s2.name)
            if always_raise_errors:
                raise align.IterationError(msg)
            logger.error(msg)
            continue
        from chimerax.atomic import Atoms
        initial_match, initial_ref = Atoms(match_atoms), Atoms(ref_atoms)
        try:
            final_match, final_ref, rmsd, full_rmsd, xf = align.align(session, initial_match, initial_ref,
                cutoff_distance=cutoff_distance, log_info=(verbose is not None), report_matrix=report_matrix)
        except align.IterationError:
            if always_raise_errors:
                raise
            logger.error("Iteration produces fewer than 3"
                " residues aligned.\nCannot match %s with %s"
                " satisfying iteration threshold."
                % (s1.name, s2.name))
            continue
        ret_vals.append({
            "full match atoms": initial_match,
            "full ref atoms": initial_ref,
            "final match atoms": final_match,
            "final ref atoms": final_ref,
            "full RMSD": full_rmsd,
            "final RMSD": rmsd,
            "transformation matrix": xf,
            "aligned ref seq": s1,
            "aligned match seq": s2,
        })
        if bring is not None:
            for m in bring:
                m.scene_position = xf * m.scene_position
        if verbose is not None:
            logger.info("") # separate matches with whitespace
        if region_info:
            by_viewer = {}
            for ra in final_ref:
                viewer, index = region_info[ra]
                by_viewer.setdefault(viewer, []).append(index)
            for viewer, indices in by_viewer.items():
                indices.sort()
                name, fill, outline = viewer.MATCHED_REGION_INFO
                viewer.new_region(name=name, columns=indices, fill=fill, outline=outline)
                viewer.status("Residues used in final fit iteration are highlighted")
        if verbose:
            for s1, s2 in seq_pairings:
                logger.info("Sequences:")
                for s in [s1,s2]:
                    logger.info(s.name + "\t" + s.characters)
                logger.info("Residues:")
                for s in [s1, s2]:
                    logger.info(", ".join([str(r) for r in s.residues]))
                logger.info("Residue usage in match (1=used, 0=unused):")
                match_residues = set([a.residue for matched in (final_match, final_ref) for a in matched])
                for s in [s1, s2]:
                    logger.info(", ".join([str(int(r in match_residues)) for r in s.residues]))

    global _dm_cleanup
    for seq in _dm_cleanup:
        delattr(seq, '_dm_rebuild_info')
    _dm_cleanup = []
    return ret_vals

def cmd_match(session, match_atoms, to=None, pairing=defaults["chain_pairing"],
        alg=defaults["alignment_algorithm"], verbose=defaults['verbose_logging'], bring=None,
        ss_fraction=defaults["ss_mixture"], matrix=defaults["matrix"], gap_open=defaults["gap_open"],
        hgap=defaults["helix_open"], sgap=defaults["strand_open"], ogap=defaults["other_open"],
        cutoff_distance=defaults["iter_cutoff"], gap_extend=defaults["gap_extend"],
        show_alignment=defaults['show_alignment'], compute_s_s=defaults["compute_ss"],
        keep_computed_s_s=defaults['overwrite_ss'], report_matrix=False,
        mat_h_h=default_ss_matrix[('H', 'H')],
        mat_s_s=default_ss_matrix[('S', 'S')],
        mat_o_o=default_ss_matrix[('O', 'O')],
        mat_h_s=default_ss_matrix[('H', 'S')],
        mat_h_o=default_ss_matrix[('H', 'O')],
        mat_s_o=default_ss_matrix[('S', 'O')]):
    """wrapper for command-line command (friendlier args)"""

    # 'to' only needed to sidestep problem with adjacent atom specs...
    ref_atoms = to

    from chimerax import sim_matrices
    if matrix not in sim_matrices.matrices(session.logger):
        raise UserError("No such matrix name: %s" % str(matrix))
    if pairing == CP_SPECIFIC_SPECIFIC:
        matches = match_atoms.residues.chains.unique()
    elif pairing == CP_SPECIFIC_BEST:
        matches = match_atoms.structures.unique()
    if pairing == CP_SPECIFIC_SPECIFIC or pairing == CP_SPECIFIC_BEST:
        refs = ref_atoms.residues.chains.unique()
        if not refs:
            raise UserError("No 'to' chains specified")
        if pairing == CP_SPECIFIC_BEST:
            if len(refs) > 1:
                raise UserError("Specify a single 'to' chain only")
    else:
        ref_mols = ref_atoms.structures.unique()
        if not ref_mols:
            raise UserError("No 'to' model specified")
        if len(ref_mols) > 1:
            raise UserError("Specify a single 'to' model only")
        refs = ref_mols
        matches = match_atoms.structures.unique()
    if not matches:
        raise UserError("No molecules/chains to match specified")
    # the .subtract() method of Collections does not preserve order (as of 10/28/16),
    # so "subtract" by hand...
    refs = [r for r in refs if r not in matches]
    if not refs:
        raise UserError("Must use different reference and match structures")
    if bring is not None:
        bring = set(bring)
        match_structures = matches if pairing != CP_SPECIFIC_SPECIFIC \
            else matches.structures.unique()
        if len(match_structures) > 1:
            raise UserError("'bring' option can only be used when exactly one structure is being"
                " matched")
        ref_structures = refs if pairing == CP_BEST_BEST else set([r.structure for r in refs])
        for test_structures, text in [(match_structures, "match"), (ref_structures, "reference")]:
            for s in test_structures:
                if s in bring:
                    bring.discard(s)
                else:
                    for b in bring:
                        if b.id == s.id[:len(b.id)]:
                            raise UserError("Cannot 'bring' parent model of %s structure" % text)
        if len(bring) == 0:
            session.logger.warning("'bring' arg specifies no non-match/ref structures")
            bring = None
    if pairing == CP_SPECIFIC_SPECIFIC:
        if len(refs) != len(matches):
            from chimerax.atomic import Chains
            num_match_structs = len(Chains(matches).structures.unique())
            if num_match_structs != len(matches) or len(refs) > 1:
                raise UserError("Different number of reference/match"
                        " chains (%d ref, %d match)" % (len(refs), len(matches)))
            match_items = [(refs[0], match) for match in matches]
        else:
            match_items = zip(refs, matches)
    else:
        match_items = (refs[0], matches)
    ss_matrix = {}
    ss_matrix[('H', 'H')] = float(mat_h_h)
    ss_matrix[('S', 'S')] = float(mat_s_s)
    ss_matrix[('O', 'O')] = float(mat_o_o)
    ss_matrix[('H', 'S')] = ss_matrix[('S', 'H')] = float(mat_h_s)
    ss_matrix[('H', 'O')] = ss_matrix[('O', 'H')] = float(mat_h_o)
    ss_matrix[('S', 'O')] = ss_matrix[('O', 'S')] = float(mat_s_o)
    ret_vals = match(session, pairing, match_items, matrix, alg, gap_open, gap_extend,
        ss_fraction=ss_fraction, ss_matrix=ss_matrix,
        cutoff_distance=cutoff_distance, show_alignment=show_alignment, bring=bring,
        domain_residues=(ref_atoms.residues.unique(), match_atoms.residues.unique()),
        gap_open_helix=hgap, gap_open_strand=sgap, gap_open_other=ogap, report_matrix=report_matrix,
        compute_ss=compute_s_s, keep_computed_ss=keep_computed_s_s, verbose=verbose)
    return ret_vals

_dm_cleanup = []
def check_domain_matching(chains, sel_residues):
    if not sel_residues:
        return chains
    chain_residues = set([r for ch in chains for r in ch.residues if r])
    sel_residues = set(sel_residues)
    if not chain_residues.issubset(sel_residues):
        # domain matching
        new_chains = []
        from chimerax.atomic import StructureSeq
        for chain in chains:
            this_chain = set([r for r in chain.residues if r])
            if this_chain.issubset(sel_residues):
                new_chains.append(chain)
                continue
            if this_chain.isdisjoint(sel_residues):
                continue
            nc = StructureSeq(structure=chain.structure, chain_id=chain.chain_id,
                polymer_type=chain.polymer_type)
            nc._dm_rebuild_info = []
            _dm_cleanup.append(nc)
            new_chains.append(nc)
            chars = []
            residues = []
            for i, c_r in enumerate(zip(chain.characters, chain.residues)):
                c, r = c_r
                if r in sel_residues:
                    chars.append(c)
                    residues.append(r)
                else:
                    nc._dm_rebuild_info.append((i, c, r))
                    chars.append('?')
                    residues.append(None)
            nc.bulk_set(residues, chars)
        chains = new_chains
    return chains

_registered = False
def register_command(logger):
    global _registered
    if _registered:
        # registration can be called for both main command and alias, so only do once...
        return
    _registered = True
    from chimerax.core.commands import CmdDesc, register, FloatArg, StringArg, \
        BoolArg, NoneArg, TopModelsArg, create_alias, Or, DynamicEnum
    # use OrderedAtomsArg so that /A-F come out in the expected order even if not ordered that way
    # internally [#7577]
    from chimerax.atomic import OrderedAtomsArg
    from chimerax import sim_matrices
    desc = CmdDesc(
        required = [('match_atoms', OrderedAtomsArg)],
        required_arguments = ['to'],
        keyword = [('to', OrderedAtomsArg), ('pairing', StringArg), ('alg', StringArg),
            ('verbose', BoolArg), ('ss_fraction', Or(FloatArg, BoolArg)),
            ('matrix', DynamicEnum(lambda logger=logger: sim_matrices.matrices(logger).keys())),
            ('gap_open', FloatArg), ('hgap', FloatArg), ('sgap', FloatArg), ('ogap', FloatArg),
            ('cutoff_distance', Or(FloatArg, NoneArg)), ('gap_extend', FloatArg),
            ('bring', TopModelsArg), ('show_alignment', BoolArg), ('compute_s_s', BoolArg),
            ('mat_h_h', FloatArg), ('mat_s_s', FloatArg), ('mat_o_o', FloatArg), ('mat_h_s', FloatArg),
            ('mat_h_o', FloatArg), ('mat_s_o', FloatArg), ('keep_computed_s_s', BoolArg),
            ('report_matrix', BoolArg)],
        synopsis = 'Align atomic structures using sequence alignment'
    )
    register('matchmaker', desc, cmd_match, logger=logger)
    create_alias('mmaker', "%s $*" % 'matchmaker', logger=logger, url="help:user/commands/matchmaker.html")
