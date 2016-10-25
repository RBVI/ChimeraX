# vim: set expandtab shiftwidth=4 softtabstop=4:

from . import CP_SPECIFIC_SPECIFIC, CP_SPECIFIC_BEST, CP_BEST
from . import AA_NEEDLEMAN_WUNSCH, AA_SMITH_WATERMAN

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
                    compute_ss=defaults["compute_ss"]):
    from chimerax.seqalign import sim_matrices
    similarity_matrix = sim_matrices.matrix(session, matrix_name)
    ssf = ss_fraction
    ssm = ss_matrix
    if ssf is not None and ssf is not False and compute_ss:
        need_compute = []
        if ref.structure not in dssp_cache:
            for r in ref.residues:
                if r and len(r.atoms) > 1:
                    # not CA only
                    need_compute.append(ref.structure)
                    dssp_cache.add(ref.structure)
                    break
        if match.structure not in dssp_cache:
            for r in match.residues:
                if r and len(r.atoms) > 1:
                    # not CA only
                    need_compute.append(match.structure)
                    dssp_cache.add(match.structure)
                    break
        if need_compute:
            """TODO
            from chimera.initprefs import ksdsspPrefs, \
                    KSDSSP_ENERGY, KSDSSP_HELIX_LENGTH, \
                    KSDSSP_STRAND_LENGTH
            """
            from chimerax.core.commands import dssp
            dssp.compute_ss(session, need_compute)
    if algorithm == "nw":
        from chimerax.seqalign.align_algs import NeedlemanWunsch
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
        from chimerax.seqalign.align_algs import SmithWaterman
        score, alignment = SmithWaterman.align(ref.characters, match.characters,
            similarity_matrix, float(gap_open), float(gap_extend),
            gap_char=".", ss_matrix=ssm, ss_fraction=ssf,
            gap_open_helix=float(gap_open_helix),
            gap_open_strand=float(gap_open_strand),
            gap_open_other=float(gap_open_other),
            ss1="".join([ss_let(r) for r in ref.residues]),
            ss2="".join([ss_let(r) for r in match.residues]))
        from chimerax.core.atomic import StructureSeq, Sequence
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
            gap_open_other=gap_open_other, compute_ss=False)
    for orig, aligned in [(ref, gapped_ref), (match, gapped_match)]:
        if hasattr(orig, '_dm_rebuild_info'):
            aligned._dm_rebuild_info = orig._dm_rebuild_info
            _dm_cleanup.append(aligned)
    return score, gapped_ref, gapped_match

def match(session, chain_pairing, match_items, matrix, alg, gap_open, gap_extend, iterate=None,
        show_alignment=False, align=align, domain_residues=(None, None), 
        verbose=False, **align_kw):
    """Superimpose structures based on sequence alignment

       'chain_pairing' is the method of pairing chains to match:
       
       CP_SPECIFIC_SPECIFIC --
       Each reference chain is paired with a specified match chain
       
       CP_SPECIFIC_BEST --
       Single reference chain is paired with best seq-aligning
       chain from one or more structures

       CP_BEST --
       Best seq-aligning pair of chains from reference structure and
       match structure(s) is used
    """
    dssp_cache = set()
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
        " amino acid chains.\nUse the command-line 'match' command" \
        " to superimpose small molecules/ligands."
    rd_res, md_res = domain_residues
    from chimerax.seqalign.sim_matrices import matrix_compatible
    if chain_pairing == CP_SPECIFIC_SPECIFIC:
        # specific chain(s) in each

        # various sanity checks
        #
        # (1) can't have same chain matched to multiple refs
        # (2) reference structure can't be a match structure
        match_chains = {}
        match_mols = {}
        ref_mols = {}
        for ref, match in match_items:
            if not matrix_compatible(session, ref, matrix):
                raise UserError("Reference chain (%s) not"
                    " compatible with %s similarity"
                    " matrix" % (ref.fullName(), matrix))
            if not matrix_compatible(session, match, matrix):
                raise UserError("Match chain (%s) not"
                    " compatible with %s similarity"
                    " matrix" % (match.fullName(), matrix))
            if match in match_chains:
                raise UserError("Cannot match the same chain"
                    " to multiple reference chains")
            match_chains[match] = ref
            if match.structure in ref_mols \
            or ref.structure in match_mols \
            or match.structure == ref.structure:
                raise UserError("Cannot have same molecule"
                    " model provide both reference and"
                    " match chains")
            match_mols[match.structure] = ref
            ref_mols[ref.structure] = match

        if not match_chains:
            raise UserError("Must select at least one reference"
                                " chain.\n")

        for match, ref in match_chains.items():
            match, ref = [check_domain_matching([ch], dr)[0] for ch, dr in
                ((match, md_res), (ref, rd_res))]
            score, s1, s2 = align(session, ref, match, matrix, alg,
                        gap_open, gap_extend, dssp_cache, **align_kw)
            pairings.setdefault(s2.structure, []).append((score, s1, s2))

    elif chain_pairing == CP_SPECIFIC_BEST:
        # specific chain in reference;
        # best seq-aligning chain in match model(s)
        ref, matches = match_items
        if not ref or not matches:
            raise UserError("Must select at least one reference and match item.\n")
        if not matrix_compatible(session, ref, matrix):
            raise UserError("Reference chain (%s) not compatible"
                        " with %s similarity matrix" % (ref.full_name, matrix))
        ref = check_domain_matching([ref], rd_res)[0]
        for match in matches:
            best_score = None
            seqs = [s for s in match.chains if matrix_compatible(session, s, matrix)]
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

    elif chain_pairing == CP_BEST:
        # best seq-aligning pair of chains between
        # reference and match structure(s)
        ref, matches = match_items
        if not ref or not matches:
            raise UserError("Must select at least one reference"
                " and match item in different models.\n")
        rseqs = [s for s in check_domain_matching(ref.chains, rd_res)
                    if matrix_compatible(session, s, matrix)]
        if not rseqs and ref.chains:
            raise UserError("No chains in reference structure"
                " %s compatible with %s similarity"
                " matrix" % (ref, matrix))
        for match in matches:
            best_score = None
            mseqs = [s for s in check_domain_matching(match.chains, md_res)
                        if matrix_compatible(session, s, matrix)]
            if not mseqs and match.chains:
                raise UserError("No chains in match structure"
                    " %s compatible with %s similarity"
                    " matrix" % (match, matrix))
            for mseq in mseqs:
                for rseq in rseqs:
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

            if not logged_params:
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
                    """ % (ss_fraction, gh, gs, go, gap_extend, ss_matrix[('H','H')],
                        ss_matrix[('H','S')], ss_matrix[('H','O')], ss_matrix[('S','S')],
                        ss_matrix[('S','O')], ss_matrix[('O','O')])
                if iterate is None:
                    iterate_row = """<tr> <td colspan="2" align="center">No iteration</td> </tr>"""
                else:
                    iterate_row = """<tr> <td>Iteration cutoff</td> <td>%g</td></tr>""" % iterate
                param_table = """
                    <table border="1">
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
                """ % (chain_pairing, alg_name, matrix, ss_rows, iterate_row)
                logger.info(param_table, is_html=True)
                logged_params = True
            logger.status("Matchmaker %s (#%s) with %s (#%s),"
                " sequence alignment score = %g" % (
                s1.name, s1.structure.id_string(), s2.name,
                s2.structure.id_string(), score), log=True)
            skip = set()
            if show_alignment:
                logger.info("Showing alignment not yet supported")
                """TODO
                from MultAlignViewer.MAViewer import MAViewer
                for s in [s1,s2]:
                    if hasattr(s, '_dm_rebuild_info'):
                        for i, c, r in s._dm_rebuild_info:
                            g = s.ungapped_to_gapped(i)
                            s[g] = c
                            s.residues[i] = r
                            skip.add(r)
                        s.resMap.clear()
                        for i, r in enumerate(s.residues):
                            if r:
                                s.resMap[r] = i
                mav = MAViewer([s1,s2], autoAssociate=None)
                mav.autoAssociate = True
                mav.hideHeaders(mav.headers(shownOnly=True))
                from MAVHeader.ChimeraExtension import CaDistanceSeq
                mav.showHeaders([h for h in mav.headers()
                            if h.name == CaDistanceSeq.name])
                """
            for i in range(len(s1)):
                if s1[i] == "." or s2[i] == ".":
                    continue
                ref_res = s1.residues[s1.gapped_to_ungapped(i)]
                match_res = s2.residues[s2.gapped_to_ungapped(i)]
                if not ref_res:
                    continue
                ref_atom = ref_res.principal_atom
                if not ref_atom:
                    continue
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
                if show_alignment and iterate is not None:
                    """TODD
                    region_info[ref_atom] = (mav, i)
                    """

            if verbose:
                seq_pairings.append((s1, s2))
        if len(match_atoms) < 3:
            logger.error("Fewer than 3 residues aligned; cannot match %s with %s"
                % (s1.name, s2.name))
            continue
        from chimerax.core.commands import align
        from chimerax.core.atomic import Atoms
        try:
            ret_vals.append(align.align(session, Atoms(match_atoms), Atoms(ref_atoms),
                cutoff_distance=iterate))
        except align.IterationError:
            logger.error("Iteration produces fewer than 3"
                " residues aligned.\nCannot match %s with %s"
                " satisfying iteration threshold."
                % (s1.name, s2.name))
            continue
        logger.info("") # separate matches with whitespace
        """TODO
        if region_info:
            by_mav = {}
            for ra in ret_vals[-1][1]:
                mav, index = region_info[ra]
                by_mav.setdefault(mav, []).append(index)
            for mav, indices in by_mav.items():
                indices.sort()
                from MultAlignViewer.MAViewer import \
                            MATCHED_REGION_INFO
                name, fill, outline = MATCHED_REGION_INFO
                mav.newRegion(name=name, columns=indices,
                        fill=fill, outline=outline)
                mav.status("Residues used in final fit"
                        " iteration are highlighted")
        """
        if verbose:
            for s1, s2 in seq_pairings:
                logger.info("Sequences:")
                for s in [s1,s2]:
                    logger.info(s.name + "\t" + str(s))
                logger.info("Residues:")
                for s in [s1, s2]:
                    logger.info(", ".join([str(r) for r in s.residues]))
                logger.info("Residue usage in match (1=used, 0=unused):")
                match_atoms1, match_atoms2 = ret_vals[-1][:2]
                match_residues = set([a.residue for matched in ret_vals[-1][:2] for a in matched])
                for s in [s1, s2]:
                    logger.info(", ".join([str(int(r in match_residues)) for r in s.residues]))

    global _dm_cleanup
    for seq in _dm_cleanup:
        delattr(seq, '_dm_rebuild_info')
    _dm_cleanup = []
    return ret_vals

def cmd_match(session, match_atoms, to=None, pairing=defaults["chain_pairing"],
        alg=defaults["alignment_algorithm"], verbose=False,
        ss_fraction=defaults["ss_mixture"], matrix=defaults["matrix"],
        gap_open=defaults["gap_open"], hgap=defaults["helix_open"],
        sgap=defaults["strand_open"], ogap=defaults["other_open"],
        iterate=defaults["iter_cutoff"], gap_extend=defaults["gap_extend"],
        show_alignment=False, compute_ss=defaults["compute_ss"],
        mat_hh=default_ss_matrix[('H', 'H')],
        mat_ss=default_ss_matrix[('S', 'S')],
        mat_oo=default_ss_matrix[('O', 'O')],
        mat_hs=default_ss_matrix[('H', 'S')],
        mat_ho=default_ss_matrix[('H', 'O')],
        mat_so=default_ss_matrix[('S', 'O')]):
    """wrapper for command-line command (friendlier args)"""

    # 'to' only needed to sidestep problem with adjacent atom specs...
    ref_atoms = to

    from chimerax.seqalign import sim_matrices
    if matrix not in sim_matrices.matrices(session):
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
    refs = refs.subtract(matches)
    if not matches:
        raise UserError("Must use different reference and match structures")
    if pairing == CP_SPECIFIC_SPECIFIC:
        if len(refs) != len(matches):
            raise UserError("Different number of reference/match"
                    " chains (%d ref, %d match)" % (len(refs), len(matches)))
        match_items = zip(refs, matches)
    else:
        match_items = (refs[0], matches)
    ss_matrix = {}
    ss_matrix[('H', 'H')] = float(mat_hh)
    ss_matrix[('S', 'S')] = float(mat_ss)
    ss_matrix[('O', 'O')] = float(mat_oo)
    ss_matrix[('H', 'S')] = ss_matrix[('S', 'H')] = float(mat_hs)
    ss_matrix[('H', 'O')] = ss_matrix[('O', 'H')] = float(mat_ho)
    ss_matrix[('S', 'O')] = ss_matrix[('O', 'S')] = float(mat_so)
    if type(iterate) == bool and not iterate:
        iterate = None
    ret_vals = match(session, pairing, match_items, matrix, alg, gap_open, gap_extend,
        ss_fraction=ss_fraction, ss_matrix=ss_matrix,
        iterate=iterate, show_alignment=show_alignment,
        domain_residues=(ref_atoms.residues.unique(), match_atoms.residues.unique()),
        gap_open_helix=hgap, gap_open_strand=sgap,
        gap_open_other=ogap, compute_ss=compute_ss, verbose=verbose)
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
        from chimerax.core.atomic import StructureSeq
        for chain in chains:
            this_chain = set([r for r in chain.residues if r])
            if this_chain.issubset(sel_residues):
                new_chains.append(chain)
                continue
            nc = StructureSeq(structure=chain.structure, chain_id=chain.chain_id)
            nc._dm_rebuild_info = []
            _dm_cleanup.append(nc)
            new_chains.append(nc)
            chars = []
            residues = []
            for c, r in zip(chain.characters, chain.residues):
                if r in sel_residues:
                    chars.append(c)
                    residues.append(r)
                else:
                    nc._dm_rebuild_info.append((len(nc.residues), c, r))
                    chars.append('?')
                    residues.append(None)
            nc.bulk_set(residues, chars)
        chains = new_chains
    return chains

_registered = False
def register_command():
    global _registered
    if _registered:
        # registration can be called for both main command and alias, so only do once...
        return
    _registered = True
    from chimerax.core.commands \
        import CmdDesc, register, AtomsArg, FloatArg, StringArg, BoolArg, create_alias, Or
    desc = CmdDesc(
        required = [('match_atoms', AtomsArg)],
        required_arguments = ['to'],
        keyword = [('to', AtomsArg), ('pairing', StringArg), ('alg', StringArg),
            ('verbose', BoolArg), ('ss_fraction', Or(FloatArg, BoolArg)), ('matrix', StringArg),
            ('gap_open', FloatArg), ('hgap', FloatArg), ('sgap', FloatArg), ('ogap', FloatArg),
            ('iterate', Or(FloatArg, BoolArg)), ('gap_extend', FloatArg),
            ('show_alignment', BoolArg), ('compute_ss', BoolArg), ('mat_hh', FloatArg),
            ('mat_ss', FloatArg), ('mat_oo', FloatArg), ('mat_hs', FloatArg),
            ('mat_ho', FloatArg), ('mat_so', FloatArg)]
    )
    register('matchmaker', desc, cmd_match)
    create_alias('mmaker', "%s $*" % 'matchmaker')
