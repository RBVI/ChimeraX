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

ALL_MISSING = "all-missing"
INTERNAL_MISSING = "internal-missing"

special_region_values = [ALL_MISSING, INTERNAL_MISSING]
protocols = ['standard', 'DOPE', 'DOPE-HR']

def model(session, targets, *, adjacent_flexible=1, block=True, chains=None, executable_location=None,
          fast=False, license_key=None, num_models=5, protocol=None, temp_path=None):
    """
    Model or remodel parts of structure, typically missing structure regions.

    Arguments:
    session
        current session
    targets
        What parts of the structures associated with a sequence (or sequences) to remodel.  It should
        be a list of (alignment, sequence, indices) tuples.  The indices should be a list of two-tuples
        of (start, end) zero-based indices into the ungapped sequence.  Alternatively, "indices" can
        be one of the string values from special_region_values above to remodel all missing structure
        or non-terminal missing structure associated with the sequence.
    adjacent_flexible
        How many residues adjacent to the remodelled region(s) on each side to allow to be adjusted
        to accomodate the remodelled segment(s).  Can be zero.
    block
        If True, wait for modelling job to finish before returning and return list of (opened) models.
        Otherwise return immediately.
    chains
        If specified, the particular chains associated with the sequence to remodel.  If omitted, all
        associated chains will be remodeled.
    executable_location
        If provided, the path to the locally installed Modeller executable.  If not provided, use the
        web service.
    fast
        Whether to use fast but crude generation of models
    license_key
        Modeller license key.  If not provided, try to use settings to find one.
    num_models
        Number of models to generate
    protocol
        Loop-modeling refinement method.  One of: standard, DOPE, or DOPE-HR. The value of None is
        treated as "standard".
    temp_path
        If provided, folder to use for temporary files
    """

    from chimerax.core.errors import UserError # , LimitationError

    by_structure = {}
    chain_indices = {}
    for alignment, seq, region_info in targets:
        model_chains = set(seq.match_maps.keys())
        if not model_chains:
            raise UserError("No chains/structures associated with sequence %s" % seq.name)
        if chains:
            model_chains = [chain for chain in chains if chain in model_chains]
        if not model_chains:
            raise UserError("Specified chains not associated with sequence %s" % seq.name)
        for chain in model_chains:
            by_structure.setdefault(chain.structure, []).append((chain, seq))

        for chain in model_chains:
            if region_info == ALL_MISSING:
                chain_indices[chain] = find_missing(chain, seq, False)
            elif region_info == INTERNAL_MISSING:
                chain_indices[chain] = find_missing(chain, seq, True)
            else:
                chain_indices[chain] = region_info
    # MAV: loop_data = (protocol, chain_indices[chain], seq, template_models)

    from .common import regularized_seq
    for s, s_chain_info in by_structure.items():
        chain_map = {chain: seq for chain, seq in s_chain_info}
        # Go through the residues of the structure: preserve het/water; for chains being modeled
        # append the complete sequence; for others append the appropriate number of '-' characters
        template_chars = []
        target_chars = []
        i = 0
        residues = s.residues
        chain_id = None
        target_offsets = {}
        offset_i = 0
        match_chains = []
        renumberings = []
        while i < len(residues):
            r = residues[i]
            if chain_id is None:
                chain_id = r.chain_id
            elif chain_id != r.chain_id:
                # Modeller apparently "keeps going" for water/het with the same chain ID,
                # but "stops" for water/het in its own chain
                if r.chain is None:
                    break
                template_chars.append('/')
                target_chars.append('/')
                chain_id = r.chain_id
            if r.chain is None:
                if r.name in r.water_res_names:
                    template_chars.append('w')
                    target_chars.append('w')
                else:
                    template_chars.append('.')
                    target_chars.append('.')
                i += 1
                offset_i += 1
            else:
                try:
                    seq = chain_map[r.chain]
                except KeyError:
                    existing = "".join([c for c, r in zip(r.chain.characters, r.chain.residues) if r])
                    template_chars.append(existing)
                    target_chars.append('-' * len(existing))
                    offset_i += len(existing)
                else:
                    prefix, suffix = [ret[0] for ret in find_affixes([r.chain], {r.chain: (seq, None)})]
                    chain_template_chars = prefix + regularized_seq(seq, r.chain).characters + suffix
                    template_chars.append(chain_template_chars)
                    # prevent Modeller from filling in unmodelled missing structure by using '-'
                    chain_target_chars = []
                    seq_chars = seq.characters
                    modeled = set()
                    # 'chain_indices' are into the *sequence*
                    for start, end in chain_indices[r.chain]:
                        modeled.update(range(start, end + 1))
                    for seq_i in range(len(seq_chars)):
                        if chain_template_chars[seq_i] == '-' and seq_i not in modeled:
                            target_char = '-'
                        else:
                            target_char = seq_chars[seq_i]
                        chain_target_chars.append(target_char)
                    renumberings.append(make_renumbering("".join(chain_target_chars), chain_template_chars,
                                        r.chain))
                    target_chars.extend(chain_target_chars)
                    target_offsets[r.chain] = offset_i
                    offset_i += len(chain_target_chars)
                    match_chains.append(r.chain)
                i += r.chain.num_existing_residues
        target_chars = ''.join(target_chars)
        compact_target_chars = target_chars.replace('/', '')

        # find the actual loop-boundaries, which needs to account for flexible extension and
        # ensure that the bounding residues actually exist
        loop_data = []
        for chain, seq in chain_map.items():
            mmap = seq.match_maps[chain]
            for start, end in chain_indices[chain]:
                start = max(start - adjacent_flexible, 0)
                while start > 0 and start-1 not in mmap:
                    start -= 1
                seq_len = len(seq)
                end = min(end + adjacent_flexible, seq_len - 1)
                while end < seq_len - 1 and end+1 not in mmap:
                    end += 1
                offset = target_offsets[chain]
                # for residue indexing, unmodeled residues (target == '-') don't count...
                unmodeled = compact_target_chars[:start + offset].count('-')

                # Modeller residue_range()'s end index is the actual index, not index+1
                loop_data.append((start + offset - unmodeled, end + offset - unmodeled))
        loop_mod_prefix = {"standard": "", "DOPE": "dope_", "DOPE-HR": "dopehr_", None: ""}[protocol]

        from .common import write_modeller_scripts, get_license_key
        _license_key = get_license_key(session, license_key)
        script_path, config_path, temp_dir = write_modeller_scripts(_license_key,
                                                                    num_models, True, True, False, fast,
                                                                    (loop_mod_prefix, loop_data), None,
                                                                    temp_path, False, None)
        config_as_json = {
            "key": _license_key
            , "version": 2
            , "numModels": num_models
            , "hetAtom": True
            , "water": True
            , "allHydrogen": False
            , "veryFast": fast
            , "loopInfo": (loop_mod_prefix, loop_data)
        }
        input_file_map = []

        # form the sequences to be written out as a PIR
        from .common import opal_safe_file_name, structure_save_name
        from chimerax.atomic import Sequence
        pir_target = Sequence(name=opal_safe_file_name(seq.name))
        pir_target.description = "sequence:%s:.:.:.:.::::" % pir_target.name
        pir_target.characters = target_chars
        pir_seqs = [pir_target]

        pir_template = Sequence(name=structure_save_name(s))
        pir_template.description = "structure:%s:FIRST:%s:LAST:%s::::" % (
            pir_template.name, residues[0].chain_id, residues[-1].chain_id)
        pir_template.characters = ''.join(template_chars)
        pir_seqs.append(pir_template)

        import os
        pir_file = os.path.join(temp_dir.name, "alignment.ali")
        aln = session.alignments.new_alignment(pir_seqs, False, auto_associate=False, create_headers=False)
        aln.save(pir_file, format_name="pir")
        session.alignments.destroy_alignment(aln)
        input_file_map.append(("alignment.ali", "text_file", pir_file))

        # write the namelist.dat file, target seq name on first line, templates on remaining lines
        name_file = os.path.join(temp_dir.name, "namelist.dat")
        input_file_map.append(("namelist.dat", "text_file", name_file))
        with open(name_file, 'w') as f:
            for template_seq in pir_seqs:
                print(template_seq.name, file=f)

        config_name = os.path.basename(config_path)
        input_file_map.append((config_name, "text_file", config_path))

        # save structure file
        struct_dir = os.path.join(temp_dir.name, "template_struc")
        if not os.path.exists(struct_dir):
            try:
                os.mkdir(struct_dir, mode=0o755)
            except FileExistsError:
                pass
        from chimerax.pdb import save_pdb, standard_polymeric_res_names as std_res_names
        base_name = structure_save_name(s) + '.pdb'
        pdb_file_name = os.path.join(struct_dir, base_name)
        input_file_map.append((base_name, "text_file", pdb_file_name))
        ATOM_res_names = s.in_seq_hets
        ATOM_res_names.update(std_res_names)
        save_pdb(session, pdb_file_name, models=[s], polymeric_res_names=ATOM_res_names)
        delattr(s, 'in_seq_hets')

        from chimerax.atomic import Chains
        match_chains = Chains(match_chains)
        if executable_location is None:
            from .common import ModellerWebService
            job_runner = ModellerWebService(session, match_chains, num_models,
                                            pir_target.name, input_file_map,
                                            config_as_json, temp_dir, [t[:2] for t in targets],
                                            res_numberings=renumberings)
        else:
            from .common import ModellerLocal
            job_runner = ModellerLocal(session, match_chains, num_models, pir_target.name,
                                       executable_location, script_path,
                                       [t[:2] for t in targets], temp_dir, loop_job=True)

        job_runner.run(block=block)
    return

def find_missing(chain, seq, internal_only):
    match_map = seq.match_maps[chain]
    missing = []
    start_missing = None
    for i in range(len(seq)):
        if i in match_map:
            if start_missing is not None:
                if not internal_only or start_missing > 0:
                    missing.append((start_missing, i - 1))
                start_missing = None
        else:
            if start_missing is None:
                start_missing = i
    if start_missing is not None and not internal_only:
        missing.append((start_missing, len(seq) - 1))
    return missing

def find_affixes(chains, chain_info):
    from chimerax.pdb import standard_polymeric_res_names as std_res_names
    in_seq_hets = []
    prefixes = []
    suffixes = []
    from chimerax.atomic import Sequence
    for chain in chains:
        try:
            aseq, target = chain_info[chain]
        except KeyError:
            prefixes.append('')
            suffixes.append('')
            continue
        match_map = aseq.match_maps[chain]
        prefix = ''
        for r in chain.existing_residues:
            if r in match_map:
                break
            if r.name not in std_res_names:
                in_seq_hets.append(r.name)
                prefix += '.'
            else:
                prefix += Sequence.rname3to1(r.name)
        prefixes.append(prefix)

        suffix = ''
        for r in reversed(chain.existing_residues):
            if r in match_map:
                break
            if r.name not in std_res_names:
                in_seq_hets.append(r.name)
                suffix = '.' + suffix
            else:
                suffix = Sequence.rname3to1(r.name) + suffix
        suffixes.append(suffix)
    s = chain.structure
    het_set = getattr(s, 'in_seq_hets', set())
    het_set.update(in_seq_hets)
    s.in_seq_hets = het_set
    return prefixes, suffixes

def make_renumbering(target_chars, template_chars, chain):
    new_nums = []
    new_inserts = []
    existing = chain.existing_residues
    existing_index = -1
    if template_chars[0] == '-':
        num_dashes = len(template_chars) - len(template_chars.lstrip('-'))
        template_res_num = existing[0].number - num_dashes - 1
        template_insert = ''

    for target_char, template_char in zip(target_chars, template_chars):
        if template_char == '-':
            template_res_num += 1
            template_insert = ''
        else:
            existing_index += 1
            template_res_num = existing[existing_index].number
            template_insert = existing[existing_index].insertion_code

        if target_char != '-':
            new_nums.append(template_res_num)
            new_inserts.append(template_insert)
    return (new_nums, new_inserts)
