# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
import os

def model(session, targets, *, block=True, multichain=True, custom_script=None,
          dist_restraints=None, executable_location=None, fast=False, het_preserve=False,
          hydrogens=False, license_key=None, num_models=5, temp_path=None,
          thorough_opt=False, water_preserve=False):
    """
    Generate comparative models for the target sequences.

    Arguments:
    session
        current session
    targets
        list of (alignment, sequence) tuples.  Each sequence will be modelled.
    block
        If True, wait for modelling job to finish before returning and return list of
        (opened) models.  Otherwise return immediately.
    multichain
        If True, the associated chains of each structure are used individually to generate
        chains in the resulting models (i.e. the models will be multimers).  If False, all
        associated chains are used together as templates to generate a single-chain model
        for the target sequence.
    custom_script
        If provided, the location of a custom Modeller script to use instead of the
        one we would otherwise generate.  Only used when executing locally.
    dist_restraints
        If provided, the location of a file containing additional distance restraints
    executable_location
        If provided, the path to the locally installed Modeller executable.  If not
        provided, use the web service.
    fast
        Whether to use fast but crude generation of models
    het_preserve
        Whether to preserve HET atoms in generated models
    hydrogens
        Whether to generate models with hydrogen atoms
    license_key
        Modeller license key.  If not provided, try to use settings to find one.
    num_models
        Number of models to generate for each template sequence
    temp_path
        If provided, folder to use for temporary files
    thorough_opt
        Whether to perform thorough optimization
    water_preserve
        Whether to preserve water in generated models
    """

    from chimerax.core.errors import LimitationError, UserError
    from .common import (
        modeller_copy, opal_safe_file_name, structure_save_name, chain_save_name
    )
    if multichain:
        # So, first find structure with most associated chains and least non-associated chains.
        # That structure is used as the multimer template.  Chains from other structures are used
        # as "standalone" templates -- each such chain will be on its own line.  Need to allow
        # space on the left and right of the target sequence so that the largest chains can be
        # accomodated.

        # Find the structure we will use as the multimer template
        by_structure = {}
        chain_info = {}
        for alignment, orig_target in targets:
            # Copy the target sequence, changing name to conform to Modeller limitations
            target = modeller_copy(orig_target)
            if not alignment.associations:
                raise UserError("Alignment %s has no associated chains" % alignment.ident)
            for chain, aseq in alignment.associations.items():
                if len(chain.chain_id) > 1:
                    raise LimitationError("Modeller cannot handle templates with multi-character chain IDs")
                by_structure.setdefault(chain.structure, []).append(chain)
                chain_info[chain] = (aseq, target, alignment.match_maps[aseq][chain])
        max_matched = min_unmatched = None
        for s, match_info in by_structure.items():
            matched = len(match_info)
            unmatched = s.num_chains - len(match_info)
            if(
               max_matched is None
               or matched > max_matched
               or (matched == max_matched and (unmatched < min_unmatched))
               ):
                multimer_template = s
                max_matched = matched
                min_unmatched = unmatched
        mm_targets = []
        mm_chains = []
        match_chains = []
        for chain in multimer_template.chains:
            mm_chains.append(chain)
            try:
                aseq, target, match_maps = chain_info[chain]
            except KeyError:
                mm_targets.append(None)
            else:
                mm_targets.append(target)
                match_chains.append(chain)
        # okay, now form single-chain lines for the other structure associations, that eventually will
        # be handled column by column in exactly the same way as the non-multichain method.
        single_template_lines = []
        for chain, info in chain_info.items():
            if chain.structure == multimer_template:
                continue
            aseq, target, match_map = info
            for i, mm_target in enumerate(mm_targets):
                if mm_target != target:
                    continue
                template_line = [None] * len(mm_targets)
                template_line[i] = chain
                single_template_lines.append(template_line)
        # The multimer template chain sequences need to have the complete PDB sequence (of existing
        # residues) even for parts of the structure not associated with the target, so may need to
        # add gaps to the target sequence to allow for that
        target_name = "target" if len(targets) > 1 else targets[0][1].name
        target_strings, templates_strings = find_target_template_strings(mm_chains, single_template_lines,
            mm_targets, chain_info)
        templates_info = [None]
        for template_line in single_template_lines:
            for chain, target in zip(template_line, mm_targets):
                if target is not None and chain is not None:
                    aseq, _, match_map = chain_info[chain]
                    templates_info.append((chain, match_map))
    else:
        if len(targets) > 1:
            raise LimitationError("Cannot have multiple targets(/alignments) unless creating multimeric model")
        alignment, orig_target = targets[0]
        # Copy the target sequence, changing name to conform to Modeller limitations
        target = modeller_copy(orig_target)
        target_name = target.name
        chain_info = {}
        templates = []
        templates_info = []
        match_chains = []
        for chain, aseq in alignment.associations.items():
            if len(chain.chain_id) > 1:
                raise LimitationError("Modeller cannot handle templates with multi-character chain IDs")
            match_map = alignment.match_maps[aseq][chain]
            chain_info[chain] = (aseq, target, match_map)
            templates.append(chain)
            templates_info.append((chain, match_map))
            if not match_chains:
                match_chains.append(chain)
        target_string, template_info = form_strings(target, templates, chain_info)
        target_strings = [target_string]
        templates_strings = []
        for template in templates:
            templates_strings.append([''.join(template_info[template][0])])

    if het_preserve or water_preserve:
        for template_strings in templates_strings:
            if len(template_strings) > 1:
                session.logger.warning("Cannot preserve water/het with more than one template per target;"
                                       " not preserving")
                het_preserve = water_preserve = False
                break

    if het_preserve or water_preserve:
        # add water/het characters to strings
        for i, target_string in enumerate(target_strings):
            template_info = templates_info[i]
            if template_info is None:
                template = mm_chains[0]
            else:
                template = template_info[0]
            template_string = templates_strings[i][0]
            het_string = ('.' if het_preserve else '-') * count_hets(template)
            target_string += het_string
            template_string += het_string
            if water_preserve:
                water_string = 'w' * count_water(template)
                target_string += water_string
                template_string += water_string
            target_strings[i] = target_string
            templates_strings[i] = [template_string]

    from .common import write_modeller_scripts, get_license_key, save_template
    _license_key = get_license_key(session, license_key)
    script_path, config_path, temp_dir = write_modeller_scripts(_license_key,
                                                                num_models, het_preserve, water_preserve,
                                                                hydrogens, fast, None, custom_script, temp_path,
                                                                thorough_opt, dist_restraints)
    config_as_json = {
            "key": _license_key
            , "version": 2
            , "numModels": num_models
            , "hetAtom": het_preserve
            , "water": water_preserve
            , "allHydrogen": hydrogens
            , "veryFast": fast
            , "loopInfo": None
    }
    input_file_map = []

    # form the sequences to be written out as a PIR
    from chimerax.atomic import Sequence
    pir_target = Sequence(name=opal_safe_file_name(target_name))
    pir_target.description = "sequence:%s:.:.:.:.::::" % pir_target.name
    pir_target.characters = '/'.join(target_strings)
    pir_seqs = [pir_target]

    structures_to_save = set()
    for strings, info in zip(templates_strings, templates_info):
        if info is None:
            # multimer template
            pir_template = Sequence(name=structure_save_name(multimer_template))
            pir_template.description = (
                "structure:%s:FIRST:%s:LAST:%s::::" %
                (pir_template.name, multimer_template.chains[0].chain_id, multimer_template.chains[-1].chain_id)
            )
            structures_to_save.add(multimer_template)
        else:
            # single-chain template
            chain, match_map = info
            first_assoc_pos = 0
            while first_assoc_pos not in match_map:
                first_assoc_pos += 1
            first_assoc_res = match_map[first_assoc_pos]
            assoc_length = 0
            for string in strings:
                length = len(string.replace('-', ''))
                if length > assoc_length:
                    assoc_length = length
            pir_template = Sequence(name=chain_save_name(chain))
            pir_template.description = "structure:%s:%d%s:%s:+%d:%s::::" % (
                structure_save_name(chain.structure), first_assoc_res.number, first_assoc_res.insertion_code,
                chain.chain_id, assoc_length, chain.chain_id)
            structures_to_save.add(chain.structure)
        pir_template.characters = '/'.join(strings)
        pir_seqs.append(pir_template)
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

    # save structure files
    struct_dir = os.path.join(temp_dir.name, "template_struc")
    if not os.path.exists(struct_dir):
        try:
            os.mkdir(struct_dir, mode=0o755)
        except FileExistsError:
            pass
    from chimerax.pdb import standard_polymeric_res_names as std_res_names
    for structure in structures_to_save:
        base_name = structure_save_name(structure) + '.pdb'
        pdb_file_name = os.path.join(struct_dir, base_name)
        input_file_map.append((base_name, "text_file", pdb_file_name))
        ATOM_res_names = structure.in_seq_hets
        ATOM_res_names.update(std_res_names)
        save_template(session, pdb_file_name, structure, ATOM_res_names)
        delattr(structure, 'in_seq_hets')

    from chimerax.atomic import Chains
    match_chains = Chains(match_chains)
    if executable_location is None:
        if custom_script is not None:
            raise LimitationError("Custom Modeller scripts only supported when executing locally")
        if dist_restraints is not None:
            raise LimitationError("Distance restraints only supported when executing locally")
        if thorough_opt:
            session.logger.warning("Thorough optimization only supported when executing locally")
        from .common import ModellerWebService
        job_runner = ModellerWebService(session, match_chains, num_models,
                                        pir_target.name, input_file_map, config_as_json, temp_dir, targets)
    else:
        # a custom script [only used when executing locally] needs to be copied into the tmp dir...
        if(
            os.path.exists(script_path)
            and os.path.normpath(temp_dir.name) != os.path.normpath(os.path.dirname(script_path))
           ):
            import shutil
            shutil.copy(script_path, temp_dir.name)
        from .common import ModellerLocal
        job_runner = ModellerLocal(session, match_chains, num_models,
                                   pir_target.name, executable_location,
                                   os.path.basename(script_path), targets, temp_dir)

    return job_runner.run(block=block)

def count_hets(chain):
    last_chain_res = chain.existing_residues[-1]
    end_located = False
    het_count = 0
    for r in chain.structure.residues:
        if end_located:
            if r.chain:
                break
            if r.name in r.water_res_names:
                break
            het_count += 1
        else:
            if r == last_chain_res:
                end_located = True
    return het_count

def count_water(chain):
    last_chain_res = chain.existing_residues[-1]
    end_located = past_hets = False
    water_count = 0
    for r in chain.structure.residues:
        if end_located:
            if r.chain:
                break
            if past_hets:
                if r.name in r.water_res_names:
                    water_count += 1
                else:
                    break
            else:
                if r.name in r.water_res_names:
                    water_count += 1
                    past_hets = True
        else:
            if r == last_chain_res:
                end_located = True
    return water_count

def find_target_template_strings(mm_chains, single_template_lines, mm_targets, chain_info):
    mm_template_strings = []
    single_strings = [[] for i in range(len(single_template_lines))]
    target_strings = []
    for i, target in enumerate(mm_targets):
        mm_chain = mm_chains[i]
        if target is None:
            target_strings.append('-' * len(mm_chain))
            tmp_str = "".join([c if r else '-' for c, r in zip(mm_chain.characters, mm_chain.residues)])
            mm_template_strings.append(tmp_str)
            for single_string in single_strings:
                single_string.append('-')
            continue
        templates = [mm_chain]
        for stl in single_template_lines:
            stl_template = stl[i]
            if stl_template:
                templates.append(stl_template)
        target_string, template_info = form_strings(target, templates, chain_info)
        target_strings.append(target_string)
        mm_template_strings.append(''.join(template_info[mm_chain][0]))
        for stl, strings in zip(single_template_lines, single_strings):
            stl_template = stl[i]
            if stl_template:
                strings.append(''.join(template_info[stl_template][0]))
            else:
                strings.append('-' * len(mm_template_strings[-1]))
    return target_strings, [mm_template_strings] + single_strings


class TemplatesFinished(ValueError):
    pass

def form_strings(target, templates, chain_info):
    target_index = -1
    template_state = {}
    target_chars = []
    for template in templates:
        aseq, target, match_map = chain_info[template]
        template_state[template] = ([], 0, match_map)
        update_in_seq_hets(template)
    while True:
        try:
            next_index = find_next_target_index(template_state)
        except TemplatesFinished:
            while target_index < len(target)-1:
                target_index += 1
                target_chars.append(target[target_index])
                for chars, *vals in template_state.values():
                    chars.append('-')
            break

        # adjust target string
        if next_index is None:
            target_chars.append('-')
        else:
            while target_index < next_index-1:
                target_index += 1
                target_chars.append(target[target_index])
                for chars, *vals in template_state.values():
                    chars.append('-')
            target_index += 1
            target_chars.append(target[target_index])

        # adjust template strings
        for template, template_info in template_state.copy().items():
            chars, template_index, mmap = template_info
            if template_index >= len(template):
                chars.append('-')
                continue
            r = template.residues[template_index]
            if r:
                check_index = mmap[r] if r in mmap else None
                if check_index is None:
                    assert(next_index is None)
                    chars.append(template.characters[template_index])
                    template_state[template] = (chars, template_index+1, mmap)
                else:
                    if next_index is None or check_index > next_index:
                        chars.append('-')
                    else:
                        char = template.characters[template_index]
                        chars.append(char if char != 'X' else '.')
                        template_state[template] = (chars, template_index+1, mmap)
            else:
                assert(next_index is None)
                # Modeller only wants characters of existing template residues...
                chars.append('-')
                template_state[template] = (chars, template_index+1, mmap)

    return ''.join(target_chars), template_state

def find_next_target_index(template_state):
    next_index = False
    for template, template_info in template_state.items():
        chars, template_index, mmap = template_info
        if template_index >= len(template):
            continue
        r = template.residues[template_index]
        if r:
            check_index = mmap[r] if r in mmap else None
            if check_index is None:
                return None
            if next_index is False or check_index < next_index:
                next_index = check_index
        else:
            return None
    if next_index is False:
        raise TemplatesFinished("done!")
    return next_index

def update_in_seq_hets(chain):
    from chimerax.pdb import standard_polymeric_res_names as std_res_names
    in_seq_hets = []
    for r in chain.existing_residues:
        if r.name not in std_res_names:
            in_seq_hets.append(r.name)
    s = chain.structure
    het_set = getattr(s, 'in_seq_hets', set())
    het_set.update(in_seq_hets)
    s.in_seq_hets = het_set
