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

ALL_MISSING = "all-missing"
NT_MISSING = "non-terminal-missing"

special_region_values = [ALL_MISSING, NT_MISSING]

def model(session, targets, *, adjacent_flexible=1, block=True, chains=None, executable_location=None,
    license_key=None, num_models=5, protocol=None, show_gui=True, temp_path=None):
    """
    Model or remodel parts of structure, typically missing structure regions.

    Arguments:
    session
        current session
    targets
        What parts of the structures associated with a sequence to remodel.  It should be an
        (alignment, sequence, indices) tuple.  The indices should be a list of two-tuples of
        (start, end) Python-style indices into the ungapped sequence.  Alternatively, "indices" can be
        one of the string values from special_region_values above to remodel all missing structure
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
    license_key
        Modeller license key.  If not provided, try to use settings to find one.
    num_models
        Number of models to generate
    protocol
        Loop-modeling refinement method.  One of: standard, DOPE, or DOPE-HR.
    show_gui
        If True, show user interface for Modeller results (if ChimeraX is in gui mode).
    temp_path
        If provided, folder to use for temporary files
    """

    from chimerax.core.errors import LimitationError, UserError

    alignment, seq, region_info = targets
    model_chains = set(seq.match_maps.keys())
    if not model_chains:
        raise UserError("No chains/structures associated with sequence %s" % seq.name)
    if chains:
        model_chains = [chain for chain in chains if chain in model_chains]
    if not model_chains:
        raise UserError("Specified chains not associated with sequence %s" % seq.name)
    by_structure = {}
    for chain in model_chains:
        by_structure.setdefault(chain.structure, []).append(chain)

    chain_indices = {}
    for chain in model_chains:
        if region_info == ALL_MISSING:
            chain_indices[chain] = find_missing(chain, seq.match_maps[chain], False)
        elif region_info == NT_MISSING:
            chain_indices[chain] = find_missing(chain, seq.match_maps[chain], True)
        else:
            chain_indices[chain] = region_info
    #MAV: loop_data = (protocol, chain_indices[chain], seq, template_models)

    for s, s_chains in by_structure.items():
        # Go through the residues of the structure: preserve het/water; for chains being modeled
        # append the complete sequence; for others append the appropriate number of '-' characters
        chars = []
        i = 0
        residues = s.residues
        chain_id = None
        while i < len(residues):
            r = residues[i]
            if chain_id is None:
                chain_id = r.chain_id
            elif chain_id != r.chain_id:
                chars.append('/')
                chain_id = r.chain_id
            if r.chain is None:
                if r.name in r.water_res_names:
                    chars.append('w')
                else:
                    chars.append('.')
                i += 1
            else:
                if r.chain in s_chains:
                    prefix, suffix = [ret[0] for ret in find_affixes([r.chain], { r.chain: (seq, None) })]
                    chars.append(prefix)
                    chars.append(r.chain.characters)
                    chars.append(suffix)
                else:
                    chars.append('-' * r.chain.num_existing_residues)
                i += r.chain.num_existing_residues
        print("seq for %s:" % s, ''.join(chars))
    return


    from .common import modeller_copy
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
                chain_info[chain] = (aseq, target)
        max_matched = min_unmatched = None
        for s, match_info in by_structure.items():
            matched = len(match_info)
            unmatched = s.num_chains - len(match_info)
            if max_matched is None or matched > max_matched or (matched == max_matched
                    and (unmatched < min_unmatched)):
                multimer_template = s
                max_matched = matched
                min_unmatched = unmatched
        mm_targets = []
        mm_chains = []
        match_chains = []
        for chain in multimer_template.chains:
            mm_chains.append(chain)
            try:
                aseq, target = chain_info[chain]
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
            aseq, target = info
            for i, mm_target in enumerate(mm_targets):
                if mm_target != target:
                    continue
                template_line = [None] * len(mm_targets)
                template_line[i] = chain
                single_template_lines.append(template_line)
        # AFAIK, the multimer template chain sequences need to have complete PDB sequence, so may need
        # to prefix and suffix he corresponding alignment sequence with characters for residues
        # outside of the alignment sequence.  For other templates/targets, affix a corresponding number
        # of '-' characters
        prefixes, suffixes = find_affixes(mm_chains, chain_info)
        target_strings = []
        for prefix, suffix, mm_target in zip(prefixes, suffixes, mm_targets):
            if mm_target is None:
                target_strings.append('-')
                continue
            target_strings.append('-' * len(prefix) + mm_target.characters + '-' * len(suffix))
        templates_strings = []
        templates_info = []
        mm_template_strings = []
        for prefix, suffix, chain in zip(prefixes, suffixes, mm_chains):
            try:
                aseq, target = chain_info[chain]
            except KeyError:
                mm_template_strings.append('-')
                continue
            mm_template_strings.append(prefix + regularized_seq(aseq, chain).characters + suffix)
        templates_strings.append(mm_template_strings)
        templates_info.append(None)
        for template_line in single_template_lines:
            template_strings = []
            for prefix, suffix, chain, target in zip(prefixes, suffixes, template_line, mm_targets):
                if target is None:
                    template_strings.append('-')
                elif chain is None:
                    template_strings.append('-' * (len(prefix) + len(target) + len(suffix)))
                else:
                    aseq, target = chain_info[chain]
                    template_strings.append('-' * len(prefix) + regularized_seq(aseq, chain).characters
                        + '-' * len(suffix))
                    templates_info.append((chain, aseq.match_maps[chain]))
            templates_strings.append(template_strings)
        target_name = "target" if len(targets) > 1 else target.name
    else:
        if len(targets) > 1:
            raise LimitationError("Cannot have multiple targets(/alignments) unless creating multimeric model")
        alignment, orig_target = targets[0]
        # Copy the target sequence, changing name to conform to Modeller limitations
        target = modeller_copy(orig_target)
        target_strings = [target.characters]

        templates_strings = []
        templates_info = []
        match_chains = []
        for chain, aseq in alignment.associations.items():
            if len(chain.chain_id) > 1:
                raise LimitationError("Modeller cannot handle templates with multi-character chain IDs")
            templates_strings.append([regularized_seq(aseq, chain).characters])
            templates_info.append((chain, aseq.match_maps[chain]))
            if not match_chains:
                match_chains.append(chain)

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

        target_name = target.name

    from .common import write_modeller_scripts, get_license_key
    script_path, config_path, temp_dir = write_modeller_scripts(get_license_key(session, license_key),
        num_models, het_preserve, water_preserve, hydrogens, fast, None, custom_script, temp_path,
        thorough_opt, dist_restraints)

    input_file_map = []

    # form the sequences to be written out as a PIR
    from chimerax.atomic import Sequence
    pir_target = Sequence(name=target_name)
    pir_target.description = "sequence:%s:.:.:.:.::::" % pir_target.name
    pir_target.characters = '/'.join(target_strings)
    pir_seqs = [pir_target]

    structures_to_save = set()
    for strings, info in zip(templates_strings, templates_info):
        if info is None:
            # multimer template
            pir_template = Sequence(name=structure_save_name(multimer_template))
            pir_template.description = "structure:%s:FIRST:%s::::::" % (
                pir_template.name, multimer_template.chains[0].chain_id)
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
    import os.path
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
    import os
    struct_dir = os.path.join(temp_dir.name, "template_struc")
    if not os.path.exists(struct_dir):
        try:
            os.mkdir(struct_dir, mode=0o755)
        except FileExistsError:
            pass
    from chimerax.pdb import save_pdb, standard_polymeric_res_names as std_res_names
    for structure in structures_to_save:
        base_name = structure_save_name(structure) + '.pdb'
        pdb_file_name = os.path.join(struct_dir, base_name)
        input_file_map.append((base_name, "text_file",  pdb_file_name))
        ATOM_res_names = structure.in_seq_hets
        ATOM_res_names.update(std_res_names)
        save_pdb(session, pdb_file_name, models=[structure], polymeric_res_names=ATOM_res_names)
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
        job_runner = ModellerWebService(session, match_chains, num_models,
            pir_target.name, input_file_map, config_name, targets, show_gui)
    else:
        #TODO: job_runner = ModellerLocal(...)
        from chimerax.core.errors import LimitationError
        raise LimitationError("Local Modeller execution not yet implemented")
        # a custom script [only used when executing locally] needs to be copied into the tmp dir...
        if os.path.exists(script_path) \
        and os.path.normpath(temp_dir.name) != os.path.normpath(os.path.dirname(script_path)):
            import shutil
            shutil.copy(script_path, temp_dir.name)

    return job_runner.run(block=block)

def find_missing(chain, match_map, terminal):
    missing = []
    start_missing = None
    for i in len(match_map):
        if match_map[i] is None:
            if start_missing is None:
                start_missing = i
        else:
            if start_missing is not None:
                if not terminal or start_missing == 0:
                    missing.append((start_missing, i))
                start_missing = None
    if start_missing is not None:
        missing.append((start_missing, len(match_map)))
    return missing

def regularized_seq(aseq, chain):
    mmap = aseq.match_maps[chain]
    from .common import modeller_copy
    rseq = modeller_copy(aseq)
    rseq.description = "structure:" + chain_save_name(chain)
    seq_chars = list(rseq.characters)
    from chimerax.atomic import Sequence
    from chimerax.pdb import standard_polymeric_res_names as std_res_names
    in_seq_hets = []
    num_res = 0
    for ungapped in range(len(aseq.ungapped())):
        gapped = aseq.ungapped_to_gapped(ungapped)
        if ungapped not in mmap:
            seq_chars[gapped] = '-'
        else:
            r = mmap[ungapped]
            num_res += 1
            if r.name not in std_res_names:
                in_seq_hets.append(r.name)
                seq_chars[gapped] = '.'
            else:
                seq_chars[gapped] = Sequence.rname3to1(mmap[ungapped].name)
    s = chain.structure
    het_set = getattr(s, 'in_seq_hets', set())
    # may want to preserve all-HET chains, so don't auto-exclude them
    if num_res != len(in_seq_hets):
        het_set.update(in_seq_hets)
    s.in_seq_hets = het_set
    rseq.characters = "".join(seq_chars)
    return rseq

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

def structure_save_name(s):
    return s.name.replace(':', '_').replace(' ', '_') + "_" + s.id_string

def chain_save_name(chain):
    return structure_save_name(chain.structure) + '/' + chain.chain_id.replace(' ', '_')

from .common import RunModeller
class ModellerWebService(RunModeller):

    def __init__(self, session, match_chains, num_models, target_seq_name, input_file_map, config_name,
            targets, show_gui):

        super().__init__(session, match_chains, num_models, target_seq_name, targets, show_gui)
        self.input_file_map = input_file_map
        self.config_name = config_name

        self.job = None

    def run(self, *, block=False):
        self.job = ModellerJob(self.session, self, self.config_name, self.input_file_map, block)

    def take_snapshot(self, session, flags):
        """For session/scene saving"""
        return {
            'base data': super().take_snapshot(session, flags),
            'input_file_map': self.input_file_map,
            'config_name': self.config_name,
        }

    @staticmethod
    def restore_snapshot(session, data):
        inst = ModellerWebService(session, None, None, None, data['input_file_map'], data['config_name'],
            None, None)
        inst.set_state_from_snapshot(data['base data'])

from chimerax.webservices.opal_job import OpalJob
class ModellerJob(OpalJob):

    OPAL_SERVICE = "Modeller9v8Service"
    SESSION_SAVE = True

    def __init__(self, session, caller, command, input_file_map, block):
        super().__init__(session)
        self.caller = caller
        self.start(self.OPAL_SERVICE, command, input_file_map=input_file_map, blocking=block)

    def monitor(self):
        super().monitor()
        stdout = self.get_file("stdout.txt")
        num_done = stdout.count('# Heavy relative violation of each residue is written to:')
        status = self.session.logger.status
        tsafe = self.session.ui.thread_safe
        if not num_done:
            tsafe(status, "No models generated yet")
        else:
            tsafe(status, "%d of %d models generated" % (num_done, self.caller.num_models))

    def next_check(self):
        return 15

    def on_finish(self):
        logger = self.session.logger
        logger.info("Modeller job ID %s finished" % self.job_id)
        if not self.exited_normally():
            err = self.get_file("stderr.txt")
            if self.fail_callback:
                self.fail_callback(self, err)
                return
            if err:
                raise RuntimeError("Modeller failure; standard error:\n" + err)
            else:
                raise RuntimeError("Modeller failure with no error output")
        try:
            model_info = self.get_file("ok_models.dat")
        except KeyError:
            try:
                stdout = self.get_file("stdout.txt")
                stderr = self.get_file("stderr.txt")
            except KeyError:
                raise RuntimeError("No output from Modeller")
            logger.info("<br><b>Modeller error output</b>", is_html=True)
            logger.info(stderr)
            logger.info("<br><b>Modeller run output</b>", is_html=True)
            logger.info(stdout)
            from chimerax.core.errors import NonChimeraError
            raise NonChimeraError("No output models from Modeller; see log for Modeller text output.")
        try:
            stdout = self.get_file("stdout.txt")
        except KeyError:
            raise RuntimeError("No standard output from Modeller job")
        def get_pdb_model(fname):
            from io import StringIO
            try:
                pdb_text = self.get_file(fname)
            except KeyError:
                raise RuntimeError("Could not find Modeller out PDB %s on server" % fname)
            from chimerax.pdb import open_pdb
            return open_pdb(self.session, StringIO(pdb_text), fname)[0][0]
        self.caller.process_ok_models(model_info, stdout, get_pdb_model)
        self.caller = None

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
