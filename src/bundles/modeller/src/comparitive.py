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

class ModelingError(ValueError):
    pass

def model(session, targets, combined_templates=False, custom_script=None,
    dist_restraints_path=None, executable_location=None, fast=False, het_preserve=False,
    hydrogens=False, license_key=None, num_models=5, temp_path=None, thorough_opt=False,
    water_preserve=False):
    """
    Generate comparitive models for the target sequences.

    Arguments:
    session
        current session
    targets
        list of (alignment, sequence) tuples.  Each sequence will be modelled.
    combined_templates
        If True, all associated chains are used together as templates to generate a single set
        of models for the target sequence.  If False, the associated chains are used individually
        to generate chains in the resulting models (i.e. the models will be multimers).
    custom_script
        If provided, the location of a custom Modeller script to use instead of the
        one we would otherwise generate.
    dist_restraints_path
        If provided, the path to a file containing additional distance restraints
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

    from chimerax.core.errors import LimitationError
    template_info = []
    for alignment, orig_target in targets:
        if not alignment.associations:
            raise ModelingError("Alignment %s has no associated chains to use as templates."
                % alignment.ident)
        # Copy the target sequence, changing name to conform to Modeller limitations
        from .common import modeller_copy
        target = modeller_copy(orig_target)
        if combined_templates:
            target_templates = []
            template_info.append((target, target_templates))
        for chain, aseq in alignment.associations.items():
            if len(chain.chain_id) > 1:
                raise LimitationError(
                    "Modeller cannot handle templates with multi-character chain IDs")
            if not combined_templates:
                target_templates = []
                template_info.append((target, target_templates))
            target_templates.append((regularized_seq(aseq, chain), chain, aseq.match_maps[chain]))

    # collate the template info in series of strings that can be joined with '/'
    target_strings = []
    templates_strings = []
    for target, templates_info in template_info:
        target_seq = target.characters
        target_strings.append(target_seq)
        target_template_strings = []
        templates_strings.append(target_template_strings)
        accum_water_het = ""
        for template, chain, match_map in templates_info:
            # match_map has the chain-to-aseq original match map
            # missing positions have already been changed to '-' in template
            end = chain.existing_residues[-1]
            template_string = template.characters + accum_water_het
            accum = ""
            if not het_preserve and not water_preserve:
                target_template_strings.append(template_string)
                continue
            # add het/water characters and get proper end residue
            before_end = True
            for r in chain.structure.residues:
                if before_end:
                    before_end = r != end
                    continue
                if r.chain_id != chain.chain_id or (r.chain and r.chain != chain):
                    break
                end = r
                if water_preserve and r.name in r.standard_water_names \
                or het_preserve and r.is_het:
                    char = '.'
                else:
                    char = '-'
                target_seq += char
                template_string += char
                accum_ += '-'
            accum_water_het += accum
            for i, tts in enumerate(target_template_strings):
                target_template_strings[i] = tts + accum
            target_template_strings.append(template_string)
    # Insert/append all-'-' strings so that each template is in it's own line
    insertions = []
    appendings = []
    for i in range(len(templates_strings)):
        insertions.append([])
        appendings.append([])
    for i, target_template_strings in enumerate(templates_strings):
        line_to_add = '-' * len(target_template_strings[0])
        for appending in appendings[:i]:
            appending.append(line_to_add)
        for insertion in insertions[i+1:]:
            insertion.append(line_to_add)
    """
    # just checking things out...
    session.logger.info("<pre> targ " + '/'.join(target_strings) + "</pre>", is_html=True) 
    for i, lines in enumerate(templates_strings):
        for line in lines:
            joined_line = line
            insertion = '/'.join(insertions[i])
            if insertion:
                joined_line = insertion + '/' + joined_line
            appending = '/'.join(appendings[i])
            if appending:
                joined_line = joined_line + '/' + appending
            session.logger.info("<pre> tmpl " + joined_line + "</pre>", is_html=True) 
    """
    # form the sequences to be written out as a PIR
    pir_seqs = []
    from chimerax.core.atomic import Sequence
    structures_to_save = set()
    for i, tmpl_strs in enumerate(templates_strings):
        for j, tmpl_str in enumerate(tmpl_strs):
            chain = template_info[i][1][j][1]
            pir_template = Sequence(name=chain_save_name(chain))
            pir_seqs.append(pir_template)
            pir_template.description = "structure:%s:FIRST:%s:+%d:%s::::" % (
                structure_save_name(chain.structure),
                chain.chain_id, len(tmpl_str) - tmpl_str.count('-'), chain.chain_id)
            structures_to_save.add(chain.structure)
            full_line = tmpl_str
            prefix = '/'.join(insertions[i])
            if prefix:
                full_line = prefix + '/' + full_line
            suffix = '/'.join(appendings[i])
            if suffix:
                full_line = full_line + '/' + suffix
            pir_template.characters = full_line
    pir_target = Sequence(name=template_info[0][0].name)
    pir_seqs.append(pir_target)
    pir_target.description = "sequence:%s:.:.:.:.::::" % pir_target.name
    pir_target.characters = '/'.join(target_strings)
    from tempfile import NamedTemporaryFile
    tf = NamedTemporaryFile(mode="w", suffix=".pir", delete=False)
    aln = session.alignments.new_alignment(pir_seqs, None, align_viewer=False, auto_associate=False)
    aln.save(tf, format_name="pir")
    session.alignments.destroy_alignment(aln)
    print("temp PIR saved to:", tf.name)
    #TODO: save structure files
    #TODO: ...
    #TODO: delete temp PIR file

def regularized_seq(aseq, chain):
    mmap = aseq.match_maps[chain]
    from .common import modeller_copy
    rseq = modeller_copy(aseq)
    rseq.descript = "structure:" + chain_save_name(chain)
    seq_chars = list(rseq.characters)
    from chimerax.core.atomic import Sequence
    for ungapped in range(len(aseq.ungapped())):
        gapped = aseq.ungapped_to_gapped(ungapped)
        if ungapped not in mmap:
            seq_chars[gapped] = '-'
        else:
            seq_chars[gapped] = Sequence.rname3to1(mmap[ungapped].name)
    rseq.characters = "".join(seq_chars)
    return rseq

def structure_save_name(s):
    return s.name.replace(':', '_').replace(' ', '_') + "_" + s.id_string()

def chain_save_name(chain):
    return structure_save_name(chain.structure) + '/' + chain.chain_id.replace(' ', '_')
