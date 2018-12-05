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

#
def sequence_model(session, targets, block=True, combined_templates=False, custom_script=None,
    dist_restraints_path=None, executable_location=None, fast=False, het_preserve=False,
    hydrogens=False, license_key=None, num_models=5, temp_path=None, thorough_opt=False,
    water_preserve=False):
    '''
    Command to generate a comparitive model of one or more chains
    '''
    from chimerax.core.errors import UserError
    seen = set()
    for alignment, seq in targets:
        if alignment in seen:
            raise UserError("Only one target sequence per alignent allowed;"
                " multiple targets chosen in alignment %s" % alignment)
        seen.add(alignment)
    #TODO: if license_key given, save in settings; otherwise try to retrieve from settings
    from . import comparitive
    try:
        comparitive.model(session, targets, block=block, combined_templates=combined_templates,
            custom_script=custom_script, dist_restraints_path=dist_restraints_path,
            executable_location=executable_location, fast=fast, het_preserve=het_preserve,
            hydrogens=hydrogens, license_key=license_key, num_models=num_models,
            temp_path=temp_path, thorough_opt=thorough_opt, water_preserve=water_preserve)
    except comparitive.ModelingError as e:
        raise UserError(e)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ListOf, BoolArg, PasswordArg, IntArg
    from chimerax.seqalign import AlignSeqPairArg
    #TODO: all the other args; make sure license_key uses PasswordArg
    desc = CmdDesc(
        required = [('targets', ListOf(AlignSeqPairArg))],
        keyword = [('block', BoolArg), ('combined_templates', BoolArg), ('license_key', PasswordArg),
            ('num_models', IntArg)],
        synopsis = 'Use Modeller to generate comparitive model'
    )
    register('modeller comparitive', desc, sequence_model, logger=logger)
