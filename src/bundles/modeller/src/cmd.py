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
def sequence_model(session, targets, *, block=None, combined_templates=False, custom_script=None,
    dist_restraints=None, executable_location=None, fast=False, het_preserve=False,
    hydrogens=False, license_key=None, num_models=5, show_gui=True, temp_path=None, thorough_opt=False,
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
    if block is None:
        block = not session.ui.is_gui
    from .settings import get_settings
    settings = get_settings(session)
    if license_key is None:
        license_key = settings.license_key
    else:
        settings.license_key = license_key
    from . import comparitive
    try:
        comparitive.model(session, targets, block=block, combined_templates=combined_templates,
            custom_script=custom_script, dist_restraints=dist_restraints,
            executable_location=executable_location, fast=fast, het_preserve=het_preserve,
            hydrogens=hydrogens, license_key=license_key, num_models=num_models, show_gui=show_gui,
            temp_path=temp_path, thorough_opt=thorough_opt, water_preserve=water_preserve)
    except comparitive.ModelingError as e:
        raise UserError(e)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ListOf, BoolArg, PasswordArg, IntArg
    from chimerax.core.commands import OpenFileNameArg, OpenFolderNameArg
    from chimerax.seqalign import AlignSeqPairArg
    desc = CmdDesc(
        required = [('targets', ListOf(AlignSeqPairArg))],
        keyword = [('block', BoolArg), ('combined_templates', BoolArg), ('custom_script', OpenFileNameArg),
            ('dist_restraints', OpenFileNameArg), ('executable_location', OpenFileNameArg), ('fast', BoolArg),
            ('het_preserve', BoolArg), ('hydrogens', BoolArg), ('license_key', PasswordArg),
            ('num_models', IntArg), ('show_gui', BoolArg), ('temp_path', OpenFolderNameArg),
            ('thorough_opt', BoolArg), ('water_preserve', BoolArg)
        ],
        synopsis = 'Use Modeller to generate comparitive model'
    )
    register('modeller comparitive', desc, sequence_model, logger=logger)
