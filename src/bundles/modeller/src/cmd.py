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

def sequence_model(session, targets, *, block=None, multichain=True, custom_script=None,
    dist_restraints=None, executable_location=None, fast=False, het_preserve=False,
    hydrogens=False, license_key=None, num_models=5, show_gui=True, temp_path=None, thorough_opt=False,
    water_preserve=False):
    '''
    Command to generate a comparative model of one or more chains
    '''
    from chimerax.core.errors import UserError
    seen = set()
    for alignment, seq in targets:
        if alignment in seen:
            raise UserError("Only one target sequence per alignment allowed;"
                " multiple targets chosen in alignment %s" % alignment)
        seen.add(alignment)
    if block is None:
        block = session.in_script or not session.ui.is_gui
    if fast:
        num_models = 1
    from . import comparative, common
    try:
        comparative.model(session, targets, block=block, multichain=multichain,
            custom_script=custom_script, dist_restraints=dist_restraints,
            executable_location=executable_location, fast=fast, het_preserve=het_preserve,
            hydrogens=hydrogens, license_key=license_key, num_models=num_models, show_gui=show_gui,
            temp_path=temp_path, thorough_opt=thorough_opt, water_preserve=water_preserve)
    except common.ModelingError as e:
        raise UserError(e)

def model_loops(session, targets, *, adjacent_flexible=1, block=None, chains=None, executable_location=None,
    license_key=None, num_models=5, protocol=None, show_gui=True, temp_path=None):
    '''
    Command to model loops or refine structure regions
    '''
    from chimerax.core.errors import UserError
    if block is None:
        block = session.in_script or not session.ui.is_gui
    if chains is not None and not chains:
        raise UserError("'chains' argument doe not match any chains")
    from . import loops, common
    try:
        loops.model(session, targets, adjacent_flexible=adjacent_flexible, block=block, chains=chains,
            executable_location=executable_location, license_key=license_key, num_models=num_models,
            protocol=protocol, show_gui=show_gui, temp_path=temp_path)
    except common.ModelingError as e:
        raise UserError(e)

def score_models(session, structures, *, block=None, license_key=None, refresh=False):
    '''
    Fetch Modeller scores for models
    '''
    if block is None:
        block = session.in_script or not session.ui.is_gui
    from . import scores
    scores.fetch_scores(session, structures, block=block, license_key=license_key, refresh=refresh)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, create_alias, RepeatOf, BoolArg, PasswordArg
    from chimerax.core.commands import IntArg, OpenFileNameArg, OpenFolderNameArg, NonNegativeIntArg, EnumOf
    from chimerax.seqalign import AlignSeqPairArg, SeqRegionArg
    from chimerax.atomic import AtomicStructuresArg, UniqueChainsArg
    desc = CmdDesc(
        required = [('targets', RepeatOf(AlignSeqPairArg))],
        keyword = [('block', BoolArg), ('multichain', BoolArg), ('custom_script', OpenFileNameArg),
            ('dist_restraints', OpenFileNameArg), ('executable_location', OpenFileNameArg),
            ('fast', BoolArg), ('het_preserve', BoolArg), ('hydrogens', BoolArg),
            ('license_key', PasswordArg), ('num_models', IntArg), ('show_gui', BoolArg),
            ('temp_path', OpenFolderNameArg), ('thorough_opt', BoolArg), ('water_preserve', BoolArg)
        ],
        synopsis = 'Use Modeller to generate comparative model'
    )
    register('modeller comparative', desc, sequence_model, logger=logger)

    class LoopsRegionArg(SeqRegionArg):
        from .loops import special_region_values

    desc = CmdDesc(
        required = [('targets', LoopsRegionArg)],
        keyword = [('adjacent_flexible', NonNegativeIntArg), ('block', BoolArg), ('chains', UniqueChainsArg),
            ('executable_location', OpenFileNameArg), ('license_key', PasswordArg), ('num_models', IntArg),
            ('protocol', EnumOf(['standard', 'DOPE', 'DOPE-HR'])), ('show_gui', BoolArg),
            ('temp_path', OpenFolderNameArg),
        ],
        synopsis = 'Use Modeller to model loops or refine structure'
    )
    register('modeller loops', desc, model_loops, logger=logger)
    create_alias('modeller refine', "%s $*" % 'modeller loops', logger=logger)
    #create_alias('modeller refine', "%s $*" % 'modeller loops', logger=logger, url="help:user/commands/matchmaker.html")

    desc = CmdDesc(
        required = [('structures', AtomicStructuresArg)],
        keyword = [('block', BoolArg), ('license_key', PasswordArg), ('refresh', BoolArg)],
        synopsis = 'Fetch scores for models from Modeller web site'
    )
    register('modeller scores', desc, score_models, logger=logger)
