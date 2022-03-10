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

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, OpenFolderNameArg, BoolArg, FloatArg
    from chimerax.map import MapArg
    from chimerax.atomic import AtomicStructureArg
    desc = CmdDesc(
        required = [('map', MapArg)],
        keyword = [('block', BoolArg),
                   ('far_water', BoolArg),
                   ('keep_input_water', BoolArg),
                   ('map_range', FloatArg),
                   ('near_model', AtomicStructureArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('residue_range', FloatArg),
                   ('server', BoolArg),
                   ('verbose', BoolArg),
        ],
        required_arguments = ['near_model'],
        synopsis = 'Place water molecules in map'
    )
    from .douse import phenix_douse
    register('phenix douse', desc, phenix_douse, logger=logger)

    desc = CmdDesc(
        optional = [('phenix_location', OpenFolderNameArg)],
        synopsis = 'Set the Phenix installation location'
    )
    from .locate import phenix_location
    register('phenix location', desc, phenix_location, logger=logger)
