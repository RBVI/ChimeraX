# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
#
def hkcage(session, h, k, radius = 100.0, orientation = '222', color = (255,255,255,255),
           sphere_factor = 0.0, edge_radius = None, mesh = False, replace = True):

    if h == 0 and k == 0:
        from chimerax.core.errors import UserError
        raise UserError('h and k must be positive, got %d %d' % (h,k))

    from .cage import show_hk_lattice
    show_hk_lattice(session, h, k, radius, orientation, color, sphere_factor,
                    edge_radius, mesh, replace)

# -----------------------------------------------------------------------------
#
def register_hkcage_command(logger):
    from chimerax.core.commands import CmdDesc, register, NonNegativeIntArg, \
                                       FloatArg, EnumOf, Color8Arg, BoolArg

    from chimerax.geometry.icosahedron import coordinate_system_names
    desc = CmdDesc(
        required = [('h', NonNegativeIntArg),
                    ('k', NonNegativeIntArg)],
        keyword = [('radius', FloatArg),
                   ('orientation', EnumOf(coordinate_system_names)),
                   ('color', Color8Arg),
                   ('sphere_factor', FloatArg),
                   ('edge_radius', FloatArg),
                   ('mesh', BoolArg),
                   ('replace', BoolArg)],
        synopsis = 'Create icosahedron mesh of hexagons and pentagons'
    )
    register('hkcage', desc, hkcage, logger=logger)
