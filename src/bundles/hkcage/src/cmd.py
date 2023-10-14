# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
#
def hkcage(session, h, k, radius = 100.0, orientation = '222', color = (255,255,255,255),
           sphere_factor = 0.0, edge_radius = None, mesh = False, replace = True, alpha = 'hexagonal'):

    if h == 0 and k == 0:
        from chimerax.core.errors import UserError
        raise UserError('h and k must be positive, got %d %d' % (h,k))

    from .cage import show_hk_lattice
    show_hk_lattice(session, h, k, radius, orientation, color, sphere_factor,
                    edge_radius, mesh, replace, alpha)

    _show_citation_message(alpha, session.logger)

# -----------------------------------------------------------------------------
#
_citation_shown = False
def _show_citation_message(alpha, log):
    global _citation_shown
    if _citation_shown or alpha == 'hexagonal':
        return

    log.info(
        'hkcage %s pattern was developed by Colin Brown and Antoni Luque supported by '
        'National Science Foundation Award #1951678. Please cite:<br>'
        '<div style="background-color:lightyellow;">'
        'Structural puzzles in virology solved with an overarching icosahedral design principle<br>'
        'Reidun Twarock, Antoni Luque<br>'
        'Nat Commun. 2019; 10: 4414. Published online 2019 Sep 27.<br>'
        '<a href="https://doi.org/10.1038/s41467-019-12367-3">https://doi.org/10.1038/s41467-019-12367-3</a><br>'
        '</div>' % alpha, is_html=True)
    _citation_shown = True

# -----------------------------------------------------------------------------
#
def register_hkcage_command(logger):
    from chimerax.core.commands import CmdDesc, register, NonNegativeIntArg, \
                                       FloatArg, EnumOf, Color8Arg, BoolArg

    from chimerax.geometry.icosahedron import coordinate_system_names
    alpha_values = ['hexagonal', 'hexagonal-dual', 'trihex', 'trihex-dual',
                    'snub', 'snub-dual', 'rhomb', 'rhomb-dual']
    desc = CmdDesc(
        required = [('h', NonNegativeIntArg),
                    ('k', NonNegativeIntArg)],
        keyword = [('radius', FloatArg),
                   ('orientation', EnumOf(coordinate_system_names)),
                   ('color', Color8Arg),
                   ('sphere_factor', FloatArg),
                   ('edge_radius', FloatArg),
                   ('mesh', BoolArg),
                   ('replace', BoolArg),
                   ('alpha', EnumOf(alpha_values))],
        synopsis = 'Create icosahedron mesh of hexagons, pentagons, squares and triangles'
    )
    register('hkcage', desc, hkcage, logger=logger)
