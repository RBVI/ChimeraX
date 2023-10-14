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

def cage(session, cage, place_model = None, polygon_sides = 6,
         surface_only = False, resolution = None):
    '''Build polygonal cages and place molecules on faces.

    Parameters
    ----------
    cage : Structure
      Cage model.
    place_model : Model
      Place copies of model on each n-sided polygon of cage.
    polygon_sides : int
      Place on polygons with this number of sides.
    surface_only : bool
      Instead of showing instances of the molecule, show instances
      of surfaces of each chain.  The chain surfaces are computed if
      they do not already exist.
    resolution : float
      Resolution for computing surfaces when surface_only is true.
    '''

    from chimerax.core.errors import UserError
    if not hasattr(cage, 'placements'):
        raise UserError('Model %s is not a cage model.' % cage.name)

    n = polygon_sides
    p = cage.placements('p%d' % n)
    if len(p) == 0:
        raise UserError('Cage %s has no %d-gons.' % (cage.name, n))
    c = place_model.bounds().center()
    pc = make_closest_placement_identity(p, c)

    # TODO: Is positioning right if cage is moved?
    from chimerax.atomic import Structure
    if surface_only and isinstance(place_model, Structure):
        from chimerax.core.commands.surface import surface
        surfs = surface(session, place_model.atoms, resolution = resolution)
        mpinv = place_model.scene_position.inverse()
        for s in surfs:
            s.positions = mpinv * pc * s.scene_position
    else:
        place_model.positions = pc * place_model.scene_position

    session.logger.info('Placed %s at %d positions on %d-sided polygons'
                        % (place_model.name, len(pc), n))

# -----------------------------------------------------------------------------
# Find the transform that maps (0,0,0) closest to the molecule center and
# multiply all transforms by the inverse of that transform.  This chooses
# the best placement for the current molecule position and makes all other
# placements relative to that one.
#
def make_closest_placement_identity(tflist, center):

    d = tflist * (0,0,0)
    d -= center
    d2 = (d*d).sum(axis = 1)
    i = d2.argmin()
    tfinv = tflist[i].inverse()

    from chimerax.geometry import Place, Places
    rtflist = Places([Place()] + [tf*tfinv for tf in tflist[:i]+tflist[i+1:]])
    return rtflist

def register_cage_command(logger):
    from chimerax.core.commands import CmdDesc, register, ModelArg, IntArg, BoolArg, FloatArg
    desc = CmdDesc(required = [('cage', ModelArg)],
                   keyword = [('place_model', ModelArg),
                              ('polygon_sides', IntArg),
                              ('surface_only', BoolArg),
                              ('resolution', FloatArg)],
                   required_arguments = ['place_model'],
                   synopsis = 'Place copies of model on polygons of cage.')
    register('cage', desc, cage, logger=logger)
