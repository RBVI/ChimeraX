# vim: set expandtab ts=4 sw=4:

def cage(session, cage, place_model = None, polygon_sides = 6):
    '''Build polygonal cages and place molecules on faces.

    Parameters
    ----------
    cage : AtomicStructure
      Cage model.
    place_model : Model
      Place copies of model on each n-sided polygon of cage.
    polygon_sides : int
      Place on polygons with this number of sides.
    '''

    from chimerax.core.commands import AnnotationError
    if place_model is None:
        raise AnnotationError('Cage command requires "place_model" argument')
    if not hasattr(cage, 'cage_placements'):
        raise AnnotationError('Model %s is not a cage model.' % cage.name)

    n = polygon_sides
    p = cage.cage_placements(n)
    if len(p) == 0:
        raise AnnotationError('Cage %s has no %d-gons.' % (cage.name, n))
    c = place_model.bounds().center()
    pc = make_closest_placement_identity(p, c)
    place_model.positions = pc * place_model.position
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
    d2 = (d*d).sum(axis = 1)
    i = d2.argmin()
    tfinv = tflist[i].inverse()

    from chimerax.core.geometry import Place, Places
    rtflist = Places([Place()] + [tf*tfinv for tf in tflist[:i]+tflist[i+1:]])
    return rtflist

def register_cage_command():
    from chimerax.core.commands import CmdDesc, register, ModelArg, IntArg
    desc = CmdDesc(required = [('cage', ModelArg)],
                   keyword = [('place_model', ModelArg),
                              ('polygon_sides', IntArg),],
                   synopsis = 'Place copies of model on polygons of cage.')
    register('cage', desc, cage)
