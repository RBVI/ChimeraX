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
# Command to perform surface operations.
#
#   Syntax: sop <operation> <surfaceSpec>
#               [spacing <d>]
#               [modelId <n>]
#               [inPlace true|false]
#
# where allowed operations are: finerMesh
#
from ..errors import UserError
def register_command(session):

    # TODO: Port other subcommands from Chimera 1.
    old_stuff = """
    ops = {
        'cap': (cap_op,
                (('onoff', string_arg),),
                (),
                (('color', string_arg), # Can be "same"
                 ('mesh', bool_arg),
                 ('subdivision', float_arg, {'min':0}),
                 ('offset', float_arg))),
        'clip': (clip_op,
                 (('volumes', volumes_arg),),
                 (),
                 (('center', string_arg),
                  ('coordinateSystem', openstate_arg),
                  ('radius', float_arg, {'min':0}),
                  ('color', color_arg),
                  ('mesh', bool_arg),
                  ('replace', bool_arg))),
        'colorCopy': (color_copy_op,
                      (('surfaces', surface_pieces_arg),
                       ('tosurfaces', surface_pieces_arg),),
                      (),
                      ()),
        'finerMesh': (subdivide_op,
                      (('surfaces', surface_pieces_arg),),
                      (),
                      (('spacing', float_arg),
                       ('inPlace', bool_arg),
                       ('modelId', model_id_arg))),
        'hidePieces': (hide_pieces_op,
                       (('pieces', surface_pieces_arg),), (), ()),
        'invertShown': (invert_shown_op,
                        (('pieces', surface_pieces_arg),), (), ()),
        'showPieces': (show_pieces_op,
                       (('pieces', surface_pieces_arg),), (), ()),
        'smooth': (smooth_op,
                      (('surfaces', surface_pieces_arg),),
                      (),
                      (('factor', float_arg),
                       ('iterations', int_arg),
                       ('inPlace', bool_arg),
                       ('modelId', model_id_arg))),
        'split': (split_op,
                      (('surfaces', surface_pieces_arg),),
                      (),
                      (('inPlace', bool_arg),
                       ('modelId', model_id_arg))),
        'transform': (transform_op,
                 (('surfaces', surface_pieces_arg),),
                 (),
                 (('scale', float_arg),
                  ('radius', float_arg, {'min':0}),
                  ('move', float3_arg),
                  ('rotate', float_arg),
                  ('axis', string_arg),
                  ('center', string_arg),
                  ('coordinateSystem', openstate_arg),)),
        }
    """

    from . import CmdDesc, register, SurfacesArg, AtomsArg, FloatArg, IntArg, BoolArg, EnumOf

    from ..surface.dust import metrics
    dust_desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                        keyword = [('metric', EnumOf(metrics)),
                                   ('size', FloatArg),
                                   ('update', BoolArg)],
                        synopsis = 'hide small connected surface patches')
    register('sop dust', dust_desc, sop_dust)

    zone_desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                        keyword = [('near_atoms', AtomsArg),
                                   ('range', FloatArg),
                                   ('bond_point_spacing', FloatArg),
                                   ('max_components', IntArg),
                                   ('update', BoolArg)],
                        required_arguments = ['near_atoms'],
                        synopsis = 'show surface near atoms')
    register('sop zone', zone_desc, sop_zone)


# -----------------------------------------------------------------------------
#
def unsop_command(cmdname, args):

    from Commands import perform_operation, string_arg, surfaces_arg
    from Commands import surface_pieces_arg, volumes_arg

    ops = {
        'clip': (unclip_op, (('volumes', volumes_arg),), (), ()),
        'hideDust': (unhide_dust_op, (('surfaces', surfaces_arg),), (), ()),
        'hidePieces': (show_pieces_op,
                       (('pieces', surface_pieces_arg),), (), ()),
        'showPieces': (hide_pieces_op,
                       (('pieces', surface_pieces_arg),), (), ()),
        'zone': (unzone_op, (('surfaces', surfaces_arg),), (), ()),
        }

    perform_operation(cmdname, args, ops)

# -----------------------------------------------------------------------------
#
def cap_op(onoff, color = None, mesh = None, subdivision = None, offset = None):

    if not onoff in ('on', 'off'):
        raise CommandError('sop cap argument must be "on" or "off", got "%s"'
                           % onoff)
    
    from SurfaceCap.surfcaps import capper
    c = capper()
    
    if not color is None:
        from Commands import color_arg
        c.set_cap_color(None if color == 'same' else color_arg(color))
    if not mesh is None:
        c.set_style(mesh)
    if not subdivision is None:
        c.set_subdivision_factor(subdivision)
    if not offset is None:
        c.set_cap_offset(offset)

    if onoff == 'on':
        c.show_caps()
    else:
        c.unshow_caps()

# -----------------------------------------------------------------------------
#
def clip_op(volumes, center = None, coordinateSystem = None, radius = None,
            color = None, mesh = False, replace = True):

    if not center is None:
        import Commands as C
        center, axis, csys = C.parse_center_axis(center, None, coordinateSystem,
                                                 'sop clip')

    import clip
    for v in volumes:
        if center is None or coordinateSystem is None:
            c = center
        else:
            import Matrix as M
            c = M.xform_xyz(center, coordinateSystem.xform, v.openState.xform)
        clip.spherical_clip(v, c, radius, color, mesh, replace)

# -----------------------------------------------------------------------------
#
def unclip_op(volumes):

    import clip
    for v in volumes:
        clip.unclip(v)

# -----------------------------------------------------------------------------
#
def color_copy_op(surfaces, tosurfaces):

    ns, nd = len(surfaces), len(tosurfaces)
    if ns == 0:
        raise CommandError('No source surfaces specified')
    if nd == 0:
        raise CommandError('No destination surfaces specified')
    if nd != ns and ns != 1:
        raise CommandError('Multiple source surfaces (%d), and unequal number of destination surfaces (%d)'
                           % (ns,nd))

    if nd == ns:
        for s,ts in zip(surfaces, tosurfaces):
            ts.color = s.color
            if s.hasVertexColors() and ts.vertexCount != s.vertexCount:
                raise CommandError('Surfaces have different number of vertices (%d and %d)'
                                   % (s.vertexCount, ts.vertexCount))
            ts.vertexColors = s.vertexColors
    elif ns == 1:
        s = surfaces[0]
        for ts in tosurfaces:
            if s.hasVertexColors() and ts.vertexCount != s.vertexCount:
                raise CommandError('Surfaces have different number of vertices (%d and %d)'
                                   % (s.vertexCount, ts.vertexCount))
            ts.color = s.color
            ts.vertexColors = s.vertexColors

# -----------------------------------------------------------------------------
#
def smooth_op(surfaces, factor = 0.3, iterations = 2, inPlace = False, modelId = None):

    from Commands import check_number, parse_model_id
    if len(surfaces) == 0:
        raise CommandError('No surfaces specified')
    plist = surfaces
    s = None if inPlace else new_surface('smoothed', plist[0].model, modelId)
    from smooth import smooth_surface_piece
    for p in plist:
        smooth_surface_piece(p, factor, iterations, s)

# -----------------------------------------------------------------------------
#
def subdivide_op(surfaces, spacing = None, inPlace = False, modelId = None):

    from Commands import check_number, parse_model_id
    if len(surfaces) == 0:
        raise CommandError('No surfaces specified')
    if spacing is None:
        raise CommandError('Must specify mesh spacing')
    check_number(spacing, 'spacing', positive = True)
    plist = surfaces
    s = None if inPlace else new_surface('finer mesh', plist[0].model, modelId)
    from subdivide import subdivide
    for p in plist:
        np = subdivide(p, spacing, s)
        if np != p:
            np.save_in_session = True

# -----------------------------------------------------------------------------
#
def new_surface(name, align_to, model_id):
    from _surface import SurfaceModel
    s = SurfaceModel()
    s.name = name
    from chimera import openModels as om
    if model_id:
        id, subid = model_id
    else:
        id, subid = om.Default, om.Default
    om.add([s], baseId = id, subid = subid)
    s.openState.xform = align_to.openState.xform
    return s

# -----------------------------------------------------------------------------
#
def sop_dust(session, surfaces, metric = 'size', size = None, update = False):
    '''
    Hide connected surface patchs smaller than a specified size.

    Parameters
    ----------
    surfaces : Models list
      Surface models to act on.
    metric : One of 'size', 'area', 'volume', 'size rank', 'area rank', 'volume rank'
      Use this size metric.  Rank metrics hide patches smaller than the N largest.
    size : float
      Hide patches smaller than this size.
    update : bool
      Whether to update dust hiding when surface shape changes.
      Not implemented.

    '''

    if len(surfaces) == 0:
        raise UserError('No surfaces specified')
    if size is None:
        raise UserError('Must specify dust size')
    from ..surface import dust
    for s in surfaces:
        dust.hide_dust(s, metric, size, update)

# -----------------------------------------------------------------------------
#
def unhide_dust_op(surfaces):

    from HideDust import dust
    for s in surfaces:
        dust.unhide_dust(s)

# -----------------------------------------------------------------------------
#
def show_pieces_op(pieces):

    for p in pieces:
        p.display = True

# -----------------------------------------------------------------------------
#
def hide_pieces_op(pieces):

    for p in pieces:
        p.display = False

# -----------------------------------------------------------------------------
#
def invert_shown_op(pieces):

    from Surface import set_visibility_method
    for p in pieces:
        m = p.triangleAndEdgeMask
        if m is not None:
            minv = m^8
            p.triangleAndEdgeMask = minv
            # Suppress surface mask auto updates
            set_visibility_method('invert', p.model)

# -----------------------------------------------------------------------------
#
def split_op(surfaces, inPlace = False, modelId = None):

    from Commands import check_number, parse_model_id
    if len(surfaces) == 0:
        raise CommandError('No surfaces specified')
    plist = surfaces
    import split
    plist = split.split_surfaces(plist, inPlace, modelId)
    return plist

# -----------------------------------------------------------------------------
#
def transform_op(surfaces, scale = None, radius = None, move = None,
                 rotate = 0, axis = (0,0,1), center = None,
                 coordinateSystem = None):

    if len(surfaces) == 0:
        return

    import Matrix as M

    csys = coordinateSystem
    if csys is None:
        csys = surfaces[0].model.openState
    c = a = None
    need_center = not (scale is None and radius is None and rotate == 0)
    if need_center:
        if not center is None or not axis is None:
            import Commands as C
            c, a = C.parse_center_axis(center, axis, csys, 'sop transform')[:2]
            if not c is None:
                ctf = M.xform_matrix(csys.xform)
                c = M.apply_matrix(ctf, c)
            if not a is None:
                ctf = M.xform_matrix(csys.xform)
                a = M.apply_matrix_without_translation(ctf, a)
        if c is None:
            from Measure import inertia
            axes, d2, c = inertia.surface_inertia(surfaces) # global coords
            if c is None:
                ctf = M.xform_matrix(csys.xform)
                M.apply_matrix_without_translation(ctf, (0,0,0))

    import transform
    if scale is None and not radius is None:
        rmax = max(transform.surface_radius(p,c) for p in surfaces)
        scale = radius / rmax if rmax > 0 else 1

    transform.transform_surface_pieces(surfaces, scale, rotate, a, c, move)
    
# -----------------------------------------------------------------------------
#
def sop_zone(session, surfaces, near_atoms = None, range = 2,
             max_components = None, bond_point_spacing = None, update = False):
    '''
    Hide parts of a surface beyond a given distance from specified atoms.

    Parameters
    ----------
    surfaces : Model list
      Surface models to act on.
    near_atoms : Atoms
      Display only surface triangles that have all vertices in range of
      at least one of these atoms.
    range : float
      Maximum distance from atoms.
    max_components : integer
      Show at most this number of connected surface patches, hiding the smaller ones.
      The limit applies for each surface model.
    bond_point_spacing : float
      Include distances from points along bonds between the given atoms at this spacing.
      Not implemented.
    update : bool
      Whether to recompute the zone when the surface geometry changes.
      Not implemented.
    '''
    if len(surfaces) == 0:
        raise UserError('No surfaces specified')
    atoms = near_atoms
    if len(atoms) == 0:
        raise UserError('No atoms specified')

#    bonds = atoms.inter_bonds if bond_point_spacing is not None else None
    bonds = None

    from ..surface import zone
    for s in surfaces:
        points = zone.path_points(atoms, bonds, bond_point_spacing)
        spoints = s.position.inverse() * points
        zone.surface_zone(s, spoints, range, auto_update = update,
                          max_components = max_components)

# -----------------------------------------------------------------------------
#
def unzone_op(surfaces):

    import SurfaceZone as SZ
    for s in surfaces:
        SZ.no_surface_zone(s)
