# vim: set expandtab shiftwidth=4 softtabstop=4:

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
# Command to perform surface operations.
#
#   Syntax: surface <operation> <surfaceSpec>
#               [spacing <d>]
#               [modelId <n>]
#               [inPlace true|false]
#
# where allowed operations are: finerMesh
#
from chimerax.core.errors import UserError
def register_surface_subcommands(logger):

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
        'showPieces': (show_pieces_op,
                       (('pieces', surface_pieces_arg),), (), ()),
        'split': (split_op,
                      (('surfaces', surface_pieces_arg),),
                      (),
                      (('inPlace', bool_arg),
                       ('modelId', model_id_arg))),
        }
    """

    from chimerax.core.commands import CmdDesc, register, SurfacesArg, FloatArg, \
        IntArg, BoolArg, EnumOf, AxisArg, CenterArg, CoordSysArg
    from chimerax.atomic import AtomsArg

    from .dust import metrics
    dust_desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                        keyword = [('metric', EnumOf(metrics)),
                                   ('size', FloatArg),
                                   ('update', BoolArg)],
                        synopsis = 'hide small connected surface patches')
    register('surface dust', dust_desc, surface_dust, logger=logger)

    undust_desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                          synopsis = 'reshow surface dust')
    register('surface undust', undust_desc, surface_undust, logger=logger)

    invert_desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                          synopsis = 'show hidden part of surface and hide shown part')
    register('surface invertShown', invert_desc, surface_invert_shown, logger=logger)

    smooth_desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                          keyword = [('factor', FloatArg),
                                     ('iterations', IntArg),
                                     ('in_place', BoolArg)],
                        synopsis = 'smooth surfaces')
    register('surface smooth', smooth_desc, surface_smooth, logger=logger)

    transform_desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                             keyword = [('scale', FloatArg),
                                        ('rotate', FloatArg),
                                        ('axis', AxisArg),
                                        ('center', CenterArg),
                                        ('move', AxisArg),
                                        ('coordinate_system', CoordSysArg)],
                            synopsis = 'scale, rotate or move surface')
    register('surface transform', transform_desc, surface_transform, logger=logger)

    zone_desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                        keyword = [('near_atoms', AtomsArg),
                                   ('distance', FloatArg),
                                   ('bond_point_spacing', FloatArg),
                                   ('max_components', IntArg),
                                   ('update', BoolArg)],
                        required_arguments = ['near_atoms'],
                        synopsis = 'show surface near atoms')
    register('surface zone', zone_desc, surface_zone, logger=logger)

    unzone_desc = CmdDesc(required = [('surfaces', SurfacesArg)],
                          synopsis = 'show full surface without zone')
    register('surface unzone', unzone_desc, surface_unzone, logger=logger)

    from chimerax.core.commands import create_alias
    create_alias('sop', 'surface $*', logger=logger,
            url="help:user/commands/surface.html#sop")

#NOTE: below is unported code
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
        raise CommandError('surface cap argument must be "on" or "off", got "%s"'
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
                                                 'surface clip')

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
def surface_smooth(session, surfaces, factor = 0.3, iterations = 2, in_place = False):

    if len(surfaces) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No surfaces specified')

    from chimerax.surface import smooth_vertex_positions
    if in_place:
        for surface in surfaces:
            va, na, ta = surface.vertices, surface.normals, surface.triangles
            smooth_vertex_positions(va, ta, factor, iterations)
            smooth_vertex_positions(na, ta, factor, iterations)
            surface.set_geometry(va, na, ta)
        return surfaces
    else:
        copies = []
        from chimerax.core.models import Surface
        for surface in surfaces:
            va, na, ta = surface.vertices.copy(), surface.normals.copy(), surface.triangles.copy()
            smooth_vertex_positions(va, ta, factor, iterations)
            smooth_vertex_positions(na, ta, factor, iterations)
            copy = Surface(surface.name + ' smooth', session)
            copy.set_geometry(va, na, ta)
            copy.positions = surface.get_scene_positions()
            _copy_surface_attributes(surface, copy)
            copies.append(copy)
        session.models.add(copies)
        return copies

# -----------------------------------------------------------------------------
#
def _copy_surface_attributes(from_surf, to_surf):
    # TODO: There are many more attributes that could be copied.
    to_surf.color = from_surf.color
    to_surf.vertex_colors = from_surf.vertex_colors
    to_surf.display_style = from_surf.display_style
    to_surf.edge_mask = from_surf.edge_mask
    to_surf.triangle_mask = from_surf.triangle_mask

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
def surface_dust(session, surfaces, metric = 'size', size = 5, update = True):
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
    '''

    if len(surfaces) == 0:
        raise UserError('No surfaces specified')
    if size is None:
        raise UserError('Must specify dust size')
    from chimerax.surface import dust
    for s in surfaces:
        dust.hide_dust(s, metric, size, auto_update = update)

# -----------------------------------------------------------------------------
#
def surface_undust(session, surfaces):
    '''
    Redisplay the entire surface on which surface_dust() was used.
    '''
    from chimerax.surface import dust
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
def surface_invert_shown(session, surfaces):

    for surface in surfaces:
        m = surface.triangle_mask
        if m is None:
            from numpy import ones
            m = ones((len(surface.triangles),), bool)
        from numpy import logical_not
        surface.triangle_mask = logical_not(m)

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
def surface_transform(session, surfaces, scale = None,
                      rotate = 0, axis = (0,0,1), center = None, move = None,
                      coordinate_system = None):

    if len(surfaces) == 0:
        return

    csys = coordinate_system
    if csys is None:
        csys = surfaces[0].scene_position

    a = None
    need_axis = (rotate != 0)
    if need_axis:
        from chimerax.core.commands import Axis
        if isinstance(axis, Axis):
            a = axis.scene_coordinates(csys)
        else:
            a = csys.transform_vector(axis)

    c = None
    need_center = not (scale is None and rotate == 0)
    if need_center:
        from chimerax.core.commands import Center
        if isinstance(center, Center):
            c = center.scene_coordinates(csys)
        elif center is not None:
            c = csys.transform_point(center)

    if move is not None:
        from chimerax.core.commands import Axis
        if isinstance(move, Axis):
            move = move.scene_coordinates(csys)
        else:
            move = csys.transform_vector(move)
            
    for surf in surfaces:
        if surf.empty_drawing():
            continue
        if need_center and c is None:
            b = surf.bounds()
            if b is not None:
                c = b.center()
            else:
                from chimerax.core.errors import UserError
                raise UserError('surface transform: Center of displayed part of surface "%s" is not defined since none of it is displayed' % surf)

        tf = _transform_matrix(scale, rotate, a, c, move)
        sp = surf.scene_position
        stf = sp.inverse() * tf * sp
        vcolors = surf.vertex_colors	# Preserve vertex colors
        surf.set_geometry(stf.transform_points(surf.vertices),
                          stf.transform_vectors(surf.normals),
                          surf.triangles,
                          edge_mask = surf.edge_mask,
                          triangle_mask = surf.triangle_mask)
        surf.vertex_colors = vcolors

# -----------------------------------------------------------------------------
#
def _transform_matrix(scale, rotate, axis, center, move):
    
    from chimerax.geometry import identity, translation, scale as scale_transform, rotation
    tf = identity()
    if center is not None:
        tf = translation([-x for x in center]) * tf
    if scale is not None:
        tf = scale_transform(scale) * tf
    if rotate != 0:
        tf = rotation(axis, rotate) * tf
    if center is not None:
        tf = translation(center) * tf
    if move is not None:
        tf = translation(move) * tf
    return tf
    
# -----------------------------------------------------------------------------
#
def surface_zone(session, surfaces, near_atoms = None, distance = 2,
                 max_components = None, bond_point_spacing = None, update = True):
    '''
    Hide parts of a surface beyond a given distance from specified atoms.

    Parameters
    ----------
    surfaces : Model list
      Surface models to act on.
    near_atoms : Atoms
      Display only surface triangles that have all vertices within a specified distance
      of at least one of these atoms.
    distance : float
      Maximum distance from atoms.
    max_components : integer
      Show at most this number of connected surface patches, hiding the smaller ones.
      The limit applies for each surface model.
    bond_point_spacing : float
      Include points along bonds between the given atoms at this spacing.
    update : bool
      Whether to recompute the zone when the surface geometry changes.
    '''
    if len(surfaces) == 0:
        raise UserError('No surfaces specified')
    atoms = near_atoms
    if len(atoms) == 0:
        raise UserError('No atoms specified')

    bonds = atoms.intra_bonds if bond_point_spacing is not None else None

    from chimerax.surface import zone
    for s in surfaces:
        points = zone.path_points(atoms, bonds, bond_point_spacing)
        spoints = s.scene_position.inverse() * points
        zone.surface_zone(s, spoints, distance, auto_update = update,
                          max_components = max_components)

# -----------------------------------------------------------------------------
#
def surface_unzone(session, surfaces):
    '''
    Redisplay the entire surface on which surface_zone() was used.
    '''
    from chimerax.surface import zone
    for s in surfaces:
        zone.surface_unzone(s)
