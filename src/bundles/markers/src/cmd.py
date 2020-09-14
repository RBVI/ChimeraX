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

def marker(session, marker_set, position, radius = 0.5, color = (255,255,0,255), coordinate_system = None):

    mset = _create_marker_set(session, marker_set)
    center = position.scene_coordinates(coordinate_system)
    m = mset.create_marker(center, color, radius)
    return m

def _create_marker_set(session, marker_set):
    if isinstance(marker_set, tuple):
        # Model id for creating a new marker set.
        model_id = marker_set
        if session.models.list(model_id = model_id):
            from chimerax.core.errors import UserError
            raise UserError('Cannot create a marker set #%s with same model id as another model'
                             % ','.join('%d'%i for i in model_id))
        from . import MarkerSet
        marker_set = MarkerSet(session)
        marker_set.id = model_id
        session.models.add([marker_set])
    return marker_set

def marker_delete(session, markers, links_only = False):
    if links_only:
        links = markers.intra_bonds
        n = len(links)
        links.delete()
        session.logger.status('Deleted %d links' % n, log=True)
    else:
        n = len(markers)
        for mset, m in markers.by_structure:
            if len(m) == mset.num_atoms:
                mset.delete()	# Delete entire marker set
            else:
                m.delete()
        session.logger.status('Deleted %d markers' % n, log=True)

def marker_change(session, some_markers, position = None, coordinate_system = None,
                  radius = None, color = None, markers = True, links = True):
    center = position.scene_coordinates(coordinate_system) if position else None
    if markers:
        for m in some_markers:
            if center is not None:
                m.scene_coord = center
            if radius is not None:
                m.radius = radius
            if color is not None:
                m.color = color
    if links:
        for l in some_markers.intra_bonds:
            if radius is not None:
                l.radius = radius
            if color is not None:
                l.color = color

def marker_link(session, markers, radius = 0.5, color = (255,255,0,255)):
    if len(markers) != 2:
        from chimerax.core.errors import UserError
        raise UserError('marker link command requires exactly 2 markers, got %d' % len(markers))
    if len(markers.intra_bonds) > 0:
        from chimerax.core.errors import UserError
        from .markergui import _markers_spec
        raise UserError('marker link already exists between %s' % _markers_spec(markers))
    
    m1, m2 = markers
    if m1.structure is not m2.structure:
        from chimerax.core.errors import UserError
        raise UserError('marker link: Cannot link markers from different models (#%s:%d, #%s:%d)'
                        % (m1.structure.id_string, m1.residue.number,
                           m2.structure.id_string, m2.residue.number))
    
    from . import create_link
    link = create_link(m1, m2, color, radius)
    return link

def marker_segment(session, marker_set, position, to_position,
                   radius = 0.5, color = (255,255,0,255), coordinate_system = None,
                   label = None, label_height = 1.0, label_color = 'default'):

    mset = _create_marker_set(session, marker_set)
    center1 = position.scene_coordinates(coordinate_system)
    center2 = to_position.scene_coordinates(coordinate_system)

    m1 = mset.create_marker(center1, color, radius)
    m2 = mset.create_marker(center2, color, radius)
    from . import create_link
    link = create_link(m1, m2, color, radius)

    if label is not None:
        from chimerax.label.label3d import label as make_label
        from chimerax.core.objects import Objects
        from chimerax.atomic import Bonds
        lo = Objects(bonds = Bonds([link]))
        make_label(session, lo, object_type = 'bonds',
                   text = label, color = label_color, height = label_height)
        
    return m1, m2

# -----------------------------------------------------------------------------
# Create a marker model from a surface mesh.
#
def markers_from_mesh(session, surfaces, edge_radius = 1, color = None,
                      markers = None):

    if len(surfaces) == 0:
        from chimerax.core.errors import UserError
        raise UserError('marker fromMesh: no surfaces specified')
    
    if markers is None:
        from . import MarkerSet
        markers = MarkerSet(session, 'Mesh ' + surfaces[0].name)

    for s in surfaces:
        varray = s.vertices
        edges = _masked_edges(s)
        if varray is None or len(edges) == 0:
            continue
        
        vcolors = s.vertex_colors
        if color is None and vcolors is None:
            color = s.color

        vmarker = {}
        for edge in edges:
            for v in edge:
                if not v in vmarker:
                    xyz = varray[v]
                    mcolor = color if vcolors is None else vcolors[v]
                    vmarker[v] = markers.create_marker(xyz, mcolor, edge_radius)

        from . import create_link
        ecolor = s.color if color is None else color
        for v1,v2 in edges:
            m1 = vmarker[v1]
            m2 = vmarker[v2]
            if color is None and vcolors is not None:
                ecolor = tuple((c1+c2)//2 for c1,c2 in zip(m1.color, m2.color))
            create_link(m1, m2, rgba=ecolor, radius=edge_radius)

    if markers.id is None:
        session.models.add([markers])
        
    return markers

def _masked_edges(surface):
    tmask = surface.triangle_mask
    emask = surface.edge_mask
    tri = surface.triangles
    if tri is None or tri.shape[1] == 1:
        return []
    if tri.shape[1] == 2:
        return tri if tmask is None else tri[tmask]

    edges = set()
    em0 = em1 = em2 = True
    for t, (v0,v1,v2) in enumerate(surface.triangles):
        if tmask is None or tmask[t]:
            if emask is not None:
                em = emask[t]
                em0,em1,em2 = em & 0x1, em & 0x2, em & 0x4
                if em & 0x1 and (v1,v0) not in edges:
                    edges.add((v0,v1))
                if em & 0x2 and (v2,v1) not in edges:
                    edges.add((v1,v2))
                if em & 0x4 and (v0,v2) not in edges:
                    edges.add((v2,v0))
    return edges

from chimerax.atomic import AtomsArg
class MarkersArg(AtomsArg):
    pass

from chimerax.core.commands import AtomSpecArg, AnnotationError
class MarkerSetArg(AtomSpecArg):
    """Marker set specifier"""
    name = "a marker set"

    @classmethod
    def parse(cls, text, session):
        aspec, text, rest = super().parse(text, session)
        from . import MarkerSet
        msets = [m for m in aspec.evaluate(session).models if isinstance(m, MarkerSet)]
        if len(msets) != 1:
            raise AnnotationError('Must specify 1 marker set, got %d for "%s"' % (len(msets), aspec))
        return msets[0], text, rest

    @classmethod
    def unparse(cls, model, session):
        return model.atomspec
    
def register_marker_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, Color8Arg, BoolArg, StringArg
    from chimerax.core.commands import CenterArg, CoordSysArg, ModelIdArg, Or, EnumOf, SurfacesArg
    desc = CmdDesc(
        required = [('marker_set', Or(MarkerSetArg, ModelIdArg))],
        keyword = [('position', CenterArg),
                   ('radius', FloatArg),
                   ('color', Color8Arg),
                   ('coordinate_system', CoordSysArg)],
        required_arguments = ['position'],
        synopsis = 'Place a marker'
    )
    register('marker', desc, marker, logger=logger)

    desc = CmdDesc(
        required = [('markers', MarkersArg)],
        keyword = [('links_only', BoolArg)],
        synopsis = 'Delete markers'
    )
    register('marker delete', desc, marker_delete, logger=logger)

    desc = CmdDesc(
        required = [('some_markers', MarkersArg)],
        keyword = [('position', CenterArg),
                   ('radius', FloatArg),
                   ('color', Color8Arg),
                   ('coordinate_system', CoordSysArg),
                   ('markers', BoolArg),
                   ('links', BoolArg)],
        synopsis = 'Change marker position or appearance'
    )
    register('marker change', desc, marker_change, logger=logger)

    desc = CmdDesc(
        required = [('markers', MarkersArg)],
        keyword = [('radius', FloatArg),
                   ('color', Color8Arg)],
        synopsis = 'Connect two markers'
    )
    register('marker link', desc, marker_link, logger=logger)

    desc = CmdDesc(
        required = [('marker_set', Or(MarkerSetArg, ModelIdArg))],
        keyword = [('position', CenterArg),
                   ('to_position', CenterArg),
                   ('radius', FloatArg),
                   ('color', Color8Arg),
                   ('coordinate_system', CoordSysArg),
                   ('label', StringArg),
                   ('label_height', FloatArg),
                   ('label_color', Or(EnumOf(['default']),Color8Arg))],
        required_arguments = ['position', 'to_position'],
        synopsis = 'Create two markers and a link between them'
    )
    register('marker segment', desc, marker_segment, logger=logger)

    desc = CmdDesc(
        required = [('surfaces', SurfacesArg)],
        keyword = [('edge_radius', FloatArg),
                   ('color', Color8Arg),
                   ('markers', MarkerSetArg)],
        synopsis = 'Create markers and links for surface mesh'
    )
    register('marker fromMesh', desc, markers_from_mesh, logger=logger)
