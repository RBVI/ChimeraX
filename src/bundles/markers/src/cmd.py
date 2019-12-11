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
    from chimerax.core.colors import Color
    rgba = color.uint8x4() if isinstance(color, Color) else color

    m = mset.create_marker(center, rgba, radius)
    return m

def _create_marker_set(session, marker_set):
    if isinstance(marker_set, tuple):
        # Model id for creating a new marker set.
        model_id = marker_set
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
    rgba8 = None if color is None else color.uint8x4()
    if markers:
        for m in some_markers:
            if center is not None:
                m.scene_coord = center
            if radius is not None:
                m.radius = radius
            if rgba8 is not None:
                m.color = rgba8
    if links:
        for l in some_markers.intra_bonds:
            if radius is not None:
                l.radius = radius
            if rgba8 is not None:
                l.color = rgba8

def marker_link(session, markers, radius = 0.5, color = (255,255,0,255)):
    if len(markers) != 2:
        from chimerax.core.errors import UserError
        raise UserError('marker link command requires exactly 2 markers, got %d' % len(markers))
    if len(markers.intra_bonds) > 0:
        from chimerax.core.errors import UserError
        from .markergui import _markers_spec
        raise UserError('marker link already exists between %s' % _markers_spec(markers))
    
    m1, m2 = markers
    from chimerax.core.colors import Color
    rgba = color.uint8x4() if isinstance(color, Color) else color
    from . import create_link
    link = create_link(m1, m2, rgba, radius)
    return link

def marker_segment(session, marker_set, position, to_position,
                   radius = 0.5, color = (255,255,0,255), coordinate_system = None,
                   label = None, label_height = 1.0, label_color = 'default'):

    mset = _create_marker_set(session, marker_set)
    center1 = position.scene_coordinates(coordinate_system)
    center2 = to_position.scene_coordinates(coordinate_system)
    from chimerax.core.colors import Color
    rgba = color.uint8x4() if isinstance(color, Color) else color

    m1 = mset.create_marker(center1, rgba, radius)
    m2 = mset.create_marker(center2, rgba, radius)
    from . import create_link
    link = create_link(m1, m2, rgba, radius)

    if label is not None:
        from chimerax.label.label3d import label as make_label
        from chimerax.core.objects import Objects
        from chimerax.atomic import Bonds
        lo = Objects(bonds = Bonds([link]))
        make_label(session, lo, object_type = 'bonds',
                   text = label, color = label_color, height = label_height)
        
    return m1, m2

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
    from chimerax.core.commands import CmdDesc, register, FloatArg, ColorArg, BoolArg, StringArg
    from chimerax.core.commands import CenterArg, CoordSysArg, ModelIdArg, Or, EnumOf
    desc = CmdDesc(
        required = [('marker_set', Or(MarkerSetArg, ModelIdArg)),
                    ('position', CenterArg)],
        keyword = [('radius', FloatArg),
                   ('color', ColorArg),
                   ('coordinate_system', CoordSysArg)],
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
                   ('color', ColorArg),
                   ('coordinate_system', CoordSysArg),
                   ('markers', BoolArg),
                   ('links', BoolArg)],
        synopsis = 'Change marker position or appearance'
    )
    register('marker change', desc, marker_change, logger=logger)

    desc = CmdDesc(
        required = [('markers', MarkersArg)],
        keyword = [('radius', FloatArg),
                   ('color', ColorArg)],
        synopsis = 'Connect two markers'
    )
    register('marker link', desc, marker_link, logger=logger)

    desc = CmdDesc(
        required = [('marker_set', Or(MarkerSetArg, ModelIdArg)),
                    ('position', CenterArg)],
        keyword = [('to_position', CenterArg),
                   ('radius', FloatArg),
                   ('color', ColorArg),
                   ('coordinate_system', CoordSysArg),
                   ('label', StringArg),
                   ('label_height', FloatArg),
                   ('label_color', Or(EnumOf(['default']),ColorArg))],
        required_arguments = ['to_position'],
        synopsis = 'Create two markers and a link between them'
    )
    register('marker segment', desc, marker_segment, logger=logger)
