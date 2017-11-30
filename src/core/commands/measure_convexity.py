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

def measure_convexity(session, surfaces, palette = None, range = None, smoothing_iterations = 5):
    '''
    Compute the convexity at each surface vertex defined as 2*pi minus the cone-angle
    spanned by the triangles incident at the vertex.  The surface vertices are colored
    based on convexity value.

    This definition of convexity is not standard and gives mottled coloring because the
    values depend strongly on the triangulation.  Vertices surrounded by large triangles
    on a smooth surface will have sharper cone angles then vertices surrounded by small
    triangles.  (Normalizing by vertex triangle areas does not help because the patch about
    a vertex is often irregular in shape.) To ameliorate this the "smoothing_iterations"
    option is useful.

    Parameters
    ----------
    surface : Model list
    palette : Colormap
      Default color palette is cyan-gray-maroon.
    range : 2-tuple of float or "full"
      Minimum and maximum convexity values corresponding to ends of color palette.
    smoothing_iterations : int
      Convexity values are averaged with neighbor vertices connected by an edge.
      This value specifies how many rounds of smoothing to perform.  Default 5.
    '''
    if palette is None:
        from .. import colors
        palette = colors.BuiltinColormaps['cyan-gray-maroon']
    if range is None and not palette.values_specified:
        range = (-1,1)
    if range is not None and range != 'full':
        rmin, rmax = range
        palette = palette.rescale_range(rmin, rmax)
    surf_drawings = []
    for s in surfaces:
        if hasattr(s, 'surface_drawings_for_vertex_coloring'):
            surf_drawings.extend(s.surface_drawings_for_vertex_coloring())
    for s in surfaces + surf_drawings:
        if s.empty_drawing():
            continue
        va,ta = s.vertices, s.triangles
        from ..surface import vertex_convexity
        c = vertex_convexity(va, ta, smoothing_iterations)
        cmap = palette.rescale_range(c.min(), c.max()) if range == 'full' else palette
        vc = cmap.interpolated_rgba8(c)
        s.vertex_colors = vc
        msg = ('Convexity %.3g - %.3g, mean %.3g, std deviation %.3g at %d vertices of %s'
               % (c.min(), c.max(), c.mean(), c.std(), len(va), s.name))
        from ..models import Model
        if isinstance(s, Model):
            msg += ' ' + s.id_string()
        session.logger.status(msg, log = True)
            
def register_command(session):
    from . import CmdDesc, register, SurfacesArg, ColormapArg, ColormapRangeArg, IntArg
    desc = CmdDesc(
        required = [('surfaces', SurfacesArg)],
        keyword = [('palette', ColormapArg),
                   ('range', ColormapRangeArg),
                   ('smoothing_iterations', IntArg),],
        synopsis = 'compute surface covexity')
    register('measure convexity', desc, measure_convexity, logger=session.logger)
