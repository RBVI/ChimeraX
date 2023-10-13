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

def set(session, bg_color=None,
        silhouettes=None, silhouette_width=None, silhouette_color=None, silhouette_depth_jump=None,
        selection_color=None, selection_width=None,
        subdivision=None, max_frame_rate = None):
    '''Set global parameters.  With no options reports the current settings.

    Parameters
    ----------
    bg_color : Color
        Set the graphics window background color.
    silhouettes : bool
        Enable or disable silhouette edges (black lines drawn at boundaries of objects
        where the depth of the scene jumps.
    silhouette_width : float
        Width in pixels of silhouette edges. Minimum width is 1 pixel.
    silhouette_color : Color
        Color of silhouette edges.
    silhouette_depth_jump : float
        Fraction of scene depth giving minimum depth change to draw a silhouette edge. Default 0.03.
    selection_color : Color
        Color to use when outlining selected objects.  Initially green.
    selection_width : float
        Width in pixels of the selection outline.  Initially 1 (or 2 for high DPI screens).
    subdivision : float
        Controls the rendering quality of spheres and cylinders for drawing atoms and bonds.
        Default value is 1, higher values give smoother spheres and cylinders.
    max_frame_rate : float
        Maximum frames per second to render graphics.  The default frame rate is 60 frames per second.
        A slower rate is sometimes useful when making movies to preview at the slower movie playback rate.
    '''
    had_arg = False
    view = session.main_view
    if bg_color is not None:
        had_arg = True
        from .graphics import graphics
        graphics(session, background_color = bg_color)
    if (silhouettes is not None
        or silhouette_width is not None
        or silhouette_color is not None
        or silhouette_depth_jump is not None):
        had_arg = True
        from .graphics import graphics_silhouettes
        graphics_silhouettes(session, enable = silhouettes, width = silhouette_width,
                             color = silhouette_color, depth_jump = silhouette_depth_jump)
    if selection_color is not None or selection_width is not None:
        had_arg = True
        from .graphics import graphics_selection
        graphics_selection(session, color = selection_color, width = selection_width)
    if subdivision is not None:
        had_arg = True
        from .graphics import graphics_quality
        graphics_quality(session, quality = subdivision)
    if max_frame_rate is not None:
        had_arg = True
        from .graphics import graphics_rate
        graphics_rate(session, max_frame_rate = max_frame_rate)

    if not had_arg:
        from chimerax import atomic
        lod = atomic.level_of_detail(session)
        msg = '\n'.join(('Current settings:',
                         '  Background color: %d,%d,%d' % tuple(100*r for r in view.background_color[:3]),
                         '  Subdivision: %.3g'  % lod.quality))
        session.logger.info(msg)

def xset(session, setting):
    # only bgColor right now...
    from chimerax.core.core_settings import settings
    view = session.main_view
    view.background_color = settings.saved_value('background_color').rgba
    view.redraw_needed = True

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ColorArg, BoolArg, FloatArg, EnumOf
    desc = CmdDesc(
        keyword=[('bg_color', ColorArg),
                 ('silhouettes', BoolArg),
                 ('silhouette_width', FloatArg),
                 ('silhouette_color', ColorArg),
                 ('silhouette_depth_jump', FloatArg),
                 ('selection_color', ColorArg),
                 ('selection_width', FloatArg),
                 ('subdivision', FloatArg),
                 ('max_frame_rate', FloatArg)],
        hidden = ['silhouettes', 'silhouette_width',
                  'silhouette_color', 'silhouette_depth_jump',
                  'selection_color', 'selection_width',
                  'max_frame_rate'],
        synopsis="set miscellaneous parameters"
    )
    register('set', desc, set, logger=logger)
    xdesc = CmdDesc(required=[('setting', EnumOf(['bgColor']))],
        synopsis="reset parameters to default"
    )
    register('~set', xdesc, xset, logger=logger)
