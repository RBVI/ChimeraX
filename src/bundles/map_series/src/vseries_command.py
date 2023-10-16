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

# ----------------------------------------------------------------------------
# Volume series command.
#
#   Syntax: vseries <operation> <mapSpec>
#
players = set()         # Active players.

def register_vseries_command(logger):

    from chimerax.core.commands import CmdDesc, register, BoolArg, EnumOf, IntArg, StringArg, FloatArg, ColorArg
    from chimerax.atomic import AtomsArg
    from chimerax.map.mapargs import MapArg, MapStepArg, MapRegionArg, ValueTypeArg, IntRangeArg

    sarg = [('series', SeriesArg)]

    align_desc = CmdDesc(required = sarg,
                         keyword = [('enclose_volume', FloatArg),
                                    ('fast_enclose_volume', FloatArg)],
                         synopsis = 'Align map to preceding map in series')
    register('vseries align', align_desc, vseries_align, logger=logger)

    save_desc = CmdDesc(required = sarg + [('path', StringArg)],
                         keyword = [
                             ('subregion', MapRegionArg),
                             ('step', MapStepArg),
                             ('value_type', ValueTypeArg),
                             ('threshold', FloatArg),
                             ('zero_mean', BoolArg),
                             ('scale_factor', FloatArg),
                             ('match_scale', SeriesArg),
                             ('enclose_volume', FloatArg),
                             ('fast_enclose_volume', FloatArg),
                             ('normalize_level', FloatArg),
                             ('align', BoolArg),
                             ('on_grid', MapArg),
                             ('mask', MapArg),
                             ('final_value_type', ValueTypeArg),
                             ('compress', BoolArg)],
                        synopsis = 'Process and save a map series')
    register('vseries save', save_desc, vseries_save, logger=logger)

    measure_desc = CmdDesc(required = sarg,
                           keyword = [('output', StringArg),
                                      ('centroids', BoolArg),
                                      ('color', ColorArg),
                                      ('radius', FloatArg),],
                           synopsis = 'Measure centroids for each map in series')
    register('vseries measure', measure_desc, vseries_measure, logger=logger)

    play_desc = CmdDesc(required = sarg,
                        keyword = [('loop', BoolArg),
                                   ('direction', EnumOf(('forward', 'backward', 'oscillate'))),
                                   ('normalize', BoolArg),
                                   ('max_frame_rate', FloatArg),
                                   ('pause_frames', IntArg),
                                   ('markers', AtomsArg),
                                   ('preceding_marker_frames', IntArg),
                                   ('following_marker_frames', IntArg),
                                   ('color_range', FloatArg),
                                   ('cache_frames', IntArg),
                                   ('jump_to', IntArg),
                                   ('range', IntRangeArg),
                                   ('start_time', IntArg),],
                        synopsis = 'Draw each map in a series in order')
    register('vseries play', play_desc, vseries_play, logger=logger)

    stop_desc = CmdDesc(required = sarg,
                        synopsis = 'Stop playing map series')
    register('vseries stop', stop_desc, vseries_stop, logger=logger)

    slider_desc = CmdDesc(required = sarg,
                          synopsis = 'Display a slider to control which map in a series is displayed')
    register('vseries slider', slider_desc, vseries_slider, logger=logger)

# -----------------------------------------------------------------------------
#
def vseries_play(session, series, direction = 'forward', loop = False, max_frame_rate = None, pause_frames = 0,
            jump_to = None, range = None, start_time = None, normalize = False, markers = None,
            preceding_marker_frames = 0, following_marker_frames = 0,
            color_range = None, cache_frames = 1):
    '''Show a sequence of maps from a volume series.'''
    if len(series) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No volume series specified')
    
    from . import play
    p = play.Play_Series(series, session, range = range, start_time = start_time,
                         play_direction = direction,
                         loop = loop,
                         max_frame_rate = max_frame_rate,
                         pause_frames = pause_frames,
                         normalize_thresholds = normalize,
                         markers = markers,
                         preceding_marker_frames = preceding_marker_frames,
                         following_marker_frames = following_marker_frames,
                         color_range = color_range,
                         rendering_cache_size = cache_frames)
    if not jump_to is None:
        p.change_time(jump_to)
    else:
        global players
        players.add(p)
        p.play()
    release_stopped_players()
    return p

# -----------------------------------------------------------------------------
#
def vseries_stop(session, series):
    '''Stop playing a map series.'''
    for p in players:
        for s in series:
            if s in p.series:
                p.stop()
    release_stopped_players()

# -----------------------------------------------------------------------------
#
def vseries_align(session, series, enclose_volume = None, fast_enclose_volume = None):
    '''Align each frame of a map series to the preceeding frame.'''
    for s in series:
        align_series(s, enclose_volume, fast_enclose_volume, session)

# -----------------------------------------------------------------------------
#
def align_series(s, enclose_volume = None, fast_enclose_volume = None, session = None):

    n = len(s.maps)
    vprev = None
    for i,v in enumerate(s.maps):
        session.logger.status('Aligning %s (%d of %d maps)' % (v.data.name, i+1, n))
        set_enclosed_volume(v, enclose_volume, fast_enclose_volume)
        if vprev:
            align(v, vprev)
        vprev = v

# -----------------------------------------------------------------------------
#
def set_enclosed_volume(v, enclose_volume, fast_enclose_volume):
    if not enclose_volume is None:
        level = v.surface_level_for_enclosed_volume(enclose_volume)
        v.set_parameters(surface_levels = [level])
    elif not fast_enclose_volume is None:
        level = v.surface_level_for_enclosed_volume(fast_enclose_volume,
                                                    rank_method = True)
        v.set_parameters(surface_levels = [level])

# -----------------------------------------------------------------------------
#
def align(v, vprev):

    v.position = vprev.position
    from chimerax.map_fit.fitmap import map_points_and_weights, motion_to_maximum
    points, point_weights = map_points_and_weights(v, above_threshold = True)
    move_tf, stats = motion_to_maximum(points, point_weights, vprev,
                                       max_steps = 2000,
                                       ijk_step_size_min = 0.01,
                                       ijk_step_size_max = 0.5,
                                       optimize_translation = True,
                                       optimize_rotation = True)
    v.position = move_tf * v.position

# -----------------------------------------------------------------------------
#
def vseries_save(session, series, path, subregion = None, step = None, value_type = None,
                 threshold = None, zero_mean = False, scale_factor = None, match_scale = None,
                 enclose_volume = None, fast_enclose_volume = None, normalize_level = None,
                 align = False, on_grid = None, mask = None, final_value_type = None, compress = False):
    '''
    Process the frames of a map series and save the result to a a file.
    Processing can normalize, align, mask and change the numeric value type of maps.
    '''
    if len(series) != 1:
        from chimerax.core.errors import UserError
        raise UserError('vseries save: Can only save one series in a file, got %d' % len(series))
    s = series[0]

    from os.path import expanduser, basename
    path = expanduser(path)         # Tilde expansion
    fname = basename(path)

    maps = s.maps
    if on_grid is None and align:
        on_grid = maps[0]

    grid = None
    if not on_grid is None:
        vtype = maps[0].data.value_type if value_type is None else value_type
        grid = on_grid.writable_copy(value_type = vtype)

    n = len(maps)
    for i,v in enumerate(maps):
        session.logger.status('Writing %s to %s (%d of %d maps)' % (v.data.name, fname, i+1, n))
        align_to = maps[i-1] if align and i > 0 else None
        mscale = match_scale[0].maps[i] if match_scale else None
        d = processed_volume(v, subregion, step, value_type, threshold, zero_mean, scale_factor, mscale,
                             enclose_volume, fast_enclose_volume, normalize_level,
                             align_to, grid, mask, final_value_type)
        d.name = '%04d' % i
        options = {'append': True, 'compress': compress}
        from chimerax.map_data import cmap
        cmap.save(d, path, options)

    if grid:
        grid.delete()

# -----------------------------------------------------------------------------
#
def processed_volume(v, subregion = None, step = None, value_type = None, threshold = None,
                     zero_mean = False, scale_factor = None, match_scale = None,
                     enclose_volume = None, fast_enclose_volume = None, normalize_level = None,
                     align_to = None, on_grid = None, mask = None, final_value_type = None):
    d = v.data
    region = None
    if not subregion is None or not step is None:
        from chimerax.map.volume import full_region
        ijk_min, ijk_max = full_region(d.size)[:2] if subregion is None else subregion
        ijk_step = (1,1,1) if step is None else step
        region = (ijk_min, ijk_max, ijk_step)
        from chimerax.map_data import GridSubregion
        d = GridSubregion(d, ijk_min, ijk_max, ijk_step)

    if (value_type is None and threshold is None and not zero_mean and
        scale_factor is None and match_scale is None and align_to is None and
        mask is None and final_value_type is None):
        return d

    m = d.full_matrix()
    if not value_type is None:
        m = m.astype(value_type)

    if not threshold is None:
        from numpy import array, putmask
        t = array(threshold, m.dtype)
        putmask(m, m < t, 0)

    if zero_mean:
        from numpy import float64
        mean = m.mean(dtype = float64)
        m = (m - mean).astype(m.dtype)

    if not scale_factor is None:
        m = (m*scale_factor).astype(m.dtype)

    if not match_scale is None:
        ms = match_scale.region_matrix(region) if region else match_scale.full_matrix()
        from numpy import float64, einsum
        m1, ms1 = m.sum(dtype = float64), ms.sum(dtype = float64)
        m2, ms2, mms = einsum('ijk,ijk',m,m,dtype=float64), einsum('ijk,ijk',ms,ms,dtype=float64), einsum('ijk,ijk',m,ms,dtype=float64)
        n = m.size
        a = (mms - m1*ms1/n) / (m2 - m1*m1/n)
        b = (ms1 - a*m1) / n
        am = a*m
        am += b
        print ('scaling #%s' % v.id_string, a, b, m1, ms1, m2, ms2, mms, am.mean(), am.std(), ms.mean(), ms.std())
        m[:] = am.astype(m.dtype)

    if not enclose_volume is None or not fast_enclose_volume is None:
        set_enclosed_volume(v, enclose_volume, fast_enclose_volume)

    if not normalize_level is None:
        level = v.maximum_surface_level
        if level is None:
            from chimerax.core.errors import UserError
            raise UserError('vseries save: normalize_level used but no level set for volume %s' % v.name)
        if zero_mean:
            level -= mean
        scale = normalize_level / level
        m = (m*scale).astype(m.dtype)

    if not align_to is None:
        align(v, align_to)

    if not on_grid is None:
        vc = v.writable_copy(value_type = m.dtype, unshow_original = False)
        vc.full_matrix()[:,:,:] = m
        m = on_grid.full_matrix()
        m[:,:,:] = 0
        on_grid.add_interpolated_values(vc)
        vc.close()
        d = on_grid.data

    if not mask is None:
        m[:,:,:] *= mask.full_matrix()

    if not final_value_type is None:
        m = m.astype(final_value_type)

    from chimerax.map_data import ArrayGridData
    d = ArrayGridData(m, d.origin, d.step, d.cell_angles, d.rotation)

    return d

# -----------------------------------------------------------------------------
#
def vseries_measure(session, series, output = None, centroids = True,
                    color = None, radius = None):
    '''Report centroid motion of a map series.'''
    rgba = (170,170,170,255) if color is None else color.uint8x4()
    from chimerax.surface import surface_volume_and_area
    from chimerax.std_commands import measure_inertia
    meas = []
    for s in series:
        n = s.number_of_times()
        start_time = s.last_shown_time
        for t in range(n):
            if t != start_time:
                s.copy_display_parameters(start_time, t)
            s.show_time(t)
            v = s.maps[t]
            v.update_drawings()	# Compute surface.  Normally does not happen until rendered.
            level = v.minimum_surface_level
            if level is None:
                from chimerax.core.errors import UserError
                raise UserError('vseries measure (#%s) requires surface style display' % s.id_string +
                                ' since the surface threshold is used for computing centroid,' +
                                ' enclosed volume, area, and size')
            vol, area, holes = surface_volume_and_area(v)
            axes, d2, c = measure_inertia.map_inertia([v])
            elen = measure_inertia.inertia_ellipsoid_size(d2)
            meas.append((level, c, vol, area, elen))
        s.show_time(start_time)

        if centroids:
            if radius is None:
                radius = min(v.data.step)
            create_centroid_path(session, 'centroid path', tuple(m[1] for m in meas), radius, rgba)

        # Make text output
        lines = ['# Volume series measurements: %s\n' % s.name,
                 '#   n        level         x            y           z           step       distance       volume        area         inertia ellipsoid size  \n']
        d = 0
        cprev = None
        step = 0
        from chimerax.geometry import distance
        for n, (level, c, vol, area, elen) in enumerate(meas):
            if not cprev is None:
                step = distance(cprev, c)
                d += step
            cprev = c
            lines.append('%5d %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g\n' %
                         (n, level, c[0], c[1], c[2], step, d, vol, area, elen[0], elen[1], elen[2]))
        text = ''.join(lines)
        if output:
            from os import path
            path = path.expanduser(output)
            f = open(path, 'w')
            f.write(text)
            f.close()
        else:
            session.logger.info(text)
  
# -----------------------------------------------------------------------------
#
def create_centroid_path(session, name, xyz, radius, color):

    from chimerax.markers import MarkerSet, create_link
    mset = MarkerSet(session, name)
    mset.save_marker_attribute_in_sessions('frame', int)
    mprev = None
    for i,p in enumerate(xyz):
        m = mset.create_marker(p, color, radius)
        m.frame = i
        if mprev:
            create_link(mprev, m, rgba = color, radius = radius/2)
        mprev = m
    session.models.add([mset])
    return mset

# -----------------------------------------------------------------------------
#
def release_stopped_players():

  players.difference_update([p for p in players if p.play_handler is None])

# -----------------------------------------------------------------------------
#
def vseries_slider(session, series):
    '''Display a graphical user interface slider to play through frames of a map series.'''
    if len(series) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No volume series specified')
    from .slider import MapSeriesSlider
    MapSeriesSlider(session, series = series).show()

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import AtomSpecArg

class SeriesArg(AtomSpecArg):
    name = 'a map series specifier'

    @classmethod
    def parse(cls, text, session):
        value, used, rest = super().parse(text, session)
        models = value.evaluate(session).models
        from .series import MapSeries
        ms = [m for m in models if isinstance(m, MapSeries)]
        return ms, used, rest
