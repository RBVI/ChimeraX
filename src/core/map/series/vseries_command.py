# vim: set expandtab shiftwidth=4 softtabstop=4:
# ----------------------------------------------------------------------------
# Volume series command.
#
#   Syntax: vseries <operation> <mapSpec>
#
players = set()         # Active players.

def register_vseries_command():

    from ...commands import CmdDesc, register, BoolArg, EnumOf, IntArg, StringArg, FloatArg, AtomsArg, ColorArg
    from ..mapargs import MapArg, MapStepArg, MapRegionArg, ValueTypeArg, IntRangeArg

    sarg = [('series', SeriesArg)]

    align_desc = CmdDesc(required = sarg,
                         keyword = [('enclose_volume', FloatArg),
                                    ('fast_enclose_volume', FloatArg)])
    register('vseries align', align_desc, vseries_align)

    save_desc = CmdDesc(required = sarg + [('path', StringArg)],
                         keyword = [
                             ('subregion', MapRegionArg),
                             ('step', MapStepArg),
                             ('value_type', ValueTypeArg),
                             ('threshold', FloatArg),
                             ('zero_mean', BoolArg),
                             ('scale_factor', FloatArg),
                             ('enclose_volume', FloatArg),
                             ('fast_enclose_volume', FloatArg),
                             ('normalize_level', FloatArg),
                             ('align', BoolArg),
                             ('on_grid', MapArg),
                             ('mask', MapArg),
                             ('final_value_type', ValueTypeArg),
                             ('compress', BoolArg),])
    register('vseries save', save_desc, vseries_save)

    measure_desc = CmdDesc(required = sarg,
                           keyword = [('output', StringArg),
                                      ('centroids', BoolArg),
                                      ('color', ColorArg),
                                      ('radius', FloatArg),])
    register('vseries measure', measure_desc, vseries_measure)

    play_desc = CmdDesc(required = sarg,
                        keyword = [('loop', BoolArg),
                                   ('direction', EnumOf(('forward', 'backward', 'oscillate'))),
                                   ('normalize', BoolArg),
                                   ('max_frame_rate', FloatArg),
                                   ('markers', AtomsArg),
                                   ('preceding_marker_frames', IntArg),
                                   ('following_marker_frames', IntArg),
                                   ('color_range', FloatArg),
                                   ('cache_frames', IntArg),
                                   ('jump_to', IntArg),
                                   ('range', IntRangeArg),
                                   ('start_time', IntArg),])
    register('vseries play', play_desc, vseries_play)

    stop_desc = CmdDesc(required = sarg)
    register('vseries stop', stop_desc, vseries_stop)

    slider_desc = CmdDesc(required = sarg)
    register('vseries slider', slider_desc, vseries_slider)

# -----------------------------------------------------------------------------
#
def vseries_play(session, series, direction = 'forward', loop = False, max_frame_rate = None,
            jump_to = None, range = None, start = None, normalize = False, markers = None,
            preceding_marker_frames = 0, following_marker_frames = 0,
            color_range = None, cache_frames = 1):
    '''Show a sequence of maps from a volume series.'''
    from . import play
    p = play.Play_Series(series, session, range = range, start_time = start,
                         play_direction = direction,
                         loop = loop,
                         max_frame_rate = max_frame_rate,
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
        session.status('Aligning %s (%d of %d maps)' % (v.data.name, i+1, n))
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
    from ..fit.fitmap import map_points_and_weights, motion_to_maximum
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
            threshold = None, zero_mean = False, scale_factor = None,
            enclose_volume = None, fast_enclose_volume = None, normalize_level = None,
            align = False, on_grid = None, mask = None, final_value_type = None, compress = False):
    '''
    Process the frames of a map series and save the result to a a file.
    Processing can normalize, align, mask and change the numeric value type of maps.
    '''
    if len(series) > 1:
        from ...commands.parse import CommandError
        raise CommandError('vseries save: Can only save one series in a file, got %d'
                           % len(series))
    s = series[0]

    import os.path
    path = os.path.expanduser(path)         # Tilde expansion

    maps = s.maps
    if onGrid is None and align:
        onGrid = maps[0]

    grid = None
    if not on_grid is None:
        vtype = maps[0].data.value_type if value_type is None else value_type
        grid = on_grid.writable_copy(value_type = vtype, show = False)

    n = len(maps)
    for i,v in enumerate(maps):
        session.status('Writing %s (%d of %d maps)' % (v.data.name, i+1, n))
        align_to = maps[i-1] if align and i > 0 else None
        d = processed_volume(v, subregion, step, value_type, threshold, zero_mean, scale_factor,
                             enclose_volume, fast_enclose_volume, normalize_level,
                             align_to, grid, mask, final_value_type)
        d.name = '%04d' % i
        options = {'append': True, 'compress': compress}
        from ..data import cmap
        cmap.write_grid_as_chimera_map(d, path, options)

    if grid:
        grid.close()

# -----------------------------------------------------------------------------
#
def processed_volume(v, subregion = None, step = None, value_type = None, threshold = None,
                     zero_mean = False, scale_factor = None,
                     enclose_volume = None, fast_enclose_volume = None, normalize_level = None,
                     align_to = None, on_grid = None, mask = None, final_value_type = None):
    d = v.data
    if not subregion is None or not step is None:
        from ..volume import full_region
        ijk_min, ijk_max = full_region(d.size)[:2] if subregion is None else subregion
        ijk_step = (1,1,1) if step is None else step
        from ..data import Grid_Subregion
        d = Grid_Subregion(d, ijk_min, ijk_max, ijk_step)

    if (value_type is None and threshold is None and not zero_mean and
        scale_factor is None and align_to is None and mask is None and
        final_value_type is None):
        return d

    m = d.full_matrix()
    if not value_type is None:
        m = m.astype(value_type)

    if not threshold is None:
        from numpy import maximum, array
        maximum(m, array((threshold,),m.dtype), m)

    if zero_mean:
        from numpy import float64
        mean = m.mean(dtype = float64)
        m = (m - mean).astype(m.dtype)

    if not scale_factor is None:
        m = (m*scale_factor).astype(m.dtype)

    if not enclose_volume is None or not fast_enclose_volume is None:
        set_enclosed_volume(v, enclose_volume, fast_enclose_volume)

    if not normalize_level is None:
        if len(v.surface_levels) == 0:
            from ...commands.parse import CommandError
            raise CommandError('vseries save: normalize_level used but no level set for volume %s' % v.name)
        level = max(v.surface_levels)
        if zero_mean:
            level -= mean
        scale = normalize_level / level
        m = (m*scale).astype(m.dtype)

    if not align_to is None:
        align(v, align_to)

    if not on_grid is None:
        vc = v.writable_copy(value_type = m.dtype, show = False, unshow_original = False)
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

    from ..data import Array_Grid_Data
    d = Array_Grid_Data(m, d.origin, d.step, d.cell_angles, d.rotation)

    return d

# -----------------------------------------------------------------------------
#
def vseries_measure(session, series, output = None, centroids = True,
               color = (.7,.7,.7,1), radius = None):
    '''Report centroid motion of a map series.'''
    from ...surface import surface_volume_and_area
    from ...measure import inertia
    meas = []
    for s in series:
        n = s.number_of_times()
        for t in range(n):
            if t > 0:
                s.copy_display_parameters(0, t)
            shown = s.time_shown(t)
            s.show_time(t)
            v = s.maps[t]
            level = min(v.surface_levels)
            vol, area, holes = surface_volume_and_area(v)
            axes, d2, c = inertia.map_inertia([v])
            elen = inertia.inertia_ellipsoid_size(d2)
            meas.append((level, c, vol, area, elen))
            if not shown:
                s.unshow_time(t, cache_rendering = False)

        if centroids:
            if radius is None:
                radius = min(v.data.step)
            mol = create_centroid_path(tuple(m[1] for m in meas), radius, color)
            session.add_model(mol)

        # Make text output
        lines = ['# Volume series measurements: %s\n' % s.name,
                 '#   n        level         x            y           z           step       distance       volume        area         inertia ellipsoid size  \n']
        d = 0
        cprev = None
        step = 0
        from ...geometry.vector import distance
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
            session.info(text)
  
# -----------------------------------------------------------------------------
#
def create_centroid_path(xyz, radius, color):

    # TODO: This is obsolete Hydra code.
    n = len(xyz)
    from numpy import zeros, array, float32, arange, empty
    from ...atomic import atom_dtype, AtomicStructure
    atoms = zeros((n,), atom_dtype)
    atoms['atom_name'] = b's'
    atoms['element_number'] = 1
    atoms['xyz'] = array(xyz, float32)
    r = empty((n,), float32)
    r[:] = radius
    atoms['radius'] = r
    atoms['residue_name'] = b'S'
    atoms['residue_number'] = arange(0,n)
    atoms['chain_id'] = b'A'
    atoms['atom_color'] = tuple(int(255*c) for c in color)
    atoms['ribbon_color'] = (255,255,255,255)
    atoms['atom_shown'] = 1
    atoms['ribbon_shown'] = 0
    m = AtomicStructure('centroids', atoms)
    return m

# -----------------------------------------------------------------------------
#
def release_stopped_players():

  players.difference_update([p for p in players if p.play_handler is None])

# -----------------------------------------------------------------------------
#
def vseries_slider(session, series):
    '''Display a graphical user interface slider to play through frames of a map series.'''
    tool_info = session.toolshed.find_tool('map_series_gui')
    if tool_info:
        from chimera.map_series_gui.gui import MapSeries
        MapSeries(session, tool_info, series).show()

# -----------------------------------------------------------------------------
#
from ...commands import Annotation
class SeriesArg(Annotation):
    name = 'map series'
    @staticmethod
    def parse(text, session):
        from ...commands import AtomSpecArg
        value, used, rest = AtomSpecArg.parse(text, session)
        models = value.evaluate(session).models
        from .series import Map_Series
        ms = [m for m in models if isinstance(m, Map_Series)]
        return ms, used, rest
