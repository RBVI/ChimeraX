# ----------------------------------------------------------------------------
# Volume series command.
#
#   Syntax: vseries <operation> <mapSpec>
#
from ...commands.parse import CommandError

players = set()         # Active players.

def vseries_command(cmd_name, args, session):

    from ...commands.parse import bool_arg, float_arg, enum_arg, int_arg, string_arg, color_arg, range_arg
    from ...commands.parse import floats_arg, parse_subregion, value_type_arg, volume_arg, molecule_arg
    from ...commands.parse import perform_operation
    ops = {
        'align': (align_op,
                  (('series', series_arg),),
                  (),
                  (('encloseVolume', float_arg),
                   ('fastEncloseVolume', float_arg))),
        'save': (save_op,
                 (('series', series_arg),
                  ('path', string_arg)),
                 (),
                 (('subregion', parse_subregion),
                  ('valueType', value_type_arg),
                  ('threshold', float_arg),
                  ('zeroMean', bool_arg),
                  ('scaleFactor', float_arg),
                  ('encloseVolume', float_arg),
                  ('fastEncloseVolume', float_arg),
                  ('normalizeLevel', float_arg),
                  ('align', bool_arg),
                  ('onGrid', volume_arg),
                  ('mask', volume_arg),
                  ('finalValueType', value_type_arg),
                  ('compress', bool_arg)),
                 ),
        'measure': (measure_op,
                    (('series', series_arg),),
                    (),
                    (('output', string_arg),
                     ('centroids', bool_arg),
                     ('color', color_arg),
                     ('radius', float_arg),
                    )),
        'play': (play_op,
                 (('series', series_arg),),
                 (),
                 (('loop', bool_arg),
                  ('direction', enum_arg,
                   {'values':('forward', 'backward', 'oscillate')}),
                  ('normalize', bool_arg),
                  ('maxFrameRate', float_arg),
                  ('markers', molecule_arg),
                  ('precedingMarkerFrames', int_arg),
                  ('followingMarkerFrames', int_arg),
                  ('colorRange', float_arg),
                  ('cacheFrames', int_arg),
                  ('jumpTo', int_arg),
                  ('range', range_arg),
                  ('startTime', int_arg),
                  )),
        'stop': (stop_op,
                 (('series', series_arg),),
                 (),
                 ()),
        'slider': (slider_op,
                  (('series', series_arg),),
                  (),
                  ()),
        }

    perform_operation(cmd_name, args, ops, session)

# -----------------------------------------------------------------------------
#
def play_op(series, direction = 'forward', loop = False, maxFrameRate = None,
            jumpTo = None, range = None, start = None, normalize = False, markers = None,
            precedingMarkerFrames = 0, followingMarkerFrames = 0,
            colorRange = None, cacheFrames = 1, session = None):

    from . import play
    p = play.Play_Series(series, session, range = range, start_time = start,
                         play_direction = direction,
                         loop = loop,
                         max_frame_rate = maxFrameRate,
                         normalize_thresholds = normalize,
                         markers = markers,
                         preceding_marker_frames = precedingMarkerFrames,
                         following_marker_frames = followingMarkerFrames,
                         color_range = colorRange,
                         rendering_cache_size = cacheFrames)
    if not jumpTo is None:
        p.change_time(jumpTo)
    else:
        global players
        players.add(p)
        p.play()
    release_stopped_players()
    return p

# -----------------------------------------------------------------------------
#
def stop_op(series):

    for p in players:
        for s in series:
            if s in p.series:
                p.stop()
    release_stopped_players()

# -----------------------------------------------------------------------------
#
def align_op(series, encloseVolume = None, fastEncloseVolume = None, session = None):
    for s in series:
        align_series(s, encloseVolume, fastEncloseVolume, session)

# -----------------------------------------------------------------------------
#
def align_series(s, enclose_volume = None, fast_enclose_volume = None, session = None):

    n = len(s.maps)
    vprev = None
    for i,v in enumerate(s.maps):
        session.show_status('Aligning %s (%d of %d maps)' % (v.data.name, i+1, n))
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
def save_op(series, path, subregion = None, valueType = None,
            threshold = None, zeroMean = False, scaleFactor = None,
            encloseVolume = None, fastEncloseVolume = None, normalizeLevel = None,
            align = False, onGrid = None, mask = None, finalValueType = None, compress = False,
            session = None):

    if len(series) > 1:
        raise CommandError('vseries save: Can only save one series in a file, got %d'
                           % len(series))
    s = series[0]

    import os.path
    path = os.path.expanduser(path)         # Tilde expansion

    maps = s.maps
    if onGrid is None and align:
        onGrid = maps[0]

    on_grid = None
    if not onGrid is None:
        vtype = maps[0].data.value_type if valueType is None else valueType
        on_grid = onGrid.writable_copy(value_type = vtype, show = False)

    n = len(maps)
    for i,v in enumerate(maps):
        session.show_status('Writing %s (%d of %d maps)' % (v.data.name, i+1, n))
        align_to = maps[i-1] if align and i > 0 else None
        d = processed_volume(v, subregion, valueType, threshold, zeroMean, scaleFactor,
                             encloseVolume, fastEncloseVolume, normalizeLevel,
                             align_to, on_grid, mask, finalValueType)
        d.name = '%04d' % i
        options = {'append': True, 'compress': compress}
        from ..data import cmap
        cmap.write_grid_as_chimera_map(d, path, options)

    if on_grid:
        on_grid.close()

# -----------------------------------------------------------------------------
#
def processed_volume(v, subregion = None, value_type = None, threshold = None,
                     zeroMean = False, scaleFactor = None,
                     encloseVolume = None, fastEncloseVolume = None, normalizeLevel = None,
                     align_to = None, on_grid = None, mask = None, final_value_type = None):
    d = v.data
    if not subregion is None:
        ijk_min, ijk_max = subregion
        from ..data import Grid_Subregion
        d = Grid_Subregion(d, ijk_min, ijk_max)

    if (value_type is None and threshold is None and not zeroMean and
        scaleFactor is None and align_to is None and mask is None and
        final_value_type is None):
        return d

    m = d.full_matrix()
    if not value_type is None:
        m = m.astype(value_type)

    if not threshold is None:
        from numpy import maximum, array
        maximum(m, array((threshold,),m.dtype), m)

    if zeroMean:
        from numpy import float64
        mean = m.mean(dtype = float64)
        m = (m - mean).astype(m.dtype)

    if not scaleFactor is None:
        m = (m*scaleFactor).astype(m.dtype)

    if not encloseVolume is None or not fastEncloseVolume is None:
        set_enclosed_volume(v, encloseVolume, fastEncloseVolume)

    if not normalizeLevel is None:
        if len(v.surface_levels) == 0:
            raise CommandError('vseries save: normalizeLevel used but no level set for volume %s' % v.name)
        level = max(v.surface_levels)
        if zeroMean:
            level -= mean
        scale = normalizeLevel / level
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
def measure_op(series, output = None, centroids = True,
               color = (.7,.7,.7,1), radius = None,
               session = None):

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
            session.show_info(text)
  
# -----------------------------------------------------------------------------
#
def create_centroid_path(xyz, radius, color):

    n = len(xyz)
    from numpy import zeros, array, float32, arange, empty
    from ...molecule import atom_dtype, Molecule
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
    m = Molecule('centroids', atoms)
    return m
  
# -----------------------------------------------------------------------------
#
def series_arg(s, session):

  from ...commands import parse
  mlist = parse.models_arg(s, session)
  from . import Map_Series
  series = [m for m in mlist if isinstance(m, Map_Series)]
  if len(series) == 0:
    raise CommandError('"%s" does not specify a volume series' % s)
  return series

# -----------------------------------------------------------------------------
#
def release_stopped_players():

  players.difference_update([p for p in players if p.play_handler is None])

# -----------------------------------------------------------------------------
#
def slider_op(series, session):

    from . import slider
    vss = slider.Volume_Series_Slider(series, session)
    vss.show()
