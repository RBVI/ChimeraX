# vim: set expandtab shiftwidth=4 softtabstop=4:
# -----------------------------------------------------------------------------
# Implementation of "volume" command.
#
def register_volume_command():

    from ..commands import CmdDesc, register
    from ..commands import BoolArg, IntArg, StringArg, FloatArg, FloatsArg, NoArg, ListOf, EnumOf, Int3Arg, ColorArg
    from .mapargs import MapsArg, MapRegionArg, MapStepArg, Float1or3Arg, Int1or3Arg

    from .data.fileformats import file_writers
    stypes = [fw[1] for fw in file_writers]
    from .volume import Rendering_Options
    ro = Rendering_Options()

    volume_desc = CmdDesc(
        required = [('volumes', MapsArg)],
        keyword = [
               ('style', EnumOf(('surface', 'mesh', 'solid'))),
               ('show', NoArg),
               ('hide', NoArg),
               ('level', FloatsArg),
               ('enclose_volume', FloatsArg),
               ('fast_enclose_volume', FloatsArg),
               ('color', ListOf(ColorArg)),
               ('brightness', FloatArg),
               ('transparency', FloatArg),
               ('step', MapStepArg),
               ('region', MapRegionArg),
               ('name_region', StringArg),
               ('expand_single_plane', BoolArg),
               ('origin', Float1or3Arg),
               ('origin_index', Float1or3Arg),
               ('voxel_size', Float1or3Arg),
#               ('planes', planes_arg),
# Symmetry assignment.
               ('symmetry', StringArg),
               ('center', StringArg),
               ('center_index', Float1or3Arg),
               ('axis', StringArg),
#               ('coordinateSystem', openstate_arg),
# File saving options.
               ('save', StringArg),
               ('save_format', EnumOf(stypes)),
               ('save_region', MapRegionArg),
               ('save_step', Int1or3Arg),
               ('mask_zone', BoolArg),
               ('chunk_shapes', ListOf(EnumOf(('zyx','zxy','yxz','yzx','xzy','xyz')))),
               ('append', BoolArg),
               ('compress', BoolArg),
               ('base_index', IntArg),
# Global options.
               ('data_cache_size', FloatArg),
               ('show_on_open', BoolArg),
               ('voxel_limit_for_open', FloatArg),
               ('show_plane', BoolArg),
               ('voxel_limit_for_plane', FloatArg),
# Rendering options.
               ('show_outline_box', BoolArg),
               ('outline_box_rgb', ColorArg),
               ('outline_box_linewidth', FloatArg),
               ('limit_voxel_count', BoolArg),
               ('voxel_limit', FloatArg),
               ('color_mode', EnumOf(ro.color_modes)),
               ('projection_mode', EnumOf(ro.projection_modes)),
               ('bt_correction', BoolArg),
               ('minimal_texture_memory', BoolArg),
               ('maximum_intensity_projection', BoolArg),
               ('linear_interpolation', BoolArg),
               ('dim_transparency', BoolArg),
               ('dim_transparent_voxels', BoolArg),
               ('line_thickness', FloatArg),
               ('smooth_lines', BoolArg),
               ('mesh_lighting', BoolArg),
               ('two_sided_lighting', BoolArg),
               ('flip_normals', BoolArg),
               ('subdivide_surface', BoolArg),
               ('subdivision_levels', IntArg),
               ('surface_smoothing', BoolArg),
               ('smoothing_iterations', IntArg),
               ('smoothing_factor', FloatArg),
               ('square_mesh', BoolArg),
               ('cap_faces', BoolArg),
               ('box_faces', BoolArg),
               ('orthoplanes', EnumOf(('xyz', 'xy', 'xz', 'yz', 'off'))),
               ('position_planes', Int3Arg),
        ])
    register('volume', volume_desc, volume)
    
# -----------------------------------------------------------------------------
#
def volume(session,
           volumes,
           style = None,
           show = None,
           hide = None,
           level = None,
           enclose_volume = None,
           fast_enclose_volume = None,
           color = None,
           brightness = None,
           transparency = None,
           step = None,
           region = None,
           name_region = None,
           expand_single_plane = None,
           origin = None,
           origin_index = None,
           voxel_size = None,
           planes = None,
# Symmetry assignment.
           symmetry = None,
           center = (0,0,0),
           center_index = None,
           axis = (0,0,1),
           coordinate_system = None,
# File saving options.
           save = None,
           save_format = None,
           save_region = None,
           save_step = None,
           mask_zone = True,
           chunk_shapes = None,
           append = None,
           compress = None,
           base_index = 1,
# Global options.
           data_cache_size = None,
           show_on_open = None,
           voxel_limit_for_open = None,
           show_plane = None,
           voxel_limit_for_plane = None,
# Rendering options.
           show_outline_box = None,
           outline_box_rgb = None,
           outline_box_linewidth = None,
           limit_voxel_count = None,          # auto-adjust step size
           voxel_limit = None,               # Mvoxels
           color_mode = None,                # solid rendering pixel formats
           projection_mode = None,           # auto, 2d-xyz, 2d-x, 2d-y, 2d-z, 3d
           bt_correction = None,             # brightness and transparency
           minimal_texture_memory = None,
           maximum_intensity_projection = None,
           linear_interpolation = None,
           dim_transparency = None,          # for surfaces
           dim_transparent_voxels = None,     # for solid rendering
           line_thickness = None,
           smooth_lines = None,
           mesh_lighting = None,
           two_sided_lighting = None,
           flip_normals = None,
           subdivide_surface = None,
           subdivision_levels = None,
           surface_smoothing = None,
           smoothing_iterations = None,
           smoothing_factor = None,
           square_mesh = None,
           cap_faces = None,
           box_faces = None,
           orthoplanes = None,
           position_planes = None,
           ):
    '''
    Control the display of density maps.

    Parameters
    ----------
    volumes : list of maps
    style : "surface", "mesh", or "solid"
    show : bool
    hide : bool
    level : sequence of 1 or 2 floats
      In solid style 2 floats are used the first being a density level and second 0-1 brightness value.
    enclose_volume : float
    fast_enclose_volume : float
    color : Color
    brightness : float
    transparency : float
    step : sequence of 3 integers
    region : sequence of 6 integers
      3 minimum grid indices and 3 maximum grid indices for x,y,z axes.
    name_region : string
    expand_single_plane : bool
    origin : sequence of 3 floats
    origin_index : sequence of 3 floats
    voxel_size : sequence of 3 floats
    planes : not currently supported

    ------------------------------------------------------------------------------------------------
    Symmetry assignment options
    ------------------------------------------------------------------------------------------------

    symmetry : string
    center : string
      Parsed as 3 comma-separated floats, or an atom specifier
    center_index : sequence of 3 floats
    axis : sequence of 3 floats
    coordinate_system : not supported, model specifier

    ------------------------------------------------------------------------------------------------
    File saving options
    ------------------------------------------------------------------------------------------------

    save : string
      File name
    save_format : string
    save_region : sequence of 6 integers
    save_step : sequence of 3 integers
    mask_zone : bool
    chunk_shapes : list of "zyx", "zxy", "yxz", "yzx", "xzy", "xyz"
    append : bool
    compress : bool
    base_index : integer

    ------------------------------------------------------------------------------------------------
    Global options
    ------------------------------------------------------------------------------------------------

    data_cache_size : float
      In Mbytes
    show_on_open : bool
    voxel_limit_for_open : float
    show_plane : bool
    voxel_limit_for_plane : float

    ------------------------------------------------------------------------------------------------
    Rendering options
    ------------------------------------------------------------------------------------------------

    show_outline_box : bool
    outline_box_rgb : Color
    outline_box_linewidth : float
    limit_voxel_count : bool
      Auto-adjust step size.
    voxel_limit : float (Mvoxels)
    color_mode : string
      Solid rendering pixel formats: 'auto4', 'auto8', 'auto12', 'auto16',
      'opaque4', 'opaque8', 'opaque12', 'opaque16', 'rgba4', 'rgba8', 'rgba12', 'rgba16',
      'rgb4', 'rgb8', 'rgb12', 'rgb16', 'la4', 'la8', 'la12', 'la16', 'l4', 'l8', 'l12', 'l16'
    projection_mode : string
      One of 'auto', '2d-xyz', '2d-x', '2d-y', '2d-z', '3d'
    bt_correction : bool
      Brightness and transparency view angle correction for solid mode.
    minimal_texture_memory : bool
      Reduce graphics memory use for solid rendering at the expense of rendering speed.
    maximum_intensity_projection : bool
    linear_interpolation : bool
      Interpolate gray levels in solid style rendering.
    dim_transparency : bool
      Makes transparent surfaces dimmer
    dim_transparent_voxels : bool
      For solid rendering.
    line_thickness : float
    smooth_lines : bool
    mesh_lighting : bool
    two_sided_lighting : bool
    flip_normals : bool
    subdivide_surface : bool
    subdivision_levels : integer
    surface_smoothing : bool
    smoothing_iterations : integer
    smoothing_factor : float
    square_mesh : bool
    cap_faces : bool
    box_faces : bool
    orthoplanes : One of 'xyz', 'xy', 'xz', 'yz', 'off'
    position_planes : sequence of 3 integers
      Intersection grid point of orthoplanes display
    '''
    vlist = volumes

    # Special defaults
    if not box_faces is None:
        defaults = (('style', 'solid'), ('color_mode', 'opaque8'),
                    ('show_outline_box', True), ('expand_single_plane', True),
                    ('orthoplanes', 'off'))
    elif not orthoplanes is None and orthoplanes != 'off':
        defaults = (('style', 'solid'), ('color_mode', 'opaque8'),
                    ('show_outline_box', True), ('expand_single_plane', True))
    elif not box_faces is None or not orthoplanes is None:
        defaults = (('color_mode', 'auto8'),)
    else:
        defaults = ()
    loc = locals()
    for opt, value in defaults:
        if loc[opt] is None:
            loc[opt] = value

    # Adjust global settings.
    loc = locals()
    gopt = ('data_cache_size', 'show_on_open', 'voxel_limit_for_open',
            'show_plane', 'voxel_limit_for_plane')
    gsettings = dict((n,loc[n]) for n in gopt if not loc[n] is None)
    if gsettings:
        apply_global_settings(gsettings)

    if len(gsettings) == 0 and len(vlist) == 0:
        from .. import errors
        raise errors.UserError('No volumes specified%s' %
                            (' by "%s"' % volumes if volumes else ''))

    # Apply volume settings.
    dopt = ('style', 'show', 'hide', 'level', 'enclose_volume', 'fast_enclose_volume',
            'color', 'brightness', 'transparency',
            'step', 'region', 'name_region', 'expand_single_plane', 'origin',
            'origin_index', 'voxel_size', 'planes',
            'symmetry', 'center', 'center_index', 'axis', 'coordinate_system')
    dsettings = dict((n,loc[n]) for n in dopt if not loc[n] is None)
    ropt = (
        'show_outline_box', 'outline_box_rgb', 'outline_box_linewidth',
        'limit_voxel_count', 'voxel_limit', 'color_mode', 'projection_mode',
        'bt_correction', 'minimal_texture_memory', 'maximum_intensity_projection',
        'linear_interpolation', 'dim_transparency', 'dim_transparent_voxels',
        'line_thickness', 'smooth_lines', 'mesh_lighting',
        'two_sided_lighting', 'flip_normals', 'subdivide_surface',
        'subdivision_levels', 'surface_smoothing', 'smoothing_iterations',
        'smoothing_factor', 'square_mesh', 'cap_faces', 'box_faces')
    rsettings = dict((n,loc[n]) for n in ropt if not loc[n] is None)
    if not orthoplanes is None:
        rsettings['orthoplanes_shown'] = ('x' in orthoplanes,
                                         'y' in orthoplanes,
                                         'z' in orthoplanes)
    if not position_planes is None:
        rsettings['orthoplane_positions'] = position_planes

    for v in vlist:
        apply_volume_options(v, dsettings, rsettings, session)

    # Save files.
    fopt = ('save', 'save_format', 'save_region', 'save_step', 'mask_zone',
            'chunk_shapes', 'append', 'compress', 'base_index')
    fsettings = dict((n,loc[n]) for n in fopt if not loc[n] is None)
    save_volumes(vlist, fsettings, session)
    
# -----------------------------------------------------------------------------
#
def apply_global_settings(gsettings):

# TODO: Unused settings part of gui in Chimera 1.
#    from .volume import default_settings
#    default_settings.update(gsettings)
    if 'data_cache_size' in gsettings:
        from .data import data_cache
        data_cache.resize(gsettings['data_cache_size'] * (2**20))
    
# -----------------------------------------------------------------------------
#
def apply_volume_options(v, doptions, roptions, session):

    if 'style' in doptions:
        v.set_representation(doptions['style'])

    kw = level_and_color_settings(v, doptions)
    kw.update(roptions)
    if kw:
        v.set_parameters(**kw)

    if 'enclose_volume' in doptions:
        levels = [v.surface_level_for_enclosed_volume(ev) for ev in doptions['enclose_volume']]
        v.set_parameters(surface_levels = levels)
    elif 'fast_enclose_volume' in doptions:
        levels = [v.surface_level_for_enclosed_volume(ev, rank_method = True)
                  for ev in doptions['fast_enclose_volume']]
        v.set_parameters(surface_levels = levels)

    if 'region' in doptions or 'step' in doptions:
        r = v.subregion(doptions.get('step', None),
                        doptions.get('region', None))
    else:
        r = None
    if not r is None:
        ijk_min, ijk_max, ijk_step = r
        v.new_region(ijk_min, ijk_max, ijk_step, show = False,
                     adjust_step = not 'step' in doptions)
    if doptions.get('expand_single_plane', False):
        v.expand_single_plane()

    if 'name_region' in doptions:
        name = doptions['name_region']
        if r is None:
            r = v.region
        if r:
            v.region_list.add_named_region(name, r[0], r[1])

    if 'planes' in doptions:
        from . import volume
        volume.cycle_through_planes(v, session, *doptions['planes'])

    d = v.data
    if 'origin_index' in doptions:
        index_origin = doptions['origin_index']
        xyz_origin = [-a*b for a,b in zip(index_origin, d.step)]
        d.set_origin(xyz_origin)
    elif 'origin' in doptions:
        origin = doptions['origin']
        d.set_origin(origin)

    if 'voxel_size' in doptions:
        vsize = doptions['voxel_size']
        if min(vsize) <= 0:
            from .. import errors
            raise errors.UserError('Voxel size must positive, got %g,%g,%g'
                                % tuple(vsize))
        # Preserve index origin.
        origin = [(a/b)*c for a,b,c in zip(d.origin, d.step, vsize)]
        d.set_origin(origin)
        d.set_step(vsize)

    if 'symmetry' in doptions:
        sym, c, a = doptions['symmetry'], doptions['center'], doptions['axis']
        csys = doptions.get('coordinate_system', v.open_state)
        if 'center_index' in doptions:
            c = v.data.ijk_to_xyz(doptions['center_index'])
            if csys != v.openState:
                c = csys.position.inverse() * (v.position * c)
        from ..SymmetryCopies import symcmd
        tflist, csys = symcmd.parse_symmetry(sym, c, a, csys, v, 'volume')
        if csys != v.openState:
            tflist = symcmd.transform_coordinates(tflist, csys, v.openState)
        d.symmetries = tflist

    if 'show' in doptions:
        v.initialize_thresholds()
        v.show()
    elif 'hide' in doptions:
        v.unshow()
    elif v.shown():
        v.show()
# TODO: If it has a surface but it is undisplayed, do I need to recalculate it?
#    else:
#        v.update_display()

# TODO:
#  Allow quoted color names.
#  Could allow region name "full" or "back".
#  Could allow voxel_size or origin to be "original".

# -----------------------------------------------------------------------------
#
def save_volumes(vlist, doptions, session):

    if not 'save' in doptions:
        return
    
    path = doptions['save']
    format = doptions.get('save_format', None)
    from .data import fileformats
    if fileformats.file_writer(path, format) is None: 
        format = 'mrc' 
    options = {}
    if 'chunk_shapes' in doptions:
        options['chunk_shapes'] = doptions['chunk_shapes']
    if 'append' in doptions and doptions['append']:
        options['append'] = True
    if 'compress' in doptions and doptions['compress']:
        options['compress'] = True
    if path in ('browse', 'browser'):
        from .data import select_save_path
        path, format = select_save_path()
    if path:
        subregion = doptions.get('save_region', None)
        step = doptions.get('save_step', (1,1,1))
        mask_zone = doptions.get('mask_zone', True)
        base_index = doptions.get('base_index', 1)
        grids = [v.grid_data(subregion, step, mask_zone) for v in vlist]
        from .data import save_grid_data
        if is_multifile_save(path):
            for i,g in enumerate(grids):
                save_grid_data(g, path % (i + base_index), session, format, options)
        else:
            save_grid_data(grids, path, session, format, options)
   
# -----------------------------------------------------------------------------
# Check if file name contains %d type format specification.
#
def is_multifile_save(path):
    try:
        path % 0
    except:
        return False
    return True
   
# -----------------------------------------------------------------------------
#
def level_and_color_settings(v, options):

    kw = {}

    levels = options.get('level', [])
    colors = options.get('color', [])

    # Allow 0 or 1 colors and 0 or more levels, or number colors matching
    # number of levels.
    if len(colors) > 1 and len(colors) != len(levels):
        from .. import errors
        raise errors.UserError('Number of colors (%d) does not match number of levels (%d)'
                            % (len(colors), len(levels)))

    style = options.get('style', v.representation)
    if style in ('mesh', None):
        style = 'surface'

    if style == 'solid':
        if len(levels) % 2:
            from .. import errors
            raise errors.UserError('Solid level must be <data-value,brightness-level>')
        if levels and len(levels) < 4:
            from .. import errors
            raise errors.UserError('Must specify 2 or more levels for solid style')
        levels = tuple(zip(levels[::2], levels[1::2]))

    if levels:
        kw[style+'_levels'] = levels

    if len(colors) == 1:
        if levels:
            clist = [colors[0].rgba]*len(levels)
        else:
            clist = [colors[0].rgba]*len(getattr(v, style + '_levels'))
        kw[style+'_colors'] = clist
    elif len(colors) > 1:
        kw[style+'_colors'] = [c.rgba for c in colors]

    if len(levels) == 0 and len(colors) == 1:
        kw['default_rgba'] = colors[0].rgba

    if 'brightness' in options:
        kw[style+'_brightness_factor'] = options['brightness']

    if 'transparency' in options:
        if style == 'surface':
            kw['transparency_factor'] = options['transparency']
        else:
            kw['transparency_depth'] = options['transparency']

    return kw

# -----------------------------------------------------------------------------
# Arguments are axis,pstart,pend,pstep,pdepth.
#
def planes_arg(planes, session):

    axis, param = (planes.split(',',1) + [''])[:2]
    from ..commands.parse import enum_arg, floats_arg
    p = [enum_arg(axis, session, ('x','y','z'))] + floats_arg(param, session)
    if len(p) < 2 or len(p) > 5:
        from .. import errors
        raise errors.UserError('planes argument must have 2 to 5 comma-separated values: axis,pstart[[[,pend],pstep],pdepth.], got "%s"' % planes)
    return p

# -----------------------------------------------------------------------------
# Find maps among models and all descendants.
#
def all_maps(models):
    maps = []
    from .volume import Volume
    from ..models import Model
    for m in models:
        if isinstance(m, Volume):
            maps.append(m)
        if isinstance(m, Model):
            maps.extend(all_maps(m.child_drawings()))
    return maps
