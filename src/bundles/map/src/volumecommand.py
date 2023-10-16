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
# Implementation of "volume" command.
#
def register_volume_command(logger):

    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import BoolArg, IntArg, StringArg, FloatArg, FloatsArg, NoArg, ListOf, EnumOf, Int3Arg, ColorArg, CenterArg, AxisArg, CoordSysArg, RepeatOf, Or
    from chimerax.atomic import SymmetryArg
    from .mapargs import MapsArg, MapRegionArg, MapStepArg, Float1or3Arg, Int1or3Arg
    from .colortables import AppearanceArg

    global_options = [
               ('data_cache_size', FloatArg),
               ('show_on_open', BoolArg),
               ('voxel_limit_for_open', FloatArg),
               ('show_plane', BoolArg),
               ('voxel_limit_for_plane', FloatArg),
        ]

    from .volume import RenderingOptions
    ro = RenderingOptions()

    rendering_options = [
               ('show_outline_box', BoolArg),
               ('outline_box_rgb', ColorArg),
#               ('outline_box_linewidth', FloatArg),
               ('limit_voxel_count', BoolArg),
               ('voxel_limit', FloatArg),
               ('colormap_on_gpu', BoolArg),
               ('color_mode', EnumOf(ro.color_modes)),
               ('colormap_size', IntArg),
               ('colormap_extend_left', BoolArg),
               ('colormap_extend_right', BoolArg),
               ('blend_on_gpu', BoolArg),
               ('projection_mode', EnumOf(ro.projection_modes)),
               ('plane_spacing', Or(EnumOf(('min', 'max', 'mean')), FloatArg)),
               ('full_region_on_gpu', BoolArg),
               ('bt_correction', BoolArg),
               ('minimal_texture_memory', BoolArg),
               ('maximum_intensity_projection', BoolArg),
               ('linear_interpolation', BoolArg),
               ('dim_transparency', BoolArg),
               ('dim_transparent_voxels', BoolArg),
#               ('line_thickness', FloatArg),
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
               ('position_planes', Int3Arg),
               ('tilted_slab_axis', AxisArg),
               ('tilted_slab_offset', FloatArg),
               ('tilted_slab_spacing', FloatArg),
               ('tilted_slab_plane_count', IntArg),
               ('backing_color', Or(ColorArg, EnumOf(['none']))),
        ]
    
    volume_desc = CmdDesc(
        optional = [('volumes', MapsArg)],
        keyword = [
               ('style', EnumOf(('surface', 'mesh', 'image', 'solid'))),
               ('change', EnumOf(('surface', 'image'))),
               ('show', NoArg),
               ('hide', NoArg),
               ('toggle', NoArg),
               ('close', EnumOf(('surface', 'image'))),
               ('level', RepeatOf(FloatsArg)),
               ('rms_level', RepeatOf(FloatsArg)),
               ('sd_level', RepeatOf(FloatsArg)),
               ('enclose_volume', FloatsArg),
               ('fast_enclose_volume', FloatsArg),
               ('color', RepeatOf(ColorArg)),
               ('brightness', FloatArg),
               ('transparency', FloatArg),
               ('appearance', AppearanceArg(logger.session)),
               ('name_appearance', StringArg),
               ('name_forget', AppearanceArg(logger.session)),
               ('step', MapStepArg),
               ('region', MapRegionArg),
#               ('name_region', StringArg),
               ('expand_single_plane', BoolArg),
               ('origin', Float1or3Arg),
               ('origin_index', Float1or3Arg),
               ('voxel_size', Float1or3Arg),
               ('planes', PlanesArg),
               ('dump_header', BoolArg),
               ('pickable', BoolArg),
               ('calculate_surfaces', BoolArg),
               ('box_faces', BoolArg),
               ('orthoplanes', EnumOf(('xyz', 'xy', 'xz', 'yz', 'off'))),
               ('tilted_slab', BoolArg),
               ('image_mode', EnumOf(('full region', 'orthoplanes', 'box faces', 'tilted slab'))),
# Symmetry assignment.
               ('symmetry', SymmetryArg),
               ('center', CenterArg),
               ('center_index', Float1or3Arg),
               ('axis', AxisArg),
               ('coordinate_system', CoordSysArg),
        ] + global_options + rendering_options,
        synopsis = 'set volume model parameters, display style and colors')
    register('volume', volume_desc, volume, logger=logger)

    # Register volume settings command
    vsettings_desc = CmdDesc(optional = [('volumes', MapsArg)],
                             synopsis = 'report volume display settings')
    register('volume settings', vsettings_desc, volume_settings, logger=logger)

    # Register volume defaultvalues command
    dsettings_desc = CmdDesc(keyword = [('save_settings', BoolArg), ('reset', BoolArg)] + global_options + rendering_options,
                             synopsis = 'set or report volume default values')
    register('volume defaultvalues', dsettings_desc, volume_default_values, logger=logger)

    # Register volume channels command
    from . import channels
    channels.register_volume_channels_command(logger)

    
# -----------------------------------------------------------------------------
#
def volume(session,
           volumes = None,
           style = None,
           change = None,
           show = None,
           hide = None,
           toggle = None,
           close = None,
           level = None,
           rms_level = None,
           sd_level = None,
           enclose_volume = None,
           fast_enclose_volume = None,
           color = None,
           brightness = None,
           transparency = None,
           appearance = None,
           name_appearance = None,
           name_forget = None,
           step = None,
           region = None,
           name_region = None,
           expand_single_plane = None,
           origin = None,
           origin_index = None,
           voxel_size = None,
           planes = None,
           dump_header = None,
           pickable = None,
# Symmetry assignment.
           symmetry = None,
           center = None,
           center_index = None,
           axis = None,
           coordinate_system = None,
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
           color_mode = None,                # image rendering pixel formats
           colormap_on_gpu = None,           # image colormapping on gpu or cpu
           colormap_size = None,             # image colormapping
           colormap_extend_left = None,
           colormap_extend_right = None,
           backing_color = None,
           blend_on_gpu = None,		     # image blending on gpu or cpu
           projection_mode = None,           # auto, 2d-xyz, 2d-x, 2d-y, 2d-z, 3d
           plane_spacing = None,	     # min, max, or numeric value
           full_region_on_gpu = None,	     # for fast cropping with image rendering
           bt_correction = None,             # brightness and transparency
           minimal_texture_memory = None,
           maximum_intensity_projection = None,
           linear_interpolation = None,
           dim_transparency = None,          # for surfaces
           dim_transparent_voxels = None,     # for image rendering
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
           tilted_slab = None,
           tilted_slab_axis = None,
           tilted_slab_spacing = None,
           tilted_slab_offset = None,
           tilted_slab_plane_count = None,
           image_mode = None,
           calculate_surfaces = None
           ):
    '''
    Control the display of density maps.

    Parameters:
        volumes: list of maps
        style: "surface", "mesh", or "image"
        change: "surface" or "image"
            Determines if level, color, brightness, transparency options apply to surface style or image style.
            If this option is not specified then the currently shown style is used.
        show: bool
        hide: bool
        toggle: bool
        close: "surface" or "image"
        level: sequence of 1 or 2 floats
            In image style 2 floats are used the first being a density level and second 0-1 brightness value.
        enclose_volume: float
        fast_enclose_volume: float
        color: Color
        brightness: float
        transparency: float
        step: sequence of 3 integers
        region: sequence of 6 integers
            3 minimum grid indices and 3 maximum grid indices for x,y,z axes.
        name_region: string
        expand_single_plane: bool
        origin: sequence of 3 floats
        origin_index: sequence of 3 floats
        voxel_size: sequence of 3 floats
        planes: tuple of (axis, start, end, increment, depth), last 3 are optional


    Symmetry assignment options

    Parameters:
        symmetry: string
        center: string
            Parsed as 3 comma-separated floats, or an atom specifier
        center_index: sequence of 3 floats
        axis: sequence of 3 floats
        coordinate_system: Place
            Coordinate system for axis and center symmetry options


    Global options

    Parameters:
        data_cache_size: float
            In Mbytes
        show_on_open: bool
        voxel_limit_for_open: float
        show_plane: bool
        voxel_limit_for_plane: float


    Rendering options

    Parameters:
        show_outline_box: bool
        outline_box_rgb: Color
        outline_box_linewidth: float
        limit_voxel_count: bool
            Auto-adjust step size.
        voxel_limit: float (Mvoxels)
        color_mode: string
            Image rendering pixel formats: 'auto4', 'auto8', 'auto12', 'auto16',
            'opaque4', 'opaque8', 'opaque12', 'opaque16', 'rgba4', 'rgba8', 'rgba12', 'rgba16',
            'rgb4', 'rgb8', 'rgb12', 'rgb16', 'la4', 'la8', 'la12', 'la16', 'l4', 'l8', 'l12', 'l16'
        colormap_on_gpu: bool
            Whether colormapping is done on gpu or cpu for image rendering.
        colormap_size: integer
            Size of colormap to use for image rendering.
        backing_color: Color or None
            Draw this color behind transparent image data if different from the background color.
        blend_on_gpu: bool
            Whether image blending is done on gpu or cpu.
        projection_mode: string
            One of 'auto', '2d-xyz', '2d-x', '2d-y', '2d-z', '3d'
        plane_spacing: "min", "max", "mean" or float
            Spacing between planes when using 3d projection mode.  "min", "max", "mean" use
            minimum, maximum or average grid spacing along x,y,z axes.
        full_region_on_gpu: bool
            Whether to cache data on GPU for fast cropping.
        bt_correction: bool
            Brightness and transparency view angle correction for image rendering mode.
        minimal_texture_memory: bool
            Reduce graphics memory use for image rendering at the expense of rendering speed.
        maximum_intensity_projection: bool
        linear_interpolation: bool
            Interpolate gray levels in image style rendering.
        dim_transparency: bool
            Makes transparent surfaces dimmer
        dim_transparent_voxels: bool
            For image rendering.
        line_thickness: float
        smooth_lines: bool
        mesh_lighting: bool
        two_sided_lighting: bool
        flip_normals: bool
        subdivide_surface: bool
        subdivision_levels: integer
        surface_smoothing: bool
        smoothing_iterations: integer
        smoothing_factor: float
        square_mesh: bool
        cap_faces: bool
        box_faces: bool
        orthoplanes: One of 'xyz', 'xy', 'xz', 'yz', 'off'
        position_planes: sequence of 3 integers
            Intersection grid point of orthoplanes display
        tilted_slab: bool
        tilted_slab_axis: sequence of 3 floats
        tilted_slab_offset: float
        tilted_slab_spacing: float
        tilted_slab_plane_count: int
        image_mode: 'full region', 'orthoplanes', 'box faces', or 'tilted slab'
        calculate_surfaces: bool
            Whether to calculate surfaces immediately instead of waiting until they are drawn.
    '''
    if volumes is None:
        from . import Volume
        vlist = session.models.list(type = Volume)
    else:
        vlist = volumes

    if style == 'solid':
        style = 'image'	# Rename solid to image.

    # Special defaults
    if box_faces or image_mode == 'box faces':
        defaults = (('style', 'image'), ('image_mode', 'box faces'), ('color_mode', 'opaque8'),
                    ('show_outline_box', True), ('expand_single_plane', True),
                    ('orthoplanes', 'off'), ('tilted_slab', False))
    elif (orthoplanes is not None and orthoplanes != 'off') or image_mode == 'orthoplanes':
        defaults = (('style', 'image'), ('image_mode', 'orthoplanes'), ('color_mode', 'opaque8'),
                    ('orthoplanes', 'xyz'), ('show_outline_box', True), ('expand_single_plane', True))
    elif tilted_slab or image_mode == 'tilted slab':
        defaults = (('style', 'image'), ('image_mode', 'tilted slab'), ('color_mode', 'auto8'),
                    ('show_outline_box', True), ('expand_single_plane', True))
    elif image_mode == 'full region':
        defaults = (('style', 'image'), ('color_mode', 'auto8'),)
    else:
        defaults = ()
    loc = locals()
    for opt, value in defaults:
        if loc[opt] is None:
            loc[opt] = value

    # Adjust global settings.
    gopt = ('data_cache_size', 'show_on_open', 'voxel_limit_for_open',
            'show_plane', 'voxel_limit_for_plane')
    if volumes is None:
        gopt += ('pickable',)
    gsettings = dict((n,loc[n]) for n in gopt if not loc[n] is None)
    if gsettings:
        apply_global_settings(session, gsettings)

    if len(gsettings) == 0 and len(vlist) == 0:
        from chimerax.core import errors
        raise errors.UserError('No volumes specified%s' %
                            (' by "%s"' % volumes if volumes else ''))

    # Apply volume settings.
    dopt = ('style', 'change', 'show', 'hide', 'toggle', 'close',
            'level', 'rms_level', 'sd_level',
            'enclose_volume', 'fast_enclose_volume',
            'color', 'brightness', 'transparency',
            'appearance', 'name_appearance', 'name_forget',
            'step', 'region', 'name_region', 'expand_single_plane', 'origin',
            'origin_index', 'voxel_size', 'planes',
            'symmetry', 'center', 'center_index', 'axis', 'coordinate_system', 'dump_header', 'pickable')
    dsettings = dict((n,loc[n]) for n in dopt if not loc[n] is None)

    rsettings = _render_settings(loc)

    if box_faces is False:
        image_mode_off = 'box faces'
    elif tilted_slab is False:
        image_mode_off = 'tilted slab'
    elif orthoplanes == 'off':
        image_mode_off = 'orthoplanes'
    else:
        image_mode_off = None

    for v in vlist:
        apply_volume_options(v, dsettings, rsettings, image_mode_off, session)

    if calculate_surfaces:
        for v in vlist:
            v._update_surfaces()

# -----------------------------------------------------------------------------
#
def _render_settings(options):
    ropt = (
        'show_outline_box', 'outline_box_rgb', 'outline_box_linewidth',
        'limit_voxel_count', 'voxel_limit', 'color_mode', 'colormap_on_gpu',
        'colormap_size', 'colormap_extend_left', 'colormap_extend_right',
        'blend_on_gpu', 'projection_mode', 'plane_spacing', 'full_region_on_gpu',
        'bt_correction', 'minimal_texture_memory', 'maximum_intensity_projection',
        'linear_interpolation', 'dim_transparency', 'dim_transparent_voxels',
        'line_thickness', 'smooth_lines', 'mesh_lighting',
        'two_sided_lighting', 'flip_normals', 'subdivide_surface',
        'subdivision_levels', 'surface_smoothing', 'smoothing_iterations',
        'smoothing_factor', 'square_mesh', 'cap_faces',
        'tilted_slab_axis', 'tilted_slab_offset',
        'tilted_slab_spacing', 'tilted_slab_plane_count', 'image_mode', 'backing_color')
    rsettings = dict((n,options[n]) for n in ropt if options.get(n) is not None)
    if options.get('orthoplanes') is not None:
        orthoplanes = options['orthoplanes']
        rsettings['orthoplanes_shown'] = ('x' in orthoplanes,
                                          'y' in orthoplanes,
                                          'z' in orthoplanes)
    if options['position_planes'] is not None:
        rsettings['orthoplane_positions'] = options['position_planes']
    if options['outline_box_rgb'] is not None:
        rsettings['outline_box_rgb'] = tuple(options['outline_box_rgb'].rgba[:3])
    if options['tilted_slab_axis'] is not None:
        rsettings['tilted_slab_axis'] = options['tilted_slab_axis'].coords
    if options['backing_color'] is not None:
        bc = options['backing_color']
        rsettings['backing_color'] = (None if bc == 'none' else tuple(bc.uint8x4()))
    return rsettings

# -----------------------------------------------------------------------------
#
def apply_global_settings(session, gsettings):

    from .volume import default_settings
    default_settings(session).update(gsettings)

    if 'data_cache_size' in gsettings:
        from .volume import data_cache
        dc = data_cache(session)
        dc.resize(gsettings['data_cache_size'] * (2**20))

    if 'pickable' in gsettings:
        from . import maps_pickable
        maps_pickable(session, gsettings['pickable'])
    
# -----------------------------------------------------------------------------
#
def apply_volume_options(v, doptions, roptions, image_mode_off, session):

    if 'close' in doptions:
        si = doptions['close']
        if si == 'surface':
            v.close_surface()
        elif si == 'image':
            v.close_image()
            
    if 'style' in doptions:
        v.set_display_style(doptions['style'])

    kw = level_and_color_settings(v, doptions)
    kw.update(roptions)

    if image_mode_off and v.rendering_options.image_mode == image_mode_off:
        if 'image_mode' not in kw:
            kw['image_mode'] = 'full region'
            if 'color_mode' not in kw:
                kw['color_mode'] = 'auto8'
                
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
        v.new_region(ijk_min, ijk_max, ijk_step,
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
            from chimerax.core import errors
            raise errors.UserError('Voxel size must positive, got %g,%g,%g'
                                % tuple(vsize))
        # Preserve index origin.
        origin = [(a/b)*c for a,b,c in zip(d.origin, d.step, vsize)]
        d.set_origin(origin)
        d.set_step(vsize)

    if 'symmetry' in doptions:
        csys = doptions.get('coordinate_system', v.position)
        if 'center_index' in doptions:
            c = v.data.ijk_to_xyz(doptions['center_index'])
            if 'coordinate_system' in doptions:
                c = csys.inverse() * (v.position * c)
            from chimerax.core.commands import Center
            center = Center(c)
        else:
            center = doptions.get('center')
        ops = doptions['symmetry'].positions(center, doptions.get('axis'), csys)
        d.symmetries = ops.transform_coordinates(v.position)

    if 'show' in doptions:
        v.display = True
    elif 'hide' in doptions:
        v.display = False
    elif 'toggle' in doptions:
        v.display = not v.display

    # TODO: Volume should note when it needs update
    v._drawings_need_update()

    if 'dump_header' in doptions and doptions['dump_header']:
        show_file_header(v.data, session.logger)

    if 'pickable' in doptions:
        v.pickable = doptions['pickable']


# TODO:
#  Allow quoted color names.
#  Could allow region name "full" or "back".
#  Could allow voxel_size or origin to be "original".
   
# -----------------------------------------------------------------------------
#
def level_and_color_settings(v, options):

    kw = {}

    # Code below modifies levels, avoid modifying options argument
    from copy import deepcopy
    levels = deepcopy(options.get('level', []))
    rms_levels = deepcopy(options.get('rms_level', []))
    sd_levels = deepcopy(options.get('sd_level', []))
    if rms_levels or sd_levels:
        mean, sd, rms = v.mean_sd_rms()
        if rms_levels:
            for lvl in rms_levels:
                lvl[0] *= rms
            levels.extend(rms_levels)
        if sd_levels:
            for lvl in sd_levels:
                lvl[0] *= sd
            levels.extend(sd_levels)

    colors = options.get('color', [])

    if 'change' in options:
        style = options['change']
    elif 'style' in options:
        style = options['style']
        if style == 'mesh':
            style = 'surface'
    elif v.surface_shown:
        style = 'surface'
        if v.image_shown:
            if levels or colors or 'brightness' in options or 'transparency' in options:
                from chimerax.core.errors import UserError
                raise UserError('Need to use the option "change surface" or "change image"\n'
                                'when levels, colors, brightness, or transparency are specified\n'
                                ' and both surface and image styles are shown.')
    elif v.image_shown:
        style = 'image'
    else:
        style = 'surface'

    if style in ('surface', 'mesh'):
        for l in levels:
            if len(l) != 1:
                from chimerax.core.errors import UserError
                raise UserError('Surface level must be a single value')
        levels = [l[0] for l in levels]
    elif style == 'image':
        for l in levels:
            if len(l) != 2:
                from chimerax.core.errors import UserError
                raise UserError('Image level must be <data-value,brightness-level>')

    if levels:
        kw[style+'_levels'] = levels

    # Allow 0 or 1 colors and 0 or more levels, or number colors matching
    # number of levels.
    if levels:
        nlev = len(levels)
    else:
        nlev = len(v.image_levels) if style == 'image' else len(v.surfaces)
    if len(colors) > 1 and len(colors) != nlev:
        from chimerax.core.errors import UserError
        raise UserError('Number of colors (%d) does not match number of levels (%d)'
                        % (len(colors), nlev))

    if len(colors) == 1:
        clist = [colors[0].rgba]*nlev
        kw[style+'_colors'] = clist
    elif len(colors) > 1:
        kw[style+'_colors'] = [c.rgba for c in colors]

    if len(levels) == 0 and len(colors) == 1:
        kw['default_rgba'] = colors[0].rgba

    if 'brightness' in options:
        if style == 'surface':
            kw['brightness'] = options['brightness']
        else:
            kw['image_brightness_factor'] = options['brightness']

    if 'transparency' in options:
        if style == 'surface':
            kw['transparency'] = options['transparency']
        else:
            kw['transparency_depth'] = options['transparency']

    if 'appearance' in options:
        from . import colortables
        akw = colortables.appearance_settings(options['appearance'], v)
        kw.update(akw)

    if 'name_appearance' in options:
        from . import colortables
        colortables.add_appearance(options['name_appearance'], v)

    if 'name_forget' in options:
        from . import colortables
        colortables.delete_appearance(options['name_forget'], v.session)
        
    return kw

# -----------------------------------------------------------------------------
# Arguments are axis,pstart,pend,pstep,pdepth.
#
def planes_arg(planes, session):

    axis, param = (planes.split(',',1) + [''])[:2]
    from chimerax.core.commands.parse import enum_arg, floats_arg
    p = [enum_arg(axis, session, ('x','y','z'))] + floats_arg(param, session)
    if len(p) < 2 or len(p) > 5:
        from chimerax.core import errors
        raise errors.UserError('planes argument must have 2 to 5 comma-separated values: axis,pstart[[[,pend],pstep],pdepth.], got "%s"' % planes)
    return p

# -----------------------------------------------------------------------------
# Find maps among models and all descendants.
#
def all_maps(models):
    maps = []
    from .volume import Volume
    from chimerax.core.models import Model
    for m in models:
        if isinstance(m, Volume):
            maps.append(m)
        if isinstance(m, Model):
            maps.extend(all_maps(m.child_drawings()))
    return maps
    
# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation
class PlanesArg(Annotation):
    '''
    Parse planes argument to volume command axis,start,end,increment,depth.
    axis can be x, y, or z, and the other values are integers with the last 3
    being optional.
    '''
    name = 'planes x|y|z[,<start>[,<end>[,<increment>[,<depth>]]]]'

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import next_token, AnnotationError
        token, text, rest = next_token(text)
        fields = token.split(',')
        if fields[0] not in ('x', 'y', 'z'):
            raise AnnotationError('Planes argument first field must be x, y, or z, got "%s"' % fields[0])
        try:
            values = [float(f) for f in fields[1:]]
        except Exception:
            raise AnnotationError('Planes arguments after axis must be numbers')
        result = tuple([fields[0]] + values)
        return result, text, rest
    
# -----------------------------------------------------------------------------
#
def show_file_header(d, log):
    if hasattr(d, 'file_header') and isinstance(d.file_header, dict):
        h = d.file_header
        klist = list(h.keys())
        klist.sort()
        msg = ('File header for %s\n' % d.path +
               '\n'.join(('%s = %s' % (k, str(h[k]))) for k in klist))
    else:
        msg = 'No header info for %s' % d.name
        log.status(msg)
    log.info(msg + '\n')
    
# -----------------------------------------------------------------------------
#
def volume_settings(session, volumes = None):
    if volumes is None:
        from . import Volume
        volumes = session.models.list(type = Volume)
    msg = '\n\n'.join(volume_settings_text(v) for v in volumes)
    session.logger.info(msg)
    
# -----------------------------------------------------------------------------
#
def volume_settings_text(v):
    from chimerax.core.colors import hex_color, rgba_to_rgba8
    lines = ['Settings for map %s' % v.name,
             'grid size = %d %d %d' % tuple(v.data.size),
             'region = %d %d %d' % tuple(v.region[0]) + ' to %d %d %d' % tuple(v.region[1]),
             'step = %d %d %d' % tuple(v.region[2]),
             'voxel size = %.4g %.4g %.4g' % tuple(v.data.step),
             'origin = %.4g %.4g %.4g' % tuple(v.data.origin),
             'origin index = %.4g %.4g %.4g' % tuple(v.data.xyz_to_ijk((0,0,0))),
             'surface levels = ' + ', '.join('%.5g' % s.level for s in v.surfaces),
             'surface colors = ' + ', '.join(hex_color(s.color) for s in v.surfaces),
             'image levels = ' + ' '.join('%.5g,%.5g' % tuple(sl) for sl in v.image_levels),
             'image colors = ' + ', '.join(hex_color(rgba_to_rgba8(c)) for c in v.image_colors),
             'image brightness factor = %.5g' % v.image_brightness_factor,
             'image transparency depth = %.5g' % v.transparency_depth,
             ] + rendering_option_strings(v.session, v.rendering_options)
    return '\n'.join(lines)

# -----------------------------------------------------------------------------
#
def rendering_option_strings(session, ro = None):
    from .volume import default_settings
    ds = default_settings(session)
    if ro is None:
        ro = ds.rendering_option_defaults()
    attrs = list(ds.rendering_option_names())
    attrs.sort()
    lines = []
    for attr in attrs:
        value = getattr(ro, attr)
        if attr == 'outline_box_rgb':
            value = tuple(100*r for r in value)		# Internally uses 0-1 values, but command line uses 0-100.
        lines.append('%s = %s' % (camel_case(attr), value))
    return lines

# -----------------------------------------------------------------------------
#
def camel_case(string):
    if '_' not in string:
        return string
    cc = []
    up = False
    for c in string:
        if c == '_':
            up = True
        else:
            cc.append(c.upper() if up else c)
            up = False
    return ''.join(cc)
    
# -----------------------------------------------------------------------------
#
def volume_default_values(session,
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
           color_mode = None,                # image rendering pixel formats
           colormap_on_gpu = None,           # image colormapping on gpu or cpu
           colormap_size = None,             # image colormapping
           colormap_extend_left = None,
           colormap_extend_right = None,
           backing_color = None,
           blend_on_gpu = None,		     # image blending on gpu or cpu
           projection_mode = None,           # auto, 2d-xyz, 2d-x, 2d-y, 2d-z, 3d
           plane_spacing = None,	     # min, max, or numeric value
           full_region_on_gpu = None,	     # for fast cropping with image rendering
           bt_correction = None,             # brightness and transparency
           minimal_texture_memory = None,
           maximum_intensity_projection = None,
           linear_interpolation = None,
           dim_transparency = None,          # for surfaces
           dim_transparent_voxels = None,     # for image rendering
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
           position_planes = None,
           tilted_slab_axis = None,
           tilted_slab_spacing = None,
           tilted_slab_offset = None,
           tilted_slab_plane_count = None,
# Save to preferences file or reset
           save_settings = None,
           reset = None,
           ):
    '''Report or set default volume values.'''

    gopt = ('data_cache_size', 'show_on_open', 'voxel_limit_for_open',
            'show_plane', 'voxel_limit_for_plane')
    loc = locals()
    gsettings = dict((n,loc[n]) for n in gopt if not loc[n] is None)
    if gsettings:
        apply_global_settings(session, gsettings)
    
    rsettings = _render_settings(locals())
    if rsettings:
        from .volume import default_settings
        default_settings(session).update(rsettings)

    if reset:
        from .volume import default_settings
        ds = default_settings(session)
        ds.restore_factory_defaults()
        
    if save_settings:
        from .volume import default_settings
        ds = default_settings(session)
        ds.save_to_preferences_file()
        session.logger.info('Saved volume settings')
        
    if len(gsettings) == 0 and len(rsettings) == 0 and not save_settings and not reset:
        msg = default_settings_text(session)
        session.logger.info(msg)

# -----------------------------------------------------------------------------
#
def default_settings_text(session):
    from .volume import default_settings
    ds = default_settings(session)
    lines = ['Default volume settings',
             'data cache size = %.3g Mbytes' % ds['data_cache_size'],
             'show on open = %s' % ds['show_on_open'],
             'show plane = %s' % ds['show_plane'],
             'voxel limit for open = %.3g Mvoxels' % ds['voxel_limit_for_open'],
             'voxel limit for plane = %.3g Mvoxels' % ds['voxel_limit_for_plane'],
             ]
    lines += ['Default rendering settings'] + rendering_option_strings(session)
    return '\n'.join(lines)
    

