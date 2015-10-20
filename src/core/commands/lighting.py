# vi: set expandtab shiftwidth=4 softtabstop=4:

def lighting(session, preset = None, direction = None, intensity = None, color = None, 
             fill_direction = None, fill_intensity = None, fill_color = None,
             ambient_intensity = None, ambient_color = None,
             depth_cue = None, depth_cue_start = None, depth_cue_end = None, depth_cue_color = None,
             move_with_camera = None,
             shadows = None, quality_of_shadows = None, depth_bias = None,
             multi_shadow = None, ms_map_size = None, ms_depth_bias = None):
    '''
    Set lighting parameters. There are 2 directional lights, a key light and a fill light,
    in addition to ambient lighting.  The key light can cast a shadow, and shadows cast
    from multiple uniformly distributed directions can produce ambient shadowing (aka "ambient occlusion").
    Parameters that are not specified retain their current value.  If no options are specified
    then the current settings are printed to the log.

    Parameters
    ----------
    preset : string
      Names a standard set of lighting parameters. Allowed values are "default",
      "simple", "full", "soft" and "flat".  Simple is the same as default and has
      no shadows.  Full includes direct and ambient shadows.  Soft includes ambient
      shadows from 64 directions and no direct lighting.  Flat has only anbient lighting
      and no shadows with silhouettes enabled.  Specifying a preset only specifies some
      of the lighting parameters. Specifying other options overrides the preset values.
    direction : 3 floats
      Key light direction as vector.  Does not have to
      have unit length -- it will be normalized.  Points in the direction the light shines.
      The viewing direction is along -z.  Initially is pointing down to the right
      (1,-1,-1).
    intensity : float
      Key light intensity. This is a brightness scale factor. Initial value 1.
    color : Color
      Key light color, initial value RGB = (1,1,1).
    fill_direction : 3 floats
      Fill light direction. Initially is pointing from lower left (-0.2,-0.2,-0.959).
    fill_intensity : float
      Fill light intensity. Initial value 0.5.
    fill_color : Color
      Fill light color, initial value RGB = (1,1,1).
    ambient_intensity : float
       Ambient light intensity. Initial value 0.4.
    ambient_color : Color
      Ambient color, initial value RGB = (1,1,1).
    depth_cue : bool
      Whether to dim scene with depth.
    depth_cue_start : float
      Fraction of distance from near to far clip plane where dimming starts. Initial value 0.5
    depth_cue_end : float
      Fraction of distance from near to far clip plane where dimming ends. Initial value 1.
    depth_cue_color : Color
      Color to fade towards, initial value RGB = (0,0,0).
    move_with_camera : bool
      Whether light directions move with the camera or are fixed in scene coordinates.
      Initial value true.
    shadows : bool
      Whether to show shadows.  Initial value false.
    quality_of_shadows : string or int
      Shadows are rendered with a 2 dimensional texture. Pixelated shadow edges result from
      using small texture sizes.  Value can be "coarse" (1024), "normal" (2048), "fine" (4096),
      "finer" (8192), or an integer value can be specified.
    depth_bias : float
      To avoid a surface shadowing itself due to numerical rounding errors an bias distance
      is used. This is a fraction of the scene diameter.  Initial value 0.005.
    multi_shadow : int
      How many directions to use for casting ambient shadows.  Value 0 means no
      ambient shadows. The soft preset uses 64 directions.  Initial value 0.
    ms_map_size : int
      Size of one 2-dimensional texture holding all the ambient shadow maps.
      Small values give coarser shadows that give a smoother appearance
      when many shadows ar rendered. Initial value 128.
    ms_depth_bias : float
      Depth bias to avoid surface self shadowing for ambient shadows as a fraction
      of the scene diameter. Because small shadow map sizes are typically used a
      larger bias is needed than for directional shadows.  Initial value 0.05.
    '''
    v = session.main_view
    lp = v.lighting()

    if len([opt for opt in (preset, direction, intensity, color, fill_direction, fill_intensity, fill_color,
                            ambient_intensity, ambient_color, depth_cue, depth_cue_start, depth_cue_end,
                            depth_cue_color, move_with_camera, shadows, depth_bias, quality_of_shadows,
                            multi_shadow, ms_map_size, ms_depth_bias)
            if not opt is None]) == 0:
        # Report current settings.
        lines = (
            'Intensity: %.5g' % lp.key_light_intensity,
            'Direction: (%.5g,%.5g,%.5g)' % tuple(lp.key_light_direction),
            'Color: (%.5g,%.5g,%.5g)' % tuple(lp.key_light_color),
            'Fill intensity: %.5g' % lp.fill_light_intensity,
            'Fill direction: (%.5g,%.5g,%.5g)' % tuple(lp.fill_light_direction),
            'Fill color: (%.5g,%.5g,%.5g)' % tuple(lp.fill_light_color),
            'Ambient intensity: %.5g' % lp.ambient_light_intensity,
            'Ambient color: (%.5g,%.5g,%.5g)' % tuple(lp.ambient_light_color),
            'Shadow: %s (depth map size %d, depth bias %.5g)'
              % (v.shadows, v.shadow_map_size, v.shadow_depth_bias),
            'Multishadows: %d (max %d, depth map size %d, depth bias %.5g)'
              % (v.multishadow, v.max_multishadow(), v.multishadow_map_size, v.multishadow_depth_bias),
        )
        msg = '\n'.join(lines)
        session.logger.info(msg)
        return

    from ..geometry.vector import normalize_vector as normalize
    from numpy import array, float32

    if preset == 'default' or preset == 'simple':
        v.shadows = False
        v.multishadow = 0
        lp.set_default_parameters()
    elif preset == 'full':
        v.shadows = True
        v.multishadow = 64
        lp.key_light_intensity = 0.7
        lp.fill_light_intensity = 0.3
        lp.ambient_light_intensity = 1
    elif preset == 'soft':
        v.shadows = False
        v.multishadow = 64
        lp.key_light_intensity = 0
        lp.fill_light_intensity = 0
        lp.ambient_light_intensity = 1.5
    elif preset == 'flat':
        v.shadows = False
        v.multishadow = 0
        lp.key_light_intensity = 0
        lp.fill_light_intensity = 0
        lp.ambient_light_intensity = 1
        v.silhouettes = True

    if not direction is None:
        lp.key_light_direction = array(normalize(direction), float32)
    if not intensity is None:
        lp.key_light_intensity = intensity
    if not color is None:
        lp.key_light_color = color.rgba[:3]
    if not fill_direction is None:
        lp.fill_light_direction = array(normalize(fill_direction), float32)
    if not fill_intensity is None:
        lp.fill_light_intensity = fill_intnsity
    if not fill_color is None:
        lp.fill_light_color = fill_color.rgba[:3]
    if not ambient_intensity is None:
        lp.ambient_light_intensity = ambient_intensity
    if not ambient_color is None:
        lp.ambient_light_color = ambient_color.rgba[:3]
    if not depth_cue is None:
        v.depth_cue = depth_cue
    if not depth_cue_start is None:
        lp.depth_cue_start = depth_cue_start
    if not depth_cue_end is None:
        lp.depth_cue_end = depth_cue_end
    if not depth_cue_color is None:
        lp.depth_cue_color = depth_cue_color.rgba[:3]
    if not move_with_camera is None:
        lp.move_lights_with_camera = move_with_camera
    if not shadows is None:
        v.shadows = shadows
    if not quality_of_shadows is None:
        sizes = {'normal':2048, 'fine':4096, 'finer':8192, 'coarse':1024}
        if quality_of_shadows in sizes:
            size = sizes[quality_of_shadows]
        else:
            try:
                size = int(quality_of_shadows)
            except:
                from ..errors import UserError
                raise UserError('qualityOfShadows value must be an integer or one of %s'
                                % ', '.join('%s (%d)' % (nm,s) for nm,s in sizes.items()))
        v.shadow_map_size = size
    if not depth_bias is None:
        v.shadow_depth_bias = depth_bias
    if not multi_shadow is None:
        v.multishadow = multi_shadow
    if not ms_map_size is None:
        v.multishadow_map_size = ms_map_size
    if not ms_depth_bias is None:
        v.multishadow_depth_bias = ms_depth_bias

    v.update_lighting = True
    v.redraw_needed = True

def register_command(session):
    from . import CmdDesc, register, BoolArg, IntArg, FloatArg, Float3Arg, StringArg, EnumOf, ColorArg
    _lighting_desc = CmdDesc(
        optional = [('preset', EnumOf(('default', 'full', 'soft', 'simple', 'flat')))],
        keyword = [
            ('direction', Float3Arg),
            ('intensity', FloatArg),
            ('color', ColorArg),
            ('fill_direction', Float3Arg),
            ('fill_intensity', FloatArg),
            ('fill_color', ColorArg),
            ('ambient_intensity', FloatArg),
            ('ambient_color', ColorArg),
            ('depth_cue', BoolArg),
            ('depth_cue_start', FloatArg),
            ('depth_cue_end', FloatArg),
            ('depth_cue_color', ColorArg),
            ('move_with_camera', BoolArg),
            ('shadows', BoolArg),
            ('quality_of_shadows', StringArg),
            ('depth_bias', FloatArg),
            ('multi_shadow', IntArg),
            ('ms_map_size', IntArg),
            ('ms_depth_bias', FloatArg),
        ],
        synopsis="report or alter lighting parameters")

    register('lighting', _lighting_desc, lighting)
