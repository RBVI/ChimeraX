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
      Key light direction as vector.  Does not have to have unit length.
      Points in the direction the light shines.
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
      when many shadows ar rendered. Initial value 1024.
    ms_depth_bias : float
      Depth bias to avoid surface self shadowing for ambient shadows as a fraction
      of the scene diameter. Because small shadow map sizes are typically used a
      larger bias is needed than for directional shadows.  Initial value 0.01.
    '''
    v = session.main_view
    lp = v.lighting

    if len([opt for opt in (preset, direction, intensity, color, fill_direction, fill_intensity, fill_color,
                            ambient_intensity, ambient_color, depth_cue, depth_cue_start, depth_cue_end,
                            depth_cue_color, move_with_camera, shadows, depth_bias, quality_of_shadows,
                            multi_shadow, ms_map_size, ms_depth_bias)
            if not opt is None]) == 0:
        # Report current settings.
        lines = (
            'Intensity: %.5g' % lp.key_light_intensity,
            'Direction: %.5g,%.5g,%.5g' % tuple(lp.key_light_direction),
            'Color: %d,%d,%d' % tuple(100*r for r in lp.key_light_color),
            'Fill intensity: %.5g' % lp.fill_light_intensity,
            'Fill direction: %.5g,%.5g,%.5g' % tuple(lp.fill_light_direction),
            'Fill color: %d,%d,%d' % tuple(100*r for r in lp.fill_light_color),
            'Ambient intensity: %.5g' % lp.ambient_light_intensity,
            'Ambient color: %d,%d,%d' % tuple(100*r for r in lp.ambient_light_color),
            'Depth cue: %d, start %.5g, end %.5g, color %d,%d,%d'
              % ((lp.depth_cue, lp.depth_cue_start, lp.depth_cue_end) + tuple(100*r for r in lp.depth_cue_color)),
            'Shadow: %s (depth map size %d, depth bias %.5g)'
              % (lp.shadows, lp.shadow_map_size, lp.shadow_depth_bias),
        )
        if session.ui.has_graphics:
            lines += (
                'Multishadows: %d (max %d, depth map size %d, depth bias %.5g)'
                  % (lp.multishadow, v.max_multishadow(), lp.multishadow_map_size, lp.multishadow_depth_bias),
            )
        msg = '\n'.join(lines)
        session.logger.info(msg)
        return

    ms_directions = lighting_settings(session).lighting_multishadow_directions
    sil = v.silhouette
    if preset == 'default' or preset == 'simple':
        lp.shadows = False
        lp.multishadow = 0
        sil.depth_jump = 0.03
        lp.set_default_parameters(v.background_color)
    elif preset == 'full':
        lp.shadows = True
        lp.multishadow = ms_directions
        lp.key_light_intensity = 0.7
        lp.fill_light_intensity = 0.3
        lp.ambient_light_intensity = 0.8
        lp.multishadow_depth_bias = 0.01
        lp.multishadow_map_size = 1024
        sil.depth_jump = 0.03
    elif preset == 'soft':
        lp.shadows = False
        lp.multishadow = ms_directions
        lp.key_light_intensity = 0
        lp.fill_light_intensity = 0
        lp.ambient_light_intensity = 1.5
        lp.multishadow_depth_bias = 0.01
        lp.multishadow_map_size = 1024
        sil.depth_jump = 0.03
    elif preset == 'gentle':
        lp.shadows = False
        lp.multishadow = ms_directions
        lp.key_light_intensity = 0
        lp.fill_light_intensity = 0
        lp.ambient_light_intensity = 1.5
        lp.multishadow_depth_bias = 0.05
        lp.multishadow_map_size = 128
        sil.depth_jump = 0.03
    elif preset == 'flat':
        lp.shadows = False
        lp.multishadow = 0
        lp.key_light_intensity = 0
        lp.fill_light_intensity = 0
        lp.ambient_light_intensity = 1.45
        sil.enabled = True
        sil.depth_jump = 0.01

    from numpy import array, float32

    if not direction is None:
        lp.key_light_direction = array(direction, float32)
    if not intensity is None:
        lp.key_light_intensity = intensity
    if not color is None:
        lp.key_light_color = color.rgba[:3]
    if not fill_direction is None:
        lp.fill_light_direction = array(fill_direction, float32)
    if not fill_intensity is None:
        lp.fill_light_intensity = fill_intensity
    if not fill_color is None:
        lp.fill_light_color = fill_color.rgba[:3]
    if not ambient_intensity is None:
        lp.ambient_light_intensity = ambient_intensity
    if not ambient_color is None:
        lp.ambient_light_color = ambient_color.rgba[:3]
    if not depth_cue is None:
        lp.depth_cue = depth_cue
    if not depth_cue_start is None:
        lp.depth_cue_start = depth_cue_start
    if not depth_cue_end is None:
        lp.depth_cue_end = depth_cue_end
    if not depth_cue_color is None:
        lp.depth_cue_color = depth_cue_color.rgba[:3]
    if not move_with_camera is None:
        lp.move_lights_with_camera = move_with_camera
    if not shadows is None:
        lp.shadows = shadows
    if not quality_of_shadows is None:
        sizes = {'normal':2048, 'fine':4096, 'finer':8192, 'coarse':1024}
        if quality_of_shadows in sizes:
            size = sizes[quality_of_shadows]
        else:
            try:
                size = int(quality_of_shadows)
            except Exception:
                from chimerax.core.errors import UserError
                raise UserError('qualityOfShadows value must be an integer or one of %s'
                                % ', '.join('%s (%d)' % (nm,s) for nm,s in sizes.items()))
        lp.shadow_map_size = size
    if not depth_bias is None:
        lp.shadow_depth_bias = depth_bias
    if not multi_shadow is None:
        lp.multishadow = multi_shadow
    if not ms_map_size is None:
        lp.multishadow_map_size = ms_map_size
    if not ms_depth_bias is None:
        lp.multishadow_depth_bias = ms_depth_bias

    v.update_lighting = True
    v.redraw_needed = True

def lighting_model(session, models, depth_cue = None, shadows = None, multi_shadow = None,
                   directional = None):
    '''
    Allow disabling depth cue and shadows for specific models even when global depth cue
    or shadows are enabled.

    Parameters
    ----------
    models : list of Models
    depth_cue : bool
      Whether models will show depth cue when global depth cue enabled.
    shadows : bool
      Whether models will show shadows when global shadows is enabled.
    multi_shadow : bool
      Whether models will show multishadows when global multishadows is enabled.
    directional : bool
      Whether models will show any directional lighting.  Turning this off gives
      objects a uniform color.  It eliminates brightness variation that depends
      on the angle between surface normal and key/fill light directions.  It also
      makes no shadows appear on the model.  It does not effect depth cue.
    '''
    if depth_cue is None and shadows is None and multi_shadow is None and directional is None:
        lines = ['Model #%s: depth_cue: %s, shadows: %s, multi_shadow: %s, directional: %s'
                 % (m.id_string, m.allow_depth_cue, m.accept_shadow, m.accept_multishadow, m.use_lighting)
                 for m in models]
        session.logger.info('\n'.join(lines))
    else:
        drawings = set()
        for m in models:
            drawings.update(m.all_drawings())
        for d in drawings:
            if depth_cue is not None:
                d.allow_depth_cue = depth_cue
            if shadows is not None:
                d.accept_shadow = shadows
            if multi_shadow is not None:
                d.accept_multishadow = multi_shadow
            if directional is not None:
                d.use_lighting = directional

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, IntArg, FloatArg, Float3Arg, \
        StringArg, EnumOf, ColorArg, ModelsArg
    _lighting_desc = CmdDesc(
        optional = [('preset', EnumOf(('default', 'full', 'soft', 'gentle', 'simple', 'flat')))],
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

    register('lighting', _lighting_desc, lighting, logger=logger)

    _lighting_model_desc = CmdDesc(
        required = [('models', ModelsArg)],
        keyword = [
            ('depth_cue', BoolArg),
            ('shadows', BoolArg),
            ('multi_shadow', BoolArg),
            ('directional', BoolArg),
        ],
        synopsis="Turn off depth cue or shadows for individual models even when globally they are enabled.")

    register('lighting model', _lighting_model_desc, lighting_model, logger=logger)

def lighting_settings(session):
    if not hasattr(session, '_lighting_settings'):
        session._lighting_settings = _LightingSettings(session, 'lighting')
    return session._lighting_settings

from chimerax.core.settings import Settings
class _LightingSettings(Settings):
    EXPLICIT_SAVE = {
        'lighting_multishadow_directions': 64,
    }
