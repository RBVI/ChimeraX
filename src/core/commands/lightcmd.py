# vi: set expandtab shiftwidth=4 softtabstop=4:
from .cli import CmdDesc, BoolArg, IntArg, FloatArg, Float3Arg, StringArg, EnumOf
from .color import ColorArg
_lighting_desc = CmdDesc(
    optional = [('preset', EnumOf(('default', 'full', 'soft', 'simple', 'flat')))],
    keyword = [
        ('direction', Float3Arg),
        ('intensity', FloatArg),
        ('color', ColorArg),
        ('fillDirection', Float3Arg),
        ('fillIntensity', FloatArg),
        ('fillColor', ColorArg),
        ('ambientIntensity', FloatArg),
        ('ambientColor', ColorArg),
        ('fixed', BoolArg),
        ('shadows', BoolArg),
        ('qualityOfShadows', StringArg),
        ('depthBias', FloatArg),
        ('multiShadow', IntArg),
        ('msMapSize', IntArg),
        ('msDepthBias', FloatArg),
    ],
    synopsis="report or alter lighting parameters")

def lighting(session, preset = None, direction = None, intensity = None, color = None, 
             fillDirection = None, fillIntensity = None, fillColor = None,
             ambientIntensity = None, ambientColor = None, fixed = None,
             shadows = None, qualityOfShadows = None, depthBias = None,
             multiShadow = None, msMapSize = None, msDepthBias = None):

    v = session.main_view
    lp = v.lighting()

    if len([opt for opt in (preset, direction, intensity, color, fillDirection, fillIntensity, fillColor,
                            ambientIntensity, ambientColor, fixed, shadows, depthBias, qualityOfShadows,
                            multiShadow, msMapSize, msDepthBias)
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
    if not fillDirection is None:
        lp.fill_light_direction = array(normalize(fillDirection), float32)
    if not fillIntensity is None:
        lp.fill_light_intensity = fillIntensity
    if not fillColor is None:
        lp.fill_light_color = fillColor.rgba[:3]
    if not ambientIntensity is None:
        lp.ambient_light_intensity = ambientIntensity
    if not ambientColor is None:
        lp.ambient_light_color = ambientColor.rgba[:3]
    if not fixed is None:
        lp.move_lights_with_camera = not fixed
    if not shadows is None:
        v.shadows = shadows
    if not qualityOfShadows is None:
        sizes = {'normal':2048, 'fine':4096, 'finer':8192, 'coarse':1024}
        if qualityOfShadows in sizes:
            size = sizes[qualityOfShadows]
        else:
            try:
                size = int(qualityOfShadows)
            except:
                from ..errors import UserError
                raise UserError('qualityOfShadows value must be an integer or one of %s'
                                % ', '.join('%s (%d)' % (nm,s) for nm,s in sizes.items()))
        v.shadow_map_size = size
    if not depthBias is None:
        v.shadow_depth_bias = depthBias
    if not multiShadow is None:
        v.multishadow = multiShadow
    if not msMapSize is None:
        v.multishadow_map_size = msMapSize
    if not msDepthBias is None:
        v.multishadow_depth_bias = msDepthBias

    v.update_lighting = True
    v.redraw_needed = True

def register_lighting_command():
    from . import cli
    cli.register('lighting', _lighting_desc, lighting)

_material_desc = CmdDesc(
    optional = [('preset', EnumOf(('default', 'shiny', 'dull')))],
    keyword = [
        ('reflectivity', FloatArg),
        ('specularReflectivity', FloatArg),
        ('exponent', FloatArg),
        ('ambientReflectivity', FloatArg),
    ],
    synopsis="report or alter material parameters")

def material(session, preset = None, reflectivity = None,
             specularReflectivity = None, exponent = None,
             ambientReflectivity = None):

    v = session.main_view
    m = v.material()

    if preset == 'default':
        m.set_default_parameters()
    elif preset == 'shiny':
        m.specular_reflectivity = 1
    elif preset == 'dull':
        m.specular_reflectivity = 0

    if not reflectivity is None:
        m.diffuse_reflectivity = reflectivity
    if not specularReflectivity is None:
        m.specular_reflectivity = specularReflectivity
    if not exponent is None:
        m.specular_exponent = exponent
    if not ambientReflectivity is None:
        m.ambient_reflectivity = ambientReflectivity

    v.update_lighting = True
    v.redraw_needed = True

def register_material_command():
    from . import cli
    cli.register('material', _material_desc, material)
