# vi: set expandtab shiftwidth=4 softtabstop=4:
from .cli import CmdDesc, BoolArg, IntArg, FloatArg, Float3Arg, EnumOf
_lighting_desc = CmdDesc(
    optional = [('preset', EnumOf(('default', 'full', 'soft', 'simple')))],
    keyword = [
        ('direction', Float3Arg),
        ('intensity', FloatArg),
        ('color', Float3Arg),               # TODO: Color arg
        ('fillDirection', Float3Arg),
        ('fillIntensity', FloatArg),
        ('fillColor', Float3Arg),           # TODO: Color arg
        ('ambientIntensity', FloatArg),
        ('ambientColor', Float3Arg),	    # TODO: Color arg
        ('fixed', BoolArg),
        ('shadows', BoolArg),
        ('qualityOfShadows', EnumOf(('normal', 'fine', 'finer', 'coarse'))),
        ('multiShadow', IntArg),
    ])

def lighting(session, preset = None, direction = None, intensity = None, color = None, 
             fillDirection = None, fillIntensity = None, fillColor = None,
             ambientIntensity = None, ambientColor = None, fixed = None,
             qualityOfShadows = None, shadows = None, multiShadow = None):

    v = session.main_view
    lp = v.lighting()

    from .geometry.vector import normalize_vector as normalize
    from numpy import array, float32
    if not direction is None:
        lp.key_light_direction = array(normalize(direction), float32)
    if not intensity is None:
        lp.key_light_intensity = intensity
    if not color is None:
        lp.key_light_color = color[:3]
    if not fillDirection is None:
        lp.fill_light_direction = array(normalize(fillDirection), float32)
    if not fillIntensity is None:
        lp.fill_light_intensity = fillIntensity
    if not fillColor is None:
        lp.fill_light_color = fillColor[:3]
    if not ambientIntensity is None:
        lp.ambient_light_intensity = ambientIntensity
    if not ambientColor is None:
        lp.ambient_light_color = ambientColor[:3]
    if not fixed is None:
        lp.move_lights_with_camera = not fixed
    if not shadows is None:
        v.shadows = shadows
    if not qualityOfShadows is None:
        sizes = {'normal':2048, 'fine':4096, 'finer':8192, 'coarse':1024}
        if qualityOfShadows in sizes:
            size = sizes[qualityOfShadows]
        v.set_shadow_map_size(size)
    if not multiShadow is None:
        v.set_multishadow(multiShadow)
    if preset == 'default':
        v.shadows = False
        v.set_multishadow(0)
        lp.set_default_parameters()
    elif preset == 'full':
        v.shadows = True
        v.set_multishadow(64)
        lp.key_light_intensity = 0.7
        lp.fill_light_intensity = 0.3
        lp.ambient_light_intensity = 1
    elif preset == 'soft':
        v.shadows = False
        v.set_multishadow(64)
        lp.key_light_intensity = 0
        lp.fill_light_intensity = 0
        lp.ambient_light_intensity = 1.5
    elif preset == 'simple':
        v.shadows = False
        v.set_multishadow(0)
        lp.key_light_intensity = 1
        lp.fill_light_intensity = 0.5
        lp.ambient_light_intensity = 0.4

    v.update_lighting = True
    v.redraw_needed = True

def register_lighting_command():
    from . import cli
    cli.register('lighting', _lighting_desc, lighting)
