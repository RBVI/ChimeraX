def lighting_command(cmdname, args, session):

    from .parse import float3_arg, float_arg, color_arg, bool_arg, int_arg, string_arg, no_arg, parse_arguments
    req_args = ()
    opt_args = ()
    kw_args = (('direction', float3_arg),
               ('intensity', float_arg),
               ('color', color_arg),
               ('fillDirection', float3_arg),
               ('fillIntensity', float_arg),
               ('fillColor', color_arg),
               ('ambientIntensity', float_arg),
               ('ambientColor', color_arg),
               ('fixed', bool_arg),
               ('shadows', bool_arg),
               ('qualityOfShadows', string_arg),
               ('multiShadow', int_arg),
               ('default', no_arg),
               ('full', no_arg),
               ('soft', no_arg),
               ('simple', no_arg),
           )

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    lighting(session, **kw)

def lighting(session, direction = None, intensity = None, color = None, 
             fillDirection = None, fillIntensity = None, fillColor = None,
             ambientIntensity = None, ambientColor = None, fixed = None,
             qualityOfShadows = None, shadows = None, multiShadow = None,
             default = None, full = None, soft = None, simple = None):

    v = session.view
    lp = v.render.lighting

    from ..geometry.vector import normalize_vector as normalize
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
        else:
            from .parse import int_arg
            size = int_arg(qualityOfShadows, session)
        v.shadowMapSize = size
    if not multiShadow is None:
        v.set_multishadow(multiShadow)
    if not default is None:
        v.shadows = False
        v.set_multishadow(0)
        lp.set_default_parameters()
    if not full is None:
        v.shadows = True
        v.set_multishadow(42)
        lp.key_light_intensity = 1
        lp.fill_light_intensity = 0
        lp.ambient_light_intensity = 1
    if not soft is None:
        v.shadows = False
        v.set_multishadow(42)
        lp.key_light_intensity = 0
        lp.fill_light_intensity = 0
        lp.ambient_light_intensity = 1.5
    if not simple is None:
        v.shadows = False
        v.set_multishadow(0)
        lp.key_light_intensity = 1
        lp.fill_light_intensity = 0.5
        lp.ambient_light_intensity = 0.4

    v.update_lighting = True
    v.redraw_needed = True
