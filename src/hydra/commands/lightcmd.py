def lighting_command(cmdname, args, session):

    from .parse import float3_arg, color_arg, bool_arg, enum_arg, parse_arguments
    req_args = ()
    opt_args = ()
    kw_args = (('direction', float3_arg),
               ('color', color_arg),
               ('fillDirection', float3_arg),
               ('fillColor', color_arg),
               ('ambientColor', color_arg),
               ('fixed', bool_arg),
               ('shadows', bool_arg),
               ('qualityOfShadows', enum_arg, {'values':('normal', 'fine', 'finer', 'coarse')}),
           )

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    lighting(session, **kw)

def lighting(session, direction = None, color = None, specularColor = None, exponent = None, 
             fillDirection = None, fillColor = None, ambientColor = None, fixed = None,
             qualityOfShadows = None, shadows = None):

    v = session.view
    lp = v.render.lighting

    from ..geometry.vector import normalize_vector as normalize
    from numpy import array, float32
    if not direction is None:
        lp.key_light_direction = array(normalize(direction), float32)
    if not color is None:
        lp.key_light_color = color[:3]
    if not fillDirection is None:
        lp.fill_light_direction = array(normalize(fillDirection), float32)
    if not fillColor is None:
        lp.fill_light_color = fillColor[:3]
    if not ambientColor is None:
        lp.ambient_light_color = ambientColor[:3]
    if not fixed is None:
        lp.move_lights_with_camera = not fixed
    if not shadows is None:
        v.shadows = shadows
    if not qualityOfShadows is None:
        size = {'normal':2048, 'fine':4096, 'finer':8192, 'coarse':1024}[qualityOfShadows]
        v.shadowMapSize = size

    v.update_lighting = True
    v.redraw_needed = True
