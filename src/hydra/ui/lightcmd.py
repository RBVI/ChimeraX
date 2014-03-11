def lighting_command(cmdname, args, session):

    from .commands import float3_arg, color_arg, float_arg, bool_arg, parse_arguments
    req_args = ()
    opt_args = ()
    kw_args = (('direction', float3_arg),
               ('color', color_arg),
               ('specularColor', color_arg),
               ('exponent', float_arg),
               ('fillDirection', float3_arg),
               ('fillColor', color_arg),
               ('ambientColor', color_arg),
               ('fixed', bool_arg),
           )

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    lighting(session, **kw)

def lighting(session, direction = None, color = None, specularColor = None, exponent = None, 
             fillDirection = None, fillColor = None, ambientColor = None, fixed = None):

    v = session.view
    lp = v.render.lighting_params

    from ..geometry.vector import normalize_vector as normalize
    if not direction is None:
        lp.key_light_direction = normalize(direction)
    if not color is None:
        lp.key_light_diffuse_color = color[:3]
    if not specularColor is None:
        lp.key_light_specular_color = specularColor[:3]
    if not exponent is None:
        lp.key_light_specular_exponent = exponent
    if not fillDirection is None:
        lp.fill_light_direction = normalize(fillDirection)
    if not fillColor is None:
        lp.fill_light_diffuse_color = fillColor[:3]
    if not ambientColor is None:
        lp.ambient_light_color = ambientColor[:3]
    if not fixed is None:
        lp.move_lights_with_camera = not fixed

    v.update_lighting = True
    v.redraw_needed = True
