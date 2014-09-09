def material_command(cmdname, args, session):

    from .parse import float_arg, parse_arguments
    req_args = ()
    opt_args = ()
    kw_args = (('ambientReflectivity', float_arg),
               ('diffuseReflectivity', float_arg),
               ('specularReflectivity', float_arg),
               ('exponent', float_arg),
           )

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    material(session, **kw)

def material(session, ambientReflectivity = None, diffuseReflectivity = None,
             specularReflectivity = None, exponent = None):

    v = session.view
    mp = v.render.material

    if not ambientReflectivity is None:
        mp.ambient_reflectivity = ambientReflectivity
    if not diffuseReflectivity is None:
        mp.diffuse_reflectivity = diffuseReflectivity
    if not specularReflectivity is None:
        mp.specular_reflectivity = specularReflectivity
    if not exponent is None:
        mp.specular_exponent = exponent

    v.update_lighting = True
    v.redraw_needed = True
