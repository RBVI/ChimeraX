# vim: set expandtab shiftwidth=4 softtabstop=4:

def material(session, preset = None, reflectivity = None,
             specular_reflectivity = None, exponent = None,
             ambient_reflectivity = None):
    '''
    Change surface material properties controlling the reflection of light.
    Currently all models use the same material properties.

    Parameters
    ----------
    preset : string
      Can be "default", "shiny" or "dull".  Default sets all material
      properties to there default values.  Shiny sets specular reflectivity to 1.
      Dull sets specular reflectivity to 0.
    reflectivity : float
      Fraction of directional light reflected diffusely (reflected equally in
      all directions). Initial value 0.8.
    specular_reflectivity : float
      Fraction of key light reflected specularly (mirror reflection
      with some spread given by exponent. Initial value 0.8.
    exponent : float
      Controls specularly reflected light scaling intensity by a factor of
      cosine(x) raised to the exponent power, where x is the between the mirror
      reflection direction and the view direction. Larger values produce smaller
      specular highlight spots on surfaces.  Initial value 30.
    ambient_reflectivity : float
      Fraction of ambient light reflected. Initial value 0.8.
    '''
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
    if not specular_reflectivity is None:
        m.specular_reflectivity = specular_reflectivity
    if not exponent is None:
        m.specular_exponent = exponent
    if not ambient_reflectivity is None:
        m.ambient_reflectivity = ambient_reflectivity

    v.update_lighting = True
    v.redraw_needed = True

def register_command(session):
    from .cli import CmdDesc, EnumOf, FloatArg, register
    _material_desc = CmdDesc(
        optional = [('preset', EnumOf(('default', 'shiny', 'dull')))],
        keyword = [
            ('reflectivity', FloatArg),
            ('specular_reflectivity', FloatArg),
            ('exponent', FloatArg),
            ('ambient_reflectivity', FloatArg),
        ],
        synopsis="report or alter material parameters")
    register('material', _material_desc, material)
