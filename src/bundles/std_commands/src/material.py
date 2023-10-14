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

def material(session, preset = None, reflectivity = None,
             specular_reflectivity = None, exponent = None,
             ambient_reflectivity = None, transparent_cast_shadows = None,
             meshes_cast_shadows = None):

    '''
    Change surface material properties controlling the reflection of light.
    Currently all models use the same material properties.

    Parameters
    ----------
    preset : string
      Can be "default", "shiny", "dull", or "chimera".  Default sets all material
      properties to there default values.  Shiny sets specular reflectivity to 1.
      Dull sets specular reflectivity to 0.  Chimera sets specular reflectivity
      and exponent to try to emulate UCSF Chimera.
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
    transparent_cast_shadows : bool
      Whether transparent objects cast shadows. If set to true then transparent
      objects cast a shadows as if the objects are opaque making everything inside
      appear black. Raytracing would be required to render a partial shadow, and we
      don't support that. Initial value is false.
    meshes_cast_shadows : bool
      Whether mesh style drawings cast shadows.  Initial value is false.
    '''
    v = session.main_view
    m = v.material

    if len([opt for opt in (preset, reflectivity, specular_reflectivity, exponent,
                            ambient_reflectivity, transparent_cast_shadows, meshes_cast_shadows)
            if not opt is None]) == 0:
        # Report current settings.
        lines = (
            'Reflectivity: %.5g' % m.diffuse_reflectivity,
            'Specular reflectivity %.5g' % m.specular_reflectivity,
            'Specular exponent: %.5g' % m.specular_exponent,
            'Ambient reflectivity %.5g' % m.ambient_reflectivity,
            'Transparent cast shadows: %s' % ('true' if m.transparent_cast_shadows else 'false'),
            'Meshes cast shadows: %s' % ('true' if m.meshes_cast_shadows else 'false'),
        )
        msg = '\n'.join(lines)
        session.logger.info(msg)
        return

    if preset == 'default':
        m.set_default_parameters()
    elif preset == 'shiny':
        m.specular_reflectivity = 1
    elif preset == 'dull':
        m.specular_reflectivity = 0
    elif preset == 'chimera':
        m.specular_reflectivity = 1
        m.specular_exponent = 6

    if not reflectivity is None:
        m.diffuse_reflectivity = reflectivity
    if not specular_reflectivity is None:
        m.specular_reflectivity = specular_reflectivity
    if not exponent is None:
        m.specular_exponent = exponent
    if not ambient_reflectivity is None:
        m.ambient_reflectivity = ambient_reflectivity
    if not transparent_cast_shadows is None:
        m.transparent_cast_shadows = transparent_cast_shadows
    if not meshes_cast_shadows is None:
        m.meshes_cast_shadows = meshes_cast_shadows

    v.material = m	# Let's viewer know the material has been changed.

def register_command(logger):
    from chimerax.core.commands import CmdDesc, EnumOf, FloatArg, BoolArg, register
    _material_desc = CmdDesc(
        optional = [('preset', EnumOf(('default', 'shiny', 'dull', 'chimera')))],
        keyword = [
            ('reflectivity', FloatArg),
            ('specular_reflectivity', FloatArg),
            ('exponent', FloatArg),
            ('ambient_reflectivity', FloatArg),
            ('transparent_cast_shadows', BoolArg),
            ('meshes_cast_shadows', BoolArg),
        ],
        synopsis="report or alter material parameters")
    register('material', _material_desc, material, logger=logger)
