# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
molsurf -- Compute molecular surfaces
=====================================
"""

from . import generic3d

class MolecularSurface(generic3d.Generic3DModel):
    pass

from . import cli, atomspec, color
_surface_desc = cli.CmdDesc(
    optional = [('atoms', atomspec.AtomSpecArg)],
    keyword = [('probeRadius', cli.FloatArg),
               ('gridSpacing', cli.FloatArg),
               ('color', color.ColorArg),
               ('transparency', cli.FloatArg),])

def surface_command(session, atoms = None, probeRadius = 1.4, gridSpacing = 0.5,
                    color = color.Color((.7,.7,.7,1)), transparency = 0):
    '''
    Compute and display a solvent excluded molecular surface for each molecule.
    TODO: When atom specs are available, specify atoms to surface.
    '''
    if atoms is None:
        from .structure import StructureModel
        atom_blobs = [(m.name, m.mol_blob.atoms, m.position)
                      for m in session.models.list()
                      if isinstance(m, StructureModel)]
    else:
        a = atoms.evaluate(session).atoms
        if a is None or len(a) == 0:
            raise cli.AnnotationError('No atoms specified by %s' % (str(atoms),))
        atom_blobs = [(str(atoms), a, None)]

    surfs = []
    for name, a, place in atom_blobs:
        xyz = a.coords
        r = a.radii
        from . import surface
        va,na,ta = surface.ses_surface_geometry(xyz, r, probeRadius, gridSpacing)

        # Create surface model to show surface
        name = '%s SES surface' % name
        rgba8 = color.uint8x4()
        rgba8[3] = int(rgba8[3] * (100.0-transparency)/100.0)
        surf = show_surface(name, va, na, ta, rgba8, place)
        session.models.add([surf])
        surfs.append(surf)
    return surfs

def show_surface(name, va, na, ta, color = (180,180,180,255), place = None):

    surf = MolecularSurface(name)
    if not place is None:
        surf.position = place
    surf.geometry = va, ta
    surf.normals = na
    surf.color = color
    return surf

def register_surface_command():
    cli.register('surface', _surface_desc, surface_command)
