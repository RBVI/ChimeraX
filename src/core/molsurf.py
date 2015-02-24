# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
molsurf -- Compute molecular surfaces
=====================================
"""

from . import generic3d

class MolecularSurface(generic3d.Generic3DModel):
    pass

from . import cli
_surface_desc = cli.CmdDesc(keyword=[('probeRadius', cli.FloatArg),
                                     ('gridSpacing', cli.FloatArg),
                                     ('color', cli.IntsArg)])
def surface_command(session, probeRadius = 1.4, gridSpacing = 0.5, color = (180,205,128,128)):
    '''
    Compute and display a solvent excluded molecular surface for each molecule.
    TODO: When atom specs are available, specify atoms to surface.
    '''
    surfs = []
    from .structure import StructureModel
    for m in session.models.list():
        if isinstance(m, StructureModel):
            mb = m.mol_blob
            a = mb.atoms
            xyz = a.coords
            r = a.radii
            from . import surface
            va,na,ta = surface.ses_surface_geometry(xyz, r, probeRadius, gridSpacing)

            # Create surface model to show surface
            name = '%s SES surface' % m.name
            surf = show_surface(name, va, na, ta, color, place = m.position)
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
