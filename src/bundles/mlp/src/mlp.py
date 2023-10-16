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

def mlp(session, atoms=None, method="fauchere", spacing=1.0, max_distance=5.0, nexp=3.0,
        color=True, palette=None, range=None, transparency=None, surfaces=[], map=False, key=False):
    '''Display Molecular Lipophilic Potential for a single model.

    Parameters
    ----------
    atoms : Atoms
        Color surfaces for these atoms using MLP map.  Only amino acid residues are used.
    method : 'dubost','fauchere','brasseur','buckingham','type5'
        Distance dependent function to use for calculation
    spacing : float
    	Grid spacing, default 1 Angstrom.
    max_distance : float
        Maximum distance from atom to sum lipophilicity.  Default 5 Angstroms.
    nexp : float
        The buckingham method uses this numerical exponent.
    color : bool
        Whether to color molecular surfaces. They are created if they don't yet exist.
    palette : Colormap
        Color palette for coloring surfaces.
        Default is lipophilicity colormap (orange lipophilic, blue lipophobic).
    range : 2-tuple of float
        Range of lipophilicity values defining ends of color map.  Default is -20,20
    transparency : float
        Percent transparency to use.  If not specified then palette transparency values used.
    surfaces : list of Surface models
        If the color options is true then these surfaces are colored instead of computing surfaces.
    map : bool
        Whether to open a volume model of lipophilicity values
    key : bool
        Whether to show a color key
    '''
    if atoms is None:
        from chimerax.atomic import all_atoms
        atoms = all_atoms(session)

    from chimerax.atomic import Residue
    patoms = atoms[atoms.residues.polymer_types == Residue.PT_AMINO]
    if len(patoms) == 0:
        from chimerax.core.errors import UserError
        raise UserError('mlp: no amino acids specified')
        
    if palette is None:
        from chimerax.core.colors import BuiltinColormaps
        cmap = BuiltinColormaps['lipophilicity']
    else:
        cmap = palette
    if range is None and not cmap.values_specified:
        range = (-20,20)
        
    # Color surfaces by lipophilicity
    if color:
        # Compute surfaces if not already created
        from chimerax.surface import surface
        surfs = surface(session, patoms) if len(surfaces) == 0 else surfaces
        from chimerax.core.undo import UndoState
        undo_state = UndoState('mlp')
        for s in surfs:
            surf_has_atoms = hasattr(s, 'atoms') and len(s.atoms.intersect(patoms)) > 0
            satoms = s.atoms if surf_has_atoms else patoms
            name = 'mlp ' + s.name.split(maxsplit=1)[0]
            v = mlp_map(session, satoms, method, spacing,
                        max_distance, nexp, name, open_map = map)
            if surf_has_atoms:
                if transparency is None:
                    opacity = None
                else:
                    opacity = min(255, max(0, int(2.56 * (100 - transparency))))
                from chimerax.surface import color_surfaces_by_map_value
                color_surfaces_by_map_value(satoms, map = v, palette = cmap, range = range,
                                            opacity = opacity, undo_state = undo_state)
            else:
                from chimerax.surface import color_sample
                color_sample(session, [s], v, palette = cmap, range = range,
                             transparency = transparency, undo_state = undo_state)
        session.undo.register(undo_state)
    else:
        name = 'mlp map'
        v = mlp_map(session, patoms, method, spacing, max_distance, nexp, name, open_map = map)

    if key:
        from chimerax.color_key import show_key
        if not cmap.values_specified:
            cmap = cmap.linear_range(*range)
        show_key(session, cmap)

def register_mlp_command(logger):
    from chimerax.core.commands import register, CmdDesc, FloatArg, EnumOf, BoolArg, SurfacesArg
    from chimerax.core.commands import ColormapArg, ColormapRangeArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(optional=[('atoms', AtomsArg)],
                   keyword=[('spacing', FloatArg),
                            ('max_distance', FloatArg),
                            ('method', EnumOf(['dubost','fauchere','brasseur','buckingham','type5'])),
                            ('nexp', FloatArg),
                            ('color', BoolArg),
                            ('palette', ColormapArg),
                            ('range', ColormapRangeArg),
                            ('transparency', FloatArg),
                            ('surfaces', SurfacesArg),
                            ('map', BoolArg),
                            ('key', BoolArg),
                            ],
                   synopsis='display molecular lipophilic potential for selected models')
    register('mlp', desc, mlp, logger=logger)

def mlp_map(session, atoms, method, spacing, max_dist, nexp, name, open_map):
    data, bounds = calculatefimap(atoms, method, spacing, max_dist, nexp)

    # m.pot is 1-dimensional if m.writedxfile() was called.  Has indices in x,y,z order.
    origin = tuple(xmin for xmin,xmax in bounds)
    s = spacing
    step = (s,s,s)
    from chimerax.map_data import ArrayGridData
    g = ArrayGridData(data, origin, step, name = name)
    g.polar_values = True
    from chimerax.map import volume_from_grid_data
    v = volume_from_grid_data(g, session, open_model = open_map, show_dialog = open_map)
    if open_map:
        v.update_drawings()  # Compute surface levels
        v.set_parameters(surface_colors = [(0, 139/255, 139/255, 1), (184/255, 134/255, 11/255, 1)])
    return v

#
# Code below is modified version of pyMLP, eliminating most the of code
# (unneeded parsing PDB files, writing dx files, ...) and optimizing the calculation speed.
#

class Defaults(object):
    """Constants"""

    def __init__(self):
        self.gridmargin = 10.0
        #
        # Elaine Meng replaced the MLP lipophilic potential values (July 2019) with values derived from
        # the Ghose paper because some of the original MLP values were misleading.
        #
        # Prediction of Hydrophobic (Lipophilic) Properties of Small Organic Molecules Using Fragmental Methods:  An Analysis of ALOGP and CLOGP Methods.
        # Ghose AK, Viswanadhan VN, Wendoloski JJ.
        # J. Phys. Chem. A1998; 102(21):3762-3772.
        #
        # The Ghose paper had values for hydrogens that she amalgamated into the heavy atoms.
        # It also used many atom types.
        #
        self.fidatadefault = {                    #Default fi table
 'ALA': {'CB': 0.4395,    #fi : lipophilic atomic potential
         'C': -0.1002,
         'CA': -0.1571,
         'O': -0.0233,
         'N': -0.6149},
 'ARG': {'C': -0.1002,
         'CA': -0.1571,
         'CB':  0.3212,
         'CG':  0.3212,
         'CD':  0.0116,
         'CZ':  0.5142,
         'N': -0.6149,
         'NE': -0.1425,
         'NH1': -0.5995,
         'NH2': -0.5995,
         'O': -0.0233},
 'ASN': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.0348,
         'CG': -0.1002,
         'N': -0.6149,
         'ND2': -0.7185,
         'O': -0.0233,
         'OD1': -0.0233},
 'ASP': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.0348,
         'CG': -0.1002,
         'N': -0.6149,
         'O': -0.0233,
         'OD1': -0.4087,
         'OD2': -0.4087},
 'CYS': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.0116,
         'N': -0.6149,
         'O': -0.0233,
         'SG': 0.5110},
 'GLN': {'C': -0.1002,
         'CA': -0.1571,
         'CB':  0.3212,
         'CG':  0.0348,
         'CD': -0.1002,
         'N': -0.6149,
         'NE2': -0.7185,
         'O': -0.0233,
         'OE1': -0.0233},
 'GLU': {'C': -0.1002,
         'CA': -0.1571,
         'CB':  0.3212,
         'CG':  0.0348,
         'CD': -0.1002,
         'N': -0.6149,
         'O': -0.0233,
         'OE1': -0.4087,
         'OE2': -0.4087},
 'GLY': {'C': -0.1002,
         'CA': -0.2018,
         'O': -0.0233,
         'N': -0.6149},
 'HIS': {'C': -0.1002,
         'CA': -0.1571,
         'CB':  0.0348,
         'CG': 0.2361,
         'CD2': 0.5185,
         'CE1': 0.1443,
         'N': -0.6149,
         'ND1': -0.2660,
         'NE2': -0.2660,
         'O': -0.0233},
 'HYP': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': -0.0504,
         'CD': 0.0116,
         'N': -0.5113,
         'O': -0.0233,
         'OD1': -0.4603},
 'ILE': {'C': -0.1002,
         'CA': -0.1571,
         'CB': -0.0015,
         'CG1': 0.4562,
         'CG2': 0.6420,
         'CD1': 0.6420,
         'N': -0.6149,
         'O': -0.0233},
 'LEU': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.0660,
         'CD1': 0.6420,
         'CD2': 0.6420,
         'N': -0.6149,
         'O': -0.0233},
 'LYS': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.4562,
         'CD': 0.4562,
         'CE': 0.0116,
         'NZ': -0.8535,
         'N': -0.6149,
         'O': -0.0233},
 'MET': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.0116,
         'CE': 0.1023,
         'N': -0.6149,
         'O': -0.0233,
         'SD': 0.5906},
 'MSE': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.0116,
         'CE': 0.1023,
         'N': -0.6149,
         'O': -0.0233,
         'SE': 0.6601},
 'UNK': {'C': -0.1002,
         'CA': -0.1571,
         'N': -0.6149,
         'O': -0.0233},
 'ACE': {'C': -0.1002,
         'CH3': 0.0099,
         'O': -0.0233},
 'NME': {'N': -0.6149,
         'C': 0.1023},
 'NH2': {'N': -0.7185},
 'PCA': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.0348,
         'CD': -0.1002,
         'N': -0.6149,
         'O': -0.0233,
         'OE': -0.0233},
 'PHE': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.1492,
         'CD1': 0.3050,
         'CD2': 0.3050,
         'CE1': 0.3050,
         'CE2': 0.3050,
         'CZ': 0.3050,
         'N': -0.6149,
         'O': -0.0233},
 'PRO': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.3212,
         'CD': 0.0116,
         'N': -0.5113,
         'O': -0.0233},
 'SER': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.0116,
         'N': -0.6149,
         'O': -0.0233,
         'OG': -0.4603},
 'THR': {'C': -0.1002,
         'CA': -0.1571,
         'CB': -0.0514,
         'CG2': 0.4395,
         'N': -0.6149,
         'O': -0.0233,
         'OG1': -0.4603},
 'TRP': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.0348,
         'CG': 0.1492,
         'CD1': 0.5185,
         'CD2': 0.1492,
         'CE2': 0.1539,
         'CE3': 0.3050,
         'CH2': 0.3050,
         'CZ2': 0.3050,
         'CZ3': 0.3050,
         'N': -0.6149,
         'NE1': 0.0223,
         'O': -0.0233},
 'TYR': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.1492,
         'CD1': 0.3050,
         'CD2': 0.3050,
         'CE1': 0.3050,
         'CE2': 0.3050,
         'CZ': 0.1539,
         'N': -0.6149,
         'O': -0.0233,
         'OH': -0.1163},
 'VAL': {'C': -0.1002,
         'CA': -0.1571,
         'CB': -0.0015,
         'CG1': 0.6420,
         'CG2': 0.6420,
         'N': -0.6149,
         'O': -0.0233}}

def assignfi(fidata, atoms):
    """assign fi parameters to each atom in the pdbfile"""
    n = len(atoms)
    from numpy import empty, float32
    fi = empty((n,), float32)
    resname = atoms.residues.names
    aname = atoms.names
    for i in range(n):
        rname = resname[i]
        rfidata = fidata.get(rname)
        if rfidata:
            fi[i]=rfidata.get(aname[i], 0)
    return fi

def _griddimcalc(listcoord, spacing, gridmargin):
    """Determination of the grid dimension"""
    coordmin = min(listcoord) - gridmargin
    coordmax = max(listcoord) + gridmargin
    adjustment = ((spacing - (coordmax - coordmin)) % spacing) / 2.
    coordmin = coordmin - adjustment
    coordmax = coordmax + adjustment
    ngrid = int(round((coordmax - coordmin) / spacing))
    return coordmin, coordmax, ngrid

def calculatefimap(atoms, method, spacing, max_dist, nexp):
    """Calculation loop"""

    #grid settings in angstrom
    gridmargin = Defaults().gridmargin
    xyz = atoms.scene_coords
    xmingrid, xmaxgrid, nxgrid = _griddimcalc(xyz[:,0], spacing, gridmargin)
    ymingrid, ymaxgrid, nygrid = _griddimcalc(xyz[:,1], spacing, gridmargin)
    zmingrid, zmaxgrid, nzgrid = _griddimcalc(xyz[:,2], spacing, gridmargin)
    bounds = [[xmingrid, xmaxgrid],
              [ymingrid, ymaxgrid],
              [zmingrid, zmaxgrid]]
    origin = (xmingrid, ymingrid, zmingrid)

    fi_table = Defaults().fidatadefault
    fi = assignfi(fi_table, atoms)

    from numpy import zeros, float32
    pot = zeros((nzgrid+1, nygrid+1, nxgrid+1), float32)
    # Make sure _mlp can runtime link shared library libarrays.
    import chimerax.arrays
    from ._mlp import mlp_sum
    mlp_sum(xyz, fi, origin, spacing, max_dist, method, nexp, pot)
                 
    return pot, bounds

def mlp_sum(xyz, fi, origin, spacing, max_dist, method, nexp, pot):
    computemethod = None
    if method == 'dubost':
        computemethod = _dubost
    elif method == 'fauchere':
        computemethod = _fauchere
    elif method == 'brasseur':
        computemethod = _brasseur
    elif method == 'buckingham':
        computemethod = _buckingham
    elif method == 'type5':
        computemethod = _type5
    else:
        raise ValueError('Unknown lipophilicity method %s\n' % computemethod)

    from numpy import zeros, float32, empty, subtract, sqrt
    grid_pt = empty((3,), float32)
    dxyz = empty((len(xyz),3), float32)
    dist = empty((len(xyz),), float32)
    nz,ny,nx = pot.shape
    x0,y0,z0 = origin
    for k in range(nz):
        grid_pt[2] = z0 + k * spacing
        for j in range(ny):
            grid_pt[1] = y0 + j * spacing
            for i in range(nx):
                #Evaluation of the distance between th grid point and each atoms
                grid_pt[0] = x0 + i * spacing
                subtract(xyz, grid_pt, dxyz)
                dxyz *= dxyz
                dist = dxyz[:,0]
                dist += dxyz[:,1]
                dist += dxyz[:,2]
                sqrt(dist, dist)
                pot[k,j,i] = computemethod(fi, dist, nexp)

def _dubost(fi, d, n):
    return (100 * fi / (1 + d)).sum()

def _fauchere(fi, d, n):
    from numpy import exp
    return (100 * fi * exp(-d)).sum()

def _brasseur(fi, d, n):
    #3.1 division is there to remove any units in the equation
    #3.1A is the average diameter of a water molecule (2.82 -> 3.2)
    from numpy import exp
    return (100 * fi * exp(-d/3.1)).sum()

def _buckingham(fi, d, n):
    return (100 * fi / (d**n)).sum()

def _type5(fi, d, n):
    from numpy import exp, sqrt
    return (100 * fi * exp(-sqrt(d))).sum()
