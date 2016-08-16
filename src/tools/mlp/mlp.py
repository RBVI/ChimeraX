# vim: set expandtab shiftwidth=4 softtabstop=4:


def mlp(session, atoms, method="fauchere", spacing=1.0, nexp=3.0,
        color=True, palette=None, range=None, map=False):
    '''Display Molecular Lipophilic Potential for a single model.

    Parameters
    ----------
    atoms : Atoms
        Color surfaces for these atoms using MLP map.
    method : 'dubost','fauchere','brasseur','buckingham','type5'
        Distance dependent function to use for calculation
    spacing : float
    	Grid spacing, default 1 Angstrom.
    nexp : float
        The buckingham method uses this numerical exponent.
    color : bool
        Whether to color molecular surfaces. They are created if they don't yet exist.
    palette : Colormap
        Color palette for coloring surfaces.
        Default is lipophilicity colormap (orange lipophilic, blue lipophobic).
    range : 2-tuple of float
        Range of lipophilicity values defining ends of color map.  Default is -20,20
    map : bool
        Whether to open a volume model of lipophilicity values
    '''
    if palette is None:
        from chimerax.core.colors import BuiltinColormaps
        cmap = BuiltinColormaps['lipophilicity']
    else:
        cmap = palette
    if range is None:
        range = (-20,20)
        
    # Color surfaces by lipophilicity
    if color:
        # Compute surfaces if not already created
        from chimerax.core.commands.surface import surface
        surfs = surface(session, atoms)
        for s in surfs:
            satoms = s.atoms
            v = mlp_map(session, satoms, method, spacing, nexp, open_map = map)
            from chimerax.core.commands.scolor import scolor
            scolor(session, satoms, map = v, palette = cmap, range = range)
    else:
        v = mlp_map(session, atoms, method, spacing, nexp, open_map = map)
            

def register_mlp_command():
    from chimerax.core.commands import register, CmdDesc, AtomsArg, SaveFileNameArg, FloatArg, EnumOf, NoArg, BoolArg, ColormapArg, ColormapRangeArg
    desc = CmdDesc(required=[('atoms', AtomsArg)],
                   keyword=[('spacing', FloatArg),
                            ('method', EnumOf(['dubost','fauchere','brasseur','buckingham','type5'])),
                            ('nexp', FloatArg),
                            ('color', BoolArg),
                            ('palette', ColormapArg),
                            ('range', ColormapRangeArg),
                            ('map', NoArg),
                            ],
                   synopsis='display molecular lipophilic potential for selected models')
    register('mlp', desc, mlp)

def mlp_map(session, atoms, method, spacing, nexp, open_map):
    data, bounds = calculatefimap(atoms, method, spacing, nexp)

    # m.pot is 1-dimensional if m.writedxfile() was called.  Has indices in x,y,z order.
    origin = tuple(xmin for xmin,xmax in bounds)
    s = spacing
    step = (s,s,s)
    from chimerax.core.map.data import Array_Grid_Data
    g = Array_Grid_Data(data, origin, step, name = 'mlp map')
    g.polar_values = True
    from chimerax.core.map import volume_from_grid_data
    v = volume_from_grid_data(g, session, open_model = open_map, show_data = open_map, show_dialog = open_map)
    return v

#
# Code below is modified version of pyMLP, eliminating most the of code
# (unneeded parsing PDB files, writing dx files, ...) and optimizing the calculation speed.
#

class Defaults(object):
    """Constants"""

    def __init__(self):
        self.gridmargin = 10.0
        self.fidatadefault = {                    #Default fi table
 'ALA': {'CB': 0.63,    #fi : lipophilic atomic potential
         'C': -0.54,
         'CA': 0.02,
         'O': -0.68,
         'N': -0.44},
 'ARG': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD': 0.45,
         'CG': 0.45,
         'CZ': -0.54,
         'N': -0.44,
         'NE': -0.55,
         'NH1': -0.11,
         'NH2': -0.83,
         'O': -0.68},
 'ASN': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.02,
         'CG': 0.45,
         'N': -0.44,
         'ND2': -0.11,
         'O': -0.68,
         'OD1': -0.68},
 'ASP': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CG': 0.54,
         'N': -0.44,
         'O': -0.68,
         'OD1': -0.68,
         'OD2': 0.53},
 'CYS': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'N': -0.44,
         'O': -0.68,
         'SG': 0.27},
 'GLN': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD': -0.54,
         'CG': 0.45,
         'N': -0.44,
         'NE2': -0.11,
         'O': -0.68,
         'OE1': -0.68},
 'GLU': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD': -0.54,
         'CG': 0.45,
         'N': -0.44,
         'O': -0.68,
         'OE1': -0.68,
         'OE2': 0.53},
 'GLY': {'C': -0.54,
         'CA': 0.45,
         'O': -0.68,
         'N': -0.55},
 'HIS': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD2': 0.31,
         'CE1': 0.31,
         'CG': 0.09,
         'N': -0.44,
         'ND1': -0.56,
         'NE2': -0.80,
         'O': -0.68},
 'HYP': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD1': 0.45,
         'CG': 0.02,
         'N': -0.92,
         'O': -0.68,
         'OD2': -0.93},
 'ILE': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.02,
         'CD': 0.63,
         'CD1': 0.63,
         'CG1': 0.45,
         'CG2': 0.63,
         'N': -0.44,
         'O': -0.68},
 'LEU': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD1': 0.63,
         'CD2': 0.63,
         'CG': 0.02,
         'N': -0.44,
         'O': -0.68},
 'LYS': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD': 0.45,
         'CE': 0.45,
         'CG': 0.45,
         'N': -0.44,
         'NZ': -1.08,
         'O': -0.68},
 'MET': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CE': 0.63,
         'CG': 0.45,
         'N': -0.44,
         'O': -0.68,
         'SD': -0.30},
 'PCA': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD': -0.54,
         'CG': 0.45,
         'N': 1.52,
         'O': -0.68,
         'OE': -0.68},
 'PHE': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD1': 0.31,
         'CD2': 0.31,
         'CE1': 0.31,
         'CE2': 0.31,
         'CG': 0.09,
         'CZ': 0.31,
         'N': -0.44,
         'O': -0.68},
 'PRO': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD': 0.45,
         'CG': 0.45,
         'N': -0.92,
         'O': -0.68},
 'SER': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'N': -0.44,
         'O': -0.68,
         'OG': -0.99},
 'THR': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.02,
         'CG2': 0.63,
         'N': -0.44,
         'O': -0.68,
         'OG1': -0.93},
 'TRP': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD1': 0.31,
         'CD2': 0.24,
         'CE2': 0.24,
         'CE3': 0.31,
         'CG': 0.09,
         'CH2': 0.31,
         'CZ2': 0.31,
         'CZ3': 0.31,
         'N': -0.44,
         'NE1': -0.55,
         'O': -0.68},
 'TYR': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.45,
         'CD1': 0.31,
         'CD2': 0.31,
         'CE1': 0.31,
         'CE2': 0.31,
         'CG': 0.09,
         'CZ': 0.09,
         'N': -0.44,
         'O': -0.68,
         'OH': -0.17},
 'VAL': {'C': -0.54,
         'CA': 0.02,
         'CB': 0.02,
         'CG1': 0.63,
         'CG2': 0.63,
         'N': -0.44,
         'O': -0.68}}

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

def calculatefimap(atoms, method, spacing, nexp):
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
    sum_fi(xyz, fi, origin, spacing, method, nexp, pot)
                 
    return pot, bounds

def sum_fi(xyz, fi, origin, spacing, method, nexp, pot):
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
