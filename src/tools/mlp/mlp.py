# vim: set expandtab shiftwidth=4 softtabstop=4:


def mlp(session, atoms, method="fauchere", spacing=1.0, nexp=3.0,
        color=True, palette=None, range=None, map=False):
    '''Display Molecular Lipophilic Potential for a single model.

    Parameters
    ----------
    atoms : Atoms
        Show MLP map for the specified model.
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
    from .pyMLP import Molecule, Defaults
    defaults = Defaults()
    m = Molecule()
    m.data = _MLPAtomicStructureAdapter(atoms)
    m.assignfi(defaults.fidatadefault)
    m.calculatefimap(method, spacing, nexp)

    # Create mlp map
    v = mlp_map(m, session, open_map = map)

    # Color surfaces by lipophilicity
    if color:
        # Compute surfaces if not already created
        from chimerax.core.commands.surface import surface
        surface(session, atoms)
        
        if palette is None:
            from chimerax.core.colors import BuiltinColormaps
            cmap = BuiltinColormaps['lipophilicity']
        else:
            cmap = palette
        if range is None:
            range = (-20,20)
        from chimerax.core.commands.scolor import scolor
        scolor(session, atoms, map = v, palette = cmap, range = range)

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

class _MLPAtomicStructureAdapter:
    '''Adapter class to enable pyMLP to access atomic structure data'''

    def __init__(self, atoms):
        self.atoms = atoms
        self.atom_map = {}

    def __iter__(self):
        amap = self.atom_map
        for a in self.atoms:
            if a in amap:
                aa = amap[a]
            else:
                aa = _MLPAtomAdapter(a)
                amap[a] = aa
            yield aa
        raise StopIteration()


class _MLPAtomAdapter:
    '''Adapter class to enable pyMLP to access atom data'''

    __slots__ = ["atom", "fi"]

    def __init__(self, atom):
        self.atom = atom
        self.fi = None

    def __str__(self):
        return str(self.atom)

    def __setitem__(self, key, value):
        if key == 'fi':
            self.fi = value
        else:
            raise KeyError("\"%s\" not supported in MLPAdapter" % key)

    def __getitem__(self, key):
        if key == 'fi':
            return self.fi
        elif key == 'atmx':
            return self.atom.coord[0]
        elif key == 'atmy':
            return self.atom.coord[1]
        elif key == 'atmz':
            return self.atom.coord[2]
        elif key == 'resname':
            return self.atom.residue.name
        elif key == 'atmname':
            return self.atom.name
        elif key == 'atmnumber':
            return self.atom.element_number
        else:
            raise KeyError("\"%s\" not supported in MLPAdapter" % key)

def mlp_map(m, session, open_map):
    # m.pot is 1-dimensional if m.writedxfile() was called.  Has indices in x,y,z order.
    data = m.pot.reshape(m.griddim).transpose()
    origin = tuple(xmin for xmin,xmax in m.gridcoord)
    s = m.spacing
    step = (s,s,s)
    from chimerax.core.map.data import Array_Grid_Data
    g = Array_Grid_Data(data, origin, step, name = 'mlp map')
    g.polar_values = True
    from chimerax.core.map import volume_from_grid_data
    v = volume_from_grid_data(g, session, open_model = open_map, show_data = open_map, show_dialog = open_map)
    return v
