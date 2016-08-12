# vim: set expandtab shiftwidth=4 softtabstop=4:


def mlp(session, atoms, dx=None, method="fauchere", spacing=1.0, nexp=3.0):
    '''Display Molecular Lipophilic Potential for a single model.

    Parameters
    ----------
    atoms : Atoms
        Show MLP map for the specified model.
    dx : dx file name
        Name of file for computed dx map.
    method : 'dubost','fauchere','brasseur','buckingham','type5'
        Distance dependent function to use for calculation
    spacing : float
    	Grid spacing, default 1 Angstrom.
    nexp : float
        The buckingham method uses this numerical exponent.
    '''
    from .pyMLP import Molecule, Defaults
    defaults = Defaults()
    m = Molecule()
    m.data = _MLPAtomicStructureAdapter(atoms)
    m.assignfi(defaults.fidatadefault)
    m.calculatefimap(method, spacing, nexp)
    delete_temp = dx is None
    if dx is None:
        import tempfile
        tf = tempfile.NamedTemporaryFile(prefix="chtmp", suffix=".dx")
        dx = tf.name
        tf.close()
    try:
        m.writedxfile(dx)
        from chimerax.core.commands import run
        run(session, "open %s" % dx)
    finally:
        if delete_temp:
            import os
            try:
                os.remove(dx)
            except OSError:
                pass


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


def register_mlp_command():
    from chimerax.core.commands import register, CmdDesc, AtomsArg, SaveFileNameArg, FloatArg, EnumOf
    desc = CmdDesc(required=[("atoms", AtomsArg)],
                   keyword=[("dx", SaveFileNameArg),
                            ("spacing", FloatArg),
                            ("method", EnumOf(['dubost','fauchere','brasseur','buckingham','type5'])),
                            ("nexp", FloatArg),
                            ],
                   synopsis='display molecular lipophilic potential for selected models')
    register('mlp', desc, mlp)
