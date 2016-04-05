# vim: set expandtab shiftwidth=4 softtabstop=4:


from chimerax.core.atomic import AtomicStructure


def mlp(session, model=None, dx=None, method="fauchere", spacing=1.0, nexp=3.0):
    '''Display Molecular Lipophilic Potential for a single model.

    Parameters
    ----------
    spec : model specifier
        Show MLP map for the specified model.
    dx : dx file name
        Name of file for computed dx map.
    '''
    if model is None:
        from chimerax.core.commands import atomspec
        structures = session.models.list(type=AtomicStructure)
        if len(structures) != 1:
            from chimerax.core.errors import UserError
            raise UserError("mlp command works with exactly one model.")
        model = structures[0]
    from .pyMLP import Molecule, Defaults
    defaults = Defaults()
    m = Molecule()
    m.data = _MLPAtomicStructureAdapter(model)
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

    def __init__(self, m):
        self.model = m
        self.atoms = {}

    def __iter__(self):
        for a in self.model.atoms:
            try:
                aa = self.atoms[a]
            except KeyError:
                aa = _MLPAtomAdapter(a)
                self.atoms[a] = aa
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


def initialize(command_name):
    from chimerax.core.commands import register, CmdDesc
    from chimerax.core.commands import ModelArg, SaveFileNameArg
    from chimerax.core.commands import Bounded, FloatArg, EnumOf
    desc = CmdDesc(optional=[("model", ModelArg)],
                   keyword=[("dx", SaveFileNameArg),
                            ("spacing", Bounded(FloatArg, 0.1, 10.0)),
                            ("method", EnumOf(['dubost','fauchere','brasseur','buckingham','type5'])),
                            ("nexp", FloatArg),
                            ],
                   synopsis='display molecular lipophilic potential for selected models')
    register(command_name, desc, mlp)
