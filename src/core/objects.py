class Objects:
    """
    Objects is a collection of models and subparts of models such as atoms.
    They are produced by evaluating command specifiers.

    Parameters
    ----------
    atoms : Atoms object
    models : readonly list of chimerax.core.models.Model
        List of models that matches the atom specifier
    """
    def __init__(self, atoms = None, models = None):
        self._models = set() if models is None else set(models)
        from .atomic import Atoms
        self._atoms = Atoms() if atoms is None else atoms

    def add_model(self, m):
        """Add model to atom spec results."""
        self._models.add(m)

    def add_atoms(self, atom_blob):
        """Add atoms to atom spec results."""
        self._atoms = self._atoms | atom_blob

    def combine(self, other):
        for m in other.models:
            self.add_model(m)
        self.add_atoms(other.atoms)

    def invert(self, session, models):
        from .atomic import Atoms, AtomicStructure
        atoms = Atoms()
        imodels = set()
        for m in models:
            if isinstance(m, AtomicStructure):
                if m in self._models:
                    # Was selected, so invert model atoms
                    keep = m.atoms - self._atoms
                else:
                    # Was not selected, so include all atoms
                    keep = m.atoms
                if len(keep) > 0:
                    atoms = atoms | keep
                    imodels.add(m)
            elif m not in self._models:
                imodels.add(m)
        self._atoms = atoms
        self._models = imodels

    @property
    def models(self):
        return self._models

    @property
    def atoms(self):
        return self._atoms

    @staticmethod
    def union(left, right):
        atom_spec = Objects()
        atom_spec._models = left._models | right._models
        atom_spec._atoms = right._atoms.merge(left._atoms)
        return atom_spec

    @staticmethod
    def intersect(left, right):
        atom_spec = Objects()
        atom_spec._models = left._models & right._models
        atom_spec._atoms = right._atoms & left._atoms
        return atom_spec

    def empty(self):
        return len(self._atoms) == 0 and len(self._models) == 0

    def displayed(self):
        '''Return Objects containing only displayed atoms and models.'''
	# Displayed models
        dmodels = set(m for m in self.models if m.display and m.parents_displayed)
        return Objects(self.atoms.shown_atoms, dmodels)

    def bounds(self):
        from .atomic import AtomicStructure
        bm = [m.bounds() for m in self.models if not isinstance(m, AtomicStructure)]
        from .geometry import union_bounds
        return union_bounds(bm + [self.atoms.scene_bounds])
