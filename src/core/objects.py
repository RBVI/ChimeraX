class Objects:
    """
    Objects is a collection of models and subparts of models such as atoms.
    They are produced by evaluating command specifiers.

    Objects also can include model instances. Currently models and model
    instances are independent parts of the collection. Probably models
    and model instances should always contain exactly the same models.
    But the current semantics are unclear.  Models with atoms appear
    in the models set even if not all the atoms are in the Objects collection.
    This needs more thought.

    Parameters
    ----------
    atoms : Atoms object
    models : readonly list of chimerax.core.models.Model
    """
    def __init__(self, atoms = None, models = None):
        self._models = set() if models is None else set(models)
        self._model_instances = {}
        # Use a list of Atoms collections so many concatenations is fast.
        self._atoms = [] if atoms is None else [atoms]
        self._cached_atoms = None	# Atoms collection containing all atoms.

    def add_model(self, m):
        """Add model to collection."""
        self._models.add(m)

    def add_model_instances(self, m, imask):
        """Add model instances to collection."""
        mi = self._model_instances
        if m in mi:
            i = mi[m]
            from numpy import logical_or
            logical_or(i, imask, i)
        else:
            mi[m] = imask.copy()

    def add_atoms(self, atoms):
        """Add atoms to collection."""
        self._atoms.append(atoms)
        self._cached_atoms = None

    def combine(self, other):
        for m in other.models:
            self.add_model(m)
        self.add_atoms(other.atoms)

    def invert(self, session, models):
        from .atomic import AtomicStructure, Atoms, concatenate
        matoms = []
        imodels = set()
        for m in models:
            if isinstance(m, AtomicStructure):
                matoms.append(m.atoms)
            elif m not in self._models:
                imodels.add(m)
        iatoms = concatenate(matoms, Atoms, remove_duplicates=True) - self.atoms
        imodels.update(iatoms.unique_structures)
        self._atoms = [iatoms]
        self._cached_atoms = iatoms
        self._models = imodels

        from numpy import logical_not
        self._model_instances = {m:logical_not(minst) for m, minst in self.model_instances.items()}

    @property
    def models(self):
        return self._models

    @property
    def model_instances(self):
        return self._model_instances

    @property
    def atoms(self):
        ca = self._cached_atoms
        if ca is None:
            from . import atomic
            ca = atomic.concatenate(self._atoms, atomic.Atoms, remove_duplicates = True)
            self._cached_atoms = None
        return ca

    @property
    def num_atoms(self):
        return  sum(len(a) for a in self._atoms) if self._cached_atoms is None else len(self._cached_atoms)

    @staticmethod
    def union(left, right):
        u = Objects(models = (left._models | right._models))
        u._atoms = left._atoms + right._atoms
        for m, minst in tuple(left.model_instances.items()) + tuple(right.model_instances.items()):
            u.add_model_instances(m, minst)
        return u

    @staticmethod
    def intersect(left, right):
        u = Objects(models = (left._models & right._models),
                    atoms = (right.atoms & left.atoms))
        lmi, rmi = left.model_instances, right.model_instances
        from numpy import logical_and
        for m in lmi.keys():
            if m in rmi.keys():
                u.add_model_instances(m, logical_and(lmi[m], rmi[m]))
        return u

    def empty(self):
        return self.num_atoms == 0 and len(self._models) == 0 and len(self._model_instances) == 0

    def displayed(self):
        '''Return Objects containing only displayed atoms and models.'''
	# Displayed models
        dmodels = set(m for m in self.models if m.display and m.parents_displayed)
        d = Objects(self.atoms.shown_atoms, dmodels)
        from numpy import logical_and
        for m, minst in self.model_instances.items():
            d.add_model_instances(m, logical_and(minst, m.displayed_positions))
        return d

    def bounds(self):
        from .atomic import AtomicStructure
        bm = [m.bounds() for m in self.models if not isinstance(m, AtomicStructure)]
        from .geometry import union_bounds, copies_bounding_box
        for m, minst in self.model_instances.items():
            b = m.bounds(positions = False)
            bm.append(copies_bounding_box(b, m.positions.mask(minst)))
        return union_bounds(bm + [self.atoms.scene_bounds])
