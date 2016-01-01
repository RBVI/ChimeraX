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
        from .atomic import Atoms
        self._atoms = Atoms() if atoms is None else atoms

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

    def add_atoms(self, atom_blob):
        """Add atoms to collection."""
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
                    # include atoms not in current collection
                    keep = m.atoms - self._atoms
                else:
                    # include all atoms
                    keep = m.atoms
                if len(keep) > 0:
                    atoms = atoms | keep
                    imodels.add(m)
            elif m not in self._models:
                imodels.add(m)
        self._atoms = atoms
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
        return self._atoms

    @staticmethod
    def union(left, right):
        u = Objects(models = (left._models | right._models),
                    atoms = right._atoms.merge(left._atoms))
        for m, minst in tuple(left.model_instances.items()) + tuple(right.model_instances.items()):
            u.add_model_instances(m, minst)
        return u

    @staticmethod
    def intersect(left, right):
        u = Objects(models = (left._models & right._models),
                    atoms = (right._atoms & left._atoms))
        lmi, rmi = left.model_instances, right.model_instances
        from numpy import logical_and
        for m in lmi.keys():
            if m in rmi.keys():
                u.add_model_instances(m, logical_and(lmi[m], rmi[m]))
        return u

    def empty(self):
        return len(self._atoms) == 0 and len(self._models) == 0 and len(self._model_instances) == 0

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
