# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
objects: Manage objects in selection
====================================

The objects module keeps track of collections of objects.
The most common use is for parsing atom specifiers and
tracking what atoms and bonds match.
"""

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
    bonds : Bonds object
    pseudobonds : Pseudobonds object
    models : readonly list of chimerax.core.models.Model
    """
    def __init__(self, atoms = None, bonds = None, pseudobonds = None, models = None):
        from .orderedset import OrderedSet
        self._models = OrderedSet() if models is None else OrderedSet(models)
        self._model_instances = {}	# Maps Model to boolean array of length equal to number of instances
        # Use a list of Atoms collections so many concatenations is fast.
        self._atoms = [] if atoms is None else [atoms]
        self._cached_atoms = None	# Atoms collection containing all atoms.
        self._bonds = [] if bonds is None else [bonds]
        self._pseudobonds = [] if pseudobonds is None else [pseudobonds]
        # Using a strong ref dict;  weak vs. strong only matters for models that are never opened.
        # Opened models will always call delete() when closed.  Unopened models will have to explicitly
        # call delete() to clean up.
        self._handlers = { m:m.triggers.add_handler("deleted", self._model_deleted_cb) for m in self._models }

    def __del__(self):
        for handler in self._handlers.values():
            handler.remove()
        self._handlers.clear()
        self._model_instances.clear()
        self._models.clear()

    def add_model(self, m):
        """Add model to collection."""
        self._models.add(m)
        # model might not previously be in self._models but be in self._handlers because of instances
        if m not in self._handlers:
            self._handlers[m] = m.triggers.add_handler("deleted", self._model_deleted_cb)

    def _model_deleted_cb(self, _trig_name, model):
        self._models.discard(model)
        if model in self._model_instances:
            del self._model_instances[model]
        del self._handlers[model]
        from .triggerset import DEREGISTER
        return DEREGISTER

    def add_model_instances(self, m, imask):
        """Add model instances to collection."""
        mi = self._model_instances
        if m in mi:
            i = mi[m]
            from numpy import logical_or
            logical_or(i, imask, i)
        else:
            mi[m] = imask.copy()
            if m not in self._handlers:
                self._handlers[m] = m.triggers.add_handler("deleted", self._model_deleted_cb)

    def add_atoms(self, atoms, bonds=False):
        """Add atoms to collection.  If bonds is true also add bonds between specified atoms."""
        self._atoms.append(atoms)
        self._cached_atoms = None
        if bonds:
            self.add_bonds(atoms.intra_bonds)

    def add_bonds(self, bonds):
        """Add bonds to collection."""
        self._bonds.append(bonds)

    def add_pseudobonds(self, pbonds):
        """Add pseudobonds to collection."""
        self._pseudobonds.append(pbonds)

    def combine(self, other):
        for m in other.models:
            self.add_model(m)
        self.add_atoms(other.atoms)
        self.add_bonds(other.bonds)
        self.add_pseudobonds(other.pseudobonds)

    def invert(self, session, models):
        from chimerax.atomic import Structure, PseudobondGroup, Atoms, Bonds, Pseudobonds, concatenate
        matoms = []
        mbonds = []
        mpbonds = []
        from .orderedset import OrderedSet
        imodels = OrderedSet()
        for m in models:
            if isinstance(m, Structure):
                matoms.append(m.atoms)
                mbonds.append(m.bonds)
            elif isinstance(m, PseudobondGroup):
                mpbonds.append(m.pseudobonds)
            elif m not in self._models:
                for sm in m.selection_coupled:
                    if sm in self._models:
                        break
                else:
                    imodels.add(m)
        iatoms = concatenate(matoms, Atoms, remove_duplicates=True) - self.atoms
        ibonds = concatenate(mbonds, Bonds, remove_duplicates=True) - self.bonds
        ipbonds = concatenate(mpbonds, Pseudobonds, remove_duplicates=True) - self.pseudobonds
        imodels.update(iatoms.unique_structures)
        self._atoms = [iatoms]
        self._bonds = [ibonds]
        self._pseudobonds = [ipbonds]
        self._cached_atoms = iatoms
        self._models = imodels

        from numpy import logical_not
        self._model_instances = {m:logical_not(minst) for m, minst in self._model_instances.items()}

        for handler in self._handlers.values():
            handler.remove()
        self._handlers.clear()
        for m in self._models:
            self._handlers[m] = m.triggers.add_handler("deleted", self._model_deleted_cb)
        for m in self._model_instances:
            if m not in self._handlers:
                self._handlers[m] = m.triggers.add_handler("deleted", self._model_deleted_cb)

    @property
    def models(self):
        return self._models

    @property
    def model_instances(self):
        self._remove_deleted_model_instances()
        return self._model_instances

    def _remove_deleted_model_instances(self):
        minst = self._model_instances
        mdel = [m for m,mask in minst.items() if len(mask) != len(m.positions)]
        if mdel:
            for m in mdel:
                del minst[m]
                if m not in self._models:
                    self._handlers[m].remove()
                    del self._handlers[m]
    @property
    def atoms(self):
        ca = self._cached_atoms
        if ca is None:
            from chimerax import atomic
            ca = atomic.concatenate(self._atoms, atomic.Atoms, remove_duplicates = True)
            self._cached_atoms = ca
        return ca

    @property
    def num_atoms(self):
        return sum(len(a) for a in self._atoms) if self._cached_atoms is None else len(self._cached_atoms)

    @property
    def bonds(self):
        from chimerax import atomic
        return atomic.concatenate(self._bonds, atomic.Bonds, remove_duplicates = True)

    @property
    def num_bonds(self):
        return sum(len(b) for b in self._bonds)

    @property
    def pseudobonds(self):
        from chimerax import atomic
        return atomic.concatenate(self._pseudobonds, atomic.Pseudobonds, remove_duplicates = True)

    @property
    def num_pseudobonds(self):
        return sum(len(b) for b in self._pseudobonds)

    @staticmethod
    def union(left, right):
        u = Objects(models = (left._models | right._models))
        u._atoms = left._atoms + right._atoms
        u._bonds = left._bonds + right._bonds
        u._pseudobonds = left._pseudobonds + right._pseudobonds
        for m, minst in tuple(left.model_instances.items()) + tuple(right.model_instances.items()):
            u.add_model_instances(m, minst)
        return u

    @staticmethod
    def intersect(left, right):
        u = Objects(models = (left._models & right._models),
                    atoms = (right.atoms & left.atoms),
                    bonds = (right.bonds & left.bonds),
                    pseudobonds = (right.pseudobonds & left.pseudobonds))
        lmi, rmi = left.model_instances, right.model_instances
        from numpy import logical_and
        for m in lmi.keys():
            if m in rmi.keys():
                u.add_model_instances(m, logical_and(lmi[m], rmi[m]))
        return u

    def empty(self):
        self._remove_deleted_model_instances()
        return (self.num_atoms == 0 and self.num_bonds == 0 and self.num_pseudobonds == 0
                and len(self._models) == 0 and len(self._model_instances) == 0)

    def displayed(self):
        '''Return Objects containing only displayed atoms, bonds, pseudobonds and models.'''
	# Displayed models
        from .orderedset import OrderedSet
        dmodels = OrderedSet(m for m in self.models if m.display and m.parents_displayed)
        bonds, pbonds = self.bonds, self.pseudobonds
        d = Objects(atoms = self.atoms.shown_atoms, bonds = bonds[bonds.displays],
                    pseudobonds = pbonds[pbonds.displays], models = dmodels)
        from numpy import logical_and
        for m, minst in self.model_instances.items():
            d.add_model_instances(m, logical_and(minst, m.display_positions))
        return d

    def bounds(self):
        '''Bounds of objects including undisplayed ones in scene coordinates.'''
        from .geometry import union_bounds, copies_bounding_box

        # Model bounds excluding structures and pseudobond groups.
        from chimerax.atomic import Structure, PseudobondGroup
        bm = [copies_bounding_box(m.geometry_bounds(), m.get_scene_positions())
              for m in self.models if not isinstance(m, (Structure, PseudobondGroup))]

        # Model instance bounds
        for m, minst in self.model_instances.items():
            b = m.geometry_bounds()
            ib = copies_bounding_box(b, m.positions.masked(minst))
            sb = copies_bounding_box(ib, m.parent.get_scene_positions())
            bm.append(sb)

        # Atom bounds
        bm.append(self.atoms.scene_bounds)

        b = union_bounds(bm)
        return b
