from .graphics import Drawing
class Model(Drawing):
    '''
    A model is an object with an id number that can be specified in commands.
    '''
    def __init__(self, name):
        Drawing.__init__(self, name)
        self.id = None          # Positive integer

    def submodels(self):
        return [d for d in self.child_drawings() if isinstance(d, Model)]

    def all_models(self):
        return [self] + sum([m.all_models() for m in self.submodels()],[])

    def add_model(self, model):
        self.add_drawing(model)
        if model.id is None:
            model.id = len(self.child_drawings())

class Models:
    '''
    Manage the list of models.
    '''
    def __init__(self):

        self._root_model = Model('root')        # Root of drawing tree
        self._models = []                       # All models in drawing tree
        self.next_id = 1
        self._selected_models = None
        self.redraw_needed = False
        self.shape_changed = True
        self.xyz_bounds = None
        self.bounds_changed = True
        self.add_model_callbacks = []
        self.close_model_callbacks = []

    def model_list(self):
        '''List of open models.'''
        return self._models

    def top_level_models(self):
        return self._root_model.submodels()

    def model_count(self):
        '''Number of open models.'''
        return len(self.model_list())

    def all_drawings(self):
        return self._root_model.all_drawings()

    def model_redraw_needed(self, shape_changed = False):
        self.redraw_needed = True
        if shape_changed:
            self.shape_changed = True
            self.bounds_changed = True

    def add_model(self, model, callbacks = True):
        '''
        Add a model to the scene.  A model is a Drawing object.
        '''
        self._root_model.add_model(model)
        self._models.extend(model.all_models())
        self.set_model_id(model)
        if model.display:
            self.redraw_needed = True
            self.bounds_changed = True
            self.shape_changed = True

        model.set_redraw_callback(self.model_redraw_needed)

        if callbacks:
            for cb in self.add_model_callbacks:
                cb([model])

    def add_models(self, models):
        '''
        Add a list of models to the scene.
        '''
        for m in models:
            self.add_model(m, callbacks = False)

        for cb in self.add_model_callbacks:
            cb(models)

    def set_model_id(self, model):
        if not model.id is None:
            return
        model.id = self.next_id
        self.next_id += 1

    def find_model_by_id(self, id):
        # TODO: handle case where id is a tuple of integers
        for m in self.top_level_models():
            if m.id == id:
                return m
        return None

    def close_models(self, models):
        '''
        Remove a list of models from the scene.
        '''
        cset = sum([m.all_models() for m in models],[])
        self._models = olist = [m for m in self.model_list() if not m in cset]
        for m in models:
            if m.display:
                self.redraw_needed = True
            m.parent.remove_drawing(m)          # Removes entire drawing tree.
        self._selected_models = None
        self.next_id = 1 if len(olist) == 0 else max(m.id for m in olist) + 1
        self.bounds_changed = True
        self.shape_changed = True

        for cb in self.close_model_callbacks:
            cb(models)
        
    def close_all_models(self):
        '''
        Remove all models from the scene.
        '''
        self.close_models(tuple(self.top_level_models()))

    def selected_models(self):
        sm = self._selected_models
        if sm is None:
            sm = tuple(m for m in self.model_list() if m.any_part_selected())
            self._selected_models = sm
        return sm

    def selected_atoms(self):
        mols = self.molecules()
        sel = self.selected_models()
        smols = [m for m in sel if m in mols]
        from .molecule import Atoms
        a = Atoms()
        for m in smols:
            a.add_atoms(m.selected_atoms())
        return a

    def all_atoms(self):
        '''Return an atom set containing all atoms of all open molecules.'''
        from .molecule import Atoms
        aset = Atoms()
        aset.add_molecules(self.molecules())
        return aset

    def clear_selection(self):
        sm = self.selected_models()
        for d in sm:
            for c in d.all_drawings():
                c.clear_selection()
        self._selected_models = ()
        if sm:
            self.redraw_needed = True

    def promote_selection(self):
        sm = self.selected_models()
        for m in sm:
            m.promote_selection()

    def demote_selection(self):
        sm = self.selected_models()
        for m in sm:
            m.demote_selection()

    def selection_changed(self):
        for m in self._selected_models:
            m.clear_selection_promotion_history()
        self._selected_models = None
        self.redraw_needed = True

    def display_models(self, mlist):
        for m in mlist:
            m.display = True

    def hide_models(self, mlist):
        for m in mlist:
            m.display = False

    def maps(self):
        '''Return a list of the Volume models in the scene.'''
        from .map import Volume
        return tuple(m for m in self.model_list() if isinstance(m,Volume))

    def molecules(self):
        '''Return a list of the Molecule models in the scene.'''
        from .molecule import Molecule
        return tuple(m for m in self.model_list() if isinstance(m,Molecule))

    def surfaces(self):
        '''Return a list of the Drawings in the scene which are not Molecules.'''
        from .molecule import Molecule
        from .map import Volume
        return tuple(m for m in self.model_list() if not isinstance(m,(Molecule,Volume)))

    def bounds(self):
        if self.bounds_changed:
            from .geometry import bounds
            b = bounds.union_bounds(m.bounds() for m in self.top_level_models() if m.display)
            self.xyz_bounds = b
            self.bounds_changed = False
        return self.xyz_bounds

    def bounds_center_and_width(self):
        from .geometry import bounds
        c,r = bounds.bounds_center_and_radius(self.bounds())
        return c,r

    def center(self, models = None):
        if models is None:
            models = [m for m in self.top_level_models() if m.display]
        from .geometry import bounds
        b = bounds.union_bounds(m.bounds() for m in models)
        c,r = bounds.bounds_center_and_radius(b)
        return c
