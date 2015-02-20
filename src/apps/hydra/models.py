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
        self.add_model_callbacks = []
        self.close_model_callbacks = []

    def model_list(self):
        '''List of open models.'''
        return self._models

    def drawing(self):
        return self._root_model

    def top_level_models(self):
        return self._root_model.submodels()

    def model_count(self):
        '''Number of open models.'''
        return len(self.model_list())

    def all_drawings(self):
        return self._root_model.all_drawings()

    def add_model(self, model, callbacks = True):
        '''
        Add a model to the scene.  A model is a Drawing object.
        '''
        self._root_model.add_model(model)
        self._models.extend(model.all_models())
        self.set_model_id(model)

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
            self.next_id = max(self.next_id, model.id+1)
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
        Remove a list of models from the scene, delete them, and call close callbacks.
        '''
        cset = sum([m.all_models() for m in models],[])
        self._models = olist = [m for m in self.model_list() if not m in cset]
        for m in models:
            m.parent.remove_drawing(m)          # Removes entire drawing tree.
        self.next_id = 1 if len(olist) == 0 else max(m.id for m in olist) + 1

        for cb in self.close_model_callbacks:
            cb(models)
        
    def close_all_models(self):
        '''
        Remove all models from the scene.
        '''
        self.close_models(tuple(self.top_level_models()))

    def selected_models(self):
        if not self._root_model.any_part_selected():
            return []
        sm = tuple(m for m in self.model_list() if m.any_part_selected())
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

    def promote_selection(self):
        sm = self.selected_models()
        for m in sm:
            m.promote_selection()

    def demote_selection(self):
        sm = self.selected_models()
        for m in sm:
            m.demote_selection()

    def clear_selection_hierarchy(self):
        for m in self.selected_models():
            m.clear_selection_promotion_history()

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
        return self._root_model.bounds()
