
class Models:
    '''
    Manage the list of models.
    '''
    def __init__(self):

        self.models = []
        self.next_id = 1
        self._selected_models = None
        self.redraw_needed = False
        self.xyz_bounds = None
        self.bounds_changed = True
        self.add_model_callbacks = []
        self.close_model_callbacks = []

    def model_list(self):
        '''List of open models.'''
        return self.models

    def model_count(self):
        '''Number of open models.'''
        return len(self.models)
    
    def model_redraw_needed(self):
        self.redraw_needed = True
        self.bounds_changed = True

    def add_model(self, model, callbacks = True):
        '''
        Add a model to the scene.  A model is a Drawing object.
        '''
        self.models.append(model)
        if model.id is None:
            model.id = self.next_id
            self.next_id += 1
        if model.display:
            self.redraw_needed = True
            self.bounds_changed = True

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
        
    def close_models(self, models):
        '''
        Remove a list of models from the scene.
        '''
        olist = self.models
        for m in models:
            olist.remove(m)
            if m.display:
                self.redraw_needed = True
            m.delete()
        self._selected_models = None
        self.next_id = 1 if len(olist) == 0 else max(m.id for m in olist) + 1
        self.bounds_changed = True

        for cb in self.close_model_callbacks:
            cb(models)
        
    def close_all_models(self):
        '''
        Remove all models from the scene.
        '''
        self.close_models(tuple(self.models))

    def selected_models(self):
        sm = self._selected_models
        if sm is None:
            sm = tuple(m for m in self.model_list() if m.selected)
            self._selected_models = sm
        return sm

    def clear_selection(self):
        sm = self.selected_models()
        for m in sm:
            m.selected = False
        self._selected_models = ()
        if sm:
            self.redraw_needed = True

    def selection_changed(self):
        self._selected_models = None
        self.redraw_needed = True

    def display_models(self, mlist):
        for m in mlist:
            m.display = True
            m.redraw_needed()

    def hide_models(self, mlist):
        for m in mlist:
            m.display = False
            m.redraw_needed()

    def maps(self):
        '''Return a list of the Volume models in the scene.'''
        from .map import Volume
        return tuple(m for m in self.models if isinstance(m,Volume))

    def molecules(self):
        '''Return a list of the Molecule models in the scene.'''
        from .molecule import Molecule
        return tuple(m for m in self.models if isinstance(m,Molecule))

    def surfaces(self):
        '''Return a list of the Drawings in the scene which are not Molecules.'''
        from .molecule import Molecule
        return tuple(m for m in self.models if not isinstance(m,(Molecule)))

    def bounds(self):
        if self.bounds_changed:
            from .geometry import bounds
            b = bounds.union_bounds(m.placed_bounds() for m in self.models if m.display)
            self.xyz_bounds = b
            self.bounds_changed = False
        return self.xyz_bounds

    def bounds_center_and_width(self):
        from .geometry import bounds
        c,r = bounds.bounds_center_and_radius(self.bounds())
        return c,r

    def center(self, models = None):
        if models is None:
            models = [m for m in self.models if m.display]
        from .geometry import bounds
        b = bounds.union_bounds(m.placed_bounds() for m in models)
        c,r = bounds.bounds_center_and_radius(b)
        return c
