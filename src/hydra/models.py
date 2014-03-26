
class Models:
    '''
    Manage the list of models.
    '''
    def __init__(self):

        self.models = []
        self.next_id = 1
        self.selected = set()
        self.redraw_needed = False
        self.xyz_bounds = None
        self.bounds_changed = True

    def model_list(self):
        '''List of open models.'''
        return self.models

    def model_count(self):
        '''Number of open models.'''
        return len(self.models)

    def add_model(self, model):
        '''
        Add a model to the scene.  A model is a Surface object.
        '''
        self.models.append(model)
        if model.id is None:
            model.id = self.next_id
            self.next_id += 1
        if model.display:
            self.redraw_needed = True
            self.bounds_changed = True

    def add_models(self, mlist):
        '''
        Add a list of models to the scene.
        '''
        for m in mlist:
            self.add_model(m)
        
    def close_models(self, mlist):
        '''
        Remove a list of models from the scene.
        '''
        olist = self.models
        for m in mlist:
            olist.remove(m)
            self.selected.discard(m)
            if m.display:
                self.redraw_needed = True
            m.delete()
        self.next_id = 1 if len(olist) == 0 else max(m.id for m in olist) + 1
        self.bounds_changed = True
        
    def close_all_models(self):
        '''
        Remove all models from the scene.
        '''
        self.close_models(tuple(self.models))

    def select_model(self, m):
        self.selected.add(m)
        if m.display:
            self.redraw_needed = True

    def unselect_model(self, m):
        self.selected.discard(m)
        if m.display:
            self.redraw_needed = True

    def clear_selection(self):
        for m in self.selected:
            m.selected = False
        if self.selected:
            self.redraw_needed = True
        self.selected.clear()

    def display_models(self, mlist):
        for m in mlist:
            m.display = True
            m.redraw_needed = True

    def hide_models(self, mlist):
        for m in mlist:
            m.display = False
            m.redraw_needed = True

    def maps(self):
        '''Return a list of the Volume models in the scene.'''
        from .map import Volume
        return tuple(m for m in self.models if isinstance(m,Volume))

    def molecules(self):
        '''Return a list of the Molecule models in the scene.'''
        from .molecule import Molecule
        return tuple(m for m in self.models if isinstance(m,Molecule))

    def surfaces(self):
        '''Return a list of the Surface models in the scene which are not Molecules.'''
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
