class Selection:

    def __init__(self, all_models):
        self._all_models = all_models	# Models object
        self._promotion = SelectionPromotion(all_models.drawing)
        self._intramodel_promotion_history = []

    def all_models(self):
        return self._all_models.list()

    def models(self):
        return [m for m in self.all_models() if m.any_part_selected()]

    def items(self, itype):
        si = []
        for m in self.models():
            s = m.selected_items(itype)
            si.extend(s)
        return si

    def empty(self):
        for m in self.all_models():
            if m.any_part_selected():
                return False
        return True

    def clear(self):
        for m in self.models():
            m.clear_selection()

    def clear_hierarchy(self):
        self._promotion.clear_selection_promotion_history()
        self._intramodel_promotion_history.clear()

    def promote(self):
        # Check for intra-model promotions
        imp = []
        for m in self.models():
            if hasattr(m, 'selection_promotion'):
                p = m.selection_promotion()
                if p:
                    imp.append(p)
        if imp:
            level = min(p.level for p in imp)
            impl = tuple(p for p in imp if p.level == level)
            self._intramodel_promotion_history.append(impl)
            for p in impl:
                p.promote()
        else:
            # Promote model level selection
            self._promotion.promote_selection()

    def demote(self):
        if not self._promotion.demote_selection():
            # No model level demotion so try demoting intramodel selection
            ph = self._intramodel_promotion_history
            if ph:
                for p in ph.pop():
                    p.demote()


class SelectionPromotion:

    def __init__(self, root_drawing):
        self._drawing = root_drawing
        self._promotion_history = []

    def promote_selection(self):
        '''
        Select the next larger containing group.  If one child is
        selected, then all become selected.
        '''
        
        pd = self._deepest_promotable_drawings(self._drawing)
        if len(pd) == 0:
            return

        plevel = min(level for level, d in pd)
        pdrawings = tuple(d for level, d in pd if level == plevel)
        prevsel = tuple((d, (None if d.selected_positions is None else d.selected_positions.copy()))
                        for d in pdrawings)
        self._promotion_history.append(prevsel)
        for d in pdrawings:
            d.selected = True

    def _deepest_promotable_drawings(self, drawing, level=0):
        '''
        A drawing is promotable if some children are fully selected and others
        are unselected, or if some copies are selected and other copies are
        unselected.
        '''
        sp = drawing.selected_positions
        if sp is not None:
            ns = sp.sum()
            if ns == len(sp):
                return []         # Fully selected

        if not drawing.any_part_selected():
            return []

        # Don't look at child-drawings if model supports intramodel promotion
        if not hasattr(drawing, 'selection_promotion'):
            cd = drawing.child_drawings()
            if cd:
                nfsel = [d for d in cd if not d.fully_selected()]
                if nfsel:
                    pd = sum((self._deepest_promotable_drawings(d, level+1) for d in nfsel), [])
#                    if len(pd) == 0 and len(nfsel) < len(cd):
                    if len(pd) == 0:
                        pd = [(level + 1, d) for d in nfsel]
                    return pd
        if sp is not None and ns < len(sp):
            return [(level, drawing)]
        return []

    def demote_selection(self):
        '''If the selection has previously been promoted, this returns
        it to the previous smaller selection.'''
        ph = self._promotion_history
        if len(ph) == 0:
            return False
        for d, sp in ph.pop():
            d.selected_positions = sp
        return True

    def clear_selection_promotion_history(self):
        '''
        Forget the selection history promotion history.
        This is used when the selection is changed manually.
        '''
        self._promotion_history.clear()


class IntraModelSelectionPromotion:
    def __init__(self, level):
        self.level = level
    def promote(self):
        pass
    def demote(self):
        pass
