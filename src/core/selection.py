class Selection:

    def __init__(self, all_models):
        self._all_models = all_models	# Models object
        self._promotion = SelectionPromoter(all_models.drawing)

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

    def promote(self):
        self._promotion.promote_selection()

    def demote(self):
        self._promotion.demote_selection()


class SelectionPromoter:

    def __init__(self, root_drawing):
        self._drawing = root_drawing
        self._promotion_history = []

    def promote_selection(self):
        '''
        Select the next larger containing group.  If one child is
        selected, then all become selected.
        '''

        plist = []
        self._find_deepest_promotions(self._drawing, plist)

        for p in plist:
            p.promote()

        if plist:
            self._promotion_history.append(plist)

    def _find_deepest_promotions(self, drawing, promotions, level = 0, sel = None):
        '''
        Find the deepest level in the hierarchy rooted at drawing
        where a selection promotion can be done.  Accumulate all 
        selection promotions at that level in promotions lists.
        '''

        if sel is None:
            # Drawings that are partially selected
            # or has some descendant partially selected.
            sel = set()

        # Check for intra-model promotion
        if hasattr(drawing, 'selection_promotion'):
            p = drawing.selection_promotion()
            if p:
                self._add_promotion(p, promotions)
            if p or drawing.any_part_selected():
                sel.add(drawing)

        # Check for deeper promotions
        children = drawing.child_models()
        for c in children:
            self._find_deepest_promotions(c, promotions, level+1, sel)
            if c in sel:
                sel.add(drawing)

        # Add drawing to selected set if it is selected.
        if drawing not in sel:
            sp = drawing.selected_positions
            if sp is not None and sp.sum() > 0:
                sel.add(drawing)

        # If no deeper promotions can this drawing be selected.
        if len(promotions) == 0 or promotions[0].level <= level:
            # Check if some but not all children are selected.
            nsel = [c for c in children if c not in sel]
            if nsel and len(nsel) < len(children):
                for c in nsel:
                    self._add_promotion(ModelSelectionPromotion(c,level+0.5), promotions)

            # Check if some but not all instances are selected.
            sp = drawing.selected_positions
            if sp is not None:
                ns = sp.sum()
                if ns < len(sp) and ns > 0:
                    self._add_promotion(ModelSelectionPromotion(drawing,level), promotions)
                    sel.add(drawing)

    def _add_promotion(self, p, promotions):
        '''
        Added a SelectionPromotion to the list of deepest promotions.
        '''
        if len(promotions) == 0 or p.level == promotions[0].level:
            promotions.append(p)
        elif p.level > promotions[0].level:
            promotions.clear()
            promotions.append(p)

    def demote_selection(self):
        '''If the selection has previously been promoted, this returns
        it to the previous smaller selection.'''
        ph = self._promotion_history
        if len(ph) == 0:
            return False
        for p in ph.pop():
            p.demote()
        return True

    def clear_selection_promotion_history(self):
        '''
        Forget the selection history promotion history.
        This is used when the selection is changed manually.
        '''
        self._promotion_history.clear()


class SelectionPromotion:
    def __init__(self, level):
        self.level = level
    def promote(self):
        pass
    def demote(self):
        pass

class ModelSelectionPromotion:
    def __init__(self, model, level):
        SelectionPromotion.__init__(self, level)
        self.model = model
        spos = model.selected_positions
        self._prev_selected = None if spos is None else spos.copy() 
    def promote(self):
        self.model.selected = True
    def demote(self):
        self.model.selected_positions = self._prev_selected
