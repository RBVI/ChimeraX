# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

SELECTION_CHANGED = 'selection changed'

class Selection:

    def __init__(self, sess):
        self._all_models = sess.models	# Models object
        self._promotion = SelectionPromoter(self._all_models.scene_root_model)
        sess.triggers.add_trigger(SELECTION_CHANGED)
        # XXX: SELECTION_CHANGED trigger is currently fired in by
        # atomic.structure.StructureGraphicsChangeManager
        # Maybe it should be moved up to Model level somehow?
        self._trigger_fire_needed = False

    def models(self, all_selected=False):
        if all_selected:
            return [m for m in self._all_models if m.get_selected(include_children=True, fully=True)]
        return [m for m in self._all_models if m.get_selected(include_children=True)]

    def items(self, itype):
        si = []
        for m in self.models():
            s = m.selected_items(itype)
            si.extend(s)
        return si

    def empty(self):
        for m in self._all_models:
            if m.selected:
                return False
        return True

    def clear(self):
        for m in self.models():
            m.clear_selection()

    def clear_promotion_history(self):
        self._promotion.clear_selection_promotion_history()

    def promote(self, session):
        from .undo import UndoState
        undo_state = UndoState("select up")
        self.undo_add_selected(undo_state, False)
        self._promotion.promote_selection()
        self.undo_add_selected(undo_state, True, old_state=False)
        session.undo.register(undo_state)

    def demote(self, session):
        from .undo import UndoState
        undo_state = UndoState("select down")
        self.undo_add_selected(undo_state, False)
        self._promotion.demote_selection()
        self.undo_add_selected(undo_state, True, old_state=False)
        session.undo.register(undo_state)

    def undo_add_selected(self, undo_state, new_state, old_state=None):
        from chimerax.atomic import Atoms, Bonds, Pseudobonds
        for oname, otype in (('atoms', Atoms), ('bonds', Bonds), ('pseudobonds', Pseudobonds)):
            items = self.items(oname)
            if items:
                if isinstance(items, otype):
                    orig = self._orig_state(items, old_state)
                    undo_state.add(items, "selected", orig, new_state)
                else:
                    for i in items:
                        orig = self._orig_state(i, old_state)
                        undo_state.add(i, "selected", orig, new_state)
        models = [m for m in self._all_models if m.selected]
        if models:
            for m in models:
                orig = old_state if old_state is not None else m.selected
                undo_state.add(m, "set_model_selected", orig, new_state, "M")

    @property
    def trigger_fire_needed(self):
        return self._trigger_fire_needed

    @trigger_fire_needed.setter
    def trigger_fire_needed(self, needed):
        self._trigger_fire_needed = needed

    def _orig_state(self, owner, old_state):
        if old_state is None:
            return owner.selected
        else:
            import numpy
            if old_state:
                return numpy.ones(len(owner), dtype=numpy.bool_)
            else:
                return numpy.zeros(len(owner), dtype=numpy.bool_)


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

        # TODO: Make this use selection instead of drawing highlighting to determine promotions.

        if sel is None:
            # Drawings that are partially selected
            # or has some descendant partially selected.
            sel = set()

        # Check for intra-model promotion
        if hasattr(drawing, 'selection_promotion'):
            p = drawing.selection_promotion()
            if p:
                self._add_promotion(p, promotions)
            if p or drawing.any_part_highlighted():
                sel.add(drawing)

        # Check for deeper promotions
        children = drawing.child_models()
        for c in children:
            self._find_deepest_promotions(c, promotions, level+1, sel)
            if c in sel:
                sel.add(drawing)

        # Add drawing to selected set if it is selected.
        if drawing not in sel:
            sp = drawing.highlighted_positions
            if sp is not None and sp.sum() > 0:
                sel.add(drawing)

        # If no deeper promotions can this drawing be selected.
        if len(promotions) == 0 or promotions[0].level <= level:
            # Check if some but not all children are selected.
            nsel = [c for c in children if c not in sel]
            if nsel:
                if len(nsel) < len(children):
                    # Some children are selected so select all children
                    for c in nsel:
                        self._add_promotion(ModelSelectionPromotion(c,level+0.5), promotions)
            elif children and not drawing.highlighted and drawing is not self._drawing:
                # All children selected so select parent
                self._add_promotion(ModelSelectionPromotion(drawing,level), promotions)

            # Check if some but not all instances are selected.
            sp = drawing.highlighted_positions
            if sp is not None:
                ns = sp.sum()
                if ns < len(sp) and ns > 0:
                    # Some instances are selected so select all instances
                    self._add_promotion(ModelSelectionPromotion(drawing,level + 0.3), promotions)
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

class ModelSelectionPromotion(SelectionPromotion):
    def __init__(self, model, level):
        SelectionPromotion.__init__(self, level)
        self.model = model
        spos = model.selected_positions
        self._prev_selected = None if spos is None else spos.copy()
    def promote(self):
        self.model.selected = True
    def demote(self):
        m = self.model
        m.selected = False	# This may clear child drawing selections.
        m.selected_positions = self._prev_selected
