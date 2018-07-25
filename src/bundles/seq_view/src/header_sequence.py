# vim: set expandtab ts=4 sw=4:

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

"""header sequence classes/functions"""

# Since the ChimeraX Sequence class only supports sequences of characters,
# implement our own class that can also contain numbers or other values.

class HeaderSequence(list):
    # sort_val determines the default ordering of headers.
    # Built-in headers change their sort_val to a value in the range
    # [1.0, 2.0) so they normally appear before registered headers.
    # Identical sort_vals tie-break on sequence name.
    sort_val = 2.0

    def __init__(self, sv, name=None, eval_while_hidden=False):
        self.name = name
        from weakref import proxy
        self.sv = proxy(sv)
        self.visible = False
        self.eval_while_hidden = eval_while_hidden

    def align_change(self, left, right):
        """alignment changed in positions from 'left' to 'right'"""
        for pos in range(left, right+1):
            self[pos] = self.evaluate(pos)

    def destroy(self):
        pass

    def evaluate(self, pos):
        raise NotImplementedError("evaluate() method must be"
            " implemented by %s subclass" % self.__class__.__name__)tic20@cam.ac.uk

    def fast_update(self):
        # if asked to update a few columns (align_change() method)
        # can it be done quickly?
        return True

    def get_state(self):
        state = {
            'name': self.name,
            'visible': self.visible,
            'eval_while_hidden': self.eval_while_hidden
        }
        return state

    def hide(self):
        """Called when sequence hidden"""
        self.visible = False

    def hist_infinity(self, position):
        """Convenience function to map arbitrary number to 0-1 range

           Used in the 'depiction_val' method for some kinds of data
        """
        raw = self[position]
        if raw is None:
            return 0.0
        from math import exp
        if raw >= 0:
            return 1.0 - 0.5 * exp(-raw)
        return 0.5 * exp(raw)

    def positive_hist_infinity(self, position):
        """Convenience function to map arbitrary positive number to 0-1 range

           Used in the 'depiction_val' method for some kinds of data
        """
        raw = self[position]
        if raw is None:
            return 0.0
        from math import exp
        return 1.0 - exp(-raw)

    def reevaluate(self):
        """sequences changed, possibly including length"""
        self[:] = []
        for pos in range(len(self.sv.alignment.seqs[0])):
            self.append(self.evaluate(pos))

    @staticmethod
    def session_restore(session, sv, state):
        inst = HeaderSequence(sv)
        inst.set_state(state)
        return inst

    def set_state(self, state):
        self.name = state['name']
        self.visible = state['visible']
        self.eval_while_hidden = state['eval_while_hidden']

    def show(self):
        """Called when sequence shown"""
        self.visible = True

class FixedHeaderSequence(HeaderSequence):

    # header relevant if alignment is a single sequence?
    single_sequence_relevant = True

    def __init__(self, sv, name=None, vals=[]):
        self.vals = vals
        HeaderSequence.__init__(self, sv, name=name)
        if vals:
            self.reevaluate()

    def align_change(self, left, right):
        pass

    def get_state(self):
        state = {
            'base state': HeaderSequence.get_state(self),
            'vals': self.vals
        }
        return state

    def reevaluate(self):
        if len(self.sv.alignment.seqs[0]) == len(self.vals):
            self[:] = self.vals
            if hasattr(self, "save_color_func"):
                self.color_func = self.save_color_func
                delattr(self, "save_color_func")
        else:
            self[:] = '?' * len(self.sv.alignment.seqs[0])
            if hasattr(self, "color_func") \
            and not hasattr(self, "save_color_func"):
                self.save_color_func = self.color_func
                self.color_func = lambda s, o: 'black'

    def set_state(self, state):
        HeaderSequence.set_state(state['base state'])
        self.vals = state['vals']
        self.reevaluate()

class DynamicHeaderSequence(HeaderSequence):

    # header relevant if alignment is a single sequence?
    single_sequence_relevant = False

    def __init__(self, *args, **kw):
        HeaderSequence.__init__(self, *args, **kw)
        self.__handlerID = None
        self._need_update = True
        if self.eval_while_hidden:
            self.reevaluate()

    def destroy(self):
        if self.__handlerID != None:
            from MAViewer import ADDDEL_SEQS
            self.sv.triggers.deleteHandler(ADDDEL_SEQS,
                            self.__handlerID)
            self.__handlerID = None
        HeaderSequence.destroy(self)

    def refresh(self, *args, **kw):
        if self.visible or self.eval_while_hidden:
            self.reevaluate()
            if not kw.get('from_show', False):
                self.sv.refreshHeader(self)
            self._need_update = False
        else:
            self._need_update = True

    def show(self):
        HeaderSequence.show(self)
        if self.eval_while_hidden:
            return
        if self._need_update:
            self.refresh(from_show=True)

    def hide(self):
        HeaderSequence.hide(self)

class DynamicStructureHeaderSequence(DynamicHeaderSequence):
    single_sequence_relevant = True
    # class is refreshed on association changes by sequence viewer

registered_headers = []
def register_header_sequence(header_class, default_on=True):
    registered_headers.append(header_class, default_on))
