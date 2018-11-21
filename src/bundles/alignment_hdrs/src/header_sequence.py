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
    numbering_start = None
    fast_update = True # can header be updated quickly if only a few columns are changed?
    single_column_updateable = True # can a single column be updated, or only the entire header?

    def __init__(self, alignment, name=None, eval_while_hidden=False):
        if name is None:
            if not hasattr(self, 'name'):
                self.name = ""
        else:
            self.name = name
        from weakref import proxy
        self.alignment = proxy(alignment)
        self.visible = False
        self.eval_while_hidden = eval_while_hidden
        self._update_needed = True
        self._edit_bounds = None
        if self.eval_while_hidden:
            self.reevaluate()

    def align_change(self, left, right, *, edit=False):
        """alignment changed in positions from 'left' to 'right'"""
        if edit and not self.fast_update:
            if self._edit_bounds is None:
                self._edit_bounds = (left, right)
            else:
                self._edit_bounds = (min(left, self._edit_bounds[0]), max(right, self._edit_bounds[1]))
            return
        if single_column_updateable:
            for pos in range(left, right+1):
                self[pos] = self.evaluate(pos)
            self._edit_bounds = None
        else:
            self.reevaluate()

    def evaluate(self, pos):
        raise NotImplementedError("evaluate() method must be"
            " implemented by %s subclass" % self.__class__.__name__)

    def position_color(self, position):
        return 'black'

    def get_state(self):
        state = {
            'name': self.name,
            'visible': self.visible,
            'eval_while_hidden': self.eval_while_hidden
        }
        return state

    def __hash__(self):
        return id(self)

    def hide(self):
        """Called when sequence hidden"""
        self.visible = False

    def hist_infinity(self, position):
        """Convenience function to map arbitrary number to 0-1 range

           Used as the 'depiction_val' method for some kinds of data
        """
        raw = self[position]
        if raw is None:
            return 0.0
        from math import exp
        if raw >= 0:
            return 1.0 - 0.5 * exp(-raw)
        return 0.5 * exp(raw)

    def __lt__(self, other):
        return self.sort_val < other.sort_val

    def positive_hist_infinity(self, position):
        """Convenience function to map arbitrary positive number to 0-1 range

           Used as the 'depiction_val' method for some kinds of data
        """
        raw = self[position]
        if raw is None:
            return 0.0
        from math import exp
        return 1.0 - exp(-raw)

    def reason_requires_update(self, reason):
        return False

    def reevaluate(self):
        """sequences changed, possibly including length"""
        self[:] = []
        for pos in range(len(self.alignment.seqs[0])):
            self.append(self.evaluate(pos))
        self._update_needed = False
        self._edit_bounds = None

    def refresh(self, reason=None):
        """Needs to be called from viewer when notified of alignment changes.

        Returns True, False, or a two-tuple indicating whether the header changed.
        If a two-tuple, it's the starting and ending changed positions (zero-based indexing)"""
        if not self._update_needed:
            if reason and self.reason_requires_update(reason):
                self._update_needed = True
        if not self.visible and not self.eval_while_hidden:
            return False
        if not self._update_needed:
            if self._edit_bounds and reason == "editing finished":
                bounds = self._edit_bounds
                self.align_change(*bounds)
                return bounds
            return False
        prev_vals = self[:]
        self.reevaluate()
        cur_vals = self[:]
        if len(prev_vals) != len(cur_vals):
            return True
        if prev_vals == cur_vals:
            return False
        if prev_vals[0] != cur_vals[0] and prev_vals[-1] != cur_vals[-1]:
            return True
        first_mismatch = last_mismatch = None
        for i, val in enumerate(prev_vals):
            if val != cur_vals[i]:
                last_mismatch = i
                if first_mismatch is None:
                    first_mismatch = i
        return (first_mismatch, last_mismatch)

    @staticmethod
    def session_restore(session, alignment, state):
        inst = HeaderSequence(alignment)
        inst.set_state(state)
        return inst

    def set_state(self, state):
        self.name = state['name']
        self.visible = state['visible']
        self.eval_while_hidden = state['eval_while_hidden']

    def show(self):
        """Called when sequence shown"""
        self.visible = True
        if self._update_needed or self._edit_bounds:
            self.refresh()

class FixedHeaderSequence(HeaderSequence):
    # header relevant if alignment is a single sequence?
    single_sequence_relevant = True

    def __init__(self, alignment, name=None, vals=[]):
        self.vals = vals
        HeaderSequence.__init__(self, alignment, name=name)

    def align_change(self, left, right):
        pass

    def get_state(self):
        state = {
            'base state': HeaderSequence.get_state(self),
            'vals': self.vals
        }
        return state

    def reevaluate(self):
        if len(self.alignment.seqs[0]) == len(self.vals):
            self[:] = self.vals
            if hasattr(self, "save_color_func"):
                self.position_color = self.save_color_func
                delattr(self, "save_color_func")
        else:
            self[:] = '?' * len(self.alignment.seqs[0])
            if self.position_color.__func__ != HeaderSequence.position_color:
                self.save_color_func = self.position_color
                self.position_color = lambda pos, *, s=self.position_color.__self__, \
                    f=HeaderSequence.position_color: f(s, pos)
        self._update_needed = False

    def set_state(self, state):
        HeaderSequence.set_state(state['base state'])
        self.vals = state['vals']
        self.reevaluate()

class DynamicHeaderSequence(HeaderSequence):
    # header relevant if alignment is a single sequence?
    single_sequence_relevant = False

    def reason_requires_update(self, reason):
        return not reason.endswith("association")

class DynamicStructureHeaderSequence(DynamicHeaderSequence):
    single_sequence_relevant = True
    # class is refreshed on association changes by sequence viewer

    def reason_requires_update(self, reason):
        return True

registered_headers = []
def register_header(header_class, default_on=True):
    registered_headers.append(header_class, default_on)
