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

    def __init__(self, alignment, refresh_callback, name=None, eval_while_hidden=False):
        if name is None:
            if not hasattr(self, 'name'):
                self.name = ""
        else:
            self.name = name
        from weakref import proxy
        self.alignment = proxy(alignment)
        self.alignment.attach_viewer(self)
        self.refresh_callback = refresh_callback
        self.visible = False
        self.eval_while_hidden = eval_while_hidden
        self._update_needed = True
        self._edit_bounds = None
        self._alignment_being_edited = False
        if self.eval_while_hidden:
            self.reevaluate()

    def add_options(self, options_container, *, category=None, verbose_labels=True):
        pass

    def align_change(self, left, right):
        """alignment changed in positions from 'left' to 'right'"""
        if self._alignment_being_edited and not self.fast_update:
            if self._edit_bounds is None:
                self._edit_bounds = (left, right)
            else:
                self._edit_bounds = (min(left, self._edit_bounds[0]), max(right, self._edit_bounds[1]))
            return
        if single_column_updateable:
            self.reevaluate(left, right)
        else:
            self.reevaluate()

    def alignment_notification(self, note_name, note_data):
        if note_name == "editing started":
            self._alignment_being_edited = True
        if note_name == "editing finished":
            self._alignment_being_edited = False

    def destroy(self):
        if not self.alignment.being_destroyed:
            self.alignment.detach_viewer(self)

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

    def num_options(self):
        return 0

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

    def reevaluate(self, pos1=0, pos2=None, *, evaluation_func=None):
        """sequences changed, possibly including length"""
        if not self.visible and not self.eval_while_hidden:
            self._update_needed = True
            return
        prev_vals = self[:]
        if pos2 is None:
            pos2 = len(self.alignment.seqs[0]) - 1
        if evaluation_func is None:
            self[:] = []
            for pos in range(pos1, pos2+1):
                self.append(self.evaluate(pos))
        else:
            evaluation_func(pos1, pos2)
        self._update_needed = False
        self._edit_bounds = None
        if self.visible and self.refresh_callback:
            cur_vals = self[:]
            if len(prev_vals) != len(cur_vals):
                bounds = None
            elif prev_vals == cur_vals:
                return
            elif prev_vals[0] != cur_vals[0] and prev_vals[-1] != cur_vals[-1]:
                bounds = None
            else:
                first_mismatch = last_mismatch = None
                for i, val in enumerate(prev_vals):
                    if val != cur_vals[i]:
                        last_mismatch = i
                        if first_mismatch is None:
                            first_mismatch = i
                bounds = (first_mismatch, last_mismatch)
            self.refresh_callback(self, bounds)

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
        if self._edit_bounds:
            self.reevaluate(*self._edit_bounds)
        elif self._update_needed:
            self.reevaluate()

    def _add_options(self, options_container, category, verbose_labels, option_data):
        for base_label, attr_name, opt_class, balloon in option_data:
            option = opt_class(self._final_option_label(base_label, verbose_labels), None, None,
                balloon=balloon, attr_name=attr_name, settings=self.settings)
            if category is not None:
                options_container.add_option(category, option)
            else:
                options_container.add_option(option)

    def _final_option_label(self, base_label, verbose_labels):
        if verbose_labels:
            return "%s: %s" % (self.name, base_label)
        return base_label[0].upper() + base_label[1:]

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

    def _reevaluate(self, bounds):
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

    def reevaluate(self, pos1=0, pos2=None, *, evaluation_func=None):
        if evaluation_func is None:
            super().reevaluate(pos1, pos2, evaluation_func=evaluation_func)
        else:
            super().reevaluate(pos1, pos2, evaluation_func=self._reevaluate)

    def set_state(self, state):
        HeaderSequence.set_state(state['base state'])
        self.vals = state['vals']
        self.reevaluate()

class DynamicHeaderSequence(HeaderSequence):
    # header relevant if alignment is a single sequence?
    single_sequence_relevant = False

class DynamicStructureHeaderSequence(DynamicHeaderSequence):
    single_sequence_relevant = True
    # class is refreshed on association changes by sequence viewer

    def alignment_notification(self, note_name, note_data):
        super().alignment_notification(note_name, note_data)
        if note_name == "association modified":
            self.reevaluate()

registered_headers = []
def register_header(header_class, default_on=True):
    registered_headers.append((header_class, default_on))
