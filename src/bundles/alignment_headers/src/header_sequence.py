# vim: set expandtab ts=4 sw=4:

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

"""header sequence classes/functions"""

# Since the ChimeraX Sequence class only supports sequences of characters,
# implement our own class that can also contain numbers or other values.

from contextlib import contextmanager

class HeaderSequence(list):
    # sort_val determines the default ordering of headers.
    # Built-in headers change their sort_val to a value in the range
    # [1.0, 2.0) so they normally appear before registered headers.
    # Identical sort_vals tie-break on sequence name.
    sort_val = 2.0
    numbering_start = None
    fast_update = True # can header be updated quickly if only a few columns are changed?
    single_column_updateable = True # can a single column be updated, or only the entire header?
    ident = None    # should be string, used to identify header in commands and as part of the
                    # generated residue attribute name, so the string should be "attribute friendly"
    value_type = float
    value_none_okay = True

    ATTR_PREFIX = "seq_"

    def __init__(self, alignment, name=None, *, eval_while_hidden=False, session_restore=False):
        if name is None:
            if not hasattr(self, 'name'):
                self.name = ""
        else:
            self.name = name
        if not session_restore and self.ident is None:
            raise AssertionError("%s header class failed to define 'ident' attribute"
                % self.__class__.__name__)
        from weakref import proxy
        self.alignment = proxy(alignment)
        self.alignment.add_observer(self)
        self._notifications_suppressed = 0
        self._shown = False
        self.eval_while_hidden = eval_while_hidden
        self._update_needed = True
        self._edit_bounds = None
        self._alignment_being_edited = False
        self._command_runner = None
        if not hasattr(self.__class__, 'settings'):
            self.__class__.settings = self.make_settings(alignment.session)
        if self.eval_while_hidden:
            self.reevaluate()

    def add_options(self, options_container, *, category=None, verbose_labels=True):
        self._add_options(options_container, category, verbose_labels, self.option_data())


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
        if note_name == self.alignment.NOTE_EDIT_START:
            self._alignment_being_edited = True
        elif note_name == self.alignment.NOTE_EDIT_END:
            self._alignment_being_edited = False
        elif note_name == self.alignment.NOTE_REALIGNMENT:
            with self.alignment_notifications_suppressed():
                self.reevaluate()

    @contextmanager
    def alignment_notifications_suppressed(self):
        self._notifications_suppressed += 1
        try:
            yield
        finally:
            self._notifications_suppressed -= 1

    def destroy(self):
        if not self.alignment.being_destroyed:
            self.alignment.remove_observer(self)

    def evaluate(self, pos):
        raise NotImplementedError("evaluate() method must be"
            " implemented by %s subclass" % self.__class__.__name__)

    def position_color(self, position):
        return 'black'

    def get_state(self):
        state = {
            'name': self.name,
            'shown': self._shown,
            'eval_while_hidden': self.eval_while_hidden,
            'contents': self[:]
        }
        return state

    def __hash__(self):
        return id(self)

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

    def make_settings(self, session):
        """For derived classes with their own settings, the settings_info()
           method must be overridden (which see)"""
        settings_name, settings_info = self.settings_info()
        settings_defaults = {}
        self.__class__._setting_cmd_annotations = cmd_annotations = {}
        for attr_name, info in settings_info.items():
            annotation, default_value = info
            settings_defaults[attr_name] = default_value
            cmd_annotations[attr_name] = annotation
        from chimerax.core.settings import Settings
        class HeaderSettings(Settings):
            EXPLICIT_SAVE = settings_defaults
        return HeaderSettings(session, settings_name)

    def notify_alignment(self, note_name, *args):
        if not self._notifications_suppressed:
            if args:
                note_data = (self, *args)
            else:
                note_data = self
            self.alignment.notify(note_name, note_data)

    def num_options(self):
        return 0

    def option_data(self):
        from chimerax.ui.options import BooleanOption
        return [
            ("show initially", 'initially_shown', BooleanOption, {},
                "Show this header when sequence/alignment initially shown")
        ]

    def option_sorting(self, option):
        for base_label, attr_name, opt_class, opt_kw, balloon in HeaderSequence.option_data(self):
            if option.attr_name == attr_name:
                return (0, option.name.casefold())
        return (1, option.name.casefold())

    def positive_hist_infinity(self, position):
        """Convenience function to map arbitrary positive number to 0-1 range

           Used as the 'depiction_val' method for some kinds of data
        """
        raw = self[position]
        if raw is None:
            return 0.0
        from math import exp
        return 1.0 - exp(-raw)

    def process_command(self, command_text):
        if self._command_runner is None:
            from chimerax.core.commands import Command
            from chimerax.core.commands.cli import RegisteredCommandInfo
            command_registry = RegisteredCommandInfo()
            self._register_commands(command_registry)
            self._command_runner = Command(self.alignment.session, registry=command_registry)
        self._command_runner.run(command_text, log=False)

    def reason_requires_update(self, reason):
        return False

    def reevaluate(self, pos1=0, pos2=None, *, evaluation_func=None):
        """sequences changed, possibly including length"""
        if not self._shown and not self.eval_while_hidden:
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
        if self._shown and not self._notifications_suppressed:
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
            self.notify_alignment(self.alignment.NOTE_HDR_VALUES, bounds)

    @property
    def relevant(self):
        return True

    @property
    def residue_attr_name(self):
        return self.ATTR_PREFIX + self.ident

    @classmethod
    def session_restore(cls, session, alignment, state):
        inst = cls(alignment, session_restore=True)
        inst.set_state(state)
        return inst

    def set_state(self, state):
        self.name = state['name']
        self._shown = state.get('shown', state.get('visible', None))
        self.eval_while_hidden = state['eval_while_hidden']
        self[:] = state['contents']

    def settings_info(self):
        """This method needs to return a (name, dict) tuple where 'name' is used to distingush
           this group of settings from settings of other headers or tools (e.g. "consensus sequence header"),
           and 'dict' is a dictionary of (attr_name: (Annotation subclass, default_value)) key/value pairs.
           Annotation subclass will be used by the "seq header header_name setting" command to parse the
           text for the value into the proper value type.

           The dictionary must include the base class settings, so super().settings_info() must be
           called and the returned dictionary updated with the derived class's settings"""
        # the code relies on the fact that the returned settings dict is a different object every
        # time (it gets update()d), so don't make it a class variable!
        from chimerax.core.commands import BoolArg
        return "base header sequence", { 'initially_shown': (BoolArg, False) }

    @property
    def shown(self):
        return self._shown

    @shown.setter
    def shown(self, show):
        if show == self._shown:
            return
        self._shown = show
        # suppress the alignment notification
        with self.alignment_notifications_suppressed():
            if show:
                if self._edit_bounds:
                    self.reevaluate(*self._edit_bounds, suppress_callback=True)
                elif self._update_needed:
                    self.reevaluate()
        self.notify_alignment(self.alignment.NOTE_HDR_SHOWN)

    def _add_options(self, options_container, category, verbose_labels, option_data):
        for base_label, attr_name, opt_class, opt_kw, balloon in option_data:
            option = opt_class(self._final_option_label(base_label, verbose_labels), None,
                self._setting_option_cb, balloon=balloon, attr_name=attr_name, settings=self.settings,
                auto_set_attr=False, **opt_kw)
            if category is not None:
                options_container.add_option(category, option)
            else:
                options_container.add_option(option)

    def _final_option_label(self, base_label, verbose_labels):
        if verbose_labels:
            return "%s: %s" % (getattr(self, "settings_name", self.name), base_label)
        return base_label[0].upper() + base_label[1:]

    def _process_setting_command(self, session, setting_arg_text):
        from chimerax.core.commands import EnumOf
        enum = EnumOf(list(self._setting_cmd_annotations.keys()))
        attr_name, arg_text, remainder = enum.parse(setting_arg_text, session)
        remainder = remainder.strip()
        if not remainder:
            from chimerax.core.errors import UserError
            raise UserError("No value provided for setting")
        val, val_text, remainder = self._setting_cmd_annotations[attr_name].parse(remainder, session)
        if remainder and not remainder.isspace():
            from chimerax.core.errors import UserError
            raise UserError("Extraneous text after command")
        setattr(self.settings, attr_name, val)

    def _register_commands(self, registry):
        from chimerax.core.commands import register, CmdDesc, RestOfLine, SaveFileNameArg
        register("show", CmdDesc(synopsis='Show %s header' % self.ident),
            lambda session, hdr=self: setattr(hdr, "shown", True), registry=registry)
        register("hide", CmdDesc(synopsis='Hide %s header' % self.ident),
            lambda session, hdr=self: setattr(hdr, "shown", False), registry=registry)
        register("setting", CmdDesc(required=[('setting_arg_text', RestOfLine)],
            synopsis="change header setting"), self._process_setting_command, registry=registry)
        register("save", CmdDesc(required=[('file_name', SaveFileNameArg)],
            synopsis="save header values to file"), self._save, registry=registry)

    def _setting_option_cb(self, opt):
        from chimerax.core.commands import run, StringArg
        session = self.alignment.session
        align_arg = "%s " % StringArg.unparse(str(self.alignment)) \
            if len(session.alignments.alignments) > 1 else ""
        run(session, "seq header %s%s setting %s %s"
            % (align_arg, self.ident, opt.attr_name, StringArg.unparse(str(opt.value))))

    def _save(self, session, file_name):
        from chimerax.io import open_output
        with open_output(file_name, encoding="utf-8") as f:
            if hasattr(self, 'save_file_preamble'):
                print(self.save_file_preamble, file=f)
            print("%s header for %s" % (self.name, self.alignment), file=f)
            for i, val in enumerate(self):
                print("%d:" % (i+1), val, file=f)


class FixedHeaderSequence(HeaderSequence):

    def __init__(self, alignment, name=None, vals=[]):
        self.vals = vals
        HeaderSequence.__init__(self, alignment, name=name)

    def align_change(self, left, right):
        pass

    def evaluate(self, pos):
        return self.vals[pos]

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
    # if subclass is in fact relevant to single-sequence alignments, override this
    single_sequence_relevant = False

    def alignment_notification(self, note_name, note_data):
        super().alignment_notification(note_name, note_data)
        if not self.single_sequence_relevant:
            if note_name == self.alignment.NOTE_ADD_SEQS and len(self.alignment.seqs) - len(note_data) == 1:
                self.notify_alignment(self.alignment.NOTE_HDR_RELEVANCE)
            elif note_name == self.alignment.NOTE_DEL_SEQS and len(self.alignment.seqs) == 1:
                self.notify_alignment(self.alignment.NOTE_HDR_RELEVANCE)

    @property
    def relevant(self):
        return self.single_sequence_relevant or len(self.alignment.seqs) > 1

class DynamicStructureHeaderSequence(HeaderSequence):

    min_chain_relevance = min_structure_relevance = None

    def alignment_notification(self, note_name, note_data):
        super().alignment_notification(note_name, note_data)
        if note_name == self.alignment.NOTE_MOD_ASSOC:
            self.reevaluate()
        elif note_name == self.alignment.NOTE_ADD_ASSOC:
            if self.min_chain_relevance is not None:
                cur_assocs = len(self.alignment.associations)
                prev_assocs = cur_assocs - len(note_data)
                if prev_assocs < self.min_chain_relevance \
                and cur_assocs >= self.min_chain_relevance:
                    self.notify_alignment(self.alignment.NOTE_HDR_RELEVANCE)
            elif self.min_structure_relevance is not None:
                added_sseqs = set([x.struct_seq for x in note_data])
                prev_structs = len(set([x.structure
                    for x in self.alignment.associations if x not in added_sseqs]))
                cur_structs = len(set([x.structure for x in self.alignment.associations]))
                if prev_structs < self.min_structure_relevance \
                and cur_structs >= self.min_structure_relevance:
                    self.notify_alignment(self.alignment.NOTE_HDR_RELEVANCE)
            else:
                if len(note_data) == len(self.alignment.associations):
                    self.notify_alignment(self.alignment.NOTE_HDR_RELEVANCE)
        elif note_name == self.alignment.NOTE_DEL_ASSOC:
            if self.min_chain_relevance is not None:
                if note_data['num remaining associations'] == self.min_chain_relevance - 1:
                    self.notify_alignment(self.alignment.NOTE_HDR_RELEVANCE)
            elif self.min_structure_relevance is not None:
                # can't be completely sure relevance changed in all cases, but be overly cautious
                if note_data['max previous structures'] >= self.min_structure_relevance \
                and note_data['num remaining structures'] < self.min_structure_relevance:
                    self.notify_alignment(self.alignment.NOTE_HDR_RELEVANCE)
            else:
                if note_data['num remaining associations'] == 0:
                    self.notify_alignment(self.alignment.NOTE_HDR_RELEVANCE)

    @property
    def relevant(self):
        if self.min_chain_relevance is not None:
            return len(self.alignment.association) >= self.min_chain_relevance
        if self.min_structure_relevance is not None:
            return len(set([x.structure
                for x in self.alignment.associations])) >= self.min_structure_relevance
        return len(self.alignment.associations) > 0

registered_headers = []
def register_header(header_class, default_on=True):
    registered_headers.append((header_class, default_on))
