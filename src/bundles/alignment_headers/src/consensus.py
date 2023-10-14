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

"""Show consensus sequence"""

from .header_sequence import DynamicHeaderSequence

class Consensus(DynamicHeaderSequence):
    name = "Consensus"
    ident = "consensus"
    sort_val = 1.3
    value_type = None # since Clustal style is characters

    def __init__(self, alignment, *args, **kw):
        self.conserved = [False] * len(alignment.seqs[0])
        super().__init__(alignment, *args, **kw)
        self.handler_ID = self.settings.triggers.add_handler('setting changed',
            lambda *args: self.reevaluate())

    def alignment_notification(self, note_name, note_data):
        super().alignment_notification(note_name, note_data)
        if note_name == self.alignment.NOTE_SEQ_CONTENTS:
            self.reevaluate()

    @property
    def capitalize_threshold(self):
        return self.settings.capitalize_threshold

    @capitalize_threshold.setter
    def capitalize_threshold(self, capitalize_threshold):
        self.settings.capitalize_threshold = capitalize_threshold

    def destroy(self):
        self.handler_ID.remove()
        super().destroy()

    def evaluate(self, pos):
        occur = {}
        for i in range(len(self.alignment.seqs)):
            let = self.alignment.seqs[i][pos]
            if self.settings.ignore_gaps and not let.isalpha():
                continue
            try:
                occur[let] += 1
            except KeyError:
                occur[let] = 1
        best = (0, None)
        for let, num in occur.items():
            if num > best[0]:
                best = (num, let)
            elif num == best[0] and not let.isalpha():
                # "gappy" characters win ties
                best = (num, let)
        num, let = best
        self.conserved[pos] = False
        if let is None:
            return ' '
        if num / len(self.alignment.seqs) >= self.settings.capitalize_threshold:
            retlet = let.upper()
            if num == len(self.alignment.seqs):
                self.conserved[pos] = True
        else:
            retlet = let.lower()
        return retlet

    def get_state(self):
        state = {
            'base state': super().get_state(),
            'capitalize_threshold': self.settings.capitalize_threshold,
            'ignore_gaps': self.settings.ignore_gaps,
        }
        return state

    @property
    def ignore_gaps(self):
        return self.settings.ignore_gaps

    @ignore_gaps.setter
    def ignore_gaps(self, ignore_gaps):
        self.settings.ignore_gaps = ignore_gaps

    def num_options(self):
        return 2

    def option_data(self):
        from chimerax.ui.options import FloatOption, BooleanOption
        return super().option_data() + [
            ("capitalization threshold", 'capitalize_threshold', FloatOption, {},
                "Capitalize consensus letter if at least this fraction of sequences are identical"),
            ("ignore gap characters", 'ignore_gaps', BooleanOption, {},
                "Whether gap characters are considered for the consensus character")
        ]

    def position_color(self, pos):
        if self[pos].isupper():
            if self.conserved[pos]:
                return 'red'
            return 'purple'
        return 'black'

    def reevaluate(self, pos1=0, pos2=None, *, evaluation_func=None):
        """sequences changed, possibly including length"""
        if len(self.conserved) != len(self.alignment.seqs[0]):
            self.conserved = [False] * len(self.alignment.seqs[0])
        else:
            r1, r2 = pos1, (len(self.conserved) if pos2 is None else pos2+1)
            self.conserved[r1:r2] = [False] * (r2-r1)
        super().reevaluate(pos1, pos2, evaluation_func=evaluation_func)

    def set_state(self, state):
        super().set_state(state['base state'])
        self.settings.capitalize_threshold = state['capitalize_threshold']
        self.settings.ignore_gaps = state['ignore_gaps']

    def settings_info(self):
        name, defaults = super().settings_info()
        from chimerax.core.commands import Bounded, FloatArg, BoolArg
        defaults.update({
            'capitalize_threshold': (Bounded(FloatArg, min=0, max=1), 0.8),
            'ignore_gaps': (BoolArg, False),
        })
        return "consensus sequence header", defaults
