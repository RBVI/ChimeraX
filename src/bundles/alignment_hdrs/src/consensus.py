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

"""Show consensus sequence"""

from .header_sequence import DynamicHeaderSequence

class Consensus(DynamicHeaderSequence):
    name = "Consensus"
    sort_val = 1.3
    def __init__(self, alignment, refresh_callback):
        self.settings = get_settings(alignment.session)
        self.handler_ID = self.settings.triggers.add_handler('setting changed', lambda *args: self.reevaluate())
        self.conserved = [False] * len(alignment.seqs[0])
        super().__init__(alignment, refresh_callback)

    def add_options(self, options_container, *, category=None, verbose_labels=True):
        from chimerax.ui.options import FloatOption, BooleanOption
        option_data =[
            ("capitalization threshold", 'capitalize_threshold', FloatOption, {},
                "Capitalize consensus letter if at least this fraction of sequences are identical"),
            ("ignore gap characters", 'ignore_gaps', BooleanOption, {},
                "Whether gap characters are considered for the consensus character")
        ]
        self._add_options(options_container, category, verbose_labels, option_data)

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

    @property
    def ignore_gaps(self):
        return self.settings.ignore_gaps

    @ignore_gaps.setter
    def ignore_gaps(self, ignore_gaps):
        self.settings.ignore_gaps = ignore_gaps

    def num_options(self):
        return 2

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

from chimerax.core.settings import Settings
class ConsensusSettings(Settings):
    EXPLICIT_SAVE = {
        'capitalize_threshold': 0.8,
        'ignore_gaps': False
    }

_settings = None
def get_settings(session):
    global _settings
    if _settings is None:
        _settings = ConsensusSettings(session, "consensus alignment header")
    return _settings
