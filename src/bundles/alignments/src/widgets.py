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

from chimerax.ui.widgets import ModelListWidget, ModelMenuButton
from .alignment import Alignment
from chimerax.atomic import Sequence
from chimerax.core.triggerset import TriggerSet

class AlignmentListWidget(ModelListWidget):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=Alignment,
            list_func=lambda ses=session: session.alignments.alignments,
            key_func=lambda aln: aln.ident,
            trigger_info=[
                (session.alignments.triggers, "new alignment"),
                (session.alignments.triggers, "destroy alignment"),
            ],
            **kw)

class AlignmentMenuButton(ModelMenuButton):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=Alignment,
            list_func=lambda ses=session: session.alignments.alignments,
            key_func=lambda aln: aln.ident,
            trigger_info=[
                (session.alignments.triggers, "new alignment"),
                (session.alignments.triggers, "destroy alignment"),
            ],
            **kw)

class AlignSeqListWidget(ModelListWidget):
    def __init__(self, session, alignment, **kw):
        self.alignment = alignment
        self.triggers = TriggerSet()
        self.triggers.add_trigger("seqs changed")
        alignment.add_observer(self)
        super().__init__(session, class_filter=Sequence,
            list_func=lambda aln=alignment: alignment.seqs,
            key_func=lambda seq: seq.name,
            item_text_func=lambda seq: seq.name,
            trigger_info=[
                (self.triggers, "seqs changed"),
            ],
            **kw)

    def destroy(self):
        self.alignment.remove_observer(self)
        super().destroy()

    def alignment_notification(self, note_name, note_data):
        if note_name == "add or remove seqs":
            self.triggers.activate_trigger("seqs changed", note_data)


class AlignSeqMenuButton(ModelMenuButton):
    def __init__(self, session, alignment, **kw):
        self.alignment = alignment
        self.triggers = TriggerSet()
        self.triggers.add_trigger("seqs changed")
        alignment.add_observer(self)
        super().__init__(session, class_filter=Sequence,
            list_func=lambda aln=alignment: alignment.seqs,
            key_func=lambda seq: seq.name,
            item_text_func=lambda seq: seq.name,
            trigger_info=[
                (self.triggers, "seqs changed"),
            ],
            **kw)

    def destroy(self):
        self.alignment.remove_observer(self)
        super().destroy()

    def alignment_notification(self, note_name, note_data):
        if note_name == "add or remove seqs":
            self.triggers.activate_trigger("seqs changed", note_data)

