# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from PyQt5.QtWidgets import QWidget, QFormLayout

class OptionsPanel(QWidget):
    """OptionsPanel is a container for Options"""

    def __init__(self, parent=None, *, sorting=None, **kw):
        """sorting:
             None; options shown in order added
             True: options sorted alphabetically by name
             func: options sorted based on the provided key function
        """
        QWidget.__init__(self, parent, **kw)
        self._sorting = sorting
        self._options = []
        self.setLayout(QFormLayout())

    def add_option(self, option):
        if self._sorting is None:
            insert_row = len(self._options)
        else:
            if self._sorting is True:
                test = lambda o1, o2: o1.name < o2.name
            else:
                test = lambda o1, o2: self._sorting(o1) < self._sorting(o2)
            for insert_row in range(len(self._options)):
                if test(option, self._options[insert_row]):
                    break
            else:
                insert_row = len(self._options)
        self.layout().insertRow(insert_row, option.name, option.widget)
