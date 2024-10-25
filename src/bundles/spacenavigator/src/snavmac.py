# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

class Space_Device_Mac:

    def __init__(self):

        self.max_delay = 0.2
        self.min_lag = 1e20

        from . import _spacenavigator
        _spacenavigator.connect()

    def last_event(self):

        from ._spacenavigator import state
        s = state()
        if s is None:
            return None

        trans = (s[0],-s[1],-s[2])
        rot = (s[3],-s[4],-s[5])
        buttons = []
        if s[6] & 1: buttons.append('N1')
        if s[6] & 2: buttons.append('N2')

        # Reject delayed events.
        from time import time
        lag = time() - s[7]*1e-9
        self.min_lag = min(lag, self.min_lag)
        if lag - self.min_lag > self.max_delay:
            return None

        return (rot, trans, buttons)
