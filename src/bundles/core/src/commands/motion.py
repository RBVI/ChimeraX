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

class CallForNFrames:
    # CallForNFrames acts like a function that keeps track of per-frame
    # functions.  But those functions need state, so that state is
    # encapsulated in instances of this class.
    #
    # Instances are automatically added to the given session 'Attribute'.

    Infinite = -1
    Attribute = 'motion_in_progress'    # session attribute

    def __init__(self, func, n, session):
        self.func = func
        self.n = n
        self.session = session
        self.frame = 0
        if n > 0 or n == self.Infinite:
            self.handler = session.triggers.add_handler('new frame', self)
            if not hasattr(session, self.Attribute):
                setattr(session, self.Attribute, set([self]))
            else:
                getattr(session, self.Attribute).add(self)

    def __call__(self, *_):
        f = self.frame
        self.func(self.session, f)
        self.frame = f + 1
        if self.n != self.Infinite and self.frame >= self.n:
            self.done()

    def done(self):
        s = self.session
        s.triggers.remove_handler(self.handler)
        getattr(s, self.Attribute).remove(self)


def motion_in_progress(session):
    # Return True if there are non-infinite motions
    if not hasattr(session, CallForNFrames.Attribute):
        return False
    for m in getattr(session, CallForNFrames.Attribute):
        if m.n != CallForNFrames.Infinite:
            return True
    return False
