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

from chimerax.mouse_modes import MouseMode
class NextDockingMouseMode(MouseMode):

  name = 'next docked'
  icon_file = 'nextdocking.png'

  def __init__(self, session):
    MouseMode.__init__(self, session)

    self._step_pixels = 50
    self._last_xy = None

  def mouse_down(self, event):
    self._last_xy = event.position()
  
  def mouse_drag(self, event):

    x,y = event.position()
    if self._last_xy is None:
      self._last_xy = (x,y)
      return

    lx,ly = self._last_xy
    dx,dy = x - lx, ly - y
    d = dx if abs(dx) > abs(dy) else -dy
    if d >= self._step_pixels:
        step = 1
    elif d <= -self._step_pixels:
        step = -1
    else:
        return

    self._last_xy = (x,y)
    self._show_next(step)

  def mouse_up(self, event):
    self._last_xy = None
    
  def wheel(self, event):
    d = event.wheel_value()
    self._show_next(-int(d))

  def _show_next(self, step):
    if not self._viewdockx_running():
        return
    from chimerax.core.commands import run
    if step > 0:
      run(self.session, 'viewdockx down')
    elif step < 0:
      run(self.session, 'viewdockx up')

  def _viewdockx_running(self):
    from .tool import TableTool
    try:
      tool = TableTool.find(name = None)
    except KeyError:
      return False
    return tool is not None

  def vr_press(self, event):
    # Use touchpad position to step to next or previous ligand
    if not event.is_touchpad:
      return
    x,y = event.touchpad_position
    if x is None or y is None:
      return
    v = x if abs(x) > abs(y) else -y
    step = 1 if v >= 0 else -1
    self._show_next(step)

  vr_update_delay_frames = 45	# Wait long enough to let GUI update
  
  def vr_motion(self, event):
    # Virtual reality hand controller motion.
    if event.is_touchpad:
      return
    step_size = 0.10	# Meters vertical motion per step
    step = int(round(-event.room_vertical_motion / step_size))
    if step == 0:
      return 'accumulate drag'
    self._show_next(step)

  def vr_thumbstick(self, event):
    # Virtual reality hand controller thumbstick tilt.
    step = event.thumbstick_step(flip_y = True)
    if step != 0:
      self._show_next(step)

  # TODO: Add a vr_trackpad() method for use with Vive hand controllers
  #   so up/down can be done by clicking on different parts of the trackpad.
  
# -----------------------------------------------------------------------------
#
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(NextDockingMouseMode(session))
