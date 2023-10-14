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

# -----------------------------------------------------------------------------
# Mouse mode to scale surfaces about their center of mass.  The vertex
# positions are changed.  If some other tool generated and is updating
# the surface geometry then the size change will be overwritten the next
# time that tool regenerates the surface.  This resize capability is for
# adjusting geometric shapes to match the size of features in density maps.
#

# ----------------------------------------------------------------------------
#
srmm = None
def enable_resize_surface_mouse_mode(enable = True, button = '3',
                                     modifiers = [], one_use = False):

  global srmm
  if srmm is None:
    srmm = Surface_Resize_Mouse_Mode()
  srmm.enable(enable, button, modifiers, one_use)

# ----------------------------------------------------------------------------
#
class Surface_Resize_Mouse_Mode:

  def __init__(self):

    self.register_resize_mouse_mode()
    self.last_xy = None
    self.one_use = None
    
  # ----------------------------------------------------------------------------
  #
  def enable(self, enable = True, button = '3', modifiers = [],
             one_use = False,):

    from chimera import mousemodes
    if enable:
      mousemodes.setButtonFunction(button, modifiers, 'resize surfaces')
      self.one_use = (button, modifiers)
    else:
      def_mode = mousemodes.getDefault(button, modifiers)
      mousemodes.setButtonFunction(button, modifiers, def_mode)
    
  # ----------------------------------------------------------------------------
  #
  def register_resize_mouse_mode(self):

    from chimera.mousemodes import addFunction
    callbacks = (self.mouse_down_cb, self.mouse_drag_cb, self.mouse_up_cb)
    addFunction('resize surfaces', callbacks)

  # ---------------------------------------------------------------------------
  #
  def mouse_down_cb(self, v, e):

    self.last_xy = (e.x, e.y)

  # ---------------------------------------------------------------------------
  #
  def mouse_drag_cb(self, v, e):

    if self.last_xy is None:
      self.last_xy = (e.x, e.y)
      return
    
    import Surface
    plist = Surface.selected_surface_pieces()
    if len(plist) == 0:
      from chimera.replyobj import status
      status('No surfaces selected for resizing')
      return

    dx, dy = (e.x - self.last_xy[0], self.last_xy[1] - e.y)
    delta = dx + dy
    self.last_xy = (e.x, e.y)

    shift_mask = 1
    shift = (e.state & shift_mask)
    if shift:
      factor_per_pixel = 1.001  # shift key held so scale by smaller amount
    else:
      factor_per_pixel = 1.01

    factor = factor_per_pixel ** delta
    for p in plist:
      scale_surface_piece(p, factor)

  # ---------------------------------------------------------------------------
  #
  def mouse_up_cb(self, v, e):

    self.last_xy = None
    if self.one_use:
      b, m = self.one_use
      self.enable(False, b, m)

# -----------------------------------------------------------------------------
#
def scale_surface_piece(p, factor):

  v, t = p.geometry
  if len(v) == 0:
    return

  center = v.sum(axis = 0) / v.shape[0]
  v -= center
  v *= factor
  v += center

  p.geometry = v, t
