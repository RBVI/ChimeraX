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

# -----------------------------------------------------------------------------
# Histogram widget and interactive canvas ramp display.
#

# -----------------------------------------------------------------------------
#
class Histogram:

  def __init__(self, canvas, scene):

    self.canvas = canvas
    self.scene = scene
    self.graphics_items = []

  # ---------------------------------------------------------------------------
  # Use QGraphicsScene coordinate ranges 0 to 1 in x and y.
  def show_data(self, heights):

    s = self.scene
    sr = s.sceneRect()
    sz = sr.size()
    w, h = sz.width(), sz.height()

    for gi in self.graphics_items:
      s.removeItem(gi)
    self.graphics_items.clear()

    bins = len(heights)
    max_height = max(heights)
    if max_height == 0:
      return            # No data was binned.
    self.graphics_items = items = []
    for b in range(bins):
      x = w * (b / bins)
      y = h * (heights[b] / max_height)
      item = s.addLine(x, h, x, h-y)
      items.append(item)
#      c.tag_lower(id)                           # keep bars below marker lines

    self.canvas.fitInView(sr)

# -----------------------------------------------------------------------------
# Draw a set of movable markers on a canvas.  The markers can be moved with
# the mouse.  They can be drawn as vertical lines ('line' type) or as small
# boxes ('box' type).  Each marker has a color.  Markers can be added or
# deleted with ctrl-button-1.  A callback is invoked when user mouse
# interaction selects or moves a marker and takes one argument that is the marker.
#
# This was designed for display and control of theshold levels shown on a
# histogram.
#
class Markers:

  def __init__(self, canvas, scene, marker_type, new_marker_color,
               connect_markers, selected_marker_callback, moved_marker_callback):

    self.canvas = canvas
    sr = scene.sceneRect()
    self.canvas_box = (0,0,sr.width(),sr.height())
    self.scene = scene
    self.marker_type = marker_type        # 'line' or 'box'
    self.box_size = 2
    self.new_marker_color = new_marker_color
    self.extend_left = False
    self.extend_right = True

    self.connect_markers = connect_markers
    self.connect_color = 'yellow'
    self.connect_graphics_items = []

    self.selected_marker_callback = selected_marker_callback
    self.moved_marker_callback = moved_marker_callback

    self.markers = []

    self.user_x_range = (0, 1)
    self.user_y_range = (0, 1)

    self.drag_marker_index = None
    self.last_mouse_xy = None

    self.shown = False
    self.show(True)

    canvas.click_callbacks.append(self.select_marker_cb)
    canvas.drag_callbacks.append(self.move_marker_cb)
    
  # ---------------------------------------------------------------------------
  #
  def show(self, show):

    if show and not self.shown:
      self.shown = show
      self.update_plot()
    elif not show and self.shown:
      self.unplot_markers()
      self.shown = show
    
  # ---------------------------------------------------------------------------
  #
  def plot_markers(self):

    s = self.scene
    bs = self.box_size
    x0, y1, x1, y0 = self.canvas_box

    from Qt.QtGui import QPen, QColor, QBrush
    from Qt.QtCore import Qt
    p = QPen(QColor('black'))
    b = QBrush()

    for m in self.markers:
      if m.graphics_item is None:
        x, y = self.user_xy_to_canvas_xy(m.xy)
        color = hex_color_name(m.rgba[:3])
        c = QColor(color)
        b.setColor(c)
        b.setStyle(Qt.BrushStyle.SolidPattern)
        if self.marker_type == 'line':
          m.graphics_item = gi = s.addRect(x-bs, y0+bs, 2*bs, y1-y0-bs, pen = p, brush = b)
          gi.setZValue(1.0)	# Show on top of histogram
        elif self.marker_type == 'box':
          m.graphics_item = gi = s.addRect(x-bs, y-bs, 2*bs, 2*bs, pen = p, brush = b)
          gi.setZValue(1.0)	# Show on top of histogram
          
  # ---------------------------------------------------------------------------
  #
  def unplot_markers(self):

    s = self.scene
    for m in self.markers:
      m.unplot(s)

    for i in self.connect_graphics_items:
      s.removeItem(i)
    self.connect_graphics_items = []
    
  # ---------------------------------------------------------------------------
  # canvas_box = (xmin, ymin, xmax, ymax)
  # The xmin and xmax values should give the positions corresponding to the
  # values passed to set_user_x_range().
  #
  def set_canvas_box(self, canvas_box):

    if canvas_box != self.canvas_box:
      self.canvas_box = canvas_box
      self.update_plot()
    
  # ---------------------------------------------------------------------------
  #
  def set_user_x_range(self, xmin, xmax):

    self.user_x_range = (xmin, xmax)
    self.update_plot()
    
  # ---------------------------------------------------------------------------
  #
  def canvas_xy_to_user_xy(self, cxy):

    xmin, ymin, xmax, ymax = self.canvas_box
    fx = float(cxy[0] - xmin) / (xmax - xmin)
    uxmin, uxmax = self.user_x_range
    ux = (1-fx) * uxmin + fx * uxmax
    fy = float(cxy[1] - ymin) / (ymax - ymin)
    uymin, uymax = self.user_y_range
    uy = (1-fy) * uymax + fy * uymin
    return ux, uy

  # ---------------------------------------------------------------------------
  #
  def user_xy_to_canvas_xy(self, uxy):

    xmin, ymin, xmax, ymax = self.canvas_box

    uxmin, uxmax = self.user_x_range
    fx = (uxy[0] - uxmin) / (uxmax - uxmin)
    cx = (1-fx) * xmin + fx * xmax

    uymin, uymax = self.user_y_range
    fy = (uxy[1] - uymin) / (uymax - uymin)
    cy = (1-fy) * ymax + fy * ymin

    return cx, cy
  
  # ---------------------------------------------------------------------------
  #
  def set_markers(self, markers, extend_left = None, extend_right = None):

    changed = False
    
    if extend_left is not None and extend_left != self.extend_left:
      self.extend_left = extend_left
      changed = True

    if extend_right is not None and extend_right != self.extend_right:
      self.extend_right = extend_right
      changed = True

    if (len(markers) != len(self.markers) or
        [m1 for m1, m2 in zip(markers, self.markers)
         if tuple(m1.xy) != tuple(m2.xy) or tuple(m1.rgba) != tuple(m2.rgba)]):
      changed = True
      
    if changed:
      for m in self.markers:
        m.unplot(self.scene)
      self.markers = markers
      self.update_plot()
  
  # ---------------------------------------------------------------------------
  #
  def clamp_canvas_xy(self, xy):
    
    x, y = xy
    xmin, ymin, xmax, ymax = self.canvas_box

    if x < xmin:      x = xmin
    elif x > xmax:    x = xmax
    
    if y < ymin:      y = ymin
    elif y > ymax:    y = ymax

    return [x, y]
  
  # ---------------------------------------------------------------------------
  #
  def update_plot(self):

    if not self.shown:
      return

    self.plot_markers()

    self.update_marker_coordinates()

    if self.connect_markers:
      self.update_connections()
  
  # ---------------------------------------------------------------------------
  #
  def update_marker_coordinates(self):

    x0, y0, x1, y1 = self.canvas_box
    bs = self.box_size

    from Qt.QtCore import QRectF
    for m in self.markers:
      cxy = self.user_xy_to_canvas_xy(m.xy)
      x, y = self.clamp_canvas_xy(cxy)
      if self.marker_type == 'line':
        m.graphics_item.setRect(QRectF(x-bs, y0+bs, 2*bs, y1-y0-2*bs))
      elif self.marker_type == 'box':
        m.graphics_item.setRect(QRectF(x-bs, y-bs, 2*bs, 2*bs))
  
  # ---------------------------------------------------------------------------
  #
  def update_connections(self):

    xy_list = [m.xy for m in self.markers]
    cxy_list = [self.user_xy_to_canvas_xy(xy) for xy in xy_list]
    cxy_list.sort()
    cxy_list = [self.clamp_canvas_xy(xy) for xy in cxy_list]

    s = self.scene
    from Qt.QtGui import QPen, QColor
    p = QPen(QColor(self.connect_color))

    graphics_items = []
    n = len(cxy_list)
    if self.extend_left and n > 0:
      xmin = self.canvas_box[0]
      x0, y0 = cxy_list[0]
      x1, y1 = xmin, y0
      gi = s.addLine(x0, y0, x1, y1, pen = p)
      gi.setZValue(1.0)	# Show on top of histogram
      graphics_items.append(gi)

    for k in range(n-1):
      x0, y0 = cxy_list[k]
      x1, y1 = cxy_list[k+1]
      gi = s.addLine(x0, y0, x1, y1, pen = p)
      gi.setZValue(1.0)	# Show on top of histogram
      graphics_items.append(gi)

    if self.extend_right and n > 0:
      xmax = self.canvas_box[2]
      x0, y0 = cxy_list[n-1]
      x1, y1 = xmax, y0
      gi = s.addLine(x0, y0, x1, y1, pen = p)
      gi.setZValue(1.0)	# Show on top of histogram
      graphics_items.append(gi)
      
    for gi in self.connect_graphics_items:
      s.removeItem(gi)

    self.connect_graphics_items = graphics_items

  # ---------------------------------------------------------------------------
  #
  def add_marker(self, canvas_x, canvas_y):
    sp = self.canvas.mapToScene(canvas_x, canvas_y)
    cxy = self.clamp_canvas_xy((sp.x(), sp.y()))
    xy = self.canvas_xy_to_user_xy(cxy)
    sm = self.selected_marker()
    if sm:
      color = sm.rgba
    else:
      color = self.new_marker_color
    m = Marker(xy, color)
    self.markers.append(m)
    self.last_mouse_xy = xy
    self.drag_marker_index = len(self.markers) - 1

    self.update_plot()

  # ---------------------------------------------------------------------------
  #
  def clicked_marker(self, canvas_x, canvas_y):
    range = 3
    i = self.closest_marker_index(canvas_x, canvas_y, range)
    if i is None:
      return None
    return self.markers[i]

  # ---------------------------------------------------------------------------
  #
  def delete_marker(self, m):
    i = self.markers.index(m)
    self.drag_marker_index = None
    m.unplot(self.scene)
    del self.markers[i]

    self.update_plot()

  # ---------------------------------------------------------------------------
  #
  def select_marker_cb(self, event):

    if not self.shown:
      return

    range = 3
    ep = event.pos()
    i = self.closest_marker_index(ep.x(), ep.y(), range)
    self.drag_marker_index = i

    if i == None:
      return

    p = self.canvas.mapToScene(ep.x(), ep.y())
    self.last_mouse_xy = self.canvas_xy_to_user_xy((p.x(), p.y()))

    cb = self.selected_marker_callback
    if cb:
      cb(self.markers[i])

  # ---------------------------------------------------------------------------
  #
  def closest_marker_index(self, x, y, range):

    items = {m.graphics_item:mi for mi,m in enumerate(self.markers)}
    c = self.canvas
    for i in c.items(x, y, 2*range, 2*range):
      if i in items:
        return items[i]

    return None

  # ---------------------------------------------------------------------------
  #
  def move_marker_cb(self, event):

    if (not self.shown or
        self.last_mouse_xy == None or
        self.drag_marker_index == None or
        self.drag_marker_index >= len(self.markers)):
      return

    ep = event.pos()
    p = self.canvas.mapToScene(ep.x(), ep.y())
    mouse_xy = self.canvas_xy_to_user_xy((p.x(),p.y()))
    dx = mouse_xy[0] - self.last_mouse_xy[0]
    dy = mouse_xy[1] - self.last_mouse_xy[1]
    self.last_mouse_xy = mouse_xy

    from Qt.QtCore import Qt
    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
      dx = .1 * dx
      dy = .1 * dy

    #
    # Don't allow dragging out of the canvas box.
    #
    m = self.markers[self.drag_marker_index]
    xy = (m.xy[0] + dx, m.xy[1] + dy)
    cxy = self.user_xy_to_canvas_xy(xy)
    cxy = self.clamp_canvas_xy(cxy)
    xy = self.canvas_xy_to_user_xy(cxy)
    m.xy = xy

    self.update_plot()

    cb = self.moved_marker_callback
    if cb:
      cb(m)
    
  # ---------------------------------------------------------------------------
  #
  def selected_marker(self):

    if (self.drag_marker_index == None or
        self.drag_marker_index >= len(self.markers)):
      if len(self.markers) > 0:
        return self.markers[0]
      else:
        return None
    return self.markers[self.drag_marker_index]
    
# -----------------------------------------------------------------------------
#
class Marker:

  def __init__(self, xy, color):

    self.xy = xy
    self.rgba = color
    self.graphics_item = None
  
  # ---------------------------------------------------------------------------
  #
  def set_color(self, rgba, canvas):

    if tuple(rgba) == tuple(self.rgba):
      return

    self.rgba = rgba
    gi = self.graphics_item
    if gi:
      color = hex_color_name(rgba[:3])
      from Qt.QtGui import QColor, QBrush
      gi.setBrush(QBrush(QColor(color)))
  
  # ---------------------------------------------------------------------------
  #
  def unplot(self, scene):

    gi = self.graphics_item
    if gi:
      scene.removeItem(gi)
      self.graphics_item = None
    
# -----------------------------------------------------------------------------
#
def hex_color_name(rgb):

  rgb8 = tuple(min(255, max(0,int(256 * c))) for c in rgb)
  return '#%02x%02x%02x' % rgb8
