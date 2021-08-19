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

from Qt.QtWidgets import QWidget, QLabel, QStackedWidget, QGraphicsView, QGraphicsScene, QFrame
from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLineEdit
from Qt.QtCore import QSize, Qt, QTimer, QRectF
from Qt.QtGui import QBrush, QColor, QPen
from chimerax.mouse_modes import mod_key_info
from chimerax.core.colors import Color

class MarkedHistogram(QWidget):
    """Histogram with color-indication markers

       MarkedHistogram shows a histogram of a data set and an optional
       label for the numeric range of the data set.  Color markers can
       be placed on the histogram by the user and moved interactively,
       either with the mouse or by typing in a particular data value.
       A color button is used to control the color of the "current" marker
       (the one most recently selected with the mouse).  Markers can
       either be vertical bars or squares.  Vertical bars can only be
       moved left/right whereas squares can also be moved up/down.
       Squares are also connected from left to right by line segments.

       A row of associated widgets (such as the marker color button) is
       placed below the histogram.  User-specified widgets can also be
       placed in this row with the add_custom_widget() method.

       Individual markers are grouped into HistogramMarkers instances,
       and several HistogramMarkers instances can be associated with
       a single histogram, though only one instance is active/shown at
       a time.

       MarkedHistogram has the following constructor options:
           [Options noted as init options can only be specified at
        widget creation.  Others can be changed later via the corresponding
        property name.]

        color_button --  controls whether a color button is offered in
            the user interface for changing marker colors.
            default: True

        data_source -- either a string or a 3-tuple.  If a string, then
            no histogram is displayed and instead the string is
            displayed in the histogram area as a text message.
            The first 2 components of a 3-tuple should be the
            minimum and maximum values of the histogram,  The
            third component should either be an array of numbers
            (i.e. the histogram) or a callback function that
            takes as its single argument the number of bins to
            histogram into and that returns a histogram array.
            default: 'no data'

        layout -- [init option] how to organize the megawidget layout.
            Choices are 'single', 'top', and 'below'.  'single'
            should be used when you are using a single histogram
            in your GUI, or histograms that aren't arrayed
            vertically.  'top' and 'below' should be used for
            histograms that are laid out in a vertical array
            ('top' for the top-most one, and 'below' for all
            others).  Certain repeated elements will be omitted
            in 'below' histograms (e.g. some widget labels).
            default: single

        max_label/min_label [init options] show the max/min histogram
            values as labels below the histogram on the right/left.
            If neither is True, then the range will be shown in a
            single label widget below the histogram.
            default: False

        redraw_delay -- amount of time (in seconds) to delay between
            needing to redraw and actually starting to redraw.
            Used to avoid multiple (possibly slow) redraws during
            a resize.
            default: 0.25

        scaling -- how to scale the vertical height of the histogram.
            Either 'logarithmic' or 'linear'.
            default: logarithmic

        select_callback -- [init option] function to call when the
            "current" marker changes.  The function receives 4
            argments:  previous active marker set/marker,
            new active marker set/marker.  The marker set can
            be None to indicate no active marker set and the
            marker can be None if no marker was/is current.

        show_marker_help -- [init option] whether to show the help
            text over the histogram describing how to add/delete
            markers.
            default: True

        status_line -- function to use to output messages (such as
            warning when trying to add more markers than allowed).
            The function should take a single string argument.
            default: None

        value_label -- [init option] label to use next to the
            entry widget describing the current marker value.
            default: 'Value'

        value_width -- width of the current marker value entry widget.
            default: 7

       Constructor options that begin with 'Markers_' specify default
       constructor options for HistogramMarkers objects created in the
       add_markers method (e.g. Markers_connect_color='red' will supply
       connect_color='red' to the HistogramMarkers constructor).  Options
       for specific instances can still be provided to the add_ markers()
       method as keyword arguments (without the 'Markers_' prefix).
    """

    def __init__(self, *args, color_button=True, data_source='no data', layout='single',
            max_label=False, min_label=False, redraw_delay=0.25, scaling='logarithmic',
            select_callback=None, show_marker_help=True, status_line=None, value_label='Value',
            value_width=7, **kw):

        # Get HistogramMarkers options and initialise base class
        self._histogram_markers_kw = markers_kw = {}
        for opt_name in list(kw.keys()):
            if opt_name.startswith('Markers_'):
                markers_kw[opt_name[8:]] = kw.pop(opt_name)
        super().__init__(*args, **kw)

        # initialize variables
        self._layout = layout
        self.status_line = status_line
        self._show_marker_help = show_marker_help
        self._active_markers = None
        self._markers = []
        self._markable = False
        self._drag_marker = None
        self._scaling = scaling
        self._select_callback = select_callback
        if select_callback:
            self._prev_markers = None
            self._prev_marker = None

        overall_layout = QVBoxLayout()
        self.setLayout(overall_layout)

        # Create the add/delete marker help
        if show_marker_help and layout != 'below':
            self._marker_help = QLabel("Ctrl-click on histogram to add or delete thresholds")
            self._marker_help.setAlignment(Qt.AlignCenter)
            overall_layout.addWidget(self._marker_help)
        else:
            self._marker_help = None

        # Create the data area
        class HistFrame(QFrame):
            def sizeHint(self):
                return QSize(300, 64)
        data_frame = QFrame()
        data_frame.setLineWidth(1)
        data_frame.setMidLineWidth(2)
        data_frame.setFrameStyle(data_frame.Panel | data_frame.Sunken)
        data_frame.setContentsMargins(0,0,0,0)
        data_frame_layout = QHBoxLayout()
        data_frame.setLayout(data_frame_layout)
        self._data_widgets = QStackedWidget()
        data_frame_layout.addWidget(self._data_widgets)

        # Crate the histogram widget
        self._hist_scene = QGraphicsScene()
        self._hist_bars = self._hist_scene.createItemGroup([])
        self._hist_view = QGraphicsView(self._hist_scene)
        self._hist_view.resizeEvent = self._redraw
        self._hist_scene.mousePressEvent = lambda event: self._add_or_delete_marker_cb(event) \
            if event.modifiers() & mod_key_info("control")[0] else self._select_marker_cb(event)
        self._hist_scene.mouseMoveEvent = lambda event: self._move_marker_cb(event) \
            if self._drag_marker else super().mouseMoveEvent(event)
        self._hist_scene.mouseReleaseEvent = self._button_up_cb
        self._redraw_timer = QTimer()
        self._redraw_timer.timeout.connect(self._redraw_cb)
        self._redraw_timer.start(1000 * redraw_delay)
        self._redraw_timer.stop()
        self._data_widgets.addWidget(self._hist_view)

        # Create the histogram replacement label
        self._no_histogram_label = QLabel()
        self._no_histogram_label.setAlignment(Qt.AlignCenter)
        self._data_widgets.addWidget(self._no_histogram_label)
        overall_layout.addWidget(self._data_widgets, stretch=1)

        # Create range label(s)
        self._widget_area = QWidget()
        self._widget_layout = QHBoxLayout()
        self._min_label = self._max_label = None
        if min_label or max_label:
            min_max_layout = QHBoxLayout()
            if min_label:
                self._min_label = QLabel()
                min_max_layout.addWidget(self._min_label, alignment=Qt.AlignLeft & Qt.AlignTop)

            if max_label:
                self._max_label = QLabel()
                min_max_layout.addWidget(self._max_label, alignment=Qt.AlignRight & Qt.AlignTop)
            overall_layout.addLayout(min_max_layout)
        else:
            self._range_label = QLabel()
            if layout == 'below':
                self._widget_layout.addWidget(self._range_label)
            else:
                lab = QLabel("Range")
                if layout == 'single':
                    self._widget_layout.addWidget(lab, alignment=Qt.AlignRight)
                    self._widget_layout.addWidget(self._range_label, alignment=Qt.AlignLeft)
                else: # layout == 'top'
                    range_layout = QVBoxLayout()
                    range_layout.addWidget(lab, alignment=Qt.AlignBottom)
                    range_layout.addWidget(self._range_label, alignment=Qt.AlignTop)
                    self._widget_layout.addLayout(range_layout)
        self._widget_area.setLayout(self._widget_layout)
        overall_layout.addWidget(self._widget_area)

        # Create value widget
        self._value_entry = QLineEdit()
        self._value_entry.setEnabled(False)
        self._value_entry.returnPressed.connect(self._set_value_cb)
        self.value_width = value_width
        if layout == 'below':
            self._widget_layout.addWidget(self._value_entry)
        else:
            lab = QLabel(value_label)
            if layout == 'single':
                self._widget_layout.addWidget(lab, alignment=Qt.AlignRight)
                self._widget_layout.addWidget(self._value_entry, alignment=Qt.AlignLeft)
            else:
                value_layout = QVBoxLayout()
                value_layout.addWidget(lab, alignment=Qt.AlignBottom)
                value_layout.addWidget(self._value_entry, alignment=Qt.AlignTop)
                self._widget_layout.addLayout(value_layout)

        # Create color button widget
        from .color_button import ColorButton
        self._color_button = cb = ColorButton()
        cb.color_changed.connect(lambda rgba8: self._color_button_cb([c/255.0 for c in rgba8]))
        cb.setEnabled(False)
        self._color_button_label = cbl = QLabel("Color")
        if layout == 'below':
            self._widget_layout.addWidget(self._color_button)
        else:
            if layout == 'single':
                self._widget_layout.addWidget(cbl, alignment=Qt.AlignRight)
                self._widget_layout.addWidget(cb, alignment=Qt.AlignLeft)
            else:
                color_layout = QVBoxLayout()
                color_layout.addWidget(cbl, alignment=Qt.AlignBottom)
                color_layout.addWidget(cb, alignment=Qt.AlignTop)
                self._widget_layout.addLayout(color_layout)
        self._color_button_shown = True

        # Show the histogram or the no-data label
        self.data_source = data_source

    def activate(self, markers):
        """Make the given set of markers the currently active set

           Any previously-active set will be hidden.
        """

        if markers is not None and markers not in self._markers:
            raise ValueError("activate() called with bad value")
        if markers == self._active_markers:
            return
        if self._active_markers is not None:
            self._active_markers._hide()
        elif self.layout != 'below' and self._show_marker_help:
            self._marker_help.setHidden(False)
        self._active_markers = markers
        if self._active_markers is not None:
            self._active_markers.shown = True
            self._set_sel_marker(self._active_markers._sel_marker)
        else:
            if self.layout != 'below' and self._show_marker_help:
                self._marker_help.setHidden(True)
            if self._select_callback:
                if self._prev_marker is not None:
                    self._select_callback(self._prev_markers, self._prev_marker, None, None)
                self._prev_markers = None
                self._prev_marker = None

    def add_custom_widget(self, widget, left_side=True):
        self._widget_layout.addWidget(0 if left_side else -1, widget)

    def add_markers(self, activate=True, **kw):
        """Create and return a new set of markers.

           If 'activate' is true, the new set will also become
           the active set.  Other keyword arguments will be passed
           to the HistogramMarkers constructor.
        """
        final_kw = { k:v for k,v in self._histogram_markers_kw.items() }
        final_kw.update(kw)
        final_kw['histogram'] = self
        markers = HistogramMarkers(**final_kw)
        self._markers.append(markers)
        if activate:
            self.activate(markers)
        return markers

    @property
    def color_button(self):
        return self._color_button_shown

    @color_button.setter
    def color_button(self, show):
        if show == self._color_button_shown:
            return
        if self.layout != 'below':
            self._color_button_label.setHidden(not show)
        self._color_button.setHidden(not show)
        self._color_button_shown = show

    def current_marker_info(self):
        """Identify the marker currently selected by the user.
           
           Returns a HistogramMarkers instance and a marker.
           The instance will be None if no marker set has been
           activated.  The marker will be None if no marker has
           been selected by the user.
        """
        if self._active_markers is None:
            return None, None
        return self._active_markers, self._active_markers._sel_marker

    @property
    def data_source(self):
        return self._data_source

    @data_source.setter
    def data_source(self, data_source):
        self._data_source = data_source
        self._new_data()

    def delete_markers(self, markers):
        """Delete the given set of markers.

           If the markers were active, there will be no active set
           of markers afterward.
        """
        if markers not in self._markers:
            raise ValueError("Bad value for delete()")
        if markers == self._active_markers:
            self.activate(None)
        self._markers.remove(markers)

    @property
    def layout(self):
        return self._layout

    @property
    def redraw_delay(self):
        return self._redraw_timer.interval() / 1000.0

    @redraw_delay.setter
    def redraw_delay(self, secs):
        self._redraw_timer.setInterval(secs * 1000)

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        if self._scaling != scaling:
            self._scaling = scaling
            self._redraw_cb()

    def snapshot_data(self):
        info = {
            'version': 1,
            'draw_min': self._draw_min,
            'draw_max': self._draw_max,
            'markers': [markers.snapshot_data() for markers in self._markers],
        }
        if self._active_markers is None:
            info['active markers'] = None
        else:
            info['active markers'] = self._markers.index(self._active_markers)
        if self['color_button']:
            info['color well'] = self._color_button.color
        else:
            info['color well'] = None
        return info

    def snapshot_restore(self, data):
        self._draw_min = data['draw_min']
        self._draw_max = data['draw_max']
        if data['color well'] is not None:
            self._color_button.color = data['color well']
        if len(data['markers']) != len(self._markers):
            # don't know how to deal with this situation
            return
        for markers, markers_data in zip(self._markers, data['markers']):
            markers.snapshot_restore(markers_data)
        if data['active markers'] is not None:
            self.activate(self._markers[data['active markers']])
            self._set_sel_marker(self._active_markers._sel_marker)

    @property
    def value_width(self):
        return self._value_width

    @value_width.setter
    def value_width(self, vw):
        self._value_width = vw
        ve = self._value_entry
        fm = ve.fontMetrics()
        tm = ve.textMargins()
        cm = ve.contentsMargins()
        w = vw*fm.width('w') + tm.left() + tm.right() + cm.left() + cm.right() + 8
        ve.setMaximumWidth(w)

    def _abs2rel(self, abs_xy):
        x, y = abs_xy
        rel_x = (x - self._min_val) / float(self._max_val - self._min_val)
        rel_y = y / float(self._ymax)
        return rel_x, rel_y

    def _abs_xy(self, scene_xy):
        scene_x, scene_y = scene_xy
        dy = min(max(self._bottom - scene_y, 0), self._hist_height - 1)
        if self.scaling == 'logarithmic':
            exp = dy / float(self._hist_height - 1)
            abs_y = (self._max_height + 1) ** exp - 1
        else:
            abs_y = self._max_height*dy / float(self._hist_height-1)

        cx = scene_x - self._border
        num_bins = len(self._bins)
        if num_bins == self._hist_width:
            fract = cx / (num_bins - 1)
            abs_x = self._min_val + fract * (self._max_val - self._min_val)
        elif num_bins == 2:
            abs_x = self._min_val + (self._max_val - self._min_val) * (
                2 * cx / self._hist_width - 0.5)
        else:
            extra = self._hist_width / (2.0*(num_bins-1))
            abs_x = self._min_val + (self._max_val - self._min_val) * (
                cx - extra) / (self._hist_width - 2.0 * extra)
        abs_x = max(self._min_val, abs_x)
        abs_x = min(self._max_val, abs_x)
        return abs_x, abs_y

    def _add_or_delete_marker_cb(self, event=None):
        if self._active_markers is None:
            return

        marker = self._active_markers._pick_marker(event.scenePos())

        if marker is None:
            max_marks = self._active_markers.max_marks
            if max_marks is not None and len(self._active_markers) >= max_marks:
                if self.status_line:
                    self.status_line("Maximum of %d markers\n" % max_marks)
                return
            xy = self._abs_xy((event.scenePos().x(), event.scenePos().y()))
            if self._active_markers.coord_type == 'relative':
                xy = self._abs2rel(xy)
            sel_marker = self._active_markers._sel_marker
            if sel_marker:
                color = sel_marker.rgba
            else:
                color = self._active_markers.new_color
            marker = self._active_markers.append((xy, color))
            self._set_sel_marker(marker, drag_start=event)
        else:
            min_marks = self._active_markers.min_marks
            if min_marks is not None and len(self._active_markers) <= min_marks:
                if self.status_line:
                    self.status_line("Minimum of %d markers\n" % min_marks)
                return
            self._active_markers.remove(marker)
            self._set_sel_marker(None)

    def _button_up_cb(self, event=None):
        if self._drag_marker:
            self._drag_marker = None
            if self._active_markers.move_callback:
                self._active_markers.move_callback('end')

    def _scene_xy(self, abs_xy):
        # minimum is in the _center_ of the first bin,
        # likewise, maximum is in the center of the last bin

        abs_x, abs_y = abs_xy

        abs_y = max(0, abs_y)
        abs_y = min(self._max_height, abs_y)
        if self.scaling == 'logarithmic':
            import math
            abs_y = math.log(abs_y+1)
        scene_y = self._bottom - (self._hist_height - 1) * (abs_y / self._max_height)

        abs_x = max(self._min_val, abs_x)
        abs_x = min(self._max_val, abs_x)
        num_bins = len(self._bins)
        if num_bins == self._hist_width:
            bin_width = (self._max_val - self._min_val) / float(num_bins - 1)
            left_edge = self._min_val - 0.5 * bin_width
            scene_x = int((abs_x - left_edge) / bin_width)
        else:
            # histogram is effectively one bin wider (two half-empty bins on each end)
            if num_bins == 1:
                scene_x = 0.5 * (self._hist_width - 1)
            else:
                extra = (self._max_val - self._min_val) / (2.0*(num_bins-1))
                eff_min_val = self._min_val - extra
                eff_max_val = self._max_val + extra
                eff_range = float(eff_max_val - eff_min_val)
                scene_x = (self._hist_width - 1) * (abs_x - eff_min_val) / eff_range
        return self._border + scene_x, scene_y

    def _color_button_cb(self, rgba):
        m = self._active_markers._sel_marker
        if not m:
            if self.status_line:
                self.status_line("No marker selected")
            return
        m.rgba = rgba

    def _marker2abs(self, marker):
        if self._active_markers.coord_type == 'absolute':
            return marker.xy
        else:
            return self._rel2abs(marker.xy)

    def _move_cur_marker(self, x, yy=None):
        #
        # Don't allow dragging out of the scene box.
        #
        m = self._active_markers._sel_marker
        if x < self._min_val:
            x = self._min_val
        elif x > self._max_val:
            x = self._max_val
        if yy is None:
            y = m.xy[1]
        else:
            y = yy
            if y < 0:
                y = 0
            elif y > self._ymax:
                y = self._ymax

        if self._active_markers.coord_type == 'absolute':
            m.xy = (x, y)
        else:
            m.xy = self._abs2rel((x,y))
        if yy is None:
            m.xy = (m.xy[0], y)

        self._set_value_entry(x)

        self._active_markers._update_plot()

        if self._active_markers.move_callback:
            self._active_markers.move_callback(m)

    def _move_marker_cb(self, event):
        mouse_xy = self._abs_xy((event.scenePos().x(), event.scenePos().y()))
        dx = mouse_xy[0] - self._last_mouse_xy[0]
        dy = mouse_xy[1] - self._last_mouse_xy[1]
        self._last_mouse_xy = mouse_xy

        if event.modifiers() & mod_key_info("shift")[0]:
            dx *= .1
            dy *= .1

        m = self._drag_marker
        mxy = self._marker2abs(m)
        x, y = mxy[0] + dx, mxy[1] + dy

        self._move_cur_marker(x, y)

    def _new_data(self):
        ds = self.data_source
        if isinstance(ds, str):
            self._no_histogram_label.setText(ds)
            self._data_widgets.setCurrentWidget(self._no_histogram_label)
            if self._min_label:
                self._min_label.setText("")
            if self._max_label:
                self._max_label.setText("")
            if self.layout != 'below' and self._show_marker_help:
                self._marker_help.setHidden(True)
            self._widget_area.setHidden(True)
        else:
            self._data_widgets.setCurrentWidget(self._hist_view)
            if self.layout != 'below' and self._show_marker_help:
                self._marker_help.setHidden(False)
            self._widget_area.setHidden(False)
        self._draw_min = self._draw_max = None
        self._redraw_cb()

    def _redraw(self, event=None):
        self._markable = False
        self._redraw_timer.start()

    def _redraw_cb(self):
        self._redraw_timer.stop()
        ds = self.data_source
        if isinstance(ds, str):
            # displaying a text label right now
            return
        view = self._hist_view
        scene = self._hist_scene
        hist_size = view.viewport().size()
        self._hist_width, self._hist_height = hist_width, hist_height = hist_size.width(), hist_size.height()
        self._min_val, self._max_val, self._bins = ds
        filled_range = self._max_val - self._min_val
        empty_ranges = [0, 0]
        if self._draw_min != None:
            empty_ranges[0] = self._min_val - self._draw_min
            self._min_val = self._draw_min
        if self._draw_max != None:
            empty_ranges[1] = self._draw_max - self._max_val
            self._max_val = self._draw_max
        if callable(self._bins):
            if empty_ranges[0] or empty_ranges[1]:
                full_range = filled_range + empty_ranges[0] + empty_ranges[1]
                filled_bins = self._bins(int(hist_width * filled_range / full_range))
                left = [0] * int(hist_width * empty_ranges[0] / full_range)
                right = [0] * (hist_width - len(filled_bins) - len(left))
                self._bins = left + filled_bins + right
            else:
                self._bins = self._bins(hist_width)
        elif empty_ranges[0] or empty_ranges[1]:
            full_range = filled_range + empty_ranges[0] + empty_ranges[1]
            left = [0] * int(len(self._bins) * empty_ranges[0] / full_range)
            right = [0] * int(len(self._bins) * empty_ranges[1] / full_range)
            self._bins = left + self._bins + right
        if self._min_label:
            self._min_label.setText(self._str_val(self._min_val))
        if self._max_label:
            self._max_label.setText(self._str_val(self._max_val))
        if not self._min_label and not self._max_label:
            self._range_label.setText("%s - %s" % (self._str_val(self._min_val), self._str_val(self._max_val)))

        bars = self._hist_bars.childItems()
        for bar in bars:
            self._hist_bars.removeFromGroup(bar)
            self._hist_scene.removeItem(bar)

        self._ymax = max(self._bins)
        if self.scaling == 'logarithmic':
            from numpy import array, log, float32
            self._bins = array(self._bins, float32)
            self._bins += 1.0
            log(self._bins, self._bins)

        max_height = max(self._bins)
        self._max_height = max_height
        h_scale = float(hist_height - 1) / max_height
        self._border = border = 0
        bottom = hist_height + border - 1
        self._bottom = bottom

        num_bins = len(self._bins)
        if num_bins == hist_width:
            for b, n in enumerate(self._bins):
                x = border + b
                h = int(h_scale * n)
                line = self._hist_scene.addLine(x, bottom, x, bottom-h)
                self._hist_bars.addToGroup(line)
                line.setZValue(-1)  # keep bars below markers
        else:
            x_scale = (hist_width - 1) / float(num_bins)
            for b, n in enumerate(self._bins):
                x1 = border + b * x_scale
                x2 = border + (b+1) * x_scale
                h = int(h_scale * n)
                rect = self._hist_scene.addRect(x1, bottom-h, x2-x1, h)
                self._hist_bars.addToGroup(rect)
                rect.setZValue(-1) # keep bars below markers
        self._markable = True
        if self._active_markers is not None:
            self._active_markers._update_plot()
            marker = self._active_markers._sel_marker
            if marker:
                self._set_value_entry(self._marker2abs(marker)[0])
        self._hist_scene.setSceneRect(self._hist_scene.itemsBoundingRect())

    def _rel2abs(self, rel_xy):
        x, y = rel_xy
        abs_x = self._min_val * (1-x) + x * self._max_val
        abs_y = y * self._ymax
        return abs_x, abs_y

    def _select_marker_cb(self, event=None):
        if self._active_markers is not None:
            marker = self._active_markers._pick_marker(event.scenePos())
            self._set_sel_marker(marker, drag_start=event)
            if marker is not None:
                return
        # show value where histogram clicked
        self._set_value_entry(self._abs_xy((event.scenePos().x(), 0))[0])

    def _set_sel_marker(self, marker, drag_start=None):
        self._active_markers._sel_marker = marker
        if not marker:
            self._color_button.color = "gray"
            self._color_button.setEnabled(False)
            self._set_value_entry("")
            self._value_entry.setEnabled(False)
        else:
            self._color_button.setEnabled(True)
            self._color_button.color = marker.rgba
            self._value_entry.setEnabled(True)
            self._set_value_entry(self._marker2abs(marker)[0])
        if self._select_callback:
            if marker is not None or self._prev_marker is not None:
                self._select_callback(self._prev_markers, self._prev_marker, self._active_markers, marker)
            self._prev_markers = self._active_markers
            self._prev_marker = marker
        if not drag_start:
            return
        self._drag_marker = marker
        if not marker:
            return

        self._last_mouse_xy = self._abs_xy((drag_start.scenePos().x(), drag_start.scenePos().y()))
        if self._active_markers.move_callback:
            self._active_markers.move_callback('start')

    def _set_value_cb(self):
        try:
            v = eval(self._value_entry.text())
        except Exception:
            raise ValueError("Invalid histogram value")
        if type(self._min_val) != type(v):
            v = type(self._min_val)(v)
        if v < self._min_val:
            self._draw_min = v
            self._redraw_cb()
        elif v > self._max_val:
            self._draw_max = v
            self._redraw_cb()
        self._move_cur_marker(v)

    def _set_value_entry(self, val):
        if isinstance(val, str):
            self._value_entry.setText(val)
            return
        if isinstance(self._min_val, int):
            val = int(val + 0.5)
        self._value_entry.setText("%g" % val)

    def _str_val(self, val):
        if isinstance(val, (int, bool)):
            return str(val)
        return "%g" % val

class HistogramMarkers:
    """Color-designating markers on a histogram

       Instances should only created via the add_markers() method of
       MarkedHistogram.  Options can be specified as keyword arguments
       to that function.

       Contained HistogramMarker instances can be accessed as if
       HistogramMarker were a sequence.  The instances are always kept
       sorted ascending in X, so sequence order can change with any
       method that adds markers (e.g. a marker added with 'append'
       may not wind up at the end of the sequence).  Methods that create
       new HistogramMarker instances (append, extend, insert, __setitem__)
       need 2-tuples/lists for each HistogramMarker instance, the
       first component of which is the XY value (i.e. another 2-tuple
       or list) and the second of which is the color info.  The color
       info can be either:
        an RGBA value (integers in the range 0-255 or floats in the range 0-1)
        a color name
        a chimerax.core.colors.Color instance
        a built-in color name

       The MarkedHistogram and HistogramMarker doc strings should be
       examined for further info on usage.

       Options are:

        box_radius -- the radius in pixels of boxes drawn when the
            marker_type is 'box'
            default: 2

        connect -- [init option] whether markers should be
            connected left-to-right with lines.  Typically
            used only when the marker_type is 'box'.
            default: False

        connect_color -- [init option] the color used to draw
            lines connecting markers ('connect' must be True)
            default: yellow

        coord_type -- either 'relative' or 'absolute'.  If the former,
            then the 'xy' option of contained HistgramMarkers are
            in the range 0-1 and indicate positioning relative to
            left/right and bottom/top of the histogram.  If the
            latter, then the x of 'xy' indicates a histogram
            bin by value and a height by count.
            default: absolute

        histogram -- [init option provided automatically by
            MarkedHistogram.add_markers()] the MarkedHistogram
            instance

        marker_type -- [init option] the type of markers to use, 
            either 'line' (vertical bars) or 'box' (squares).
            default: line

        max_marks/min_marks -- the maximum/minimum amount of marks the
            user is allowed to place on the histogram.  A value of
            None indicates no limit.  Can always be exceeded
            programmatically.
            default: None

        move_callback -- [init option] function to call when the user
            moves a marker.  The function receives a value of
            'start' at the beginning of a move and 'end' at the
            end.  During the move the value is the marker being
            moved.
            default: None

        new_color -- the default color assigned to newly-created
            markers.
            default: yellow
    """

    def __init__(self, *args, box_radius=2, connect=False, connect_color='yellow', coord_type='absolute',
            histogram=None, marker_type='line', max_marks=None, min_marks=None, move_callback=None,
            new_color='yellow', **kw):

        self._box_radius = box_radius
        self._connect = connect
        self._connect_color = connect_color
        self._coord_type = coord_type
        self._histogram = histogram
        self._marker_type = marker_type
        self._max_marks = max_marks
        self._min_marks = min_marks
        self._move_callback = move_callback
        self._new_color = new_color

        # Check keywords and initialise options
        self._shown = False
        self._sel_marker = None
        self._prev_box_radius = None
        self._markers = []
        self._connector_items = []
        self._prev_coord_type = self.coord_type

        # values derived from options
        self._marker_func = lambda v: HistogramMarker(self, v[0], self._rgba(v[1]))
        # convenience
        self._scene = self.histogram._hist_scene

    def append(self, val):
        marker = self._marker_func(val)
        self._markers.append(marker)
        self._update_plot()
        return marker

    @property
    def box_radius(self):
        return self._box_radius

    @box_radius.setter
    def box_radius(self, box_radius):
        if box_radius == self._box_radius:
            return
        self._box_radius = box_radius
        self._new_box_radius()

    @property
    def connect(self):
        return self._connect

    @property
    def connect_color(self):
        return self._connect_color

    @property
    def coord_type(self):
        return self._coord_type

    @coord_type.setter
    def coord_type(self, coord_type):
        if coord_type == self._coord_type:
            return
        self._coord_type = coord_type
        self._convert_coords()

    def __delitem__(self, i):
        del self._markers[i]
        self._update_plot()

    def destroy(self):
        self._unplot_markers()

    def extend(self, vals):
        markers = [self._marker_func(v) for v in vals]
        self._markers.extend(markers)
        self._update_plot()
        return markers

    def __getitem__(self, i):
        return self._markers[i]

    @property
    def histogram(self):
        return self._histogram

    def index(self, marker):
        return self._markers.index(marker)

    def insert(self, i, val):
        marker = self._marker_func(val)
        self._markers.insert(i, marker)
        self._update_plot()
        return marker

    def __iter__(self):
        return self._markers.__iter__()

    def __len__(self):
        return len(self._markers)

    @property
    def marker_type(self):
        return self._marker_type

    @property
    def max_marks(self):
        return self._max_marks

    @max_marks.setter
    def max_marks(self, max_marks):
        self._max_marks = max_marks

    @property
    def min_marks(self):
        return self._min_marks

    @min_marks.setter
    def min_marks(self, min_marks):
        self._min_marks = min_marks

    @property
    def move_callback(self):
        return self._move_callback

    @property
    def new_color(self):
        return self._new_color

    @new_color.setter
    def new_color(self, new_color):
        self._new_color = new_color

    def pop(self):
        ret = self._markers.pop()
        if ret == self._sel_marker:
            self._sel_marker = None
        self._unplot_markers(ret)
        self._update_plot()
        return ret

    def remove(self, marker):
        self._markers.remove(marker)
        if marker is self._sel_marker:
            self._sel_marker = None
        self._unplot_markers(marker)
        self._update_plot()

    @property
    def shown(self):
        return self._shown

    @shown.setter
    def shown(self, s):
        if s == self._shown:
            return
        self._shown = s
        self._update_plot()

    def snapshot_data(self):
        info = {
            'marker data': [(m.xy, m.rgba) for m in self._markers],
            'shown': self._shown
        }
        if self._sel_marker:
            info['sel marker'] = self._markers.index(self._sel_marker)
        else:
            info['sel marker'] = None
        return info

    def snapshot_restore(self, data):
        self._unplot_markers()
        self._shown = data['shown']
        self._markers[:] = []
        self.extend(data['marker data'])
        if data['sel marker'] is None:
            self._sel_marker = None
        else:
            self._sel_marker = self._markers[data['sel marker']]

    def __setitem__(self, i, val):
        if isinstance(i, slice):
            new_markers = [self._marker_func(v) for v in val]
            sel_replaced = self._sel_marker in self._markers[i]
        else:
            new_markers = self._marker_func(val)
            sel_replaced = self._sel_marker is self._markers[i]
        if sel_replaced:
            self._sel_marker = None
        self._unplot_markers(self._markers[i])
        self._markers[i] = new_markers
        self._update_plot()

    def sort(self, *args, **kw):
        self._markers.sort(*args, **kw)

    def _scene_xy(self, xy):
        if self.coord_type == 'relative':
            abs_xy = self.histogram._rel2abs(xy)
        else:
            abs_xy = xy
        return self.histogram._scene_xy(abs_xy)

    def _convert_coords(self):
        if self.coord_type == self._prev_coord_type:
            return
        if self.coord_type == 'relative':
            conv_func = self.histogram._abs2rel
        else:
            conv_func = self.histogram._rel2abs
        for m in self._markers:
            m.xy = conv_func(m.xy)
        self._prev_coord_type = self.coord_type

    def _drag_region(self):
        rect = self.histogram._hist_bars.boundingRect()
        x1, y1, x2, y2 = rect.left(), rect.bottom(), rect.right(), rect.top()
        br = self.box_radius
        y1 -= br + 1
        y2 += br + 1
        return x1, y1, x2, y2

    def _hide(self):
        if not self._shown:
            return
        self._shown = False
        self._unplot_markers()

    def _new_box_radius(self):
        box_radius = self.box_radius
        if box_radius <= 0:
            raise ValueError("box_radius must be > 0")
        if self._prev_box_radius != None:
            diff = box_radius - self._prev_box_radius
            canvas = self._scene
            box = self.marker_type == 'box'
            for marker in self._markers:
                rect = marker.scene_item.rect()
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                x -= diff
                w += 2 * diff
                if box:
                    y -= diff
                    h += 2 *diff
                marker.setRect(x, y, w, h)
        self._prev_box_radius = box_radius

    def _pick_marker(self, scene_pos):
        marker_items = {m.scene_item:m for m in self._markers}
        for i in self._scene.items(scene_pos):
            if i in marker_items:
                return marker_items[i]
        # allow for some "fuzziness"
        x, y = scene_pos.x(), scene_pos.y()
        for i in self._scene.items(QRectF(x-3, y-3, 7, 7)):
            if i in marker_items:
                return marker_items[i]
        return None

    def _plot_markers(self):
        scene = self._scene
        br = self.box_radius

        marker_type = self.marker_type
        if marker_type == 'line':
            x1, y1, x2, y2 = self._drag_region()
        for m in self._markers:
            if m.scene_item:
                continue
            x, y = self._scene_xy(m.xy)
            brush = QBrush(QColor(*[int(255 * chan + 0.5) for chan in m.rgba[:3]]))
            if marker_type == 'line':
                m.scene_item = scene.addRect(x-br, y2, 2*br, y1-y2, brush=brush)
            else:
                m.scene_item = scene.addRect(x-br, y-br, 2*br, 2*br, brush=brush)

    def _rgba(self, color_info):
        if color_info is None:
            color_info = self.new_color
        return Color(color_info).rgba

    def _unplot_markers(self, markers=None):
        if markers is None:
            markers = self._markers
        elif isinstance(markers, HistogramMarker):
            markers = [markers]
        scene = self._scene
        for m in markers:
            if m.scene_item != None:
                scene.removeItem(m.scene_item)
                m.scene_item = None
        for i in self._connector_items:
            scene.removeItem(i)
        self.connector_items = []

    def _update_connections(self):
        cxy_list = [self._scene_xy(m.xy) for m in self._markers]

        scene = self._scene
        pen = QPen(QColor(*[int(255 * chan + 0.5) for chan in Color(self.connect_color).rgba[:3]]))
        items = []
        for k in range(len(cxy_list) - 1):
            x0, y0 = cxy_list[k]
            x1, y1 = cxy_list[k+1]
            item = scene.addLine(x0, y0, x1, y1, pen=pen)
            items.append(item)

        for item in self._connector_items:
            scene.removeItem(item)
        self._connector_items = items

    def _update_marker_coordinates(self):
        br = self.box_radius

        marker_type = self.marker_type
        if marker_type == 'line':
            x1, y1, x2, y2 = self._drag_region()
        for m in self._markers:
            x, y = self._scene_xy(m.xy)
            if marker_type == 'line':
                m.scene_item.setRect(x-br, y2, 2*br, y1-y2)
            else:
                m.scene_item.setRect(x-br, y-br, 2*br, 2*br)

    def _update_plot(self):
        self._markers.sort()
        if not self._shown:
            return
        if not self.histogram._markable:
            return

        self._plot_markers()

        self._update_marker_coordinates()

        if self.connect:
            self._update_connections()

class HistogramMarker:
    """a marker on a histogram

       Should only be created (or destroyed) with methods of a
       HistogramMarkers instance.  See that class's doc string
       for details.

       The only options relevant externally are 'rgba' and 'xy'.
       'xy' should be treated as if it were read-only (use
       HistogramMarkers methods to delete/add a marker if it
       is necessary to get one to "move" programatically).  'xy'
       values will depend on HistogramMarkers' 'coord_type' option.
    """

    def __init__(self, markers, xy, rgba):
        self.markers = markers
        self.xy = xy
        self._rgba = rgba
        self.scene_item = None

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError("Cannot compare HistogramMarker to %s" % other.__class__.__name__)
        return self.xy < other.xy

    @property
    def rgba(self):
        return self._rgba

    @rgba.setter
    def rgba(self, rgba):
        if self.scene_item == None:
            return
        self.scene_item.setBrush(QBrush(QColor(*[int(255 * chan + 0.5) for chan in rgba[:3]])))
        self._rgba = rgba
        histo = self.markers.histogram
        if histo.current_marker_info()[-1] == self:
            histo._color_button.color = self.rgba
