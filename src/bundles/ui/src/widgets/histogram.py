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

from PyQt5.QtWidgets import QWidget, QLabel, QStackedWidget, QGraphicsView, QGraphicsScene, QFrame
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLineEdit
from PyQt5.QtCore import QSize, Qt, QTimer

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
            **kw):

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

        # Create the add/delete marker help
        if show_marker_help and layout != 'below':
            self.marker_help = QLabel("Ctrl-click on histogram to add or delete thresholds")
        else:
            self.marker_help = None

        overall_layout = QVBoxLayout()
        self.setLayout(overall_layout)

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
        from chimerax.mousemodes import mod_key_info
        self._hist_scene.mousePressEvent = lambda event: self._add_or_delete_marker_cb(event) \
            if event.modifiers() & mod_key_info("control")[0] else self._select_marker_cb(event)
        self._hist_scene.mouseReleaseEvent = self._button_up_cb
        self._resize_timer = QTimer()
        self._resize_timer.timeout.connect(self._redraw_cb)
        self._resize_timer.start(1000 * redraw_delay)
        self._resize_timer.stop()
        self._data_widgets.addWidget(self._hist_view)

        # Create the histogram replacement label
        self._no_histogram_label = QLabel()
        self._data_widgets.addWidget(self._no_histogram_label)
        overall_layout.addWidget(self._data_widgets, stretch=1)

        # Show the histogram or the no-data label
        self.data_source = data_source

        # Create range label(s)
        self._widget_layout = QHBoxLayout()
        if min_label or max_label:
            min_max_layout = QHBoxLayout()
            if min_label:
                self._min_label = QLabel()
                min_max_layout.addWidget(self._min_label, alignment=Qt.AlignLeft & Qt.AlignTop)
            else:
                self._min_label = None

            if max_label:
                self._max_label = QLabel()
                min_max_layout.addWidget(self._max_label, alignment=Qt.AlignRight & Qt.AlignTop)
            else:
                self._max_label = None
            overall_layout.addLayout(min_max_layout)
        else:
            self._range_label = QLabel()
            if layout == 'below':
                self._widget_layout.addWidget(self._range_label))
            else:
                lab = QLabel("Range")
                if layout == 'single':
                    self._widget_layout.addWidget(lab, alignment=Qt.AlignRight))
                    self._widget_layout.addWidget(self._range_label, alignment=Qt.AlignLeft))
                else: # layout == 'top'
                    range_layout = QVBoxLayout()
                    range_layout.addWidget(lab, alignment=Qt.AlignBottom)
                    range_layout.addWidget(self._range_label, alignment=Qt.AlignTop)
                    self._widget_layout.addLayout(range_layout)
        overall_layout.addLayout(self._widget_layout)

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
                self._widget_layout.addWidget(lab, alignment=Qt.AlignRight))
                self._widget_layout.addWidget(self._value_entry, alignment=Qt.AlignLeft))
            else:
                value_layout = QVBoxLayout()
                value_layout.addWidget(lab, alignment=Qt.AlignBottom)
                value_layout.addWidget(self._value_entry, alignment=Qt.AlignTop)
                self._widget_layout.addLayout(value_layout)

        # Create color button widget
        from .color_button import ColorButton
        self._color_button = cb = ColorButton()
        self._color_button.color+changed.connect(self._color_button_cb)
        self._color_button_label = cbl = QLabel("Color")
        if layout == 'below':
            self._widget_layout.addWidget(self._color_button)
        else:
            if layout == 'single':
                self._widget_layout.addWidget(cbl, alignment=Qt.AlignRight))
                self._widget_layout.addWidget(cb, alignment=Qt.AlignLeft))
            else:
                color_layout = QVBoxLayout()
                color_layout.addWidget(cbl, alignment=Qt.AlignBottom)
                color_layout.addWidget(cb, alignment=Qt.AlignTop)
                self._widget_layout.addLayout(color_layout)
        self._color_button_shown = True
        self.color_button = color_button

    def activate(self, markers):
        """Make the given set of markers the currently active set
        
           Any previously-active set will be hidden.
        """

        if markers is not None and markers not in self._markers:
            raise ValueError, "activate() called with bad value"
        if markers == self._active_markers:
            return
        if self._active_markers is not None:
            self._active_markers._hide()
        elif self.layout != 'below' and self._show_marker_help:
            self.marker_help.grid(row=2, column=2, columnspan=2)
        self._active_markers = markers
        if self._active_markers is not None:
            self._active_markers._show()
            self._setSelMarker(self._active_markers._selMarker)
        else:
            if self.layout != 'below' and self._show_marker_help:
                self.marker_help.grid_forget()
            if self['select_callback']:
                if self._prev_marker is not None:
                    self['select_callback'](
                        self._prev_markers,
                        self._prev_marker, None, None)
                self._prev_markers = None
                self._prev_marker = None

    def add_custom_widget(self, widget, left_side=True):
        #TODO

    def addmarkers(self, activate=True, **kw):
        """Create and return a new set of markers.

           If 'activate' is true, the new set will also become
           the active set.  Other keyword arguments will be passed
           to the HistogramMarkers constructor.
        """
        if self._markers:
            newName = "markers" + str(int(
                    self._markers[-1]._name[7:]) + 1)
        else:
            newName = "markers1"
        kw['histogram'] = self
        markers = self.createcomponent(newName, (), 'Markers', 
                        HistogramMarkers, (), **kw)
        markers._name = newName
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

    def currentmarkerinfo(self):
        """Identify the marker currently selected by the user.
           
           Returns a HistogramMarkers instance and a marker.
           The instance will be None if no marker set has been
           activated.  The marker will be None if no marker has
           been selected by the user.
        """
        if self._active_markers is None:
            return None, None
        return self._active_markers, self._active_markers._selMarker

    @property
    def data_source(self):
        return self._data_source

    @data_source.setter
    def data_source(self, data_source):
        self._data_source = data_source
        self._new_data()

    def deletemarkers(self, markers):
        """Delete the given set of markers.

           If the markers were active, there will be no active set
           of markers afterward.
        """
        if markers not in self._markers:
            raise ValueError, "Bad value for delete()"
        if markers == self._active_markers:
            self.activate(None)
        self._markers.remove(markers)
        self.destroycomponent(markers._name)

    @property
    def layout(self):
        return self._layout

    @property
    def redraw_delay(self):
        return self._resize_timer.interval() / 1000.0

    @redraw_delay.setter(self, secs):
        self._resize_time.setInterval(secs * 1000)

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        if self._scaling != scaling:
            self._scaling = scaling
            self._redraw_cb()

    def sceneData(self):
        info = {
            'version': 1,
            'draw_min': self._draw_min,
            'draw_max': self._draw_max,
            'markers': [markers.sceneData() for markers in self._markers],
        }
        if self._active_markers is None:
            info['active markers'] = None
        else:
            info['active markers'] = self._markers.index(self._active_markers)
        if self['color_button']:
            info['color well'] = self._color_button.rgba
        else:
            info['color well'] = None
        return info

    def sceneRestore(self, data):
        self._draw_min = data['draw_min']
        self._draw_max = data['draw_max']
        if data['color well'] is not None:
            self._color_button.showColor(data['color well'], doCallback=False)
        if len(data['markers']) != len(self._markers):
            # don't know how to deal with this situation
            return
        for markers, markersData in zip(self._markers, data['markers']):
            markers.sceneRestore(markersData)
        if data['active markers'] is not None:
            self.activate(self._markers[data['active markers']])
            self._setSelMarker(self._active_markers._selMarker)

    @property
    def value_width(self):
        return self._value_width

    @value_width.setter(self, vw):
        self._value_width = vw
        ve = self._value_entry
        fm = ve.fontMetrics()
        tm = ve.textMargins()
        cm = ve.contentsMargins()
        w = 4*fm.width('w') + tm.left() + tm.right() + cm.left() + cm.right() + 8
        ve.setMaximumWidth(w)

    def _abs2rel(self, absXY):
        x, y = absXY
        relX = (x - self._min_val) / float(self._max_val - self._min_val)
        relY = y / float(self._ymax)
        return relX, relY

    def _absXY(self, canvasXY):
        canvasX, canvasY = canvasXY
        dy = min(max(self._bottom - canvasY, 0), self._hist_height - 1)
        if self.scaling == 'logarithmic':
            exp = dy / float(self._hist_height - 1)
            absY = (self._maxHeight + 1) ** exp - 1
        else:
            absY = self._maxHeight*dy / float(self._hist_height-1)

        cx = canvasX - self._border
        numBins = len(self._bins)
        if numBins == self._hist_width:
            fract = cx / (numBins - 1)
            absX = self._min_val + fract * (self._max_val - self._min_val)
        elif numBins == 2:
            absX = self._min_val + (self._max_val - self._min_val) * (
                2 * cx / self._hist_width - 0.5)
        else:
            extra = self._hist_width / (2.0*(numBins-1))
            absX = self._min_val + (self._max_val - self._min_val) * (
                cx - extra) / (self._hist_width - 2.0 * extra)
        absX = max(self._min_val, absX)
        absX = min(self._max_val, absX)
        return absX, absY

    def _add_or_delete_marker_cb(self, event=None):
        if self._active_markers is None:
            return

        marker = self._active_markers._pickMarker(event.x, event.y)

        if marker is None:
            maxMarks = self._active_markers['maxmarks']
            if maxMarks is not None \
            and len(self._active_markers) >= maxMarks:
                if self.status_line:
                    self.status_line("Maximum of %d"
                        " markers\n" % maxMarks)
                return
            xy = self._absXY((event.x, event.y))
            if self._active_markers['coordtype'] == 'relative':
                xy = self._abs2rel(xy)
            selMarker = self._active_markers._selMarker
            if selMarker:
                color = selMarker['rgba']
            else:
                color = self._active_markers['newcolor']
            marker = self._active_markers.append((xy, color))
            self._setSelMarker(marker, dragStart=event)
        else:
            minMarks = self._active_markers['minmarks']
            if minMarks is not None \
            and len(self._active_markers) <= minMarks:
                if self.status_line:
                    self.status_line("Minimum of %d"
                        " markers\n" % minMarks)
                return
            self._active_markers.remove(marker)
            self._setSelMarker(None)
            
    def _button_up_cb(self, event=None):
        if self._drag_marker:
            self.canvas.bind("<Button1-Motion>", "")
            self._drag_marker = None
            if self._active_markers['movecallback']:
                self._active_markers['movecallback']('end')

    def _canvasXY(self, absXY):
        # minimum is in the _center_ of the first bin,
        # likewise, maximum is in the center of the last bin

        absX, absY = absXY

        absY = max(0, absY)
        absY = min(self._maxHeight, absY)
        if self.scaling == 'logarithmic':
            import math
            absY = math.log(absY+1)
        canvasY = self._bottom - (self._hist_height - 1) * (
                        absY / self._maxHeight)

        absX = max(self._min_val, absX)
        absX = min(self._max_val, absX)
        numBins = len(self._bins)
        if numBins == self._hist_width:
            binWidth = (self._max_val - self._min_val) / float(
                                numBins - 1)
            leftEdge = self._min_val - 0.5 * binWidth
            canvasX = int((absX - leftEdge) / binWidth)
        else:
            # histogram is effectively one bin wider
            # (two half-empty bins on each end)
            if numBins == 1:
                canvasX = 0.5 * (self._hist_width - 1)
            else:
                extra = (self._max_val - self._min_val) / (2.0*(numBins-1))
                effMinVal = self._min_val - extra
                effMaxVal = self._max_val + extra
                effRange = float(effMaxVal - effMinVal)
                canvasX = (self._hist_width - 1) * (absX - effMinVal) \
                                / effRange
        return self._border + canvasX, canvasY

    def _color_button_cb(self, rgba):
        m = self._active_markers._selMarker
        if not m:
            if self.status_line:
                self.status_line("No marker selected")
            return
        if rgba is None:
            if self.status_line:
                self.status_line(
                    "Cannot set marker color to None")
            # can't reset the color in the middle of the callback
            self.interior().after_idle(lambda rgba=m['rgba']:
                    self._color_button.showColor(rgba,
                    doCallback=False))
            return
        m['rgba'] = rgba

    def _marker2abs(self, marker):
        if self._active_markers['coordtype'] == 'absolute':
            return marker['xy']
        else:
            return self._rel2abs(marker['xy'])

    def _moveCurMarker(self, x, yy=None):
        #
        # Don't allow dragging out of the canvas box.
        #
        m = self._active_markers._selMarker
        if x < self._min_val:
            x = self._min_val
        elif x > self._max_val:
            x = self._max_val
        if yy is None:
            y = m['xy'][1]
        else:
            y = yy
            if y < 0:
                y = 0
            elif y > self._ymax:
                y = self._ymax

        if self._active_markers['coordtype'] == 'absolute':
            m['xy'] = (x, y)
        else:
            m['xy'] = self._abs2rel((x,y))
        if yy is None:
            m['xy'] = (m['xy'][0], y)

        self._setValueEntry(x)

        self._active_markers._updatePlot()

        if self._active_markers['movecallback']:
            self._active_markers['movecallback'](m)

    def _moveMarkerCB(self, event):
        mouseXY = self._absXY((event.x, event.y))
        dx = mouseXY[0] - self._lastMouseXY[0]
        dy = mouseXY[1] - self._lastMouseXY[1]
        self._lastMouseXY = mouseXY

        shiftMask = 1
        if event.state & shiftMask:
            dx *= .1
            dy *= .1

        m = self._drag_marker
        mxy = self._marker2abs(m)
        x, y = mxy[0] + dx, mxy[1] + dy

        self._moveCurMarker(x, y)

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
                self.marker_help.setHidden(True)
            self._widget_layout.setHidden(True)
        else:
            self._data_widgets.setCurrentWidget(self._hist_view)
            if self.layout != 'below' and self._show_marker_help:
                self.marker_help.setHidden(False)
            self._widget_layout.setHidden(False)
        self._draw_min = self._draw_max = None
        self._redraw_cb()

    def _redraw(self, event=None):
        self._markable = False
        self._redraw_timer.start()

    def _redraw_cb(self):
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

        for bar in self._hist_bars.childItems():
            self._hist_scene.removeItem(bar)

        self._ymax = max(self._bins)
        if self.scaling == 'logarithmic':
            from numpy import array, log, float32
            self._bins = array(self._bins, float32)
            self._bins += 1.0
            log(self._bins, self._bins)

        #TODO
        maxHeight = max(self._bins)
        self._maxHeight = maxHeight
        hScale = float(hist_height - 1) / maxHeight
        bottom = hist_height + border - 1
        self._bottom = bottom

        numBins = len(self._bins)
        if numBins == hist_width:
            for b, n in enumerate(self._bins):
                x = border + b
                h = int(hScale * n)
                id = canvas.create_line(x, bottom, x, bottom-h,
                                tags=('bar',))
                canvas.tag_lower(id)  # keep bars below markers
        else:
            xScale = (hist_width - 1) / float(numBins)
            for b, n in enumerate(self._bins):
                x1 = border + b * xScale
                x2 = border + (b+1) * xScale
                h = int(hScale * n)
                id = canvas.create_rectangle(x1, bottom,
                        x2, bottom-h, tags=('bar',))
                canvas.tag_lower(id)  # keep bars below markers
        self._markable = True
        if self._active_markers is not None:
            self._active_markers._updatePlot()
            marker = self._active_markers._selMarker
            if marker:
                self._setValueEntry(self._marker2abs(marker)[0])

    def _rel2abs(self, relXY):
        x, y = relXY
        absX = self._min_val * (1-x) + x * self._max_val
        absY = y * self._ymax
        return absX, absY

    def _select_marker_cb(self, event=None):
        if self._active_markers is not None:
            marker = self._active_markers._pickMarker(event.x,
                                event.y)
            self._setSelMarker(marker, dragStart=event)
            if marker is not None:
                return
        # show value where histogram clicked
        self._setValueEntry(self._absXY((event.x, 0))[0])
    
    def _setSelMarker(self, marker, dragStart=None):
        self._active_markers._selMarker = marker
        if not marker:
            self._color_button.showColor(None, doCallback=False)
            self._setValueEntry("")
            self._value_entry.component('entry').config(
                            state='disabled')
        else:
            self._color_button.showColor(marker['rgba'],
                            doCallback=False)
            self._value_entry.component('entry').config(
                            state='normal')
            self._setValueEntry(self._marker2abs(marker)[0])
        if self['select_callback']:
            if marker is not None or self._prev_marker is not None:
                self['select_callback'](self._prev_markers,
                    self._prev_marker, self._active_markers,
                    marker)
            self._prev_markers = self._active_markers
            self._prev_marker = marker
        if not dragStart:
            return
        self._drag_marker = marker
        if not marker:
            return

        self._lastMouseXY = self._absXY((dragStart.x, dragStart.y))
        self._motionHandler = self.canvas.bind("<Button1-Motion>",
                            self._moveMarkerCB)
        if self._active_markers['movecallback']:
            self._active_markers['movecallback']('start')
    
    def _set_value_cb(self):
        try:
            v = eval(self._value_entry.getvalue())
        except:
            raise ValueError, "Invalid histogram value"
        if type(self._min_val) != type(v):
            v = type(self._min_val)(v)
        if v < self._min_val:
            self._draw_min = v
            self._redraw_cb()
        elif v > self._max_val:
            self._draw_max = v
            self._redraw_cb()
        self._moveCurMarker(v)

    def _setValueEntry(self, val):
        if isinstance(val, basestring):
            self._value_entry.setvalue(val)
            return
        if isinstance(self._min_val, int):
            val = int(val + 0.5)
        self._value_entry.setvalue("%g" % val)

    def _str_val(self, val):
        if isinstance(val, (int, bool)):
            return str(val)
        return "%g" % val

Pmw.forwardmethods(MarkedHistogram, Tkinter.Canvas, 'canvas')

from CGLtk.color import rgba2tk
class HistogramMarkers(Pmw.MegaArchetype):
    """Color-designating markers on a histogram

       Instances should only created via the addmarkers() method of
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
        an RGBA value
        None (use the 'newmarker' color)
        a color name
        an instance that has either an 'rgba' attribute or
            an argless 'rgba' method (e.g. a MaterialColor)

       The MarkedHistogram and HistogramMarker doc strings should be
       examined for further info on usage.

       Options are:

        boxradius -- the radius in pixels of boxes drawn when the
            markertype is 'box'
            default: 2

           connect -- [init option] whether markers should be
            connected left-to-right with lines.  Typically
            used only when the markertype is 'box'.
            default: False

        connectcolor -- [init option] the color used to draw
            lines connecting markers ('connect' must be True)
            default: yellow

        coordtype -- either 'relative' or 'absolute'.  If the former,
            then the 'xy' option of contained HistgramMarkers are
            in the range 0-1 and indicate positioning relative to
            left/right and bottom/top of the histogram.  If the
            latter, then the x of 'xy' indicates a histogram
            bin by value and a height by count.
            default: absolute

        histogram -- [init option provided automatically by
            MarkedHistogram.addmarkers()] the MarkedHistogram
            instance

        markertype -- [init option] the type of markers to use, 
            either 'line' (vertical bars) or 'box' (squares).
            default: line

        maxmarks/minmarks -- the maximum/minimum amount of marks the
            user is allowed to place on the histogram.  A value of
            None indicates no limit.  Can always be exceeded
            programmatically.
            default: None

        movecallback -- [init option] function to call when the user
            moves a marker.  The function receives a value of
            'start' at the beginning of a move and 'end' at the
            end.  During the move the value is the marker being
            moved.
            default: None

        newcolor -- the default color assigned to newly-created
            markers.
            default: yellow
    """

    def __init__(self, parent=None, **kw):
    
        # Define the megawidget options
        optiondefs = (
            ('boxradius',    2,        self._newBoxRadius),
            ('connect',    False,        Pmw.INITOPT),
            ('connectcolor','yellow',    Pmw.INITOPT),
            ('coordtype',    'absolute',    self._convertCoords),
            ('histogram',    None,        Pmw.INITOPT),
            ('markertype',    'line',        Pmw.INITOPT),
            ('maxmarks',    None,        None),
            ('minmarks',    None,        None),
            ('movecallback',None,        Pmw.INITOPT),
            ('newcolor',    'yellow',    None),
        )
        self.defineoptions(kw, optiondefs, dynamicGroups=['Marker'])

        # Initialise base class (after defining options)
        Pmw.MegaArchetype.__init__(self, parent)

        # Check keywords and initialise options
        self._shown = False
        self._selMarker = None
        self._prevBoxRadius = None
        self.markers = []
        self.connectIds = []
        self._prevCoordType = self['coordtype']
        self.initialiseoptions(HistogramMarkers)

        # values derived from options
        self.markerFunc = lambda v: HistogramMarker(markers=self,
                    xy=v[0], rgba=self._rgba(v[1]))
        # convenience
        self._canvas = self['histogram'].component('canvas')

    def append(self, val):
        marker = self.markerFunc(val)
        self.markers.append(marker)
        self._updatePlot()
        return marker

    def __delitem__(self, i):
        if isinstance(i, basestring):
            return Pmw.MegaArchetype.__delitem__(self, i)
        del self.markers[i]
        self._updatePlot()

    def destroy(self):
        self._unplotMarkers()
        Pmw.MegaArchetype.destroy(self)

    def extend(self, vals):
        markers = map(self.markerFunc, vals)
        self.markers.extend(markers)
        self._updatePlot()
        return markers

    def __getitem__(self, i):
        if isinstance(i, basestring):
            return Pmw.MegaArchetype.__getitem__(self, i)
        return self.markers[i]

    def index(self, marker):
        return self.markers.index(marker)

    def insert(self, i, val):
        marker = self.markerFunc(val)
        self.markers.insert(i, marker)
        self._updatePlot()
        return marker

    def __iter__(self):
        return self.markers.__iter__()

    def __len__(self):
        return len(self.markers)

    def pop(self):
        ret = self.markers.pop()
        if ret == self._selMarker:
            self._selMarker = None
        self._unplotMarkers(ret)
        self._updatePlot()
        return ret

    def remove(self, marker):
        self.markers.remove(marker)
        if marker is self._selMarker:
            self._selMarker = None
        self._unplotMarkers(marker)
        self._updatePlot()

    def sceneData(self):
        info = {
            'marker data': [(m['xy'], m['rgba']) for m in self.markers],
            'shown': self._shown
        }
        if self._selMarker:
            info['sel marker'] = self.markers.index(self._selMarker)
        else:
            info['sel marker'] = None
        return info

    def sceneRestore(self, data):
        self._unplotMarkers()
        self._shown = data['shown']
        self.markers[:] = []
        self.extend(data['marker data'])
        if data['sel marker'] is None:
            self._selMarker = None
        else:
            self._selMarker = self.markers[data['sel marker']]

    def __setitem__(self, i, val):
        if isinstance(i, basestring):
            return Pmw.MegaArchetype.__setitem__(self, i, val)
        if isinstance(i, slice):
            newMarkers = map(self.markerFunc, val)
            selReplaced = self._selMarker in self.markers[i]
        else:
            newMarkers = self.markerFunc(val)
            selReplaced = self._selMarker is self.markers[i]
        if selReplaced:
            self._selMarker = None
        self._unplotMarkers(self.markers[i])
        self.markers[i] = newMarkers
        self._updatePlot()
    
    def sort(self, sortFunc=None):
        self.markers.sort(sortFunc)

    def _canvasXY(self, xy):
        if self['coordtype'] == 'relative':
            absXY = self['histogram']._rel2abs(xy)
        else:
            absXY = xy
        return self['histogram']._canvasXY(absXY)

    def _convertCoords(self):
        if self['coordtype'] == self._prevCoordType:
            return
        if self['coordtype'] == 'relative':
            convFunc = self['histogram']._abs2rel
        else:
            convFunc = self['histogram']._rel2abs
        for m in self.markers:
            m['xy'] = convFunc(m['xy'])
        self._prevCoordType = self['coordtype']

    def _dragRegion(self):
        x1, y1, x2, y2 = self._canvas.bbox('bar')
        br = self['boxradius']
        y1 += br + 1
        y2 -= br + 1
        return x1, y1, x2, y2

    def _hide(self):
        if not self._shown:
            return
        self._shown = False
        self._unplotMarkers()

    def _newBoxRadius(self):
        boxRadius = self['boxradius']
        if boxRadius <= 0:
            raise ValueError, "boxradius must be > 0"
        if self._prevBoxRadius != None:
            diff = boxRadius - self._prevBoxRadius
            canvas = self._canvas
            box = self['markertype'] == 'box'
            for marker in self.markers:
                x0, y0, x1, y1 = canvas.coords(marker['id'])
                x0 += diff
                x1 += diff
                if box:
                    y0 += diff
                    y1 += diff
                canvas.coords(marker['id'], x0, y0, x1, y1)
        self._prevBoxRadius = boxRadius

    def _pickMarker(self, cx, cy):
        close = self._canvas.find('closest', cx, cy, 3)
        for c in close:
            for m in self.markers:
                if m['id'] == c:
                    return m
        return None

    def _plotMarkers(self):
        canvas = self._canvas
        br = self['boxradius']

        markerType = self['markertype']
        if markerType == 'line':
            x1, y1, x2, y2 = self._dragRegion()
        for m in self.markers:
            if m['id'] != None:
                continue
            x, y = self._canvasXY(m['xy'])
            color = rgba2tk(m['rgba'])
            if markerType == 'line':
                m['id'] = canvas.create_rectangle(x-br, y1,
                            x+br, y2, fill=color)
            else:
                m['id'] = canvas.create_rectangle(x-br, y-br,
                            x+br, y+br, fill=color)

    def _rgba(self, colorInfo):
        if colorInfo is None:
            colorInfo = self['newcolor']
        if isinstance(colorInfo, basestring):
            from chimera.colorTable import getColorByName
            colorInfo = getColorByName(colorInfo)
        if hasattr(colorInfo, 'rgba'):
            if callable(colorInfo.rgba):
                return colorInfo.rgba()
            return colorInfo.rgba
        return colorInfo

    def _show(self):
        if self._shown:
            return
        self._shown = True
        self._updatePlot()

    def _unplotMarkers(self, markers=None):
        if markers is None:
            markers = self.markers
        elif isinstance(markers, HistogramMarker):
            markers = [markers]
        canvas = self._canvas
        for m in markers:
            if m['id'] != None:
                canvas.delete(m['id'])
                m['id'] = None
        for i in self.connectIds:
            canvas.delete(i)
        self.connect_ids = []

    def _updateConnections(self):
        cxy_list = map(lambda m: self._canvasXY(m['xy']), self.markers)

        canvas = self._canvas
        color = rgba2tk(self._rgba(self['connectcolor']))
        ids = []
        for k in range(len(cxy_list) - 1):
            x0, y0 = cxy_list[k]
            x1, y1 = cxy_list[k+1]
            id = canvas.create_line(x0, y0, x1, y1, fill=color)
            ids.append(id)

        for id in self.connectIds:
            c.delete(id)

        self.connectIds = ids

        for m in self.markers:
            canvas.tag_raise(m['id'])

    def _updateMarkerCoordinates(self):
        canvas = self._canvas
        br = self['boxradius']

        markerType = self['markertype']
        if markerType == 'line':
            x1, y1, x2, y2 = self._dragRegion()
        for m in self.markers:
            x, y = self._canvasXY(m['xy'])
            if markerType == 'line':
                canvas.coords(m['id'], x-br, y1, x+br, y2)
            else:
                canvas.coords(m['id'], x-br, y-br, x+br, y+br)

    def _updatePlot(self):
        self.markers.sort()
        if not self._shown:
            return
        if not self['histogram']._markable:
            return

        self._plotMarkers()

        self._updateMarkerCoordinates()

        if self['connect']:
            self._updateConnections()

class HistogramMarker(Pmw.MegaArchetype):
    """a marker on a histogram
       
       Should only be created (or destroyed) with methods of a
       HistogramMarkers instance.  See that class's doc string 
       for details.

       The only options relevant externally are 'rgba' and 'xy'.
       'xy' should be treated as if it were read-only (use 
       HistogramMarkers methods to delete/add a marker if it
       is necessary to get one to "move" programatically).  'xy'
       values will depend on HistogramMarkers' 'coordtype' option.
    """

    def __init__(self, parent=None, **kw):
    
        # Define the megawidget options
        optiondefs = (
            ('id',        None,        None),
            ('markers',    None,        Pmw.INITOPT),
            ('rgba',    (1,1,0,0),    self._setRgba),
            ('xy',        (0.5, 0.5),    None)
        )
        self.defineoptions(kw, optiondefs)

        # Initialise base class (after defining options)
        Pmw.MegaArchetype.__init__(self, parent)

        # Check keywords and initialise options
        self.initialiseoptions(HistogramMarker)

        # convenience
        self._canvas = self['markers']['histogram'].component('canvas')

    def __cmp__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return cmp(self['xy'], other['xy'])

    def _setRgba(self):
        if self['id'] == None:
            return
        self._canvas.itemconfigure(self['id'],
                        fill=rgba2tk(self['rgba']))
        histo = self['markers']['histogram']
        if histo.currentmarkerinfo()[-1] == self:
            histo._color_button.showColor(self['rgba'], doCallback=False)
