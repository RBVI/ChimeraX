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

from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class Slider(ToolInstance):

    def __init__(self, session, tool_name, value_name, title, value_range = (1,10),
                 loop = True, pause_frames = 50, pause_when_recording = True,
                 movie_filename = 'movie.mp4', movie_framerate = 25, placement = 'side'):
        ToolInstance.__init__(self, session, tool_name)

        self.value_range = value_range
        self.loop = loop
        self.pause_frames = pause_frames
        self.pause_when_recording = pause_when_recording
        self._pause_count = 0
        self.movie_filename = movie_filename
        self.movie_framerate = movie_framerate
        self._last_shown_value = None
        
        self._play_handler = None
        self.recording = False
        self._block_update = False

        self.display_name = title	# Text shown on panel title-bar

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from Qt.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QSlider, QPushButton
        from Qt.QtGui import QPixmap, QIcon
        from Qt.QtCore import Qt
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(4)
        vl = QLabel(value_name)
        layout.addWidget(vl)
        self.value_box = vb = QSpinBox()
        vb.setRange(value_range[0], value_range[1])
        vb.valueChanged.connect(self.value_changed_cb)
        layout.addWidget(vb)
        self.slider = sl = QSlider(Qt.Horizontal)
        sl.setRange(value_range[0], value_range[1])
        sl.valueChanged.connect(self.slider_moved_cb)
        layout.addWidget(sl)
        self.play_button = pb = QPushButton()
        pb.setCheckable(True)
        pb.pressed.connect(self.play_cb)
        layout.addWidget(pb)
        self.record_button = rb = QPushButton()
        rb.setCheckable(True)
        rb.clicked.connect(self.record_cb)
        layout.addWidget(rb)
        parent.setLayout(layout)

        self.set_button_icon(play=True, record=True)

        tw.manage(placement=placement)

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def set_slider(self, position):
        self.slider.setValue(position)
        
    def value_changed_cb(self, event):
        v = self.value_box.value()
        self.slider.setValue(v)

    def slider_moved_cb(self, event):
        v = self.slider.value()
        self.value_box.setValue(v)
        self.update_value(v)

    def update_value(self, v, playing = False):
        if self._block_update:
            return
        self._last_shown_value = v
        self.change_value(v, playing = playing)

    def change_value(self, v, playing = False):
        '''Override this in derived class to do the action the slider controls.'''
        pass
    
    def play_cb(self, checked = None):
        if self.recording:
            return
        if self._play_handler:
            self.set_button_icon(play=True)
            self.stop()
        else:
            self.set_button_icon(play=False)
            self.play()

    def play(self):
        if self._play_handler is None:
            t = self.session.triggers
            self._play_handler = t.add_handler('new frame', self.next_value_cb)

    def stop(self):
        if self._play_handler:
            t = self.session.triggers
            t.remove_handler(self._play_handler)
            self._play_handler = None
            if self.recording:
                self.record_cb()

    def next_value_cb(self, *_):
        if (not self.recording) or self.pause_when_recording:
            self._pause_count += 1
            if self._pause_count >= self.pause_frames:
                self._pause_count = 0
            else:
                if self.recording:
                    # Make sure frame is drawn during pause.
                    self.session.main_view.redraw_needed = True
                return
        v = self._last_shown_value
        if v is None or v >= self.value_range[1]:
            if self.recording or not self.loop:
                self.stop()
                return
            v = self.value_range[0]
        else:
            v += 1
        while not self.valid_value(v):
            v += 1
                    
        self._block_update = True # Don't update display when slider changes
        self.value_box.setValue(v)
        self._block_update = False
        self.update_value(v, playing = True)

    def valid_value(self, v):
        return True
    
    def set_button_icon(self, play = None, record = None):
        from chimerax.ui.icons import get_qt_icon
        if play is not None:
            pi = get_qt_icon('play' if play else 'pause')
            pb = self.play_button
            pb.setIcon(pi)

        if record is not None:
            pi = get_qt_icon('record' if record else 'stop')
            rb = self.record_button
            rb.setIcon(pi)

    def record_cb(self, event=None):
        from chimerax.core.commands import run
        ses = self.session
        if not self.recording:
            self.set_button_icon(record=False)
            self.recording = True
            run(ses, 'movie record')
            self.play()
        else:
            self.set_button_icon(record=True)
            self.recording = False
            self.stop()
            run(ses, 'movie encode ~/Desktop/%s framerate %.1f'
                % (self.movie_filename, self.movie_framerate))

    # Override ToolInstance method
    def delete(self):
        t = self.session.triggers
        if self._play_handler:
            t.remove_handler(self._play_handler)
            self._play_handler = None
        super().delete()

    
# -----------------------------------------------------------------------------
#
class LogSlider:
    '''
    Slider with floating point values and logarithmic scale.
    '''
    def __init__(self, parent, label = '', range = (1,1000), decimal_step = True,
                 value_change_cb = None, release_cb = None):
        self._range = range
        self._int_range = (0,10000)  # Qt 5.15 only has integer value sliders
        self._value_changed = value_change_cb
        
        from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel, QDoubleSpinBox, QSlider
        self.frame = f = QFrame(parent)
        layout = QHBoxLayout(f)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(4)

        if label:
            vl = QLabel(label, f)
            layout.addWidget(vl)

        self._entry = se = QDoubleSpinBox(f)
        se.setRange(range[0], range[1])
        if decimal_step:
            se.setStepType(se.AdaptiveDecimalStepType)
        se.valueChanged.connect(self._entry_changed_cb)
        layout.addWidget(se)

        from Qt.QtCore import Qt
        self._slider = sl = QSlider(Qt.Horizontal, f)
        sl.setRange(self._int_range[0], self._int_range[1])
        sl.valueChanged.connect(self._slider_moved_cb)
        if release_cb:
            sl.sliderReleased.connect(release_cb)
        layout.addWidget(sl)

    # ---------------------------------------------------------------------------
    #
    def set_range(self, value_min, value_max):
        self._range = (value_min, value_max)
        self._entry.setRange(value_min, value_max)

    # ---------------------------------------------------------------------------
    #
    def set_precision(self, precision):
        '''Decimals to right of decimal point shown in slider entry field.'''
        self._entry.setDecimals(precision)

    # ---------------------------------------------------------------------------
    #
    def get_value(self):
        return self._entry.value()
    def set_value(self, value):
        self._entry.setValue(value)
    value = property(get_value, set_value)
  
    # ---------------------------------------------------------------------------
    #
    def _entry_changed_cb(self):
        value = self._entry.value()
        if self._slider_int_to_value(self._slider.value()) != value:
            i = self._value_to_slider_int(value)
            self._slider.setValue(i)
        if self._value_changed:
            self._value_changed(value, slider_down = self._slider.isSliderDown())
    
    # ---------------------------------------------------------------------------
    #
    def _slider_moved_cb(self):
        i = self._slider.value()
        if self._value_to_slider_int(self._entry.value()) != i:
            value = self._slider_int_to_value(i)
            self._entry.setValue(value)

    # ---------------------------------------------------------------------------
    # Map desired slider value range to integer slider position.
    #
    def _value_to_slider_int(self, value):
        vmin, vmax = self._range
        if value <= vmin:
            f = 0
        elif value >= vmax:
            f = 1
        else:
            from math import log10
            f = log10(value/vmin) / log10(vmax/vmin)

        imin, imax = self._int_range
        i = imin + f * (imax-imin)
        i = int(i + 0.5)
        return i

    # ---------------------------------------------------------------------------
    # Map integer slider position to value in desired floating point value range.
    #
    def _slider_int_to_value(self, i):
        imin, imax = self._int_range
        f = (i - imin) / (imax - imin)
        vmin, vmax = self._range
        from math import log10, pow
        value = vmin * pow(10, f * log10(vmax/vmin))
        return value
