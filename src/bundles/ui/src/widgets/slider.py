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
        self.value_name = value_name
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
        tool_tip = "Number of frames to show before advancing to next image"
        rl = QLabel()
        from chimerax.ui.icons import get_icon_path
        rl.setPixmap(QPixmap(get_icon_path("snail")).scaledToHeight(19))
        rl.setToolTip(tool_tip)
        layout.addWidget(rl)
        self.rate_box = rb = QSpinBox()
        rb.setMinimum(1)
        rb.setValue(pause_frames)
        rb.valueChanged.connect(self.rate_changed_cb)
        rb.setToolTip(tool_tip)
        layout.addWidget(rb)
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

    def rate_changed_cb(self, event):
        self.pause_frames = self.rate_box.value()

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

from Qt.QtWidgets import QWidget
from Qt.QtCore import Qt, Signal

class FloatSlider(QWidget):

    valueChanged = Signal(float)

    def __init__(self, minimum, maximum, step, decimal_places, continuous_callback, *,
            ignore_wheel_event=False, **kw):
        from Qt.QtWidgets import QGridLayout, QSlider, QLabel, QSizePolicy
        super().__init__()
        layout = QGridLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.setLayout(layout)
        if ignore_wheel_event:
            class Slider(QSlider):
                def wheelEvent(self, event):
                    event.ignore()
        else:
            Slider = QSlider
        self._slider = Slider(**kw)
        self._slider.setOrientation(Qt.Horizontal)
        self._slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._minimum = minimum
        self._maximum = maximum
        self._continuous = continuous_callback
        # slider are only integer, so have to do conversions
        self._slider.setMinimum(0)
        self._slider.setMaximum(5000)
        int_step = max(1, int(5000 * step / (maximum - minimum)))
        self._slider.setSingleStep(int_step)
        layout.addWidget(self._slider, 0, 0, 1, 3)
        # for word-wrapped text, set the alignment within the label widget itself (instead of the layout)
        # so that the label is given the full width of the layout to work with, otherwise you get unneeded
        # line wrapping
        from chimerax.ui import shrink_font
        self._left_text = QLabel()
        self._left_text.setWordWrap(True)
        self._left_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        shrink_font(self._left_text)
        layout.addWidget(self._left_text, 1, 0)
        self._value_text = QLabel()
        self._value_text.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        layout.addWidget(self._value_text, 1, 1, alignment=Qt.AlignCenter | Qt.AlignTop)
        self._right_text = QLabel()
        self._right_text.setWordWrap(True)
        self._right_text.setAlignment(Qt.AlignRight | Qt.AlignTop)
        shrink_font(self._right_text)
        layout.addWidget(self._right_text, 1, 2)
        self._decimal_places = decimal_places
        self._slider.valueChanged.connect(self._slider_value_changed)
        self._slider.sliderReleased.connect(self._slider_released)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

    def blockSignals(self, *args):
        self._slider.blockSignals(*args)

    def set_left_text(self, text):
        self._left_text.setText(text)

    def set_right_text(self, text):
        self._right_text.setText(text)

    def set_text(self, text):
        self._value_text.setText(text)

    def setValue(self, float_val):
        fract = (float_val - self._minimum) / (self._maximum - self._minimum)
        self._slider.setValue(int(5000 * fract + 0.5))
        if self._slider.signalsBlocked():
            self.set_text(self._value_to_text(float_val))

    def special_value_shown(self):
        # effectively always False, unlike a SpinBox, the option's value is always accurate
        return False

    def value(self):
        return self._int_val_to_float(self._slider.value())

    def _int_val_to_float(self, int_val):
        fract = int_val / 5000
        return (1-fract) * self._minimum + fract * self._maximum

    def _slider_released(self):
        if not self._continuous:
            self.valueChanged.emit(self.value())

    def _slider_value_changed(self, int_val):
        float_val = self._int_val_to_float(int_val)
        self._value_text.setText(self._value_to_text(float_val))
        if self._continuous:
            self.valueChanged.emit(float_val)

    def _value_to_text(self, val):
        std_text = "%.*f" % (self._decimal_places, val)
        # for %.Xg, X is the total number of significant digits, both before and after the decimal
        return "%.*g" % (len(std_text)-1, val)
