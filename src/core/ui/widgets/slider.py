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
        self._recording = False
        self._block_update = False

        self.display_name = title	# Text shown on panel title-bar

        from chimerax.core.ui.gui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QSlider, QPushButton
        from PyQt5.QtGui import QPixmap, QIcon
        from PyQt5.QtCore import Qt
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
        pb.clicked.connect(self.play_cb)
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

    def play_cb(self, event):
        if self._recording:
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
            if self._recording:
                self.record_cb()

    def next_value_cb(self, *_):
        if (not self._recording) or self.pause_when_recording:
            self._pause_count += 1
            if self._pause_count >= self.pause_frames:
                self._pause_count = 0
            else:
                if self._recording:
                    # Make sure frame is drawn during pause.
                    self.session.main_view.redraw_needed = True
                return
        v = self._last_shown_value
        if v >= self.value_range[1]:
            if self._recording or not self.loop:
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
        from os.path import dirname, join
        dir = dirname(__file__)
        if play is not None:
            bitmap_path = (join(dir, 'icons', 'play.png' if play else 'pause.png'))
            pb = self.play_button
            from PyQt5.QtGui import QPixmap, QIcon
            ppix = QPixmap(bitmap_path)
            pi = QIcon(ppix)
            pb.setIcon(pi)
                
        if record is not None:
            bitmap_path = (join(dir, 'icons', 'record.png' if record else 'stop.png'))
            rb = self.record_button
            from PyQt5.QtGui import QPixmap, QIcon
            ppix = QPixmap(bitmap_path)
            pi = QIcon(ppix)
            rb.setIcon(pi)

    def record_cb(self, event=None):
        from chimerax.core.commands import run
        ses = self.session
        if not self._recording:
            self.set_button_icon(record=False)
            self._recording = True
            run(ses, 'movie record')
            self.play()
        else:
            self.set_button_icon(record=True)
            self._recording = False
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
