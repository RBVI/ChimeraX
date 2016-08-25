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
class ResidueFit(ToolInstance):

    SIZE = (-1, -1)
    SESSION_SKIP = True

    def __init__(self, session, bundle_info, residues, map, residue_range = (-2,1),
                 pause_frames = 50, motion_frames = 50, movie_framerate = 25):
        ToolInstance.__init__(self, session, bundle_info)

        self.residues = {r.number:r for r in residues}
        self.map = map

        r0 = residues[0]
        self.structure = s = r0.structure
        self.chain_id = cid = r0.chain_id
        rnums = residues.numbers
        self.rmin, self.rmax = rmin, rmax = rnums.min(), rnums.max()
        self.residue_range = residue_range
        self.pause_frames = pause_frames
        self._pause_count = 0
        self.motion_frames = motion_frames
        self.movie_framerate = movie_framerate
        self._last_shown_resnum = None
        self._last_pos = None
        self._label = None	# Graphics Label showing residue number
        
        self._play_handler = None
        self._recording = False
        self._block_update = False

        self.display_name = 'Residue fit %s chain %s %d-%d' % (s.name, cid, rmin, rmax)

        from chimerax.core import window_sys
        kw = {'size': self.SIZE} if window_sys == 'wx' else {}
        from chimerax.core.ui.gui import MainToolWindow
        tw = MainToolWindow(self, **kw)
        self.tool_window = tw
        parent = tw.ui_area

        if window_sys == 'wx':
            import wx
            label = wx.StaticText(parent, label="Residue")
            self.resnum = rn = wx.SpinCtrl(parent, value=str(rmin), min=rmin, max=rmax, size=(50, -1))
            rn.Bind(wx.EVT_SPINCTRL, self.resnum_changed_cb)
            self.slider = sl = wx.Slider(parent, value=rmin, minValue=rmin, maxValue=rmax)
            sl.Bind(wx.EVT_SLIDER, self.slider_moved_cb)
            self.play_button = pb = wx.ToggleButton(parent, label=' ', style=wx.BU_EXACTFIT)
            pb.Bind(wx.EVT_TOGGLEBUTTON, self.play_cb)
            self.record_button = rb = wx.ToggleButton(parent, label=' ', style=wx.BU_EXACTFIT)
            rb.Bind(wx.EVT_TOGGLEBUTTON, self.record_cb)

            sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(label, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
            sizer.Add(rn, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
            sizer.Add(sl, 1, wx.EXPAND)
            sizer.Add(pb, 0, wx.FIXED_MINSIZE)
            sizer.Add(rb, 0, wx.FIXED_MINSIZE)
            parent.SetSizerAndFit(sizer)
        elif window_sys == 'qt':
            from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QSlider, QPushButton
            from PyQt5.QtGui import QPixmap, QIcon
            from PyQt5.QtCore import Qt
            layout = QHBoxLayout()
            layout.setContentsMargins(0,0,0,0)
            layout.setSpacing(4)
            rl = QLabel('Residue')
            layout.addWidget(rl)
            self.resnum = rn = QSpinBox()
            rn.setRange(rmin, rmax)
            rn.valueChanged.connect(self.resnum_changed_cb)
            layout.addWidget(rn)
            self.slider = rs = QSlider(Qt.Horizontal)
            rs.setRange(rmin,rmax)
            rs.valueChanged.connect(self.slider_moved_cb)
            layout.addWidget(rs)
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

        tw.manage(placement="top")

        from chimerax.core.models import REMOVE_MODELS
        self._model_close_handler = session.triggers.add_handler(REMOVE_MODELS, self.models_closed_cb)

        map.new_region(ijk_step = (1,1,1), adjust_step = False)
        map.show(representation = 'mesh')
        s.atoms.displays = False
        s.residues.ribbon_displays = False
        self.update_resnum(rmin)

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def resnum_changed_cb(self, event):
        from chimerax.core import window_sys
        if window_sys == 'wx':
            rn = self.resnum.GetValue()
            self.slider.SetValue(rn)
            self.update_resnum(rn)
        elif window_sys == 'qt':
            rn = self.resnum.value()
            self.slider.setValue(rn)

    def slider_moved_cb(self, event):
        from chimerax.core import window_sys
        if window_sys == 'wx':
            rn = self.slider.GetValue()
            self.resnum.SetValue(rn)
        elif window_sys == 'qt':
            rn = self.slider.value()
            self.resnum.setValue(rn)
        self.update_resnum(rn)

    def update_resnum(self, rnum, motion = False):
        if self._block_update:
            return
        res =  self.residues.get(rnum, None)
        if res is not None:
            self.structure.atoms.displays = False
            self._last_shown_resnum = rnum
            mf = self.motion_frames if motion else 0
            lp = show_residue_fit(self.session, self.zone_residues(res), self.map,
                                  last_pos = self._last_pos, motion_frames = mf)
            if lp:
                self._last_pos = lp
            self.update_label(res)
        else:
            log = self.session.logger
            log.status('No residue %d in %s chain %s'
                       % (rnum, self.structure.name, self.chain_id))

    def zone_residues(self, res):
        zone_res = [res]
        rnum = res.number
        r0,r1 = self.residue_range
        for offset in range(r0,r1+1):
            if offset != 0:
                ro = self.residues.get(rnum+offset,None)
                if ro:
                    zone_res.append(ro)
        return zone_res

    def update_label(self, res):
        if self._label is None:
            from chimerax.label.label import Label
            self._label = Label(self.session, 'resfit', xpos = 0.7, ypos = 0.9)
        l = self._label
        l.text = '%s %d' % (res.name, res.number)
        l.update_drawing()

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
            self._play_handler = t.add_handler('new frame', self.next_residue_cb)

    def stop(self):
        if self._play_handler:
            t = self.session.triggers
            t.remove_handler(self._play_handler)
            self._play_handler = None
            if self._recording:
                self.record_cb()

    def next_residue_cb(self, *_):
        if not (self._recording and self.motion_frames == 0):
            self._pause_count += 1
            if self._pause_count >= self.pause_frames:
                self._pause_count = 0
            else:
                return
        rn = self._last_shown_resnum
        if rn >= self.rmax:
            if self._recording:
                self.stop()
                return
            rn = self.rmin
        else:
            rn += 1
        while rn not in self.residues:
            rn += 1
                    
        from chimerax.core import window_sys
        if window_sys == 'wx':
            self.resnum.SetValue(rn)
        elif window_sys == 'qt':
            self._block_update = True # Don't update display when slider changes
            self.resnum.setValue(rn)
            self._block_update = False
        self.update_resnum(rn, motion = True)

    def set_button_icon(self, play = None, record = None):
        from os.path import dirname, join
        dir = dirname(__file__)
        if play is not None:
            bitmap_path = (join(dir, 'play.png' if play else 'pause.png'))
            pb = self.play_button
            from chimerax.core import window_sys
            if window_sys == 'wx':
                import wx
                pbm = wx.Bitmap(bitmap_path)
                pb.SetBitmap(pbm)
            elif window_sys == 'qt':
                from PyQt5.QtGui import QPixmap, QIcon
                ppix = QPixmap(bitmap_path)
                pi = QIcon(ppix)
                pb.setIcon(pi)
                
        if record is not None:
            bitmap_path = (join(dir, 'record.png' if record else 'stop.png'))
            rb = self.record_button
            from chimerax.core import window_sys
            if window_sys == 'wx':
                import wx
                pbm = wx.Bitmap(bitmap_path)
                rb.SetBitmap(pbm)
            elif window_sys == 'qt':
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
            run(ses, 'movie encode ~/Desktop/resfit.mp4 framerate %.1f' % self.movie_framerate)
            
    def models_closed_cb(self, name, models):
        if self.structure in models or self.map in models:
            self.delete()

    # Override ToolInstance method
    def delete(self):
        t = self.session.triggers
        t.remove_handler(self._model_close_handler)
        self._model_close_handler = None
        if self._play_handler:
            t.remove_handler(self._play_handler)
            self._play_handler = None
        super().delete()
        self.structure = None
        self.residues = None
        self.map = None
        if self._label is not None:
            self._label.delete()
            self._label = None
            

def show_residue_fit(session, residues, map, range = 2, last_pos = None, motion_frames = 20):
    '''Set camera to show first residue in list and display map in zone around given residues.'''
    res = residues[0]
    ratoms = res.atoms
    anames = tuple(ratoms.names)
    try:
        i = [anames.index(aname) for aname in ('N', 'CA', 'C')]
    except ValueError:
        return None		# Missing backbone atom
    xyz = ratoms.filter(i).scene_coords

    from chimerax.core.geometry import align_points, Place
    from numpy import array
    # Template backbone atom coords and camera view
    txyz = array([[ 12.83300018,   6.83900023,   6.73799992],
                  [ 12.80800056,   7.87400055,   5.70799971],
                  [ 11.91800022,   9.06700039,   5.9920001 ]])
    c = session.main_view.camera
    cp = c.position
    if last_pos is None:
        tc = Place(((-0.46696,0.38225,-0.79739,-3.9125),
                    (0.81905,-0.15294,-0.55296,-4.3407),
                    (-0.33332,-0.91132,-0.24166,-1.4889)))
    else:
        # Maintain same relative camera position to backbone.
        tc = last_pos.inverse() * cp
    p, rms = align_points(txyz, xyz)

    # Smooth interpolation
    np = p*tc
    if motion_frames > 1:
        def interpolate_camera(session, f, cp=cp, np=np, center=np.inverse()*xyz[1], frames=motion_frames):
            c = session.main_view.camera
            c.position = cp.interpolate(np, center, frac = (f+1)/frames)
        from chimerax.core.commands import motion
        motion.CallForNFrames(interpolate_camera, motion_frames, session)
    else:
        c.position = np

    from numpy import concatenate
    zone_points = concatenate([r.atoms.scene_coords for r in residues])
    from chimerax.core.surface import zone
    zone.surface_zone(map, zone_points, range)

    for r in residues:
        r.atoms.displays = True
    
    return p
