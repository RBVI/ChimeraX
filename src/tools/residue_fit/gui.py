# vim: set expandtab ts=4 sw=4:

from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class ResidueFit(ToolInstance):

    SIZE = (-1, -1)
    SESSION_SKIP = True

    def __init__(self, session, bundle_info, residues, map, pause_frames = 50,
                 motion_frames = 50, movie_framerate = 25):
        ToolInstance.__init__(self, session, bundle_info)

        self.residues = {r.number:r for r in residues}
        self.map = map

        r0 = residues[0]
        self.structure = s = r0.structure
        self.chain_id = cid = r0.chain_id
        rnums = residues.numbers
        self.rmin, self.rmax = rmin, rmax = rnums.min(), rnums.max()
        self.neighbors = (-2,-1,1)
        self.pause_frames = pause_frames
        self._pause_count = 0
        self.motion_frames = motion_frames
        self.movie_framerate = movie_framerate
        self._last_shown_resnum = None
        
        self._play_handler = None
        self._recording = False

        self.display_name = 'Residue fit %s chain %s %d-%d' % (s.name, cid, rmin, rmax)

        from chimerax.core.ui.gui import MainToolWindow
        tw = MainToolWindow(self, size=self.SIZE)
        self.tool_window = tw
        parent = tw.ui_area

        import wx
        label = wx.StaticText(parent, label="Residue")
        self.resnum = rn = wx.SpinCtrl(parent, value=str(rmin), min=rmin, max=rmax, size=(50, -1))
        rn.Bind(wx.EVT_SPINCTRL, self.resnum_changed_cb)
        self.slider = sl = wx.Slider(parent, value=rmin, minValue=rmin, maxValue=rmax)
        sl.Bind(wx.EVT_SLIDER, self.slider_moved_cb)
        self.play_button = pb = wx.ToggleButton(parent, style=wx.BU_EXACTFIT)
        pb.Bind(wx.EVT_TOGGLEBUTTON, self.play_cb)
        self.record_button = rb = wx.ToggleButton(parent, style=wx.BU_EXACTFIT)
        rb.Bind(wx.EVT_TOGGLEBUTTON, self.record_cb)
        self.set_button_icon(play=True, record=True)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(label, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        sizer.Add(rn, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        sizer.Add(sl, 1, wx.EXPAND)
        sizer.Add(pb, 0, wx.FIXED_MINSIZE)
        sizer.Add(rb, 0, wx.FIXED_MINSIZE)
        parent.SetSizerAndFit(sizer)

        tw.manage(placement="top")

        from chimerax.core.models import REMOVE_MODELS
        self._model_close_handler = session.triggers.add_handler(REMOVE_MODELS, self.models_closed_cb)
        
        map.show(representation = 'mesh')
        s.atoms.displays = False
        self.update_resnum(rmin)

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def resnum_changed_cb(self, event):
        rn = self.resnum.GetValue()
        self.slider.SetValue(rn)
        self.update_resnum(rn)

    def slider_moved_cb(self, event):
        rn = self.slider.GetValue()
        self.resnum.SetValue(rn)
        self.update_resnum(rn)

    def update_resnum(self, rnum, motion = False):
        res =  self.residues.get(rnum, None)
        if res is not None:
            self.structure.atoms.displays = False
            self._last_shown_resnum = rnum
            mf = self.motion_frames if motion else 0
            show_residue_fit(self.session, self.zone_residues(res), self.map,
                             motion_frames = mf)
            self.update_label(res)
        else:
            log = self.session.logger
            log.status('No residue %d in %s chain %s'
                       % (rnum, self.structure.name, self.chain_id))

    def zone_residues(self, res):
        zone_res = [res]
        rnum = res.number
        for offset in self.neighbors:
            if offset != 0:
                ro = self.residues.get(rnum+offset,None)
                if ro:
                    zone_res.append(ro)
        return zone_res

    def update_label(self, res):
        ses = self.session
        if not hasattr(ses, '_resfit_label'):
            from chimerax.label.label import Label
            ses._resfit_label = Label(ses, 'resfit', xpos = 0.7, ypos = 0.9)
        l = ses._resfit_label
        l.text = '%s %d' % (res.name, res.number)
        l.make_drawing()

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
            t.delete_handler(self._play_handler)
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
        self.resnum.SetValue(rn)
        self.slider.SetValue(rn)
        self.update_resnum(rn, motion = True)

    def set_button_icon(self, play = None, record = None):
        import wx
        from os.path import dirname, join
        dir = dirname(__file__)
        if play is not None:
            bitmap_path = (join(dir, 'play.png' if play else 'pause.png'))
            pbm = wx.Bitmap(bitmap_path)
            self.play_button.SetBitmap(pbm)
        if record is not None:
            bitmap_path = (join(dir, 'record.png' if record else 'stop.png'))
            pbm = wx.Bitmap(bitmap_path)
            self.record_button.SetBitmap(pbm)

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
        if self.structure in models:
            self.delete()

    # Override ToolInstance method
    def delete(self):
        t = self.session.triggers
        t.delete_handler(self._model_close_handler)
        self._model_close_handler = None
        if self._play_handler:
            t.delete_handler(self._play_handler)
            self._play_handler = None
        super().delete()

def show_residue_fit(session, residues, map, range = 2, motion_frames = 20):
    '''Set camera to show first residue in list and display map in zone around given residues.'''
    res = residues[0]
    ratoms = res.atoms
    anames = tuple(ratoms.names)
    try:
        i = [anames.index(aname) for aname in ('N', 'CA', 'C')]
    except ValueError:
        return False		# Missing backbone atom
    xyz = ratoms.filter(i).scene_coords

    from chimerax.core.geometry import align_points, Place
    from numpy import array
    # Template backbone atom coords and camera view
    txyz = array([[ 12.83300018,   6.83900023,   6.73799992],
                  [ 12.80800056,   7.87400055,   5.70799971],
                  [ 11.91800022,   9.06700039,   5.9920001 ]])
    tc = Place(((-0.46696,0.38225,-0.79739,-3.9125),
                (0.81905,-0.15294,-0.55296,-4.3407),
                (-0.33332,-0.91132,-0.24166,-1.4889)))
    p, rms = align_points(txyz, xyz)

    # Smooth interpolation
    c = session.main_view.camera
    if motion_frames > 1:
        cp, np = c.position, p*tc
        def interpolate_camera(session, f, cp=cp, np=np, center=np.inverse()*xyz[1], frames=motion_frames):
            c = session.main_view.camera
            c.position = cp.interpolate(np, center, frac = (f+1)/frames)
        from chimerax.core.commands import motion
        motion.CallForNFrames(interpolate_camera, motion_frames, session)
    else:
        c.position = p*tc

    from numpy import concatenate
    zone_points = concatenate([r.atoms.scene_coords for r in residues])
    from chimerax.core.surface import zone
    zone.surface_zone(map, zone_points, range)

    for r in residues:
        r.atoms.displays = True
    
    return True
