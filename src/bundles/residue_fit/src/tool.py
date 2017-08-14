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

from chimerax.core.ui.widgets.slider import Slider

# ------------------------------------------------------------------------------
#
class ResidueFit(Slider):

    def __init__(self, session, tool_name, residues, map, residue_range = (-2,1),
                 pause_frames = 50, motion_frames = 50, movie_framerate = 25):

        self.residues = {r.number:r for r in residues}
        self.map = map

        r0 = residues[0]
        self.structure = s = r0.structure
        self.chain_id = cid = r0.chain_id
        rnums = residues.numbers
        self.rmin, self.rmax = rmin, rmax = rnums.min(), rnums.max()
        self.residue_range = residue_range
        self.motion_frames = motion_frames
        self._last_pos = None
        self._label = None	# Graphics Label showing residue number

        title = 'Residue fit %s chain %s %d-%d' % (s.name, cid, rmin, rmax)
        Slider.__init__(self, session, tool_name, 'Residue', title, value_range = (rmin,rmax),
                        pause_frames = max(pause_frames, motion_frames+1),
                        pause_when_recording = (motion_frames <= 1),
                        movie_filename = 'resfit.mp4', movie_framerate = movie_framerate)

        from chimerax.core.models import REMOVE_MODELS
        self._model_close_handler = session.triggers.add_handler(REMOVE_MODELS, self.models_closed_cb)

        map.new_region(ijk_step = (1,1,1), adjust_step = False)
        map.show(representation = 'mesh')
        s.atoms.displays = False
        s.residues.ribbon_displays = False
        self.update_value(rmin)

    def change_value(self, rnum, playing = False):
        res =  self.residues.get(rnum, None)
        if res is not None:
            self.structure.atoms.displays = False
            mf = self.motion_frames if playing else 0
            lp = show_residue_fit(self.session, self.zone_residues(res), self.map,
                                  last_pos = self._last_pos, motion_frames = mf)
            if lp:
                self._last_pos = lp
            self.update_label(res)
        else:
            log = self.session.logger
            log.status('No residue %d in %s chain %s'
                       % (rnum, self.structure.name, self.chain_id))

    def valid_value(self, rnum):
        return rnum in self.residues
    
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
            
    def models_closed_cb(self, name, models):
        if self.structure in models or self.map in models:
            self.delete()

    # Override ToolInstance method
    def delete(self):
        t = self.session.triggers
        t.remove_handler(self._model_close_handler)
        self._model_close_handler = None
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

    # Align backbone to template backbone coords
    from chimerax.core.geometry import align_points, Place
    from numpy import array
    txyz = array([[ 12.83300018,   6.83900023,   6.73799992],
                  [ 12.80800056,   7.87400055,   5.70799971],
                  [ 11.91800022,   9.06700039,   5.9920001 ]])
    p, rms = align_points(txyz, xyz)

    # Set camera view relative to template.
    c = session.main_view.camera
    cp = c.position
    if last_pos is None:
        tc = Place(((-0.46696,0.38225,-0.79739,-3.9125),
                    (0.81905,-0.15294,-0.55296,-4.3407),
                    (-0.33332,-0.91132,-0.24166,-1.4889)))
    else:
        # Maintain same relative camera position to backbone.
        tc = last_pos.inverse() * cp

    # Smooth interpolation
    np = p*tc
    if motion_frames > 1:
        def interpolate_camera(session, f, cp=cp, np=np, center=np.inverse()*xyz[1], frames=motion_frames):
            c = session.main_view.camera
            p = np if f+1 == frames else cp.interpolate(np, center, frac = (f+1)/frames)
            c.position = p
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
