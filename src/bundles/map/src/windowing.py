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

#
# Adjust gray-levels in medical images by translating brightness/transparency curve
# or scaling the width of the curve about its center.
#
from chimerax.mouse_modes import MouseMode
class WindowingMouseMode(MouseMode):
    name = 'windowing'
    icon_file = 'icons/windowing.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._maps = []

        # Allow adjusting multiple windows (non-zero regions of brightness curve).
        self._multiwindow = True
        self._centers = []
        self._window_num = 0
        self._vr_window_switch_motion = 0.02	# meters
        self._dragged = False
        
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        self._maps = self._visible_maps()
        if self._multiwindow:
            self._centers = self._window_centers()
        self._dragged = False

    def _visible_maps(self):
        from . import Volume
        return [m for m in self.session.models.list(type = Volume)
                if m.visible and m.image_shown]

    def _window_centers(self):
        wn = self._window_num
        centers = []
        for v in self._maps:
            wc = [0.5*(lev0 + lev1) for lev0,lev1 in _windows(v)]
            centers.append(wc[wn%len(wc)])
        return centers
            
    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        if abs(dx) > abs(dy):
            if self._multiwindow:
                _scale_window(self._maps, 0.002 * dx, self._centers)
            else:
                scale_levels(self._maps, 0.002 * dx)
        else:
            if self._multiwindow:
                self._centers = _translate_window(self._maps, -0.001 * dy, self._centers)
            else:
                translate_levels(self._maps, -0.001 * dy)

        self._dragged = True

        # Make sure new level is shown before another mouse event causes another level change.
        self.session.update_loop.update_graphics_now()
    
    def wheel(self, event):
        d = event.wheel_value()
        f = d/30
        maps = self._visible_maps()
        scale_levels(maps, f)

        # Make sure new level is shown before another mouse event causes another level change.
        self.session.update_loop.update_graphics_now()

    def mouse_up(self, event):
        if self._dragged:
            self.log_volume_command()
        elif self._multiwindow:
            self._window_num += 1
        self._maps = []
        MouseMode.mouse_up(self, event)
        
    def vr_press(self, event):
        # Virtual reality hand controller button press.
        self._maps = self._visible_maps()
        if self._multiwindow:
            self._centers = self._window_centers()
        self._dragged = 0

    def vr_motion(self, event):
        # Virtual reality hand controller motion.
        c = self.session.main_view.camera
        # Get hand controller motion in room in meters
        motion = event.tip_motion
        hand_motion = event.position.inverse().transform_vector(motion)  # Hand coordinate system
        horz_shift, vert_shift = hand_motion[0], hand_motion[1]
        if abs(horz_shift) > abs(vert_shift):
            if self._multiwindow:
                _scale_window(self._maps, horz_shift, self._centers)
            else:
                scale_levels(self._maps, horz_shift)
        else:
            if self._multiwindow:
                self._centers = _translate_window(self._maps, vert_shift, self._centers)
            else:
                translate_levels(self._maps, vert_shift)
        self._dragged += max(abs(horz_shift), abs(vert_shift))
        
    def vr_release(self, event):
        # Virtual reality hand controller button release.
        self.log_volume_command()
        self._maps = []
        if self._multiwindow and self._dragged <= self._vr_window_switch_motion:
            self._window_num += 1

    def log_volume_command(self):
        for v in self._maps:
            if isinstance(v, tuple):
                v = v[0]
            log_volume_levels_command(v)


def translate_levels(maps, f):
    for v in maps:
        ms = v.matrix_value_statistics()
        vrange = (ms.maximum - ms.minimum)
        shift = f*vrange
        levels = [(lev+shift,y) for lev,y in v.image_levels]
        v.set_parameters(image_levels = levels)

def scale_levels(maps, f):
    from numpy import mean
    for v in maps:
        center = mean([lev for lev,y in v.image_levels])
        levels = [(lev+f*(lev-center),y) for lev,y in v.image_levels]
        v.set_parameters(image_levels = levels)

def _translate_window(maps, f, centers):
    new_centers = []
    for v, c in zip(maps, centers):
        ms = v.matrix_value_statistics()
        vrange = (ms.maximum - ms.minimum)
        shift = f*vrange
        min_spacing = 0.0001 * vrange
        
        lev0, lev1, lim0, lim1 = _window_range(v, c)
        # Clamp shift so this window is not moved over adjacent window
        if shift < 0 and lim0 is not None and lev0 + shift <= lim0:
            shift = min(0, lim0 - lev0 + min_spacing)
        elif shift > 0 and lim1 is not None and lev1 + shift >= lim1:
            shift = min(0, lim1 - lev1 - min_spacing)

        levels = [((lev+shift if lev >= lev0 and lev <= lev1 else lev), y)
                  for lev,y in v.image_levels]
        v.set_parameters(image_levels = levels)
        new_centers.append(c+shift)

    return new_centers

def _scale_window(maps, f, centers):
    for v, c in zip(maps, centers):
        lev0, lev1, lim0, lim1 = _window_range(v, c)
        # Clamp shift so this window is not moved over adjacent window
        if f > 0:
            if lim0 is not None and lev0 + f*(lev0-c) <= lim0:
                f = max(0, (lim0 - lev0) / (lev0 - c) - 0.001)
            if lim1 is not None and lev1 + f*(lev1-c) >= lim1:
                f = max(0, (lim1 - lev1) / (lev1 - c) - 0.001)

        levels = [((lev+f*(lev-c) if lev >= lev0 and lev <= lev1 else lev), y)
                  for lev,y in v.image_levels]
        v.set_parameters(image_levels = levels)

def _window_range(v, center):
    lev0 = max([lev for lev,y in v.image_levels if y == 0 and lev <= center], default = None)
    if lev0 is None:
        lev0 = min([lev for lev,y in v.image_levels])
    lev1 = min([lev for lev,y in v.image_levels if y == 0 and lev >= center], default = None)
    if lev1 is None:
        lev1 = max([lev for lev,y in v.image_levels])
    lim0 = max([lev for lev,y in v.image_levels if lev < lev0], default = None)
    lim1 = min([lev for lev,y in v.image_levels if lev > lev1], default = None)
    return lev0, lev1, lim0, lim1

def _windows(v):
    levels = list(v.image_levels)
    levels.sort(key = lambda k: tuple(k))
    n = len(levels)
    wins = []
    for i in range(n):
        y0 = levels[i-1][1] if i > 0 else None
        lev,y = levels[i]
        y1 = levels[i+1][1] if i+1<n else None
        if ((i == 0 and y != 0) or
            ((i == 0 or y0 == 0) and y == 0 and i+1<n and y1 != 0)):
            lev0 = lev	# Start of window
        elif ((y != 0 and y1 is None) or
              (i > 0 and y0 != 0 and y == 0 and (y1 is None or y1 == 0))):
            wins.append((lev0,lev))	# End of window
    return wins
        
    
def log_volume_levels_command(v):
    if v.image_shown:
        levels = ' '.join('level %.4g,%.4g' % tuple(sl) for sl in v.image_levels)
        command = 'volume #%s %s' % (v.id_string, levels)
        from chimerax.core.commands import log_equivalent_command
        log_equivalent_command(v.session, command)

def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(WindowingMouseMode(session))
