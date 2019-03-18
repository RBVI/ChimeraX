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

'''
Standard mouse modes
====================
'''

from .mousemodes import MouseMode

class SelectMouseMode(MouseMode):
    '''Mouse mode to select objects by clicking on them.'''
    name = 'select'
    icon_file = 'icons/select.png'

    _menu_entry_info = []

    def __init__(self, session):
        MouseMode.__init__(self, session)

        self.mode = {'select': 'replace',
                     'select add': 'add',
                     'select subtract': 'subtract',
                     'select toggle': 'toggle'}[self.name]
        self.minimum_drag_pixels = 5
        self.drag_color = (0,255,0,255)	# Green
        self._drawn_rectangle = None

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

    def mouse_drag(self, event):
        if self._is_drag(event):
            self._undraw_drag_rectangle()
            self._draw_drag_rectangle(event)

    def mouse_up(self, event):
        self._undraw_drag_rectangle()
        if self._is_drag(event):
            # Select objects in rectangle
            mouse_drag_select(self.mouse_down_position, event, self.mode, self.session, self.view)
        elif not self.double_click:
            # Select object under pointer
            mouse_select(event, self.mode, self.session, self.view)
        MouseMode.mouse_up(self, event)

    def mouse_double_click(self, event):
        '''Show a context menu when double-clicking in select mode.'''
        MouseMode.mouse_double_click(self, event)
        entries = []
        dangerous_entries = []
        ses = self.session
        for entry in SelectMouseMode._menu_entry_info:
            if entry.criteria(ses):
                if entry.dangerous:
                    dangerous_entries.append(entry)
                else:
                    entries.append(entry)
        entries.sort(key = lambda e: e.label(ses))
        dangerous_entries.sort(key = lambda e: e.label(ses))
        from PyQt5.QtWidgets import QMenu, QAction
        menu = QMenu()
        actions = []
        all_entries = entries
        if dangerous_entries:
            all_entries = all_entries + [None] + dangerous_entries
        if all_entries:
            for entry in all_entries:
                if entry is None:
                    menu.addSeparator()
                    continue
                action = QAction(entry.label(ses))
                action.triggered.connect(lambda arg, cb=entry.callback, sess=ses: cb(sess))
                menu.addAction(action)
                actions.append(action) # keep reference
        else:
            menu.addAction("No applicable actions")
        # this will prevent atom-spec balloons from showing up
        menu.exec(event._event.globalPos())

    @staticmethod
    def register_menu_entry(menu_entry):
        '''Register a context-menu entry shown when double-clicking in select mode.

        menu_entry is a SelectContextMenuAction instance.
        '''
        SelectMouseMode._menu_entry_info.append(menu_entry)

    def _is_drag(self, event):
        dp = self.mouse_down_position
        if dp is None:
            return False
        dx,dy = dp
        x, y = event.position()
        mp = self.minimum_drag_pixels
        return abs(x-dx) > mp or abs(y-dy) > mp

    def _draw_drag_rectangle(self, event):
        dx,dy = self.mouse_down_position
        x, y = event.position()
        v = self.session.main_view
        w,h = v.window_size
        v.draw_xor_rectangle(dx, h-dy, x, h-y, self.drag_color)
        self._drawn_rectangle = (dx,dy), (x,y)

    def _undraw_drag_rectangle(self):
        dr = self._drawn_rectangle
        if dr:
            (dx,dy), (x,y) = dr
            v = self.session.main_view
            w,h = v.window_size
            v.draw_xor_rectangle(dx, h-dy, x, h-y, self.drag_color)
            self._drawn_rectangle = None

    def vr_press(self, xyz1, xyz2):
        # Virtual reality hand controller button press.
        from . import picked_object_on_segment
        pick = picked_object_on_segment(xyz1, xyz2, self.view)
        select_pick(self.session, pick, self.mode)

    def vr_motion(self, position, move, delta_z):
        # Virtual reality hand controller motion.
        ses = self.session
        sel = ses.selection
        if delta_z > 0.20:
            sel.promote(ses)
        elif delta_z < -0.20:
            sel.demote(ses)
        else:
            return 'accumulate drag'

class SelectContextMenuAction:
    '''Methods implementing a context-menu entry shown when double-clicking in select mode.'''
    def label(self, session):
        '''Returns the text of the menu entry.'''
        return 'unknown'
    def criteria(self, session):
        '''
        Return a boolean that indicates whether the menu should include this entry
        (usually based on the current contents of the selection).
        '''
        return False
    def callback(self, session):
        '''Perform the entry's action.'''
        pass
    dangerous = False
    '''If a menu is hazardous to click accidentally, 'dangerous' should be True.
    Such entries will be organized at the bottom of the menu after a separator.
    '''

class SelectAddMouseMode(SelectMouseMode):
    '''Mouse mode to add objects to selection by clicking on them.'''
    name = 'select add'
    icon_file = None

class SelectSubtractMouseMode(SelectMouseMode):
    '''Mouse mode to subtract objects from selection by clicking on them.'''
    name = 'select subtract'
    icon_file = None
    
class SelectToggleMouseMode(SelectMouseMode):
    '''Mouse mode to toggle selected objects by clicking on them.'''
    name = 'select toggle'
    icon_file = None

def mouse_select(event, mode, session, view):
    x,y = event.position()
    from . import picked_object
    pick = picked_object(x, y, view)
    select_pick(session, pick, mode)

def mouse_drag_select(start_xy, event, mode, session, view):
    sx, sy = start_xy
    x,y = event.position()
    from .mousemodes import unpickable
    pick = view.rectangle_intercept(sx,sy,x,y,exclude=unpickable)
    select_pick(session, pick, mode)

def select_pick(session, pick, mode = 'replace'):
    sel = session.selection
    from chimerax.core.undo import UndoState
    undo_state = UndoState("select")
    sel.undo_add_selected(undo_state, False)
    if pick is None:
        if mode == 'replace':
            sel.clear()
            session.logger.status('cleared selection')
    else:
        if mode == 'replace':
            sel.clear()
            mode = 'add'
        if isinstance(pick, list):
            for p in pick:
                p.select(mode)
        else:
            pick.select(mode)
    sel.clear_promotion_history()
    sel.undo_add_selected(undo_state, True, old_state=False)
    session.undo.register(undo_state)

class RotateMouseMode(MouseMode):
    '''
    Mouse mode to rotate objects (actually the camera is moved) by dragging.
    Mouse drags initiated near the periphery of the window cause a screen z rotation,
    while other mouse drags use rotation axes lying in the plane of the screen and
    perpendicular to the direction of the drag.
    '''
    name = 'rotate'
    icon_file = 'icons/rotate.png'
    click_to_select = False

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self.mouse_perimeter = False

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        x,y = event.position()
        w,h = self.view.window_size
        cx, cy = x-0.5*w, y-0.5*h
        from math import sqrt
        r = sqrt(cx*cx + cy*cy)
        fperim = 0.9
        self.mouse_perimeter = (r > fperim*0.5*min(w,h))

    def mouse_up(self, event):
        if self.click_to_select:
            if event.position() == self.mouse_down_position:
                mode = 'toggle' if event.shift_down() else 'replace'
                mouse_select(event, mode, self.session, self.view)
        MouseMode.mouse_up(self, event)

    def mouse_drag(self, event):
        axis, angle = self.mouse_rotation(event)
        self.rotate(axis, angle)

    def wheel(self, event):
        d = event.wheel_value()
        psize = self.pixel_size()
        self.rotate((0,1,0), 10*d)

    def rotate(self, axis, angle):
        # Convert axis from camera to scene coordinates
        saxis = self.camera_position.transform_vector(axis)
        self.view.rotate(saxis, angle, self.models())

    def mouse_rotation(self, event):

        dx, dy = self.mouse_motion(event)
        import math
        angle = 0.5*math.sqrt(dx*dx+dy*dy)
        if self.mouse_perimeter:
            # z-rotation
            axis = (0,0,1)
            w, h = self.view.window_size
            x, y = event.position()
            ex, ey = x-0.5*w, y-0.5*h
            if -dy*ex+dx*ey < 0:
                angle = -angle
        else:
            axis = (dy,dx,0)
        return axis, angle

    def models(self):
        return None

    def vr_motion(self, position, move, delta_z):
        # Virtual reality hand controller motion.
        self.view.move(move, self.models())

class RotateAndSelectMouseMode(RotateMouseMode):
    '''
    Mouse mode to rotate objects like RotateMouseMode.
    Also clicking without dragging selects objects.
    This mode allows click with no modifier keys to perform selection,
    while click and drag produces rotation.
    '''
    name = 'rotate and select'
    icon_file = 'icons/rotatesel.png'
    click_to_select = True

class RotateSelectedMouseMode(RotateMouseMode):
    '''
    Mouse mode to rotate objects like RotateMouseMode but only selected
    models are rotated. Selected models are actually moved in scene
    coordinates instead of moving the camera. If nothing is selected,
    then the camera is moved as if all models are rotated.
    '''
    name = 'rotate selected models'
    icon_file = 'icons/rotate_h2o.png'

    def models(self):
        return top_selected(self.session)

def top_selected(session):
    # Don't include parents of selected models.
    mlist = [m for m in session.selection.models()
             if ((len(m.child_models()) == 0 or m.selected) and not any_parent_selected(m))]
    return None if len(mlist) == 0 else mlist

def any_parent_selected(m):
    if m.parent is None:
        return False
    p = m.parent
    return p.selected or any_parent_selected(p)

class TranslateMouseMode(MouseMode):
    '''
    Mouse mode to move objects in x and y (actually the camera is moved) by dragging.
    '''
    name = 'translate'
    icon_file = 'icons/translate.png'

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        self.translate((dx, -dy, 0))

    def wheel(self, event):
        d = event.wheel_value()
        self.translate((0,0,100*d))

    def translate(self, shift):

        psize = self.pixel_size()
        s = tuple(dx*psize for dx in shift)     # Scene units
        step = self.camera_position.transform_vector(s)    # Scene coord system
        self.view.translate(step, self.models())

    def models(self):
        return None

    def vr_motion(self, position, move, delta_z):
        # Virtual reality hand controller motion.
        self.view.move(move, self.models())

class TranslateSelectedMouseMode(TranslateMouseMode):
    '''
    Mouse mode to move objects in x and y like TranslateMouseMode but only selected
    models are moved. Selected models are actually moved in scene
    coordinates instead of moving the camera. If nothing is selected,
    then the camera is moved as if all models are shifted.
    '''
    name = 'translate selected models'
    icon_file = 'icons/move_h2o.png'

    def models(self):
        return top_selected(self.session)

class ZoomMouseMode(MouseMode):
    '''
    Mouse mode to move objects in z, actually the camera is moved
    and the objects remain at their same scene coordinates.
    '''
    name = 'zoom'
    icon_file = 'icons/zoom.png'

    def mouse_drag(self, event):        

        dx, dy = self.mouse_motion(event)
        psize = self.pixel_size()
        delta_z = 3*psize*dy
        self.zoom(delta_z, stereo_scaling = not event.alt_down())

    def wheel(self, event):
        d = event.wheel_value()
        psize = self.pixel_size()
        self.zoom(100*d*psize, stereo_scaling = not event.alt_down())

    def zoom(self, delta_z, stereo_scaling = False):
        v = self.view
        c = v.camera
        if stereo_scaling and c.name == 'stereo':
            v.stereo_scaling(delta_z)
        if c.name == 'orthographic':
            c.field_width = max(c.field_width - delta_z, self.pixel_size())
            # TODO: Make camera field_width a property so it knows to redraw.
            c.redraw_needed = True
        else:
            shift = c.position.transform_vector((0, 0, delta_z))
            v.translate(shift)
        
class ObjectIdMouseMode(MouseMode):
    '''
    Mouse mode to that shows the name of an object in a popup window
    when the mouse is hovered over the object for 0.5 seconds.
    '''
    name = 'identify object'
    def __init__(self, session):
        MouseMode.__init__(self, session)
        session.triggers.add_trigger('mouse hover')
        
    def pause(self, position):
        ui = self.session.ui
        if ui.activeWindow() is None:
            # Qt 5.7 gives app mouse events on Mac even if another application has the focus,
            # and even if the this app is minimized, it gets events for where it used to be on the screen.
            return
        # ensure that no other top-level window is above the graphics
        from PyQt5.QtGui import QCursor
        if ui.topLevelAt(QCursor.pos()) != ui.main_window:
            return
        # ensure there's no popup menu above the graphics
        apw = ui.activePopupWidget()
        from PyQt5.QtCore import QPoint
        if apw and ui.topLevelAt(apw.mapToGlobal(QPoint())) == ui.main_window:
            return
        x,y = position
        from . import picked_object
        p = picked_object(x, y, self.view)

        # Show atom spec balloon
        pu = ui.main_window.graphics_window.popup
        if p:
            pu.show_text(p.description(), (x+10,y))
            res = getattr(p, 'residue', None)
            if res:
                chain = res.chain
                if chain and chain.description:
                    self.session.logger.status("chain %s: %s" % (chain.chain_id, chain.description))
                elif res.name in getattr(res.structure, "_hetnam_descriptions", {}):
                    self.session.logger.status(res.structure._hetnam_descriptions[res.name])
            if p.distance is not None:
                f = p.distance
                xyz1, xyz2 = self.view.clip_plane_points(x, y)
                xyz = (1-f)*xyz1 + f*xyz2
                self.session.triggers.activate_trigger('mouse hover', xyz)
        else:
            pu.hide()

    def move_after_pause(self):
        # Hide atom spec balloon
        self.session.ui.main_window.graphics_window.popup.hide()

class AtomCenterOfRotationMode(MouseMode):
    '''Clicking on an atom sets the center of rotation at that position.'''
    name = 'pivot'
    icon_file = 'icons/pivot.png'

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        x,y = event.position()
        view = self.session.main_view
        from . import picked_object
        pick = picked_object(x, y, view)
        from chimerax.atomic import PickedResidue, PickedBond, PickedPseudobond
        if hasattr(pick, 'atom'):
            xyz = pick.atom.scene_coord
        elif isinstance(pick, PickedResidue):
            r = pick.residue
            xyz = sum([a.scene_coord for a in r.atoms]) / r.num_atoms
        elif isinstance(pick, PickedBond):
            b = pick.bond
            xyz = sum([a.scene_coord for a in b.atoms]) / 2
        elif isinstance(pick, PickedPseudobond):
            b = pick.pbond
            xyz = sum([a.scene_coord for a in b.atoms]) / 2
        else:
            return
        from chimerax.std_commands import cofr
        cofr.cofr(self.session, pivot=xyz)
           
class NullMouseMode(MouseMode):
    '''Used to assign no mode to a mouse button.'''
    name = 'none'

class ClipMouseMode(MouseMode):
    '''
    Move clip planes.
    Move front plane with no modifiers, back plane with alt,
    both planes with shift, and slab thickness with alt and shift.
    Move scene planes unless only near/far planes are enabled.
    If the planes do not exist create them.
    '''
    name = 'clip'
    icon_file = 'icons/clip.png'

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        front_shift, back_shift = self.which_planes(event)
        self.clip_move((dx,-dy), front_shift, back_shift)

    def which_planes(self, event):
        shift, alt = event.shift_down(), event.alt_down()
        front_shift = 1 if shift or not alt else 0
        back_shift = 0 if not (alt or shift) else (1 if alt and shift else -1)
        return front_shift, back_shift
    
    def wheel(self, event):
        d = event.wheel_value()
        psize = self.pixel_size()
        front_shift, back_shift = self.which_planes(event)
        self.clip_move(None, front_shift, back_shift, delta = 100*psize*d)

    def clip_move(self, delta_xy, front_shift, back_shift, delta = None):
        pf, pb = self._planes(front_shift, back_shift)
        if pf is None and pb is None:
            return

        p = pf or pb
        if delta is not None:
            d = delta
        elif p and p.camera_normal is None:
            # Move scene clip plane
            d = self._tilt_shift(delta_xy, self.view.camera, p.normal)
        else:
            # near/far clip
            d = delta_xy[1]*self.pixel_size()

        # Check if slab thickness becomes less than zero.
        dt = -d*(front_shift+back_shift)
        if pf and pb and dt < 0:
            from chimerax.core.geometry import inner_product
            sep = inner_product(pb.plane_point - pf.plane_point, pf.normal)
            if sep + dt <= 0:
                # Would make slab thickness less than zero.
                return

        if pf:
            pf.plane_point = pf.plane_point + front_shift*d*pf.normal
        if pb:
            pb.plane_point = pb.plane_point + back_shift*d*pb.normal

    def _planes(self, front_shift, back_shift):
        v = self.view
        p = v.clip_planes
        pfname, pbname = (('front','back') if p.find_plane('front') or p.find_plane('back') or not p.planes() 
                          else ('near','far'))
        
        pf, pb = p.find_plane(pfname), p.find_plane(pbname)
        from chimerax.std_commands.clip import adjust_plane
        c = v.camera
        cfn, cbn = ((0,0,-1), (0,0,1)) if pfname == 'near' else (None, None)

        if front_shift and pf is None:
            b = v.drawing_bounds()
            if pb:
                offset = -1 if b is None else -0.2*b.radius()
                pf = adjust_plane(pfname, offset, pb.plane_point, -pb.normal, p, v, cfn)
            elif b:
                normal = v.camera.view_direction()
                offset = 0
                pf = adjust_plane(pfname, offset, b.center(), normal, p, v, cfn)

        if back_shift and pb is None:
            b = v.drawing_bounds()
            offset = -1 if b is None else -0.2*b.radius()
            if pf:
                pb = adjust_plane(pbname, offset, pf.plane_point, -pf.normal, p, v, cbn)
            elif b:
                normal = -v.camera.view_direction()
                pb = adjust_plane(pbname, offset, b.center(), normal, p, v, cbn)

        return pf, pb

    def _tilt_shift(self, delta_xy, camera, normal):
        # Measure drag direction along plane normal direction.
        nx,ny,nz = camera.position.inverse().transform_vector(normal)
        from math import sqrt
        d = sqrt(nx*nx + ny*ny)
        if d > 0:
            nx /= d
            ny /= d
        else:
            nx = 0
            ny = 1
        dx,dy = delta_xy
        shift = (dx*nx + dy*ny) * self.pixel_size()
        return shift

    def vr_motion(self, position, move, delta_z):
        # Virtual reality hand controller motion.
        for p in self._planes(front_shift = 1, back_shift = 0):
            if p:
                p.normal = move.transform_vector(p.normal)
                p.plane_point = move * p.plane_point

class ClipRotateMouseMode(MouseMode):
    '''
    Rotate clip planes.
    '''
    name = 'clip rotate'
    icon_file = 'icons/cliprot.png'

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        axis, angle = self._drag_axis_angle(dx, dy)
        self.clip_rotate(axis, angle)

    def _drag_axis_angle(self, dx, dy):
        '''Axis in camera coords, angle in degrees.'''
        from math import sqrt
        d = sqrt(dx*dx + dy*dy)
        axis = (dy/d, dx/d, 0) if d > 0 else (0,1,0)
        angle = d
        return axis, angle

    def wheel(self, event):
        d = event.wheel_value()
        self.clip_rotate(axis = (0,1,0), angle = 10*d)

    def clip_rotate(self, axis, angle):
        v = self.view
        scene_axis = v.camera.position.transform_vector(axis)
        from chimerax.core.geometry import rotation
        r = rotation(scene_axis, angle, v.center_of_rotation)
        for p in self._planes():
            p.normal = r.transform_vector(p.normal)
            p.plane_point = r * p.plane_point

    def _planes(self):
        v = self.view
        cp = v.clip_planes
        rplanes = [p for p in cp.planes() if p.camera_normal is None]
        if len(rplanes) == 0:
            from chimerax.std_commands.clip import adjust_plane
            pn, pf = cp.find_plane('near'), cp.find_plane('far')
            if pn is None and pf is None:
                # Create clip plane since none are enabled.
                b = v.drawing_bounds()
                p = adjust_plane('front', 0, b.center(), v.camera.view_direction(), cp)
                rplanes = [p]
            else:
                # Convert near/far clip planes to scene planes.
                if pn:
                    rplanes.append(adjust_plane('front', 0, pn.plane_point, pn.normal, cp))
                    cp.remove_plane('near')
                if pf:
                    rplanes.append(adjust_plane('back', 0, pf.plane_point, pf.normal, cp))
                    cp.remove_plane('far')
        return rplanes

    def vr_motion(self, position, move, delta_z):
        # Virtual reality hand controller motion.
        for p in self._planes():
            p.normal = move.transform_vector(p.normal)
            p.plane_point = move * p.plane_point

def standard_mouse_mode_classes():
    '''List of core MouseMode classes.'''
    mode_classes = [
        SelectMouseMode,
        SelectAddMouseMode,
        SelectSubtractMouseMode,
        SelectToggleMouseMode,
        RotateMouseMode,
        TranslateMouseMode,
        ZoomMouseMode,
        RotateAndSelectMouseMode,
        TranslateSelectedMouseMode,
        RotateSelectedMouseMode,
        ClipMouseMode,
        ClipRotateMouseMode,
        ObjectIdMouseMode,
        AtomCenterOfRotationMode,
        NullMouseMode,
    ]
    return mode_classes
