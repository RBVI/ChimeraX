# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
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
        mode = self.mode
        if event.shift_down() and mode == 'replace':
            mode = 'toggle'

        if not self.double_click:
            if self._is_drag(event):
                # Select objects in rectangle
                mouse_drag_select(self.mouse_down_position, event, mode, self.session, self.view)
            else:
                # Select object under pointer
                mouse_select(event, mode, self.session, self.view)
        MouseMode.mouse_up(self, event)

    def mouse_double_click(self, event):
        '''Show a context menu when double-clicking in select mode.'''
        MouseMode.mouse_double_click(self, event)
        entries = []
        dangerous_entries = []
        ses = self.session
        import inspect
        for entry in SelectMouseMode._menu_entry_info:
            # SelectContextMenuAction methods used to only take session arg, so subclasses of old
            # definition might expect only session arg, therefore inspect the callable
            sig = inspect.signature(entry.criteria)
            args = (ses,) if len(sig.parameters) == 1 else (ses, event)
            if entry.criteria(*args):
                if entry.dangerous:
                    dangerous_entries.append((entry, args))
                else:
                    entries.append((entry, args))
        entries.sort(key = lambda e: e[0].label(*e[1]))
        dangerous_entries.sort(key = lambda e: e[0].label(*e[1]))
        from Qt.QtWidgets import QMenu
        from Qt.QtGui import QAction
        menu = QMenu(ses.ui.main_window)
        actions = []
        all_entries = entries
        if dangerous_entries:
            all_entries = all_entries + [(None, None)] + dangerous_entries
        if all_entries:
            for entry, args in all_entries:
                if entry is None:
                    menu.addSeparator()
                    continue
                action = QAction(entry.label(*args))
                action.triggered.connect(lambda *, cb=entry.callback, args=args: cb(*args))
                menu.addAction(action)
                actions.append(action) # keep reference
        else:
            menu.addAction("No applicable actions")
        # this will prevent atom-spec balloons from showing up
        from Qt.QtCore import QPoint
        p = QPoint(*event.global_position())
        ses.ui.post_context_menu(menu, p)

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
        v = self.view
        w,h = v.window_size
        v.draw_xor_rectangle(dx, h-dy, x, h-y, self._xor_color)
        self._drawn_rectangle = (dx,dy), (x,y)

    def _undraw_drag_rectangle(self):
        dr = self._drawn_rectangle
        if dr:
            (dx,dy), (x,y) = dr
            v = self.view
            w,h = v.window_size
            v.draw_xor_rectangle(dx, h-dy, x, h-y, self._xor_color)
            self._drawn_rectangle = None

    @property
    def _xor_color(self):
        from chimerax.core.colors import rgba_to_rgba8
        bg_color = rgba_to_rgba8(self.view.background_color)
        xor_color = tuple(bc ^ dc for bc,dc in zip(bg_color, self.drag_color))
        return xor_color

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        pick = event.picked_object(self.view)
        select_pick(self.session, pick, self.mode)

    def vr_motion(self, event):
        # Virtual reality hand controller motion.
        delta_z = event.room_vertical_motion  # meters
        if delta_z > 0.10:
            from chimerax.core.commands import run
            run(self.session, 'select up')
        elif delta_z < -0.10:
            from chimerax.core.commands import run
            run(self.session, 'select down')
        else:
            return 'accumulate drag'

class SelectContextMenuAction:
    '''Methods implementing a context-menu entry shown when double-clicking in select mode.'''
    def label(self, session, event):
        '''Returns the text of the menu entry.'''
        return 'unknown'
    def criteria(self, session, event):
        '''
        Return a boolean that indicates whether the menu should include this entry
        (usually based on the current contents of the selection).
        '''
        return False
    def callback(self, session, event):
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
    pick = view.picked_object(x, y)
    select_pick(session, pick, mode)

def mouse_drag_select(start_xy, event, mode, session, view):
    sx, sy = start_xy
    x,y = event.position()
    pick = view.rectangle_pick(sx,sy,x,y)
    select_pick(session, pick, mode)

def select_pick(session, pick, mode = 'replace'):
    sel = session.selection
    from chimerax.core.undo import UndoState
    undo_state = UndoState("select")
    sel.undo_add_selected(undo_state, False)
    if pick is None or (isinstance(pick, list) and len(pick) == 0):
        if mode == 'replace':
            from chimerax.core.commands import run
            run(session, 'select clear')
            session.logger.status('cleared selection')
    else:
        if mode == 'replace':
            sel.clear()
            mode = 'add'
        if isinstance(pick, list):
            for p in pick:
                p.select(mode)
            if pick:
                session.logger.info('Drag select of %s' % _pick_description(pick))
        else:
            spec = pick.specifier()
            if mode == 'add' and spec:
                from chimerax.core.commands import run
                run(session, 'select %s' % spec)
            elif mode == 'toggle' and spec and hasattr(pick, 'selected'):
                from chimerax.core.commands import run
                operation = 'subtract' if pick.selected() else 'add'
                run(session, 'select %s %s' % (operation, spec))
            else:
                pick.select(mode)
    sel.clear_promotion_history()
    sel.undo_add_selected(undo_state, True, old_state=False)
    session.undo.register(undo_state)

def _pick_description(picks):
    pdesc = []
    item_counts = {}
    for p in picks:
        d = p.description()
        if d is not None:
            try:
                count, name = d.split(maxsplit = 1)
                c = int(count)
                item_counts[name] = item_counts.get(name,0) + c
            except Exception:
                pdesc.append(d)
    pdesc.extend('%d %s' % (count, name) for name, count in item_counts.items())
    desc = ', '.join(pdesc)
    return desc

class MoveMouseMode(MouseMode):
    '''
    Mouse mode to rotate and translate models by dragging.
    Actually the camera is moved if acting on all models in the scene.
    '''
    click_to_select = False
    mouse_action = 'translate'	# translate or rotate
    move_atoms = False		# Move atoms, else move whole models

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self.speed = 1
        self._z_rotate = False
        self._independent_model_rotation = False  # Rotate each model about its center
        self._moved = False

        # Restrict rotation to this axis using coordinate system of first model.
        self._restrict_to_axis = None

        # Restrict translation to the plane perpendicular to this axis.
        # Axis is in coordinate system of first model.
        self._restrict_to_plane = None

        # Moving atoms
        self._atoms = None

        # Undo
        self._starting_atom_scene_coords = None
        self._starting_model_positions = None

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        a = self.action(event)
        self._set_z_rotation(event)
        if self.move_atoms:
            from chimerax.atomic import selected_atoms
            self._atoms = selected_atoms(self.session)
        self._undo_start()

    def mouse_drag(self, event):
        a = self.action(event)
        if a == 'rotate' or a == 'rotate z':
            axis, angle = self._rotation_axis_angle(event, z_rotate = (a == 'rotate z'))
            self._rotate(axis, angle)
        elif a == 'translate' or a == 'translate z':
            shift = self._translation(event, z_translate = (a == 'translate z'))
            self._translate(shift)
        self._moved = True
        self._log_motion()

    def mouse_up(self, event):
        if self.click_to_select:
            if event.position() == self.mouse_down_position:
                mode = 'toggle' if event.shift_down() else 'replace'
                mouse_select(event, mode, self.session, self.view)
        MouseMode.mouse_up(self, event)

        self._undo_save()
        self._log_command()

        if self.move_atoms:
            self._atoms = None

    def wheel(self, event):
        d = event.wheel_value()
        if self.move_atoms:
            from chimerax.atomic import selected_atoms
            self._atoms = selected_atoms(self.session)
        a = self.action(event)
        if a == 'rotate':
            self._rotate((0,1,0), 10*d)
        elif a == 'translate':
            self._translate((0,0,100*d))

    def action(self, event):
        a = self.mouse_action
        if event.shift_down():
            # Holding shift key switches between rotation and translation
             if a == 'rotate':
                 a = 'translate'
             elif a == 'translate':
                 a = 'rotate'
        if event.ctrl_down():
            # Holding control restricts to z-axis rotation or translation
            a = a + ' z'
        if self._z_rotate and a == 'rotate':
            a = 'rotate z'
        return a

    def touchpad_two_finger_trans(self, event):
        move = event.two_finger_trans
        if self.mouse_action=='rotate':
            tp = self.session.ui.mouse_modes.trackpad
            from math import sqrt
            dx, dy = move
            turns = sqrt(dx*dx + dy*dy)*tp.full_width_translation_distance/tp.full_rotation_distance
            angle = tp.trackpad_speed*360*turns
            self._rotate((dy, dx, 0), angle)

    def touchpad_three_finger_trans(self, event):
        dx, dy = event.three_finger_trans
        if self.mouse_action=='translate':
            tp = self.session.ui.mouse_modes.trackpad
            ww = self.session.view.window_size[0] # window width in pixels
            s = tp.trackpad_speed*ww
            self._translate((s*dx, -s*dy, 0))

    def touchpad_two_finger_twist(self, event):
        angle = event.two_finger_twist
        if self.mouse_action=='rotate':
            self._rotate((0,0,1), angle)

    def _set_z_rotation(self, event):
        x,y = event.position()
        w,h = self.view.window_size
        cx, cy = x-0.5*w, y-0.5*h
        from math import sqrt
        r = sqrt(cx*cx + cy*cy)
        fperim = 0.9
        self._z_rotate = (r > fperim*0.5*min(w,h))

    def _rotate(self, axis, angle):
        # Convert axis from camera to scene coordinates
        saxis = self.camera_position.transform_vector(axis)
        angle *= self.speed
        if self._moving_atoms:
            from chimerax.geometry import rotation
            self._move_atoms(rotation(saxis, angle, center = self._atoms_center()))
        elif self._independent_model_rotation:
            for model in self.models():
                self.view.rotate(saxis, angle, [model])
            # Make sure rotation shown before another mouse event causes another rotation.
            # Otherwise dozens of mouse events can be handled with no redrawing.
            self.session.update_loop.update_graphics_now()
        else:
            self.view.rotate(saxis, angle, self.models())

    def _rotation_axis_angle(self, event, z_rotate = False):
        '''Returned axis is in camera coordinate system.'''
        dx, dy = self.mouse_motion(event)
        import math
        angle = 0.5*math.sqrt(dx*dx+dy*dy)
        if self._restrict_to_axis is not None:
            axis = self._restricted_axis()
            if dy*axis[0]+dx*axis[1] < 0:
                angle = -angle
        elif z_rotate:
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

    def _restricted_axis(self):
        '''Return restricted axis of rotation.'''
        raxis = self._restrict_to_axis
        models = self.models()
        if models is None:
            scene_axis = raxis
        else:
            scene_axis = models[0].position.transform_vector(raxis)
        axis = self.camera_position.inverse().transform_vector(scene_axis)	# Camera coords
        return axis

    def _translate(self, shift):
        psize = self.pixel_size()
        s = tuple(dx*psize*self.speed for dx in shift)     # Scene units
        step = self.camera_position.transform_vector(s)    # Scene coord system
        if self._moving_atoms:
            from chimerax.geometry import translation
            self._move_atoms(translation(step))
        else:
            self.view.translate(step, self.models(), move_near_far_clip_planes = True)

    def _translation(self, event, z_translate = False):
        '''Returned shift is in camera coordinates.'''
        dx, dy = self.mouse_motion(event)
        shift = (dx, -dy, 0)
        if self._restrict_to_plane is not None:
            shift = self._restricted_shift(shift)
        elif z_translate:
            shift = (0, 0, dy)
        return shift

    def _restricted_shift(self, shift):
        '''Return shift resticted to be in a plane.'''
        raxis = self._restrict_to_plane
        models = self.models()
        if models is None:
            scene_axis = raxis
        else:
            scene_axis = models[0].position.transform_vector(raxis)
        axis = self.camera_position.inverse().transform_vector(scene_axis)	# Camera coords
        from chimerax.geometry import normalize_vector, inner_product
        axis = normalize_vector(axis)
        rshift = -inner_product(axis, shift) * axis + shift
        return rshift

    def models(self):
        return None

    @property
    def _moving_atoms(self):
        return self.move_atoms and self._atoms is not None and len(self._atoms) > 0

    def _move_atoms(self, transform):
        atoms = self._atoms
        atoms.scene_coords = transform * atoms.scene_coords

    def _atoms_center(self):
        return self._atoms.scene_coords.mean(axis=0)

    def _undo_start(self):
        if self._moving_atoms:
            self._starting_atom_scene_coords = self._atoms.scene_coords
        else:
            models = self.models()
            self._starting_model_positions = None if models is None else [(m,m.position) for m in models]
        self._moved = False

    def _undo_save(self):
        if self._moved:
            if self._moving_atoms:
                if self._starting_atom_scene_coords is not None:
                    from chimerax.core.undo import UndoState
                    undo_state = UndoState('move atoms')
                    a = self._atoms
                    undo_state.add(a, "scene_coords", self._starting_atom_scene_coords, a.scene_coords)
                    self.session.undo.register(undo_state)
            elif self._starting_model_positions is not None:
                from chimerax.core.undo import UndoState
                undo_state = UndoState('move models')
                smp = self._starting_model_positions
                models = [m for m, pos in smp]
                start_model_positions = [pos for m, pos in smp]
                new_model_positions = [m.position for m in models]
                undo_state.add(models, "position", start_model_positions, new_model_positions,
                               option='S')
                self.session.undo.register(undo_state)

        self._starting_atom_scene_coords = None
        self._starting_model_positions = None

    def _log_command(self):
        if not self._moved:
            return
        cmd = self._move_command()
        if not cmd:
            return
        from chimerax.core.commands import log_equivalent_command
        log_equivalent_command(self.session, cmd)

    def _move_command(self):
        models = self.models()
        if models and not self._independent_model_rotation:
            from chimerax.std_commands.view import model_positions_string
            cmd = 'view matrix models %s' % model_positions_string(models)
        else:
            cmd = None
        return cmd

    def _log_motion(self):
        from chimerax.core.commands import motion_commands_enabled, motion_command
        if not motion_commands_enabled(self.session):
            return
        cmd = self._move_command()
        if not cmd:
            return
        motion_command(self.session, cmd)

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        if self.move_atoms:
            from chimerax.atomic import selected_atoms
            self._atoms = selected_atoms(self.session)
        self._undo_start()

    def vr_motion(self, event):
        # Virtual reality hand controller motion.
        if self._moving_atoms:
            self._move_atoms(event.motion)
        else:
            self.view.move(event.motion, self.models())
        self._moved = True
        self._log_motion()

    def vr_release(self, event):
        # Virtual reality hand controller button release.
        self._undo_save()
        self._log_command()

class RotateMouseMode(MoveMouseMode):
    '''
    Mouse mode to rotate objects (actually the camera is moved) by dragging.
    Mouse drags initiated near the periphery of the window cause a screen z rotation,
    while other mouse drags use rotation axes lying in the plane of the screen and
    perpendicular to the direction of the drag.
    '''
    name = 'rotate'
    icon_file = 'icons/rotate.png'
    mouse_action = 'rotate'

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

class RotateSelectedModelsMouseMode(RotateMouseMode):
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

class RotateZSelectedModelsMouseMode(RotateSelectedModelsMouseMode):
    '''
    Rotate selected models about first model z axis.
    '''
    name = 'rotate z selected models'
    def __init__(self, session):
        RotateSelectedModelsMouseMode.__init__(self, session)
        self._restrict_to_axis = (0,0,1)
        self._restrict_to_plane = (0,0,1)

class RotateIndependentMouseMode(MoveMouseMode):
    '''
    Mouse mode to rotate each displayed model about its own center.
    '''
    name = 'rotate independent'
    icon_file = None  # TODO: Make icon
    mouse_action = 'rotate'
    def __init__(self, session):
        MoveMouseMode.__init__(self, session)
        self._independent_model_rotation = True
    def models(self):
        models = [m for m in self.session.models.list()
                  if m.visible and len(m.id) == 1]
        if len(models) == 1:
            # If we have one grouping model then tile the child models.
            m = models[0]
            from chimerax.core.models import Model
            if m.empty_drawing() and type(m) is Model and len(m.child_models()) > 1:
                models = m.child_models()
        return models

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

class TranslateMouseMode(MoveMouseMode):
    '''
    Mouse mode to move objects in x and y (actually the camera is moved) by dragging.
    '''
    name = 'translate'
    icon_file = 'icons/translate.png'
    mouse_action = 'translate'

class TranslateSelectedModelsMouseMode(TranslateMouseMode):
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

class TranslateXYSelectedModelsMouseMode(TranslateSelectedModelsMouseMode):
    '''
    Translate selected models only in x and y of the first selected models coordinate system.
    '''
    name = 'translate xy selected models'
    def __init__(self, session):
        TranslateSelectedModelsMouseMode.__init__(self, session)
        self._restrict_to_plane = (0,0,1)
        self._restrict_to_axis = (0,0,1)

class MovePickedModelsMouseMode(TranslateMouseMode):
    '''
    Mouse mode to translate picked models.
    '''
    name = 'move picked models'
    icon_file = 'icons/move_picked_model.png'

    def __init__(self, session):
        TranslateMouseMode.__init__(self, session)
        self._picked_models = None

    def mouse_down(self, event):
        x,y = event.position()
        pick = self.view.picked_object(x, y)
        self._pick_model(pick)
        TranslateMouseMode.mouse_down(self, event)

    def _pick_model(self, pick):
        self._picked_models = None
        if pick and hasattr(pick, 'drawing'):
            m = pick.drawing()
            from chimerax.core.models import Model
            if isinstance(m, Model):
                self._picked_models = [m]

    def mouse_up(self, event):
        TranslateMouseMode.mouse_up(self, event)
        self._picked_models = None

    def models(self):
        return self._picked_models

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        pick = event.picked_object(self.view)
        self._pick_model(pick)
        TranslateMouseMode.vr_press(self, event)

    def vr_release(self, event):
        # Virtual reality hand controller button release.
        TranslateMouseMode.vr_release(self, event)
        self._picked_models = None

class TranslateSelectedAtomsMouseMode(TranslateMouseMode):
    '''
    Mouse mode to translate selected atoms.
    '''
    name = 'translate selected atoms'
    icon_file = 'icons/move_atoms.png'
    move_atoms = True

class RotateSelectedAtomsMouseMode(RotateMouseMode):
    '''
    Mouse mode to rotate selected atoms.
    '''
    name = 'rotate selected atoms'
    icon_file = 'icons/rotate_atoms.png'
    move_atoms = True

class ZoomMouseMode(MouseMode):
    '''
    Mouse mode to move objects in z, actually the camera is moved
    and the objects remain at their same scene coordinates.
    '''
    name = 'zoom'
    icon_file = 'icons/zoom.png'
    def __init__(self, session):
        MouseMode.__init__(self, session)
        self.speed = 1

    def mouse_drag(self, event):

        dx, dy = self.mouse_motion(event)
        psize = self.pixel_size()
        delta_z = 3*psize*dy*self.speed
        self.zoom(delta_z, stereo_scaling = not event.alt_down())

    def wheel(self, event):
        d = event.wheel_value()
        psize = self.pixel_size()
        delta_z = 100*d*psize*self.speed
        self.zoom(delta_z, stereo_scaling = not event.alt_down())

    def touchpad_two_finger_scale(self, event):
        scale = event.two_finger_scale
        v = self.session.view
        wpix = v.window_size[0]
        psize = v.pixel_size()
        d = (scale-1)*wpix*psize
        self.zoom(d)

    def zoom(self, delta_z, stereo_scaling = False):
        v = self.view
        c = v.camera
        if stereo_scaling and hasattr(c, 'eye_separation_scene'):
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
        from Qt.QtGui import QCursor
        if ui.topLevelAt(QCursor.pos()) != ui.main_window:
            return
        # ensure there's no popup menu above the graphics
        apw = ui.activePopupWidget()
        from Qt.QtCore import QPoint
        if apw and ui.topLevelAt(apw.mapToGlobal(QPoint())) == ui.main_window:
            return
        x,y = position
        p = self.view.picked_object(x, y)

        # Show atom spec balloon
        pu = ui.main_window.graphics_window.popup
        if p:
            pu.show_text(p.description(), (x+10,y))
            self.session.triggers.activate_trigger('mouse hover', p)
        else:
            pu.hide()

    def move_after_pause(self):
        # Hide atom spec balloon
        self.session.ui.main_window.graphics_window.popup.hide()

class CenterOfRotationMode(MouseMode):
    '''
    Clicking on an atom, bond, ribbon, pseudobond or volume surface
    sets the center of rotation at that position.
    '''
    name = 'pivot'
    icon_file = 'icons/pivot.png'

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        xyz = _picked_xyz(event, self.session)
        if xyz is not None:
            from chimerax.std_commands import cofr
            cofr.cofr(self.session, pivot=xyz)

def _picked_xyz(event, session):
    x,y = event.position()
    view = session.main_view
    pick = view.picked_object(x, y)
    from chimerax.atomic import PickedResidue, PickedBond, PickedPseudobond
    from chimerax.map import PickedMap
    from chimerax.graphics import PickedTriangle
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
    elif isinstance(pick, (PickedMap, PickedTriangle)) and hasattr(pick, 'position'):
        xyz = pick.position
    else:
        xyz = None
    return xyz

class MoveToCenterMode(MouseMode):
    '''
    Clicking on an atom, bond, ribbon, pseudobond or volume surface
    centers the view on that point and sets the center of rotation at that position.
    '''
    name = 'center'

    frames = 10		# Animate motion over this number of frames

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        xyz = _picked_xyz(event, self.session)
        if xyz is None:
            return

        # Move camera so it is centered on picked point.
        c = self.session.main_view.camera
        cx,cy,cz = c.position.inverse() * xyz
        steps = self.frames
        from chimerax.core.commands import Axis
        mxy = (-cx/steps, -cy/steps, 0)
        axis = Axis(coords = mxy)
        from chimerax.std_commands.move import move
        move(self.session, axis, frames = steps)

        # Set center of rotation
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

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

        # Create new clip planes if needed
        cp = self.view.clip_planes
        nplanes = len(cp.planes())
        front_shift, back_shift = self.which_planes(event)
        self._planes(front_shift, back_shift)
        self._created_planes = (len(cp.planes()) > nplanes)

    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        front_shift, back_shift = self.which_planes(event)
        self.clip_move((dx,-dy), front_shift, back_shift)

    def mouse_up(self, event):
        moved = (event.position() != self.mouse_down_position)
        MouseMode.mouse_up(self, event)	# This clears mouse down position.
        if not moved and not self._created_planes:
            # Click without drag -> turn off clipping.
            self.view.clip_planes.clear()

    def which_planes(self, event):
        shift, alt = event.shift_down(), event.alt_down()
        front_shift = 1 if shift or not alt else 0
        back_shift = 0 if not (alt or shift) else (1 if alt and shift else -1)
        return front_shift, back_shift

    def wheel(self, event):
        d = event.wheel_value()
        psize = self.pixel_size()
        front_shift, back_shift = self.which_planes(event)
        self.clip_move(None, front_shift, back_shift, delta = 10*psize*d)

    def clip_move(self, delta_xy, front_shift, back_shift, delta = None):
        pf, pb = self._planes(front_shift, back_shift)
        if pf is None and pb is None:
            return

        from chimerax.graphics import SceneClipPlane, CameraClipPlane
        p = pf or pb
        if delta is not None:
            d = delta
        elif isinstance(p, SceneClipPlane):
            # Move scene clip plane
            d = self._tilt_shift(delta_xy, self.view.camera, p.normal)
        elif isinstance(p, CameraClipPlane):
            # near/far clip
            d = delta_xy[1]*self.pixel_size()

        # Check if slab thickness becomes less than zero.
        dt = -d*(front_shift+back_shift)
        if pf and pb and dt < 0:
            from chimerax.geometry import inner_product
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

        if not p.planes():
            from .settings import clip_settings
            use_scene_planes = (clip_settings.mouse_clip_plane_type == 'scene planes')
        else:
            use_scene_planes = (p.find_plane('front') or p.find_plane('back'))

        pfname, pbname = ('front','back') if use_scene_planes else ('near','far')

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

    def vr_motion(self, event):
        # Virtual reality hand controller motion.
        move = event.motion
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

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)

        # Create new clip planes if needed
        cp = self.view.clip_planes
        nplanes = len(cp.planes())
        self._planes()
        self._created_planes = (len(cp.planes()) > nplanes)

    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        axis, angle = self._drag_axis_angle(dx, dy)
        self.clip_rotate(axis, angle)

    def mouse_up(self, event):
        moved = (event.position() != self.mouse_down_position)
        MouseMode.mouse_up(self, event)	# This clears mouse down position.
        if not moved and not self._created_planes:
            # Click without drag -> turn off clipping.
            self.view.clip_planes.clear()

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
        from chimerax.geometry import rotation
        r = rotation(scene_axis, angle, v.center_of_rotation)
        for p in self._planes():
            p.normal = r.transform_vector(p.normal)
            p.plane_point = r * p.plane_point

    def _planes(self):
        v = self.view
        cp = v.clip_planes
        from chimerax.graphics import SceneClipPlane
        rplanes = [p for p in cp.planes() if isinstance(p, SceneClipPlane)]
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

    def vr_motion(self, event):
        # Virtual reality hand controller motion.
        move = event.motion
        for p in self._planes():
            p.normal = move.transform_vector(p.normal)
            p.plane_point = move * p.plane_point

class SwipeAsScrollMouseMode(MouseMode):
    '''
    Reinterprets the vertical component of a multi-touch swiping action as a
    mouse scroll, and passes it on to the currently-mapped mode.
    '''
    name = 'swipe as scroll'
    def _scrollwheel_mode(self, modifiers):
        return self.session.ui.mouse_modes.mode(button='wheel', modifiers=modifiers)

    def _wheel_value(self, dy):
        tp = self.session.ui.mouse_modes.trackpad
        speed = tp.trackpad_speed
        wcp = tp.wheel_click_pixels
        fwd = tp.full_width_translation_distance
        delta = speed * dy * fwd / wcp
        return delta

    def touchpad_two_finger_trans(self, event):
        self._pass_event(event, 'two_finger_trans')

    def touchpad_three_finger_trans(self, event):
        self._pass_event(event, 'three_finger_trans')

    def touchpad_four_finger_trans(self, event):
        self._pass_event(event, 'four_finger_trans')

    def _pass_event(self, event, event_type):
        _, dy = getattr(event, event_type)
        swm = self._scrollwheel_mode(event.modifiers)
        if swm:
            wv = self._wheel_value(dy)
            from .mousemodes import MouseEvent
            swm.wheel(MouseEvent(position=event.position(), wheel_value=wv, modifiers=event.modifiers))



def standard_mouse_mode_classes():
    '''List of core MouseMode classes.'''
    mode_classes = [
        SelectMouseMode,
        SelectAddMouseMode,
        SelectSubtractMouseMode,
        SelectToggleMouseMode,
        RotateMouseMode,
        RotateAndSelectMouseMode,
        RotateSelectedModelsMouseMode,
        RotateZSelectedModelsMouseMode,
        RotateSelectedAtomsMouseMode,
        RotateIndependentMouseMode,
        TranslateMouseMode,
        TranslateSelectedModelsMouseMode,
        TranslateXYSelectedModelsMouseMode,
        MovePickedModelsMouseMode,
        TranslateSelectedAtomsMouseMode,
        ZoomMouseMode,
        ClipMouseMode,
        ClipRotateMouseMode,
        ObjectIdMouseMode,
        CenterOfRotationMode,
        MoveToCenterMode,
        SwipeAsScrollMouseMode,
        NullMouseMode,
    ]
    return mode_classes
