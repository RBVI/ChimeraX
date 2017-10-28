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


# -----------------------------------------------------------------------------
#
def label(session, objects = None, object_type = None, text = None,
          offset = None, orient = None, color = None, size = None, height = None, font = None,
          on_top = None):
    '''Create atom labels. The belong to a child model named "labels" of the structure.

    Parameters
    ----------
    objects : Objects or None
      Create labels on specified atoms, residues, pseudobonds, or bonds.
      If None then adjust settings of all existing labels.
    object_type : 'atoms', 'residues', 'pseudobonds', 'bonds'
      What type of object to label.
    text : string or "default"
      Displayed text of the label.
    offset : float 3-tuple or "default"
      Offset of label from atom center in screen coordinates in physical units (Angstroms)
    orient : float
      Reorient the labels to face the view direction only when the view direction changes
      changes by the specified number of degrees.  Default 0 makes the labels always face
      the view direction.  This option is primarily of interest with virtual reality viewing.
    color : Color or "default"
      Color of the label text.  If no color is specified black is used on light backgrounds
      and white is used on dark backgrounds.
    size : int or "default"
      Font size in pixels. Default 24.
    height : float or "fixed"
      Text height in scene units.  Or if "fixed" use fixed pixel height on screen.
    font : string or "default"
      Font name.  This must be a true type font installed on Mac in /Library/Fonts
      and is the name of the font file without the ".ttf" suffix.  Default "Arial".
    on_top : bool
      Whether labels always appear on top of other graphics (cannot be occluded).
      This is a per-structure attribute.  Default True.
    '''
    if object_type is None:
        if objects is None:
            otypes = ['atoms', 'residues', 'pseudobonds', 'bonds']
        elif len(objects.atoms) == 0:
            otypes = ['pseudobonds']
        else:
            otypes = ['atoms']
    else:
        otypes = [object_type]

    settings = {}
    if text == 'default':
        settings['text'] = None
    elif text is not None:
        settings['text'] = text
    if offset == 'default':
        settings['offset'] = None
    elif offset is not None:
        settings['offset'] = offset
    if orient is not None:
        settings['orient'] = orient
    from chimerax.core.colors import Color
    if isinstance(color, Color):
        settings['color'] = color.uint8x4()
    elif color == 'default':
        settings['color'] = None
    if size == 'default':
        settings['size'] = 24
    elif size is not None:
        settings['size'] = size
    if height == 'fixed':
        settings['height'] = None
    elif height is not None:
        settings['height'] = height
    if font == 'default':
        settings['font'] = 'Arial'
    elif font is not None:
        settings['font'] = font
        
    view = session.main_view
    lcount = 0
    for otype in otypes:
        if objects is None:
            mo = labeled_objects_by_model(session, otype)
        else:
            mo = objects_by_model(objects, otype)
        object_class = label_object_class(otype)
        for m, mobjects in mo:
            lm = labels_model(m, create = True)
            lm.add_labels(mobjects, object_class, view, settings, on_top)
            lcount += len(mobjects)
    if objects is None and lcount == 0:
        from chimerax.core.errors import UserError
        raise UserError('Label command requires an atom specifier to create labels.')

# -----------------------------------------------------------------------------
#
def label_delete(session, objects = None, object_type = None):
    '''Delete object labels.

    Parameters
    ----------
    objects : Objects or None
      Delete labels for specified atoms, residues, pseudobonds or bonds.  If None delete all labels.
    object_type : 'atoms', 'residues', 'pseudobonds', 'bonds'
      What type of object label to delete.
    '''
    if object_type is None:
        otypes = ['atoms', 'residues', 'pseudobonds', 'bonds']
    else:
        otypes = [object_type]

    delete_count = 0
    for otype in otypes:
        if objects is None:
            mo = labeled_objects_by_model(session, otype)
        else:
            mo = objects_by_model(objects, otype)
        for m, lbl_objects in mo:
            lm = labels_model(m)
            if lm is not None:
                delete_count += lm.delete_labels(lbl_objects)

    return delete_count

# -----------------------------------------------------------------------------
#
def label_object_class(object_type):
    if object_type == 'atoms':
        object_class = AtomLabel
    elif object_type == 'residues':
        object_class = ResidueLabel
    elif object_type == 'pseudobonds':
        object_class = PseudobondLabel
    elif object_type == 'bonds':
        object_class = BondLabel
    else:
        object_class = None
    return object_class

# -----------------------------------------------------------------------------
#
def objects_by_model(objects, object_type):
    atoms = objects.atoms
    if object_type == 'atoms':
        model_objects = atoms.by_structure
    elif object_type == 'residues':
        res = atoms.residues.unique()
        model_objects = res.by_structure
    elif object_type == 'pseudobonds':
        pbonds = objects.pseudobonds
        model_objects = pbonds.by_group
    elif object_type == 'bonds':
        bonds = objects.bonds
        model_objects = bonds.by_structure
    return model_objects

# -----------------------------------------------------------------------------
#
def labeled_objects_by_model(session, otype):
    oclass = label_object_class(otype)
    return [(lm.parent, lm.labeled_objects(oclass))
            for lm in session.models.list(type = ObjectLabels)]
        
# -----------------------------------------------------------------------------
#
def labels_model(parent, create = False):
    for lm in parent.child_models():
        if isinstance(lm, ObjectLabels):
            return lm
    if create:
        lm = ObjectLabels(parent.session)
        parent.add([lm])
    else:
        lm = None
    return lm

# -----------------------------------------------------------------------------
#
def register_label_command(logger):

    from chimerax.core.commands import CmdDesc, register, ObjectsArg, StringArg, FloatArg
    from chimerax.core.commands import Float3Arg, ColorArg, IntArg, BoolArg, EnumOf, Or, EmptyArg

    otype = EnumOf(('atoms','residues','pseudobonds','bonds'))
    DefArg = EnumOf(['default'])
    desc = CmdDesc(required = [('objects', Or(ObjectsArg, EmptyArg))],
                   optional = [('object_type', otype)],
                   keyword = [('text', Or(DefArg, StringArg)),
                              ('offset', Or(DefArg, Float3Arg)),
                              ('orient', FloatArg),
                              ('color', Or(DefArg, ColorArg)),
                              ('size', Or(DefArg, IntArg)),
                              ('height', Or(EnumOf(['fixed']), FloatArg)),
                              ('font', StringArg),
                              ('on_top', BoolArg)],
                   synopsis = 'Create atom labels')
    register('label', desc, label, logger=logger)
    desc = CmdDesc(required = [('objects', Or(ObjectsArg, EmptyArg))],
                   optional = [('object_type', otype)],
                   synopsis = 'Delete atom labels')
    register('label delete', desc, label_delete, logger=logger)
    desc = CmdDesc(synopsis = 'List available fonts')
    from .label2d import label_listfonts
    register('label listfonts', desc, label_listfonts, logger=logger)

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class ObjectLabels(Model):
    '''Model holding labels appearing next to atoms, residues, pseudobonds or bonds.'''

    pickable = False		# Don't allow mouse selection of labels
    
    def __init__(self, session):
        Model.__init__(self, 'labels', session)

        self.on_top = True		# Should labels always appear above other graphics
        
        self._label_drawings = {}	# Map object (Atom, Residue, Pseudobond, Bond) to ObjectLabel

        t = session.triggers
        self._handler = t.add_handler('graphics update', self._update_graphics_if_needed)

    def delete(self):
        if self._handler is not None:
            self.session.triggers.remove_handler(self._handler)
            self._handler = None
        Model.delete(self)

    def draw(self, renderer, place, draw_pass, selected_only=False):
        if self.on_top:
            renderer.enable_depth_test(False)
        Drawing.draw(self, renderer, place, draw_pass, selected_only)
        if self.on_top:
            renderer.enable_depth_test(True)
    
    def add_labels(self, objects, label_class, view, settings = {}, on_top = None):
        if on_top is not None:
            self.on_top = on_top
        ld = self._label_drawings
        for o in objects:
            if o not in ld:
                ld[o] = lo = label_class(o, view, **settings)
                self.add_drawing(lo)
            if settings:
                lo = ld[o]
                for k,v in settings.items():
                    setattr(lo, k, v)
                lo._needs_update = True
        if objects:
            self.redraw_needed()

    def delete_labels(self, objects):
        ld = self._label_drawings
        count = 0
        for o in objects:
            if o in ld:
                self.remove_drawing(ld[o])
                del ld[o]
                count += 1
        if len(ld) == 0:
            self.session.models.close([self])
        return count

    def label_count(self):
        return len(self._label_drawings)

    def labeled_objects(self, label_class = None):
        return [o for o,l in self._label_drawings.items()
                if label_class is None or isinstance(l, label_class)]

    def _update_graphics_if_needed(self, *_):
        if not self.visible:
            return
        # TODO: Only update if camera moved, atom display changed, label property text, color, offset changed...
        #  Currently every label has position recomputed every graphics update and it is slow for 50 labels.
        delo = []
        for o,ld in self._label_drawings.items():
            ld._update_graphics()
            if ld.object_deleted:
                delo.append(o)
        if delo:
            self.delete_labels(delo)

    SESSION_SAVE = True
    
    def take_snapshot(self, session, flags):
        lattrs = ('object', 'text', 'offset', 'orient', 'color', 'size', 'height', 'font')
        lstate = tuple({attr:getattr(ld, attr) for attr in lattrs}
                       for ld in self._label_drawings.values())
        data = {'model state': Model.take_snapshot(self, session, flags),
                'labels state': lstate,
                'on_top': self.on_top,
                'version': 1}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        s = ObjectLabels(session)
        s.set_state_from_snapshot(session, data)
        return s

    def set_state_from_snapshot(self, session, data):
        Model.set_state_from_snapshot(self, session, data['model state'])
        self.on_top = data['on_top']
        self._label_drawings = ld = {}
        v = self.session.main_view
        for ls in data['labels state']:
            o = ls['object']
            kw = {attr:ls[attr] for attr in ('text', 'offset', 'orient', 'color', 'size', 'height', 'font')}
            cls = label_class(o)
            ld[o] = ol = cls(o, v, **kw)
            self.add_drawing(ol)

    def reset_state(self, session):
        pass

# -----------------------------------------------------------------------------
#
def label_class(object):
    from chimerax.core.atomic import Atom, Residue, Pseudobond, Bond
    if isinstance(object, Atom):
        return AtomLabel
    elif isinstance(object, Residue):
        return ResidueLabel
    elif isinstance(object, Pseudobond):
        return PseudobondLabel
    elif isinstance(object, Bond):
        return BondLabel
    return None

# -----------------------------------------------------------------------------
#
from chimerax.core.graphics import Drawing
class ObjectLabel(Drawing):

    pickable = False		# Don't allow mouse selection of labels
    casts_shadows = False
    
    def __init__(self, object, view, offset = None, orient = 0, text = None, color = None,
                 size = 24, height = None, font = 'Arial'):
        Drawing.__init__(self, 'label %s' % self.default_text())

        self.object = object
        self.view = view	# View is used to update label position to face camera
        self._offset = offset
        self.orient = orient	# Degrees change in view direction before reorienting labels
        self._last_camera_position = None
        self._text = text
        self._color = color
        self.size = size
        self.height = height	# None or height in world coords.  If None used fixed screen size.
        self._pixel_size = (100,10)	# Size of label in pixels, calculated from size attribute

        self.font = font

        # Set initial billboard geometry
        from numpy import array, float32, int32
        vlist = array(((0,0,0), (1,0,0), (1,1,0), (0,1,0)), float32)
        tlist = array(((0, 1, 2), (0, 2, 3)), int32)
        tc = array(((0, 0), (1, 0), (1, 1), (0, 1)), float32)
        self.geometry = vlist, tlist
        self.texture_coordinates = tc
        self.use_lighting = False

        self._needs_update = True		# Has text, color, size, font changed.

    def default_text(self):
        '''Override this to define the default label text for object.'''
        return ''

    def default_offset(self):
        '''Override this to define the default offset for label.'''
        return (0,0,0)
        
    def location(self):
        '''Override this with position of label lower left corner in model coordinate system.'''
        return (0,0,0)

    def visible(self):
        '''Override this to control visibility of label based on visibility of atom, residue...'''
        return True

    def _get_offset(self):
        return self.default_offset() if self._offset is None else self._offset
    def _set_offset(self, offset):
        self._offset = offset
        self._needs_update = True
    offset = property(_get_offset, _set_offset)
    
    def _get_text(self):
        return self.default_text() if self._text is None else self._text
    def _set_text(self, text):
        self._text = text
        self._needs_update = True
    text = property(_get_text, _set_text)
    
    def _get_color(self):
        c = self._color
        if c is None:
            light_bg = (sum(self.view.background_color[:3]) > 1.5)
            rgba8 = (0,0,0,255) if light_bg else (255,255,255,255)
        else:
            rgba8 = c
        return rgba8
    def _set_color(self, color):
        self._color = color
        self._needs_update = True
    color = property(_get_color, _set_color)

    @property
    def object_deleted(self):
        return self.location() is None
            
    def draw(self, renderer, place, draw_pass, selected_only=False):
        if not self.display:
            return
        self._update_label_texture()  # This needs to be done during draw in case texture delete needed.
        Drawing.draw(self, renderer, place, draw_pass, selected_only)

    def _update_graphics(self):
        disp = self.visible()
        if disp != self.display:
            self.display = disp
        if self.display:
            self._position_label()

    def _update_label_texture(self):
        if not self._needs_update:
            return
        self._needs_update = False
        s = self.size
        rgba8 = (255,255,255,255)
        from .label2d import text_image_rgba
        text = self.text
        rgba = text_image_rgba(text, rgba8, s, self.font)
        if rgba is None:
            raise RuntimeError("Can't find font %s size %d for label '%s'" % (self.font, s, text))
        if self.texture is not None:
            self.texture.delete_texture()
        from chimerax.core.graphics import opengl
        t = opengl.Texture(rgba)
        self.texture = t
        Drawing.set_color(self, self.color)
        h,w,c = rgba.shape
        ps = (s*w/h, s)
        if ps != self._pixel_size:
            self._pixel_size = ps
            self._position_label()	# Size of billboard changed.

    def _position_label(self):
        # TODO: For VR when fixed scene height and orientation used, don't recalculate label position.
        xyz = self.location()
        if xyz is None:
            return	# Label deleted
        view = self.view
        spos = self.scene_position
        pw,ph = self._pixel_size
        sh = self.height	# Scene height
        if sh is None:
            psize = view.pixel_size(spos*xyz)
            w,h = psize * pw, psize * ph
        else:
            w, h = sh*pw/ph, sh
        cpos = view.camera.position	# Camera position in scene coords
        if self.orient > 0:
            lcp = self._last_camera_position
            from math import degrees
            if lcp is not None and degrees((lcp.inverse()*cpos).rotation_angle()) < self.orient:
                cpos = lcp
            else:
                self._last_camera_position = cpos
        clpos = spos.inverse() * cpos  # Camera pos in label drawing coords
        cam_xaxis, cam_yaxis, cam_zaxis = clpos.axes()
        from numpy import array, float32
        va = array((xyz, xyz + w*cam_xaxis, xyz + w*cam_xaxis + h*cam_yaxis, xyz + h*cam_yaxis), float32)
        offset = self.offset
        if offset is not None:
            va += clpos.apply_without_translation(offset)
        if (va == self.vertices).all():
            return 	# Don't set vertices causing redraw if label has not moved.
        self.vertices = va

# -----------------------------------------------------------------------------
#
class AtomLabel(ObjectLabel):
    def __init__(self, object, view, offset = None, orient = 0, text = None, color = None,
                 size = 24, height = None, font = 'Arial'):
        self.atom = object
        ObjectLabel.__init__(self, object, view, offset=offset, orient=orient, text=text, color=color,
                             size=size, height=height, font=font)
    def default_text(self):
        aname = self.atom.name
        return aname if aname else ('%d' % self.atom.residue.number)
    def default_offset(self):
        return (0.2+self.atom.display_radius, 0, 0.5)
    def location(self):
        a = self.atom
        return None if a.deleted else a.coord
    def visible(self):
        a = self.atom
        return (not a.deleted) and a.visible

# -----------------------------------------------------------------------------
#
class ResidueLabel(ObjectLabel):
    def __init__(self, object, view, offset = None, orient = 0, text = None, color = None,
                 size = 24, height = None, font = 'Arial'):
        self.residue = object
        ObjectLabel.__init__(self, object, view, offset=offset, orient=orient, text=text, color=color,
                             size=size, height=height, font=font)
    def default_text(self):
        r = self.residue
        return '%s %d' % (r.name, r.number)
    def location(self):
        r = self.residue
        return None if r.deleted else r.center
    def visible(self):
        r = self.residue
        return (not r.deleted) and (r.ribbon_display or r.atoms.displays.any())

# -----------------------------------------------------------------------------
#
class PseudobondLabel(ObjectLabel):
    def __init__(self, object, view, offset = None, orient = 0, text = None, color = None,
                 size = 24, height = None, font = 'Arial'):
        self.pseudobond = object
        ObjectLabel.__init__(self, object, view, offset=offset, orient=orient, text=text, color=color,
                             size=size, height=height, font=font)
    def default_text(self):
        return '%.2f' % self.pseudobond.length
    def default_offset(self):
        return (0.2+self.pseudobond.radius, 0, 0.5)
    def location(self):
        pb = self.pseudobond
        if pb.deleted:
            return None
        a1,a2 = pb.atoms
        sxyz = 0.5 * (a1.scene_coord + a2.scene_coord)	# Midpoint
        xyz = self.scene_position.inverse() * sxyz
        return xyz
    def visible(self):
        pb = self.pseudobond
        return (not pb.deleted) and pb.shown

# -----------------------------------------------------------------------------
#
BondLabel = PseudobondLabel
