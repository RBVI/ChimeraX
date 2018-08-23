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


# -----------------------------------------------------------------------------
#
def label(session, objects = None, object_type = None, text = None,
          offset = None, orient = None, color = None, background = None,
          size = None, height = None, font = None, on_top = None):
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
      by the specified number of degrees.  Default 0 makes the labels always face
      the view direction.  This option is primarily of interest with virtual reality viewing.
      This is a per-structure setting.
    color : Color or "default"
      Color of the label text.  If no color is specified black is used on light backgrounds
      and white is used on dark backgrounds.
    background : Color or "none"
      Draw rectangular label background in this color, or if "none", background is transparent.
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
    from chimerax.core.colors import Color
    if isinstance(color, Color):
        settings['color'] = color.uint8x4()
    elif color == 'default':
        settings['color'] = None
    if isinstance(background, Color):
        settings['background'] = background.uint8x4()
    elif background == 'none':
        settings['background'] = None
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
            if orient is not None:
                lm._reorient_angle = orient
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
    NoneArg = EnumOf(['none'])
    desc = CmdDesc(required = [('objects', Or(ObjectsArg, EmptyArg))],
                   optional = [('object_type', otype)],
                   keyword = [('text', Or(DefArg, StringArg)),
                              ('offset', Or(DefArg, Float3Arg)),
                              ('orient', FloatArg),
                              ('color', Or(DefArg, ColorArg)),
                              ('background', Or(NoneArg, ColorArg)),
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
        self._window_size = session.main_view.window_size

        self._labels = []		# list of ObjectLabel
        self._object_label = {}		# Map object (Atom, Residue, Pseudobond, Bond) to ObjectLabel

        t = session.triggers
        self._update_graphics_handler = t.add_handler('graphics update', self._update_graphics_if_needed)
        self._background_color_handler = t.add_handler('background color changed', self._background_changed_cb)

        from chimerax.atomic import get_triggers
        ta = get_triggers(session)
        self._structure_change_handler = ta.add_handler('changes', self._structure_changed)
        
        # Optimize label repositioning when minimum camera move specified.
        self._reorient_angle = 0
        self._last_camera_position = None

        self.use_lighting = False
        Model.set_color(self, (255,255,255,255))	# Do not modulate texture colors

        self._texture_width = 2048			# Pixels.
        self._texture_needs_update = True		# Has text, color, size, font changed.
        self._positions_need_update = True		# Has label position changed relative to atom?
        self._visibility_needs_update = True		# Does an atom hide require a label to hide?
        
    def delete(self):
        for hattr in ('_update_graphics_handler', '_background_color_handler'):
            h = getattr(self, hattr)
            if h is not None:
                self.session.triggers.remove_handler(h)
                setattr(self, hattr, None)
            
        h = self._structure_change_handler
        if h is not None:
            from chimerax.atomic import get_triggers
            get_triggers(self.session).remove_handler(h)
            self._structure_change_handler = None
        
        Model.delete(self)

    def _get_single_color(self):
        from chimerax.core.colors import most_common_color
        lcolors = [ld.color for ld in self._labels]
        c = most_common_color(lcolors) if lcolors else None
        return c
    def _set_single_color(self, color):
        for ld in self._labels:
            ld.color = color
        self._texture_needs_update = True
        self.redraw_needed()
    single_color = property(_get_single_color, _set_single_color)

    def draw(self, renderer, place, draw_pass, selected_only=False):
        if self.on_top:
            renderer.enable_depth_test(False)
        Model.draw(self, renderer, place, draw_pass, selected_only)
        if self.on_top:
            renderer.enable_depth_test(True)
    
    def add_labels(self, objects, label_class, view, settings = {}, on_top = None):
        if on_top is not None:
            self.on_top = on_top
        ol = self._object_label
        for o in objects:
            if o not in ol:
                ol[o] = lo = label_class(o, view, **settings)
                self._labels.append(lo)
            if settings:
                lo = ol[o]
                for k,v in settings.items():
                    setattr(lo, k, v)
        if objects:
            self._texture_needs_update = True
            self.redraw_needed()

    def delete_labels(self, objects):
        ol = self._object_label
        count = 0
        for o in objects:
            if o in ol:
                del ol[o]
                count += 1
        if count > 0:
            self._labels = [l for l in self._labels if l.object in ol]
            self._texture_needs_update = True
        if len(ol) == 0:
            self.session.models.close([self])
        return count

    def label_count(self):
        return len(self._labels)

    def labeled_objects(self, label_class = None):
        return [l.object for l in self._labels
                if label_class is None or isinstance(l, label_class)]

    def _structure_changed(self, tname, changes):
        # If atoms undisplayed, or radii change, or names change, can effect label display.
        # TODO: Update textures if label text changes
        # self._texture_needs_update = True
        self._visibility_needs_update = True
        self._positions_need_update = True
        if changes.num_deleted_atoms() > 0 or changes.num_deleted_bonds() > 0 or changes.num_deleted_pseudobonds() > 0:
            self.delete_labels([l.object for l in self._labels if l.object_deleted])
        self.redraw_needed()

    def _background_changed_cb(self, *_):
        self._texture_needs_update = True
            
    def _update_graphics_if_needed(self, *_):
        if not self.visible:
            return

        v = self.session.main_view
        resize = (v.window_size != self._window_size)
        if resize:
            self._window_size = v.window_size
            self._positions_need_update = True
            
        camera_move = v.camera.redraw_needed
        if camera_move:
            ra = self._reorient_angle
            if ra == 0:
                self._positions_need_update = True
            elif ra > 0:
                # Don't update label positions if minimum camera motion has not occured.
                # This optimization is to maintain high frame rate with virtual reality.
                cpos = self.session.main_view.camera.position
                lcpos = self._last_camera_position
                from math import degrees
                if lcpos is None:
                    self._last_camera_position = cpos
                elif degrees((cpos.inverse() * lcpos).rotation_angle()) >= ra:
                    self._positions_need_update = True
                    self._last_camera_position = cpos

        if self._texture_needs_update:
            self._rebuild_label_graphics()
        else:
            if self._positions_need_update:
                self._reposition_label_graphics()
            if self._visibility_needs_update:
                self._update_triangles()
                
    def _rebuild_label_graphics(self):
        trgba, tcoord, opaque = self._packed_texture()	# Compute images first since vertices depend on image size
        va = self._label_vertices()
        normals = None
        ta = self._visible_label_triangles()
        self.set_geometry(va, normals, ta)
        if self.texture is not None:
            self.texture.delete_texture()
        from chimerax.core.graphics import Texture
        self.texture = Texture(trgba)
        self.texture_coordinates = tcoord
        self.opaque_texture = opaque
        self._positions_need_update = False
        self._texture_needs_update = False
        self._visibility_needs_update = False

    def _packed_texture(self):
        images = [l._label_image() for l in self._labels]

        tw = self._texture_width	# texture width in pixels
        x = y = 0			# Corner for placing next image
        hr = 0				# Height of row
        positions = []
        for rgba in images:
            h,w = rgba.shape[:2]
            if x == 0 or x + w < tw:
                # Place image at end of this row.
                if h > hr:
                    hr = h
            else:
                # Place image on next row.
                y += hr
                x = 0
                hr = h
            positions.append((x,y,w,h))
            x += w
        th = y + hr	# Teture height in pixels

        print ('label texture size', tw, th)
        
        # Create single image with packed label images
        opaque = True
        from numpy import empty, uint8
        trgba = empty((th, tw, 4), uint8)
        for (x,y,w,h),rgba in zip(positions, images):
            h,w = rgba.shape[:2]
            trgba[y:y+h,x:x+w,:] = rgba
            if opaque and not (rgba[:,3] == 255).all():
                opaque = False

        # Create texture coordinates for each label.
        tclist = []
        for (x,y,w,h) in positions:
            x0,y0,x1,y1 = ((x+0.5)/tw, (y+0.5)/th, (x+w-0.5)/tw, (y+h-0.5)/th)
            tclist.extend(((x0,y0), (x1,y0), (x1,y1), (x0,y1)))
        from numpy import array, float32
        tcoord = array(tclist, float32)

        return trgba, tcoord, opaque

    def _reposition_label_graphics(self):
        self.set_geometry(self._label_vertices(), self.normals, self.triangles)
        self._positions_need_update = False
        
    def _label_vertices(self):
        spos = self.scene_position
        cpos = self.session.main_view.camera.position	# Camera position in scene coords
        cposd = spos.inverse() * cpos  # Camera pos in label drawing coords
        from numpy import concatenate
        va = concatenate([l._label_rectangle(spos, cposd) for l in self._labels])
        return va

    def _update_triangles(self):
        self.set_geometry(self.vertices, self.normals, self._visible_label_triangles())
        self._visibility_needs_update = False

    def _visible_label_triangles(self):
        tlist = []
        for i,l in enumerate(self._labels):
            if l.visible():
                c = 4*i
                tlist.extend(((c,c+1,c+2), (c,c+2,c+3)))
        from numpy import array, int32
        ta = array(tlist, int32)
        return ta
        
    SESSION_SAVE = True
    
    def take_snapshot(self, session, flags):
        lattrs = ('object', 'text', 'offset', 'color', 'background', 'size', 'height', 'font')
        lstate = tuple({attr:getattr(l, attr) for attr in lattrs}
                       for l in self._labels)
        data = {'model state': Model.take_snapshot(self, session, flags),
                'labels state': lstate,
                'orient': self._reorient_angle,
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
        self._reorient_angle = data.get('orient', 0)
        self._labels = []
        self._object_label = ol = {}
        v = self.session.main_view
        for ls in data['labels state']:
            o = ls['object']
            kw = {attr:ls[attr] for attr in ('text', 'offset', 'color', 'background',
                                             'size', 'height', 'font') if attr in ls}
            cls = label_class(o)
            ol[o] = l = cls(o, v, **kw)
            self._labels.append(l)

# -----------------------------------------------------------------------------
#
def label_class(object):
    from chimerax.atomic import Atom, Residue, Pseudobond, Bond
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
class ObjectLabel:

    pickable = False		# Don't allow mouse selection of labels
    casts_shadows = False
    
    def __init__(self, object, view, offset = None, text = None,
                 color = None, background = None,
                 size = 24, height = None, font = 'Arial'):

        self.object = object
        self.view = view	# View is used to update label position to face camera
        self._offset = offset
        self._text = text
        self._color = color
        self.background = background
        self.size = size
        self.height = height	# None or height in world coords.  If None used fixed screen size.
        self._pixel_size = (100,10)	# Size of label in pixels, calculated from size attribute

        self.font = font
        
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
    offset = property(_get_offset, _set_offset)
    
    def _get_text(self):
        return self.default_text() if self._text is None else self._text
    def _set_text(self, text):
        self._text = text
    text = property(_get_text, _set_text)
    
    def _get_color(self):
        c = self._color
        if c is None:
            bg = self.background
            if bg is None:
                bg = self.view.background_color
            light_bg = (sum(bg[:3]) > 1.5)
            rgba8 = (0,0,0,255) if light_bg else (255,255,255,255)
        else:
            rgba8 = c
        return rgba8
    def _set_color(self, color):
        self._color = color
    color = property(_get_color, _set_color)

    @property
    def object_deleted(self):
        return self.location() is None

    def _label_image(self):
        s = self.size
        rgba8 = tuple(self.color)
        bg = self.background
        xpad = 0 if bg is None else int(.2*s)
        from .label2d import text_image_rgba
        text = self.text
        rgba = text_image_rgba(text, rgba8, s, self.font, background_color = bg, xpad=xpad)
        if rgba is None:
            raise RuntimeError("Can't find font %s size %d for label '%s'" % (self.font, s, text))
        h,w = rgba.shape[:2]
        self._label_size = w,h
        return rgba

    def _label_rectangle(self, scene_position, camera_position):
        # Camera position is in label drawing coordinate system.
        xyz = self.location(scene_position)
        if xyz is None:
            return	# Label deleted
        pw,ph = self._label_size
        sh = self.height	# Scene height
        if sh is None:
            psize = self.view.pixel_size(scene_position*xyz)
            w,h = psize * pw, psize * ph
        else:
            w, h = sh*pw/ph, sh
        offset = self.offset
        if offset is not None:
            xyz += camera_position.apply_without_translation(offset)
        wa = camera_position.apply_without_translation((w,0,0))
        ha = camera_position.apply_without_translation((0,h,0))
        from numpy import array, float32
        va = array((xyz, xyz + wa, xyz + wa + ha, xyz + ha), float32)
        return va
    
# -----------------------------------------------------------------------------
#
class AtomLabel(ObjectLabel):
    def __init__(self, object, view, offset = None, text = None,
                 color = None, background = None,
                 size = 24, height = None, font = 'Arial'):
        self.atom = object
        ObjectLabel.__init__(self, object, view, offset=offset, text=text,
                             color=color, background=background,
                             size=size, height=height, font=font)
    def default_text(self):
        aname = self.atom.name
        return aname if aname else ('%d' % self.atom.residue.number)
    def default_offset(self):
        return (0.2+self.atom.display_radius, 0, 0.5)
    def location(self, scene_position = None):
        a = self.atom
        return None if a.deleted else a.coord
    def visible(self):
        a = self.atom
        return (not a.deleted) and a.visible

# -----------------------------------------------------------------------------
#
class ResidueLabel(ObjectLabel):
    def __init__(self, object, view, offset = None, text = None,
                 color = None, background = None,
                 size = 24, height = None, font = 'Arial'):
        self.residue = object
        ObjectLabel.__init__(self, object, view, offset=offset, text=text,
                             color=color, background=background,
                             size=size, height=height, font=font)
    def default_text(self):
        r = self.residue
        return '%s %d' % (r.name, r.number)
    def location(self, scene_position = None):
        r = self.residue
        return None if r.deleted else r.center
    def visible(self):
        r = self.residue
        return (not r.deleted) and ((r.ribbon_display and r.polymer_type != r.PT_NONE) or r.atoms.displays.any())

# -----------------------------------------------------------------------------
#
class PseudobondLabel(ObjectLabel):
    def __init__(self, object, view, offset = None, text = None,
                 color = None, background = None,
                 size = 24, height = None, font = 'Arial'):
        self.pseudobond = object
        ObjectLabel.__init__(self, object, view, offset=offset, text=text,
                             color=color, background=background,
                             size=size, height=height, font=font)
    def default_text(self):
        dm = self.pseudobond.session.pb_dist_monitor
        return dm.distance_format % self.pseudobond.length
    def default_offset(self):
        return (0.2+self.pseudobond.radius, 0, 0.5)
    def location(self, scene_position = None):
        pb = self.pseudobond
        if pb.deleted:
            return None
        a1,a2 = pb.atoms
        sxyz = 0.5 * (a1.scene_coord + a2.scene_coord)	# Midpoint
        xyz = scene_position.inverse() * sxyz
        return xyz
    def visible(self):
        pb = self.pseudobond
        return (not pb.deleted) and pb.shown

# -----------------------------------------------------------------------------
#
BondLabel = PseudobondLabel
