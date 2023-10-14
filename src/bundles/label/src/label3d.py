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


# -----------------------------------------------------------------------------
#
def label(session, objects = None, object_type = None, text = None,
          offset = None, color = None, bg_color = None, attribute = None,
          size = None, height = None, default_height = None, font = None, on_top = None):
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
    color : (r,g,b,a) or "auto" or "default"
      Color of the label text.  If no color is specified black is used on light backgrounds
      and white is used on dark backgrounds.
    bg_color : (r,g,b,a) or "none"
      Draw rectangular label background in this color, or if "none", background is transparent.
    attribute : string
      Attribute name whose value to display as text
    size : int or "default"
      Font size in points (1/72 inch). Default 48.
    height : float or "fixed"
      Text height in scene units.  Or if "fixed" use fixed pixel height on screen.  Initial value 0.7.
    default_height : float
      Default height value if not specified.  Initial value 0.7.
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
            otypes = ['residues']
    else:
        otypes = [object_type]

    from chimerax.core.errors import UserError
    if text is not None and attribute is not None:
        raise UserError("Cannot specify both 'text' and 'attribute' keywords")

    has_graphics = session.main_view.render is not None
    if not has_graphics:
        from chimerax.core.errors import LimitationError
        raise LimitationError("Unable to draw 3D labels without rendering images")

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
    from numpy import ndarray
    if isinstance(color, Color):
        settings['color'] = color.uint8x4()
    elif isinstance(color, str) and color in ('default', 'auto'):
        settings['color'] = None
    elif isinstance(color, (tuple, list, ndarray)):
        settings['color'] = tuple(color)
    if isinstance(bg_color, Color):
        settings['background'] = bg_color.uint8x4()
    elif isinstance(bg_color, str) and bg_color == 'none':
        settings['background'] = None
    elif isinstance(bg_color, (tuple, list, ndarray)):
        settings['background'] = tuple(bg_color)
    if size == 'default':
        settings['size'] = 48
    elif size is not None:
        settings['size'] = size
    if height == 'fixed':
        settings['height'] = None
    elif height is not None:
        settings['height'] = height
    if default_height is not None:
        from .settings import settings as prefs
        prefs.label_height = default_height
    if font == 'default':
        settings['font'] = 'Arial'
    elif font is not None:
        settings['font'] = font
    if 'text' in settings:
        settings['attribute'] = False
    elif attribute is not None:
        settings['text'] = False
        settings['attribute'] = attribute

    if objects is None and len(settings) == 0 and on_top is None:
        return	# Get this when setting default height.
    
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
    if objects is None and lcount == 0 and default_height is None:
        raise UserError('Label command requires an atom specifier to create labels.')

# -----------------------------------------------------------------------------
#
def label_orient(session, orient = None):
    '''Set how often labels reorient to face the viewer.

    Parameters
    ----------
    orient : float
      Reorient the labels to face the view direction only when the view direction changes
      by the specified number of degrees.  Default 0 makes the labels always face
      the view direction.  This option is primarily of interest with virtual reality viewing.
      This is a global setting.
    '''
    if orient is None:
        session.logger.status('Labels reorient at %.0f degrees' % _reorient_angle(session), log = True)
    else:
        session._label_orient = orient

# -----------------------------------------------------------------------------
#
def _reorient_angle(session):
    return getattr(session, '_label_orient', 0)
    
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
def label_objects(objects, object_types = ['atoms', 'residues', 'pseudobonds', 'bonds']):
    lobjects = []
    lmodels = set()
    for otype in object_types:
        for m, lbl_objects in objects_by_model(objects, otype):
            lm = labels_model(m)
            if lm is not None:
                lo = lm.labels(lbl_objects)
                if lo:
                    lobjects.extend(lo)
                    lmodels.add(lm)
    return lmodels, lobjects
        
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

# DefArg/NoneArg also use by 2D labels
from chimerax.core.commands import EnumOf
DefArg = EnumOf(['default'])
NoneArg = EnumOf(['none'])
def register_label_command(logger):

    from chimerax.core.commands import CmdDesc, register, create_alias, ObjectsArg, StringArg, FloatArg
    from chimerax.core.commands import Float3Arg, Color8Arg, IntArg, BoolArg, EnumOf, Or, EmptyArg

    otype = EnumOf(('atoms','residues','pseudobonds','bonds'))
    desc = CmdDesc(required = [('objects', Or(ObjectsArg, EmptyArg))],
                   optional = [('object_type', otype)],
                   keyword = [('text', Or(EnumOf(['default'],abbreviations=False), StringArg)),
                              ('offset', Or(DefArg, Float3Arg)),
                              ('color', Or(EnumOf(['default', 'auto']), Color8Arg)),
                              ('bg_color', Or(NoneArg, Color8Arg)),
                              ('size', Or(DefArg, IntArg)),
                              ('height', Or(EnumOf(['fixed']), FloatArg)),
                              ('default_height', FloatArg),
                              ('font', StringArg),
                              ('attribute', StringArg),
                              ('on_top', BoolArg)],
                   synopsis = 'Create atom labels')
    register('label', desc, label, logger=logger)
    desc = CmdDesc(optional = [('orient', FloatArg)],
                   synopsis = 'Set label orientation updating')
    register('label orient', desc, label_orient, logger=logger)
    desc = CmdDesc(required = [('objects', Or(ObjectsArg, EmptyArg))],
                   optional = [('object_type', otype)],
                   synopsis = 'Delete atom labels')
    register('label delete', desc, label_delete, logger=logger)
    desc = CmdDesc(synopsis = 'List available fonts')
    from .label2d import label_listfonts
    register('label listfonts', desc, label_listfonts, logger=logger)
    create_alias('~label', 'label delete $*', logger=logger)

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class ObjectLabels(Model):
    '''Model holding labels appearing next to atoms, residues, pseudobonds or bonds.'''

    pickable = False		# Don't allow mouse selection of labels
    casts_shadows = False
    
    def __init__(self, session):
        Model.__init__(self, 'labels', session)

        self.on_top = True		# Should labels always appear above other graphics
        self._window_size = session.main_view.window_size

        self._labels = []		# list of ObjectLabel
        self._object_label = {}		# Map object (Atom, Residue, Pseudobond, Bond) to ObjectLabel
        self._num_pixel_labels = 0	# Number of labels sized in pixels.

        t = session.triggers
        self._update_graphics_handler = t.add_handler('graphics update', self._update_graphics_if_needed)
        self._model_display_handler = t.add_handler('model display changed', self._model_display_changed)
        from chimerax.core.core_settings import settings as core_settings
        self._background_color_handler = core_settings.triggers.add_handler(
            'setting changed', self._background_changed_cb)

        from chimerax.atomic import get_triggers
        ta = get_triggers()
        self._structure_change_handler = ta.add_handler('changes', self._structure_changed)
        
        self.use_lighting = False
        Model.set_color(self, (255,255,255,255))	# Do not modulate texture colors

        self._texture_width = 4096			# Pixels.
        self._texture_needs_update = True		# Has text, color, size, font changed.
        self._positions_need_update = True		# Has label position changed relative to atom?
        self._visibility_needs_update = True		# Does an atom hide require a label to hide?
        self._monitored_attr_info = {}
        
    def delete(self):
        h = self._update_graphics_handler
        if h is not None:
            h.remove()
            self._update_graphics_handler = None

        h = self._model_display_handler
        if h is not None:
            h.remove()
            self._model_display_handler = None

        h = self._background_color_handler
        if h is not None:
            h.remove()
            self._background_color_handler = None

        h = self._structure_change_handler
        if h is not None:
            h.remove()
            self._structure_change_handler = None
        
        Model.delete(self)

    def _get_model_color(self):
        from chimerax.core.colors import most_common_color
        lcolors = [ld.color for ld in self._labels]
        c = most_common_color(lcolors) if lcolors else None
        return c
    def _set_model_color(self, color):
        for ld in self._labels:
            ld.color = color
        self.update_labels()
    model_color = property(_get_model_color, _set_model_color)

    def labels(self, objects = None):
        if objects is None:
            return self._labels
        ol = self._object_label
        return [ol[o] for o in objects if o in ol]
    
    def add_labels(self, objects, label_class, view, settings = {}, on_top = None):
        if on_top is not None:
            self.on_top = on_top
        ol = self._object_label
        for o in objects:
            if o not in ol:
                ol[o] = lo = label_class(o, view, **settings)
                self._labels.append(lo)
            elif settings:
                lo = ol[o]
                for k,v in settings.items():
                    setattr(lo, k, v)
            else:
                continue
            if lo.attribute is not None:
                self._monitor_attribute(lo)
        self._count_pixel_sized_labels()
        if objects:
            self.update_labels()

    def update_labels(self):
        self._texture_needs_update = True
        self.redraw_needed()
            
    def _count_pixel_sized_labels(self):
        self._num_pixel_labels = len([l for l in self._labels if l.height is None])
        
    def delete_labels(self, objects):
        ol = self._object_label
        count = 0
        for o in objects:
            try:
                lo = ol[o]
            except KeyError:
                pass
            else:
                if lo.attribute is not None:
                    self._demonitor_attribute(lo)
                del ol[o]
                count += 1
        if count > 0:
            self._labels = [l for l in self._labels if l.object in ol]
            self._texture_needs_update = True
            self._count_pixel_sized_labels()

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
        if 'name changed' in changes.atom_reasons() or 'name changed' in changes.residue_reasons():
            self._texture_needs_update = True
        else:
            for monitored_class, attr_name_info in self._monitored_attr_info.items():
                reasons = getattr(changes, monitored_class.__name__.lower() + '_reasons')()
                for attr_name in attr_name_info.keys():
                    if attr_name + ' changed' in reasons:
                        self._texture_needs_update = True
                        break
                if self._texture_needs_update:
                    break
        self._visibility_needs_update = True
        self._positions_need_update = True
        if changes.num_deleted_atoms() > 0 or changes.num_deleted_bonds() > 0 or changes.num_deleted_pseudobonds() > 0:
            self.delete_labels([l.object for l in self._labels if l.object_deleted])
        self.redraw_needed()

    def _monitor_attribute(self, lo):
        self._monitored_attr_info.setdefault(lo.object.__class__, {}).setdefault(lo.attribute, set()).add(lo)

    def _demonitor_attribute(self, lo):
        try:
            by_class = self._monitored_attr_info[lo.object.__class__]
        except KeyError:
            return
        try:
            by_attr = by_class[lo.attribute]
        except KeyError:
            return
        by_attr.discard(lo)
        if not by_attr:
            del by_class[lo.attribute]
            if not by_class:
                del self._monitored_attr_info[lo.object.__class__]

    def _model_display_changed(self, trig_name, model):
        # If a model is hidden global pseudobond labels may need to be hidden.
        self._visibility_needs_update = True
        
    def _background_changed_cb(self, trig_name, info):
        setting_name, old_val, new_val = info
        if setting_name == "background_color":
            self._texture_needs_update = True
            
    def _update_graphics_if_needed(self, *_):
        if not self.visible:
            return

        ses = self.session
        v = ses.main_view
        resize = (v.window_size != self._window_size)
        if resize:
            self._window_size = v.window_size
            self._positions_need_update = True
            
        camera_move = v.camera.redraw_needed
        if camera_move:
            ra = _reorient_angle(ses)
            if ra == 0 or self._num_pixel_labels > 0:
                self._positions_need_update = True
            elif ra > 0:
                # Don't update label positions if minimum camera motion has not occured.
                # This optimization is to maintain high frame rate with virtual reality.
                cpos = v.camera.position
                lcpos = getattr(ses, '_last_label_view', None)
                from math import degrees
                if lcpos is None:
                    ses._last_label_view = cpos
                elif degrees((cpos.inverse() * lcpos).rotation_angle()) >= ra:
                    self._positions_need_update = True
                    ses._last_label_view = cpos

        if self._texture_needs_update:
            self._rebuild_label_graphics()
        else:
            if self._positions_need_update:
                self._reposition_label_graphics()
            if self._visibility_needs_update:
                self._update_triangles()

    def _rebuild_label_graphics(self):
        if len(self._labels) == 0:
            return
        trgba, tcoord = self._packed_texture()	# Compute images first since vertices depend on image size
        opaque = self._all_labels_opaque()
        va = self._label_vertices()
        normals = None
        ta = self._visible_label_triangles()
        self.set_geometry(va, normals, ta)
        self._set_label_texture(trgba, tcoord)
        self.opaque_texture = opaque
        self._positions_need_update = False
        self._texture_needs_update = False
        self._visibility_needs_update = False

    def _set_label_texture(self, trgba, tcoord):
        # Check for too many labels.
        from chimerax.graphics import Texture
        if hasattr(Texture, 'MAX_TEXTURE_SIZE') and trgba.shape[0] > Texture.MAX_TEXTURE_SIZE:
            msg = ('Too many labels (%d),' % len(self._labels) +
                   ' label texture size (%d,%d)' % (trgba.shape[1], trgba.shape[0]) +
                   ' exceeded maximum OpenGL texture size (%d)' % Texture.MAX_TEXTURE_SIZE +
                   ', some labels will be blank.')
            self.session.logger.warning(msg)
            tcoord[:,1] *= trgba.shape[0] / Texture.MAX_TEXTURE_SIZE
            trgba = trgba[:Texture.MAX_TEXTURE_SIZE]

        if self.texture is not None:
            self.texture.reload_texture(trgba)
        else:
            self.texture = Texture(trgba)
        self.texture_coordinates = tcoord

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
            if w > tw:
                msg = f'Label width {w} exceeds maximum {tw} and will be clipped'
                self.session.logger.warning(msg)
        th = y + hr	# Teture height in pixels

        # Create single image with packed label images
        from numpy import empty, uint8
        trgba = empty((th, tw, 4), uint8)
        for (x,y,w,h),rgba in zip(positions, images):
            h,w = rgba.shape[:2]
            trgba[y:y+h,x:x+w,:] = rgba[:,:tw,:]

        # Create texture coordinates for each label.
        tclist = []
        for (x,y,w,h) in positions:
            x0,y0,x1,y1 = ((x+0.5)/tw, (y+0.5)/th, (x+w-0.5)/tw, (y+h-0.5)/th)
            tclist.extend(((x0,y0), (x1,y0), (x1,y1), (x0,y1)))
        from numpy import array, float32
        tcoord = array(tclist, float32)

        return trgba, tcoord

    def _all_labels_opaque(self):
        for l in self._labels:
            bg = l.background
            if bg is None or bg[3] < 255:
                return False
        return True
    
    def _reposition_label_graphics(self):
        self.set_geometry(self._label_vertices(), self.normals, self.triangles)
        self._positions_need_update = False
        
    def _label_vertices(self):
        spos = self.scene_position
        cpos = self.session.main_view.camera.position	# Camera position in scene coords
        cposd = spos.inverse() * cpos  # Camera pos in label drawing coords
        rects = [l._label_rectangle(spos, cposd) for l in self._labels if not l.object_deleted]
        if len(rects) == 0:
            from numpy import empty, float32
            va = empty((0,3), float32)
        else:
            from numpy import concatenate
            va = concatenate(rects)
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
        if len(tlist) == 0:
            from numpy import empty, int32
            ta = empty((0,3), int32)
        else:
            from numpy import array, int32
            ta = array(tlist, int32)
        return ta

    def picked_label(self, triangle_number):
        lnum = triangle_number//2
        vlabels = [l for l in self._labels if l.visible()]
        if lnum < len(vlabels):
            return vlabels[lnum]
        return None
        
        
    SESSION_SAVE = True
    
    def take_snapshot(self, session, flags):
        lattrs = ('object', 'text', 'offset', 'color', 'background', 'size', 'height', 'font')
        lstate = tuple({attr:getattr(l, attr) for attr in lattrs}
                       for l in self._labels)
        data = {'model state': Model.take_snapshot(self, session, flags),
                'labels state': lstate,
                'orient': _reorient_angle(session),
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
        if 'orient' in data:
            label_orient(session, data['orient'])
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
        self._count_pixel_sized_labels()

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

    def __init__(self, object, view, offset = None, text = None,
                 color = None, background = None, attribute = None,
                 size = 48, height = 'default', font = 'Arial'):

        self.object = object
        self.view = view	# View is used to update label position to face camera
        self._offset = offset
        if text is False:
            self._text = None
        else:
            self._text = text
        self._color = color
        self.background = background
        if attribute is False:
            self._attribute = None
        else:
            self._attribute = attribute
        self.size = size	# Points (1/72 inch) so high and normal DPI displays look the same.
        if height == 'default':
            from .settings import settings
            height = settings.label_height
        self.height = height	# None or height in world coords.  If None used fixed screen size.
        self._pixel_size = (100,10)	# Size of label in pixels, derived from size attribute

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
        if self._attribute is None:
            base_text = self.default_text() if self._text is None else self._text
            try:
                final_text = base_text.format(self.object)
            except AttributeError:
                # don't label objects missing the requested attribute(s)
                # and treat them as having no label assigned rather than
                # one with a missing attribute
                final_text = ""
                self._text = None
            except Exception:
                final_text = base_text
        else:
            attrs = self._attribute.split('.')
            val = self.object
            try:
                for attr in attrs:
                    val = getattr(val, attr)
            except AttributeError:
                final_text = ""
            else:
                if isinstance(val, float):
                    final_text = "%.3g" % val
                elif val is None:
                    final_text = ""
                else:
                    final_text = str(val)
        return final_text
    def _set_text(self, text):
        if text is False:
            self._text = None
        else:
            self._text = text
    text = property(_get_text, _set_text)
    
    def _get_color(self):
        c = self._color
        if c is None:
            bg = self.background
            if bg is None:
                bg = [255*r for r in self.view.background_color]
            from chimerax.core.colors import contrast_with
            if contrast_with([c/255 for c in bg[:3]])[0] == 0.0:
                rgba8 = (0, 0, 0, 255)
            else:
                rgba8 = (255, 255, 255, 255)
        else:
            rgba8 = c
        return rgba8
    def _set_color(self, color):
        self._color = color
    color = property(_get_color, _set_color)
    
    def _get_attribute(self):
        return self._attribute

    def _set_attribute(self, attribute):
        if attribute is False:
            self._attribute = None
        else:
            self._attribute = attribute
    attribute = property(_get_attribute, _set_attribute)

    @property
    def object_deleted(self):
        return self.location() is None

    def _label_image(self):
        s = self.size
        rgba8 = tuple(self.color)
        bg = self.background
        xpad = 0 if bg is None else int(.2*s)
        from chimerax.graphics import text_image_rgba
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
            return None	# Label deleted
        pw,ph = self._label_size
        sh = self.height	# Scene height
        if sh is None:
            psize = self.view.pixel_size(scene_position*xyz)
            w,h = psize * pw, psize * ph
        else:
            w, h = sh*pw/ph, sh
        offset = self.offset
        if offset is not None:
            xyz += camera_position.transform_vector(offset)
        wa = camera_position.transform_vector((w,0,0))
        ha = camera_position.transform_vector((0,h,0))
        from numpy import array, float32
        va = array((xyz, xyz + wa, xyz + wa + ha, xyz + ha), float32)
        return va
    
# -----------------------------------------------------------------------------
#
class AtomLabel(ObjectLabel):
    def __init__(self, object, view, offset = None, text = None,
                 color = None, background = None, attribute = None,
                 size = 48, height = 'default', font = 'Arial'):
        self.atom = object
        ObjectLabel.__init__(self, object, view, offset=offset, text=text,
                 color=color, background=background, attribute=attribute,
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
                 color = None, background = None, attribute = None,
                 size = 48, height = 'default', font = 'Arial'):
        self.residue = object
        ObjectLabel.__init__(self, object, view, offset=offset, text=text,
                 color=color, background=background, attribute=attribute,
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
class EdgeLabel(ObjectLabel):
    def __init__(self, object, view, offset = None, text = None,
                 color = None, background = None, attribute = None,
                 size = 48, height = 'default', font = 'Arial'):
        self.pseudobond = object
        ObjectLabel.__init__(self, object, view, offset=offset, text=text,
                 color=color, background=background, attribute=attribute,
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
        xyz = scene_position.inverse() * sxyz if scene_position else sxyz
        return xyz
    def visible(self):
        pb = self.pseudobond
        if pb.deleted or not pb.shown:
            return False
        a1,a2 = pb.atoms
        vis = a1.structure.visible and a2.structure.visible
        return vis

# -----------------------------------------------------------------------------
#
class BondLabel(EdgeLabel):
    pass

# -----------------------------------------------------------------------------
#
class PseudobondLabel(EdgeLabel):
    pass

# -----------------------------------------------------------------------------
#
def picked_3d_label(session, win_x, win_y):
    xyz1, xyz2 = session.main_view.clip_plane_points(win_x, win_y)
    if xyz1 is None or xyz2 is None:
        return None
    pick = None
    from chimerax.core.models import PickedModel
    for m in session.models.list(type = ObjectLabels):
        mtf = m.parent.scene_position.inverse()
        mxyz1, mxyz2 =  mtf*xyz1, mtf*xyz2
        p = m.first_intercept(mxyz1, mxyz2)
        if isinstance(p, PickedModel) and (pick is None or p.distance < pick.distance):
            pick = p

    if pick:
        # Return ObjectLabel instance
        lmodel = pick.drawing()
        lobject = lmodel.picked_label(pick.picked_triangle.triangle_number)
        if lobject:
            lobject._label_model = lmodel
            return lobject

    return None
