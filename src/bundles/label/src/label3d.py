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
def label(session, atoms, text = None, offset = None, color = None, size = None, typeface = None, on_top = True):
    '''Create atom labels. The belong to a child model named "labels" of the structure.

    Parameters
    ----------
    atoms : Atoms
      Create labels on specified atoms.
    offset : float 3-tuple
      Offset of label from atom center in screen coordinates in physical units (Angstroms)
    text : string
      Displayed text of the label.
    color : Color
      Color of the label text.  If no color is specified black is used on light backgrounds
      and white is used on dark backgrounds.
    size : int
      Font size in pixels. Default 24.
    typeface : string
      Font name.  This must be a true type font installed on Mac in /Library/Fonts
      and is the name of the font file without the ".ttf" suffix.  Default "Arial".
    on_top : bool
      Whether labels always appear on top of other graphics (cannot be occluded).  This is a per-structure
      attribute.  Default True.
    '''
    rgba = None if color is None else color.uint8x4()
    for s, satoms in atoms.by_structure:
        al = structure_atom_labels(s, create = True)
        al.add_labels(satoms, offset, text, rgba, size, typeface, on_top)

# -----------------------------------------------------------------------------
#
def label_delete(session, atoms):
    '''Delete atoms labels.

    Parameters
    ----------
    atoms : Atoms
      Delete labels for specified atoms.
    '''
    for s, satoms in atoms.by_structure:
        al = structure_atom_labels(s)
        if al is not None:
            al.delete_labels(satoms)
            if al.label_count() == 0:
                session.models.close([al])

# -----------------------------------------------------------------------------
#
def structure_atom_labels(s, create = False):
    for al in s.child_models():
        if isinstance(al, AtomLabels):
            return al
    if create:
        al = AtomLabels(s.session)
        s.add([al])
    else:
        al = None
    return al

# -----------------------------------------------------------------------------
#
def register_label_command(logger):

    from chimerax.core.commands import CmdDesc, register, AtomsArg, StringArg, Float3Arg, ColorArg, IntArg, BoolArg

    desc = CmdDesc(required = [('atoms', AtomsArg)],
                   keyword = [('text', StringArg),
                              ('offset', Float3Arg),
                              ('color', ColorArg),
                              ('size', IntArg),
                              ('typeface', StringArg),
                              ('on_top', BoolArg)],
                   synopsis = 'Create atom labels')
    register('label', desc, label, logger=logger)
    desc = CmdDesc(required = [('atoms', AtomsArg)],
                   synopsis = 'Delete atom labels')
    register('label delete', desc, label_delete, logger=logger)

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class AtomLabels(Model):
    '''Model holding atom labels for one Structure.'''

    pickable = False		# Don't allow mouse selection of labels
    
    def __init__(self, session):
        Model.__init__(self, 'labels', session)

        self.on_top = True		# Should labels always appear above other graphics
        
        self._label_drawings = {}	# Map Atom to AtomLabel

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
    
    def add_labels(self, atoms, offset = None, text = None,
                   color = None, size = 24, typeface = 'Arial', on_top = True):
        self.on_top = on_top
        ld = self._label_drawings
        opts = [('offset', offset), ('text', text), ('color', color),
                ('size', size), ('typeface', typeface)]
        kw = {k:v for k,v in opts if v is not None}
        for a in atoms:
            if a not in ld:
                ld[a] = AtomLabel(a, **kw)
                self.add_drawing(ld[a])
            elif kw:
                al = ld[a]
                for k,v in kw.items():
                    if v is not None:
                        setattr(al, k, v)
                al._needs_update = True
        if atoms:
            self.redraw_needed()

    def delete_labels(self, atoms):
        ld = self._label_drawings
        for a in atoms:
            if a in ld:
                self.remove_drawing(ld[a])
                del ld[a]

    def label_count(self):
        return len(self._label_drawings)

    def _update_graphics_if_needed(self, *_):
        for ld in self._label_drawings.values():
            ld._update_graphics()

    def take_snapshot(self, session, flags):
        lattrs = ('atom', 'offset', 'text', 'color', 'size', 'typeface')
        lstate = tuple({attr:getattr(ld, attr) for attr in lattrs}
                       for ld in self._label_drawings.values())
        data = {'model state': Model.take_snapshot(self, session, flags),
                'labels state': lstate,
                'on_top': self.on_top,
                'version': 1}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        s = AtomLabels(session)
        s.set_state_from_snapshot(session, data)
        return s

    def set_state_from_snapshot(self, session, data):
        Model.set_state_from_snapshot(self, session, data['model state'])
        self.on_top = data['on_top']
        als = [AtomLabel(**ls) for ls in data['labels state']]
        self._label_drawings = {al.atom:al for al in als}
        for al in als:
            self.add_drawing(al)

    def reset_state(self, session):
        pass

# -----------------------------------------------------------------------------
#
from chimerax.core.graphics import Drawing
class AtomLabel(Drawing):

    pickable = False	# Don't allow mouse selection of labels

    def __init__(self, atom, offset = None, text = None, color = None, size = 24, typeface = 'Arial'):
        Drawing.__init__(self, 'label %s' % atom.name)

        self.atom = atom
        self.session = atom.structure.session
        self.offset = offset
        self.text = atom.name if text is None else text

        v = self.session.main_view
        if color is None:
            light_bg = (sum(v.background_color[:3]) > 1.5)
            rgba8 = (0,0,0,255) if light_bg else (255,255,255,255)
        else:
            rgba8 = color
        self.color = rgba8
        
        self.size = size
        self._pixel_size = (100,10)	# Size of label in pixels, calculated from size attribute

        self.typeface = typeface

        # Set initial billboard geometry
        from numpy import array, float32, int32
        vlist = array(((0,0,0), (1,0,0), (1,1,0), (0,1,0)), float32)
        tlist = array(((0, 1, 2), (0, 2, 3)), int32)
        tc = array(((0, 0), (1, 0), (1, 1), (0, 1)), float32)
        self.geometry = vlist, tlist
        self.texture_coordinates = tc
        self.use_lighting = False

        self._needs_update = True		# Has text, color, size, font changed.
        
    def draw(self, renderer, place, draw_pass, selected_only=False):
        self._update_label_texture()  # This needs to be done during draw in case texture delete needed.
        Drawing.draw(self, renderer, place, draw_pass, selected_only)

    def _update_graphics(self):
        self._position_label()
        disp = self.atom.visible
        if disp != self.display:
            self.display = disp

    def _update_label_texture(self):
        if not self._needs_update:
            return
        self._needs_update = False
        s = self.size
        rgba8 = (255,255,255,255)
        from chimerax import app_data_dir
        from .label2d import text_image_rgba
        rgba = text_image_rgba(self.text, rgba8, s, self.typeface, app_data_dir)
        if rgba is None:
            self.session.logger.info("Can't find font for label")
            return
        if self.texture is not None:
            self.texture.delete_texture()
        from chimerax.core.graphics import opengl
        t = opengl.Texture(rgba)
        self.texture = t
        h,w,c = rgba.shape
        ps = (s*w/h, s)
        if ps != self._pixel_size:
            self._pixel_size = ps
            self._position_label()	# Size of billboard changed.

    def _position_label(self):
        xyz = self.atom.coord
        view = self.session.main_view
        psize = view.pixel_size(xyz)
        pw,ph = self._pixel_size
        w,h = psize * pw, psize * ph
        cpos = view.camera.position
        cam_xaxis, cam_yaxis, cam_zaxis = cpos.axes()
        from numpy import array, float32
        va = array((xyz, xyz + w*cam_xaxis, xyz + w*cam_xaxis + h*cam_yaxis, xyz + h*cam_yaxis), float32)
        if self.offset is not None:
            va += cpos.apply_without_translation(self.offset)
        if (va == self.vertices).all():
            return 	# Don't set vertices causing redraw if label has not moved.
        self.vertices = va
