# vim: set expandtab shiftwidth=4 softtabstop=4:

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
from chimerax.core.tools import ToolInstance
class AlphaFoldPAEOpen(ToolInstance):

    name = 'AlphaFold Error Plot'
    help = 'help:user/tools/alphafold.html#pae'

    def __init__(self, session, tool_name):
        
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,5,0))

        from chimerax.atomic import AtomicStructure
        from chimerax.ui.widgets import ModelMenu
        m = ModelMenu(self.session, parent, label = 'AlphaFold structure',
                      model_types = [AtomicStructure],
                      model_chosen_cb = self._structure_chosen)
        self._structure_menu = m
        layout.addWidget(m.frame)

        self._source_file = 'file (.json or .pkl)'
        self._source_alphafold_db = 'AlphaFold database (UniProt id)'
        from chimerax.ui.widgets import EntriesRow
        ft = EntriesRow(parent, 'Predicted aligned error (PAE) from',
                        (self._source_file, self._source_alphafold_db))
        self._source = sf = ft.values[0]
        sf.widget.menu().triggered.connect(lambda action, self=self: self._guess_for_source())
        layout.addWidget(ft.frame)
        
        from Qt.QtWidgets import QLineEdit
        fp = QLineEdit(parent)
        layout.addWidget(fp)
        self._pae_file = fp

        # Color Domains button
        from chimerax.ui.widgets import button_row
        bf = button_row(parent,
                        [('Open', self._open_pae),
                         ('Browse', self._choose_pae_file),
                         ('Help', self._show_help)],
                        spacing = 10)
        bf.setContentsMargins(0,5,0,0)
        layout.addWidget(bf)
        
        layout.addStretch(1)    # Extra space at end

        # Set initial menu entry to an AlphaFold model
        amod = [m for m in session.models.list(type = AtomicStructure) if hasattr(m, 'alphafold')]
        if amod:
            self._structure_menu.value = amod[-1]

        tw.manage(placement=None)	# Start floating

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, AlphaFoldPAEOpen, 'AlphaFold Error Plot',
                                   create=create)

    # ---------------------------------------------------------------------------
    #
    def _structure_chosen(self):
        self._guess_pae_file_or_uniprot_id()

    # ---------------------------------------------------------------------------
    #
    def _guess_pae_file_or_uniprot_id(self):
        structure_path = self._structure_path
        if structure_path is None:
            return

        self._pae_file.setText('')
        
        uniprot_id = _guess_uniprot_id(structure_path)
        if uniprot_id:
            self._pae_file.setText(uniprot_id)
            self._source.value = self._source_alphafold_db
            return

        pae_path = _matching_pae_file(structure_path)
        if pae_path:
            self._pae_file.setText(pae_path)
            self._source.value = self._source_file

    # ---------------------------------------------------------------------------
    #
    def _guess_for_source(self):
        structure_path = self._structure_path
        if structure_path is None:
            return

        source = self._source.value
        if source == self._source_file:
            pae_path = _matching_pae_file(structure_path)
            if pae_path:
                self._pae_file.setText(pae_path)
        elif source == self._source_alphafold_db:
            uniprot_id = _guess_uniprot_id(structure_path)
            if uniprot_id:
                self._pae_file.setText(uniprot_id)
                
    # ---------------------------------------------------------------------------
    #
    @property
    def _structure_path(self):
        s = self._structure_menu.value
        structure_path = getattr(s, 'filename', None)
        return structure_path
    
    # ---------------------------------------------------------------------------
    #
    def _choose_pae_file(self):
        s = self._structure_menu.value
        if s and hasattr(s, 'filename'):
            from os import path
            dir = path.split(s.filename)[0]
        elif self._pae_file.text():
            from os import path
            dir = path.split(self._pae_file.text())[0]
        else:
            dir = None
            
        parent = self.tool_window.ui_area
        from Qt.QtWidgets import QFileDialog
        path, ftype  = QFileDialog.getOpenFileName(parent, caption = 'Predicted aligned error',
                                                   directory = dir,
                                                   filter = 'PAE file (*.json *.pkl)')
        if path:
            self._pae_file.setText(path)
            self._open_pae()

    # ---------------------------------------------------------------------------
    #
    def _open_pae(self):

        s = self._structure_menu.value
        if s is None:
            from chimerax.core.errors import UserError
            raise UserError('Must choose structure to associate with predicted aligned error')

        source = self._source.value
        if source == self._source_file:
            self._open_pae_from_file(s)
        elif source == self._source_alphafold_db:
            self._open_pae_from_alphafold_db(s)

        self.display(False)

    # ---------------------------------------------------------------------------
    #
    def _open_pae_from_file(self, structure):
        from chimerax.core.errors import UserError

        path = self._pae_file.text()
        if not path:
            raise UserError('Must choose path to predicted aligned file')

        from os.path import isfile
        if not isfile(path):
            raise UserError(f'File "{path}" does not exist.')

        if not path.endswith('.json') and not path.endswith('.pkl'):
            raise UserError(f'PAE file suffix must be ".json" or ".pkl".')

        from chimerax.core.commands import run, quote_if_necessary
        cmd = 'alphafold pae #%s file %s' % (structure.id_string, quote_if_necessary(path))
        run(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def _open_pae_from_alphafold_db(self, structure):
        uniprot_id = self._pae_file.text()
        from chimerax.core.commands import run, quote_if_necessary
        cmd = 'alphafold pae #%s uniprot %s' % (structure.id_string, uniprot_id)

        version = _alphafold_db_structure_version(self._structure_menu.value)
        if version is not None:
            from .database import default_database_version
            if str(version) != default_database_version(self.session):
                cmd += f' version {version}'
                self.session.logger.warning(f'Fetching PAE using AlphaFold database version {version}')

        run(self.session, cmd)
        
    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

# -----------------------------------------------------------------------------
# ChimeraX Google colab predictions names files
#
#	best_model.pdb
#	best_model_pae.json
#	model_3_unrelaxed.pdb
#	model_3_pae.json
#
# Full AlphaFold 2.2.0 runs name files
#
#	unrelaxed_model_1_multimer_v2_pred_0.pdb
#	relaxed_model_1_multimer_v2_pred_0.pdb
#	result_model_1_multimer_v2_pred_0.pkl
#
# ColabFold 1.3.0 runs name files where 7qfc was user assigned name and efb9b was
# server assigned id.
#
#	7qfc_efb9b_unrelaxed_rank_1_model_4.pdb
#	7qfc_efb9b_unrelaxed_rank_1_model_4_scores.json
#
# AlphaFold database files
#
#	AF-P01445-F1-model_v2.cif
#	AF-P01445-F1-predicted_aligned_error_v2.json
#
# Finding json/pkl with matching prefix works except for full alphafold which
# wants matching suffix.
#
def _matching_pae_file(structure_path):
    from os.path import split, isdir, join
    dir, filename = split(structure_path)
    if not isdir(dir):
        return None

    from os import listdir
    dfiles = listdir(dir)
    pfiles = [f for f in dfiles if f.endswith('.json') or f.endswith('.pkl')]

    if len(pfiles) == 0:
        return None
    if len(pfiles) == 1:
        return join(dir, pfiles[0])
    
    mfile = _longest_matching_prefix(filename, pfiles, min_length = 6)
    if mfile is None:
        mfile = _longest_matching_suffix(filename, pfiles, min_length = 6)
    if mfile is None:
        return None
    return join(dir, mfile)

# -----------------------------------------------------------------------------
#
def _longest_matching_prefix(filename, filenames, min_length = 1):
    m = [(len(_matching_prefix(pf,filename)),pf) for pf in filenames]
    m.sort(reverse = True)
    if m[0][0] >= min_length and m[0][0] > m[1][0]:
        return m[0][1]
    return None

# -----------------------------------------------------------------------------
#
def _matching_prefix(s1, s2):
    for i in range(min(len(s1), len(s2))):
        if s2[i] != s1[i]:
            break
    return s1[:i]

# -----------------------------------------------------------------------------
#
def _longest_matching_suffix(filename, filenames, min_length = 1):
    m = [(len(_matching_suffix(pf,filename)),pf) for pf in filenames]
    m.sort(reverse = True)
    if m[0][0] >= min_length and m[0][0] > m[1][0]:
        return m[0][1]
    return None

# -----------------------------------------------------------------------------
#
def _matching_suffix(s1, s2):
    # Ignore last "." and beyond
    from os.path import splitext
    s1,s2 = splitext(s1)[0], splitext(s2)[0]
    for i in range(min(len(s1), len(s2))):
        if s2[-i] != s1[-i]:
            break
    return s1[-i:]

# ---------------------------------------------------------------------------
#
def _guess_uniprot_id(structure_path):
    from os.path import split
    filename = split(structure_path)[1]
    from .database import uniprot_id_from_filename
    uniprot_id = uniprot_id_from_filename(filename)
    return uniprot_id
        
# ---------------------------------------------------------------------------
#
def _alphafold_db_structure_version(structure):
    '''
    Parse the structure filename to get the AlphaFold database version.
    Example database file name AF-A0A4T0DZS4-F1-model_v3.cif
    '''
    if structure is None:
        return None
    path = getattr(structure, 'filename', None)
    if path is None:
        return None
    from os.path import split, splitext
    filename = split(path)[1]
    if filename.startswith('AF') and (filename.endswith('.cif') or filename.endswith('.pdb')):
        fields = splitext(filename)[0].split('_')
        if len(fields) > 1 and fields[-1].startswith('v'):
            try:
                version = int(fields[-1][1:])
            except ValueError:
                return None
            return version

    return None

# -----------------------------------------------------------------------------
#
def alphafold_error_plot_panel(session, create = False):
    return AlphaFoldPAEOpen.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_alphafold_error_plot_panel(session):
    p = alphafold_error_plot_panel(session, create = True)
    p.display(True)
    return p
    
# -----------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class AlphaFoldPAEPlot(ToolInstance):

    name = 'AlphaFold Predicted Aligned Error Plot'
    help = 'help:user/tools/alphafold.html#pae'

    def __init__(self, session, tool_name, pae, colormap = None, divider_lines = True):

        self._pae = pae		# AlphaFoldPAE instance

        self._drag_colors_structure = True
        self._showing_chain_dividers = divider_lines
        
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        tw.fill_context_menu = self._fill_context_menu
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        sname = f'for {pae.structure}' if pae.structure else ''
        title = (f'<html>Predicted aligned errors (PAE) {sname}'
                 '<br>Drag a box to color structure residues.</html>')
        from Qt.QtWidgets import QLabel
        self._heading = hl = QLabel(title)
        layout.addWidget(hl)

        self._pae_view = gv = PAEView(parent, self._rectangle_select, self._rectangle_clear,
                                      self._report_residues)
        from Qt.QtWidgets import QSizePolicy
        from Qt.QtCore import Qt
        gv.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gv.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#        gv.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(gv)

        from Qt.QtWidgets import QGraphicsScene
        self._scene = gs = QGraphicsScene(gv)
        gs.setSceneRect(0, 0, 500, 500)
        gv.setScene(gs)

        # Color Domains button
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        self._info_label = QLabel(bf)
        bf.layout().insertWidget(3, self._info_label)
        
        self.set_colormap(colormap)
        self.show_chain_dividers(self._showing_chain_dividers)

        if pae.structure is not None:
            h = pae.structure.triggers.add_handler('deleted', self._structure_deleted)
            self._structure_delete_handler = h

        tw.manage(placement=None)	# Start floating

    # ---------------------------------------------------------------------------
    #
    def closed(self):
        return self.tool_window.tool_instance is None

    # ---------------------------------------------------------------------------
    #
    def _structure_deleted(self, *args):
        '''Remove plot if associated structure closed.'''
        if not self.closed():
            self.delete()

    # ---------------------------------------------------------------------------
    #
    def _fill_context_menu(self, menu, x, y):
        def _set_drag_colors_structure(checked, self=self):
            self._drag_colors_structure = checked
        self._add_menu_toggle(menu, 'Dragging box colors structure',
                              self._drag_colors_structure,
                              _set_drag_colors_structure)

        menu.addAction('Color plot from structure', self._color_from_structure)
        menu.addAction('Color plot rainbow', self.set_colormap_rainbow)
        menu.addAction('Color plot green', self.set_colormap_green)

        self._add_menu_toggle(menu, 'Show chain divider lines',
                              self._showing_chain_dividers, self.show_chain_dividers)

        menu.addAction('Save image', self._save_image)
        
    # ---------------------------------------------------------------------------
    #
    def _add_menu_toggle(self, menu, text, checked, callback):
        from Qt.QtGui import QAction
        a = QAction(text, menu)
        a.setCheckable(True)
        a.setChecked(checked)
        a.triggered.connect(callback)
        menu.addAction(a)
        
    # ---------------------------------------------------------------------------
    #
    def set_colormap(self, colormap = None):
        if colormap is None:
            from chimerax.core.colors import BuiltinColormaps
            colormap = BuiltinColormaps[self._default_colormap_name]
        self._pae_view._make_image(self._pae.pae_matrix, colormap)
        
    # ---------------------------------------------------------------------------
    #
    def set_colormap_rainbow(self):
        from chimerax.core.colors import BuiltinColormaps
        self.set_colormap(BuiltinColormaps['pae'])
        
    # ---------------------------------------------------------------------------
    #
    def set_colormap_green(self):
        from chimerax.core.colors import BuiltinColormaps
        self.set_colormap(BuiltinColormaps['paegreen'])

    # ---------------------------------------------------------------------------
    #
    def _color_from_structure(self, colormap = None):
        '''colormap is used for plot points where residues have different colors.'''
        if colormap is None:
            colormap = self._pae_view._block_colormap((0,0,0,255))
        color_blocks = self._residue_color_blocks()
        self._pae_view._make_image(self._pae.pae_matrix, colormap, color_blocks)

    # ---------------------------------------------------------------------------
    #
    def show_chain_dividers(self, show = True, thickness = 4):
        self._showing_chain_dividers = show
        if show:
            chain_ids = self._pae.structure.residues.chain_ids
            # Residue indices for start of each chain excluding the first.
            dividers = tuple((chain_ids[:-1] != chain_ids[1:]).nonzero()[0]+1)
        else:
            dividers = []
        self._pae_view._show_chain_dividers(dividers, thickness)
        
    # ---------------------------------------------------------------------------
    #
    def _residue_color_blocks(self):
        colors = self._pae.structure.residues.ribbon_colors
        color_indices = {}
        for i,color in enumerate(colors):
            c = tuple(color)
            if c in color_indices:
                color_indices[c].append(i)
            else:
                color_indices[c] = [i]
        return tuple(color_indices.items())
        
    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Color PAE Domains', self._color_domains),
                        ('Color pLDDT', self._color_plddt),
                        ('Help', self._show_help)],
                       spacing = 10)
        return f

    # ---------------------------------------------------------------------------
    #
    def _color_domains(self):
        # TODO: Log the command to do the coloring.
        self._pae.color_domains(log_command = True)

    # ---------------------------------------------------------------------------
    #
    def _color_plddt(self):
        self._pae.color_plddt(log_command = True)

    # ---------------------------------------------------------------------------
    #
    def _save_image(self, default_suffix = '.png'):
        from os.path import splitext, basename, join
        filename = splitext(basename(self._pae._pae_path))[0] + default_suffix
        from os import getcwd
        suggested_path = join(getcwd(), filename)
        from Qt.QtWidgets import QFileDialog
        parent = self.tool_window.ui_area
        path, ftype  = QFileDialog.getSaveFileName(parent,
                                                   'AlphaFold PAE Image',
                                                   suggested_path)
        if path:
            self._pae_view.save_image(path)
        
    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

    # ---------------------------------------------------------------------------
    #
    def _report_residues(self, index1, index2):
        m = self._pae.pae_matrix
        size = m.shape[0]
        s = self._pae.structure
        if s is None:
            return
        res = s.residues
        if index1 < 0 or index2 < 0 or index1 >= size or index2 >= size or len(res) < size:
            msg = ''
        else:
            d = m[index1,index2]
            dstr = '%.1f' % d
            r1,r2 = res[index1], res[index2]
            if s.num_chains == 1:
                msg = f'{r1.label_one_letter_code}{r1.number} {r2.label_one_letter_code}{r2.number} = {dstr}'
            else:
                msg = f'/{r1.chain_id} {r1.label_one_letter_code}{r1.number} /{r2.chain_id} {r2.label_one_letter_code}{r2.number} = {dstr}'
        self._info_label.setText(msg)
        
    # ---------------------------------------------------------------------------
    #
    def _rectangle_select(self, xy1, xy2):
        x1,y1 = xy1
        x2,y2 = xy2
        r1, r2 = int(min(x1,x2)), int(max(x1,x2))
        r3, r4 = int(min(y1,y2)), int(max(y1,y2))
        off_diagonal_drag = (r2 < r3 or r4 < r1)
        if self._drag_colors_structure:
            # Color residues
            if off_diagonal_drag:
                # Use two colors
                self._color_residues(r1, r2, 'lime', 'gray')
                self._color_residues(r3, r4, 'magenta')
            else:
                # Use single color
                self._color_residues(min(r1,r3), max(r2,r4), 'lime', 'gray')
        else:
            # Select residues
            ranges = [(r1,r2), (r3,r4)] if off_diagonal_drag else [(min(r1,r3), max(r2,r4))]
            self._select_residue_ranges(ranges)

    # ---------------------------------------------------------------------------
    #
    def _color_residues(self, r1, r2, colorname, other_colorname = None):
        m = self._pae.structure
        if m is None:
            return

        from chimerax.core.colors import BuiltinColors
        if other_colorname:
            other_color = BuiltinColors[other_colorname].uint8x4()
            all_res = m.residues
            all_res.ribbon_colors = other_color
            all_res.atoms.colors = other_color
            
        color = BuiltinColors[colorname].uint8x4()
        residues = m.residues[r1:r2+1]
        residues.ribbon_colors = color
        residues.atoms._colors = color

        if len(residues) > 0:
            cmd = 'color %s %s' % (_residue_range_spec(residues), colorname)
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def _select_residue_ranges(self, ranges):
        m = self._pae.structure
        if m is None:
            return

        self.session.selection.clear()

        res = m.residues
        specs = []
        for r1,r2 in ranges:
            rres = res[r1:r2+1]
            atoms = rres.atoms
            atoms.selected = True
            if len(rres) > 0:
                specs.append(_residue_range_spec(rres))

        if specs:
            cmd = 'select %s' % ' '.join(specs)
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def _rectangle_clear(self):
        pass

# ---------------------------------------------------------------------------
#
def _residue_range_spec(residues):
    s = residues[0].structure
    if s.num_chains == 1:
        nums = residues.numbers
        rspec = ':%d-%d' % (nums.min(), nums.max())
    else:
        specs = []
        for s, cid, cres in residues.by_chain:
            nums = cres.numbers
            specs.append('/%s:%d-%d' % (cid, nums.min(), nums.max()))
        rspec = ''.join(specs)
    spec = '#%s%s' % (s.id_string, rspec)
    return spec
    
from Qt.QtWidgets import QGraphicsView
class PAEView(QGraphicsView):
    def __init__(self, parent, rectangle_select_cb=None, rectangle_clear_cb=None,
                 report_residues_cb=None):
        QGraphicsView.__init__(self, parent)
        self.click_callbacks = []
        self.drag_callbacks = []
        self._mouse_down = False
        self._drag_box = None
        self._down_xy = None
        self._rectangle_select_callback = rectangle_select_cb
        self._rectangle_clear_callback = rectangle_clear_cb
        self._report_residues_callback = report_residues_cb
        self._pixmap_item = None
        self._divider_items = []
        # Report residues as mouse hovers over plot.
        self.setMouseTracking(True)

    def sizeHint(self):
        from Qt.QtCore import QSize
        return QSize(500,500)

    def resizeEvent(self, event):
        # Rescale histogram when window resizes
        self.fitInView(self.sceneRect())
        QGraphicsView.resizeEvent(self, event)

    def mousePressEvent(self, event):
        from Qt.QtCore import Qt
        if event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return	# Ignore ctrl-click that shows context menu
        self._mouse_down = True
        for cb in self.click_callbacks:
            cb(event)
        self._down_xy = self._scene_position(event)
        self._clear_drag_box()
        if self._rectangle_clear_callback:
            self._rectangle_clear_callback()

    def mouseMoveEvent(self, event):
        if self._report_residues_callback:
            x,y = self._scene_position(event)
            self._report_residues_callback(int(x),int(y))
        if self._mouse_down:
            self._drag(event)

    def _drag(self, event):
        # Only process mouse move once per graphics frame.
        for cb in self.drag_callbacks:
            cb(event)
        self._draw_drag_box(event)

    def mouseReleaseEvent(self, event):
        if self._mouse_down:
            self._mouse_down = False
            self._drag(event)
            if self._rectangle_select_callback and self._down_xy:
                self._rectangle_select_callback(self._down_xy, self._scene_position(event))

    def _scene_position(self, event):
        p = self.mapToScene(event.pos())
        return p.x(), p.y()

    def _draw_drag_box(self, event):
        x1,y1 = self._down_xy
        x2,y2 = self._scene_position(event)
        x,y,w,h = min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)
        box = self._drag_box
        if box is None:
            scene = self.scene()
            self._drag_box = scene.addRect(x,y,w,h)
        else:
            box.setRect(x,y,w,h)

    def _clear_drag_box(self):
        box = self._drag_box
        if box:
            self.scene().removeItem(box)
            self._drag_box = None

    def _make_image(self, pae_matrix, colormap, color_blocks = None):
        scene = self.scene()
        pi = self._pixmap_item
        if pi is not None:
            scene.removeItem(pi)

        rgb = pae_rgb(pae_matrix, colormap)
        if color_blocks is not None:
            self._color_blocks(rgb, pae_matrix, color_blocks)
        pixmap = pae_pixmap(rgb)
        self._pixmap_item = scene.addPixmap(pixmap)
        scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

    def _show_chain_dividers(self, chain_dividers = [], thickness = 4):
        scene = self.scene()
        di = self._divider_items
        if di:
            for i in di:
                scene.removeItem(i)
            di.clear()

        if len(chain_dividers) == 0:
            return

        r = scene.sceneRect()
        w, h = r.width(), r.height()
        for i in chain_dividers:
            x = y = i
            from Qt.QtGui import QBrush, QColor
            brush = QBrush(QColor(0,0,0))
            t = thickness-1  # Rectangle is one pixel thicker than this value due to outline
            di.extend([scene.addRect(0,y-t/2,w,t,brush=brush), scene.addRect(x-t/2,0,t,h,brush=brush)])
        for d in di:
            d.setZValue(1)  # Make sure lines are drawn above pixmap

    def _color_blocks(self, rgb, pae_matrix, color_blocks):
        from numpy import ix_
        for color, indices in color_blocks:
            colormap = self._block_colormap(color)
            subsquare = ix_(indices, indices)
            rgb[subsquare] = pae_rgb(pae_matrix[subsquare], colormap)

    def _block_colormap(self, color, bg_color = (255,255,255,255)):
        from numpy import array, float32, linspace, sqrt
        pae_range = (0, 5, 10, 15, 20, 25, 30)
        fracs = sqrt(linspace(0, 1, len(pae_range)))
        bgcolor = array([c/255 for c in bg_color], float32)
        from chimerax.core.colors import Colormap
        fcolor = array([c/255 for c in color], float32)
        colors = [((1-f)*fcolor + f*bgcolor) for f in fracs]
        colormap = Colormap(pae_range, colors)
        return colormap

    def save_image(self, path):
        pixmap = self.grab()
        pixmap.save(path)
        
# -----------------------------------------------------------------------------
#
class AlphaFoldPAE:
    def __init__(self, pae_path, structure):
        self._pae_path = pae_path
        self._pae_matrix = read_pae_matrix(pae_path)
        self.structure = structure
        self._residue_indices = None	# Map residue to index
        self._cluster_max_pae = 5
        self._cluster_clumping = 0.5
        self._cluster_min_size = 10
        self._clusters = None		# Cache computed clusters
        self._cluster_colors = None
        self._plddt_palette = 'alphafold'

    def reduce_matrix_to_residues_in_structure(self):
        '''
        Delete rows and columns of PAE matrix for which there are no structure residues.
        This allows using PAE matrix data after an AlphaFold structure has been trimmed
        by deleting N-terminal and C-terminal residues to match an experimental structure.
        Returns True if successful, False if structure does not appear to have the right
        sequence length for the PAE matrix.
        '''
        structure = self.structure
        if structure.num_residues == self.matrix_size:
            return True

        chains = structure.chains
        num_res = sum([chain.num_residues for chain in chains], 0)
        if num_res != self.matrix_size:
            # Structure does not appear to represent same number of residues as matrix.
            # This could happen if a whole chain was deleted.
            return False

        cres = [(chain.chain_id, chain.residues) for chain in chains]
        cres.sort(key = lambda cid_res: cid_res[0])	  # Sort by chain identifier
        res = sum([r for cid,r in cres], [])	          # List of residues with None for deleted ones
        res_in_structure = [i for i,r in enumerate(res) if r is not None]  # Indices of structure residues
        self._pae_matrix = self._pae_matrix[res_in_structure,:][:,res_in_structure]

        return True

    # ---------------------------------------------------------------------------
    #
    @property
    def pae_matrix(self):
        return self._pae_matrix

    # ---------------------------------------------------------------------------
    #
    @property
    def matrix_size(self):
        return self._pae_matrix.shape[0]

    # ---------------------------------------------------------------------------
    #
    def value(self, aligned_residue, scored_residue):
        ai = self._residue_index(aligned_residue)
        si = self._residue_index(scored_residue)
        return self._pae_matrix[ai,si]

    # ---------------------------------------------------------------------------
    #
    def _residue_index(self, residue):
        ri = self._residue_indices
        if ri is None:
            self._residue_indices = ri = {r:i for i,r in enumerate(self.structure.residues)}
        return ri[residue]
    
    # ---------------------------------------------------------------------------
    #
    def color_domains(self, cluster_max_pae = None, cluster_clumping = None,
                      cluster_min_size = None, log_command = False):
        m = self.structure
        if m is None:
            return

        self.set_default_domain_clustering(cluster_max_pae, cluster_clumping,
                                           cluster_min_size)

        if self._clusters is None:
            self._clusters = pae_domains(self._pae_matrix,
                                         pae_cutoff = self._cluster_max_pae,
                                         graph_resolution = self._cluster_clumping,
                                         min_size = self._cluster_min_size)
            from chimerax.core.colors import random_colors
            self._cluster_colors = random_colors(len(self._clusters), seed=0)

        if log_command:
            cmd = f'alphafold pae #{m.id_string} colorDomains true'
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(m.session, cmd)

        color_by_pae_domain(m.residues, self._clusters, colors=self._cluster_colors)
        set_pae_domain_residue_attribute(m.residues, self._clusters)
        
    # ---------------------------------------------------------------------------
    #
    def set_default_domain_clustering(self, cluster_max_pae = None, cluster_clumping = None,
                                      cluster_min_size = None):
        changed = False
        if cluster_max_pae is not None and cluster_max_pae != self._cluster_max_pae:
            self._cluster_max_pae = cluster_max_pae
            changed = True
        if cluster_clumping is not None and cluster_clumping != self._cluster_clumping:
            self._cluster_clumping = cluster_clumping
            changed = True
        if cluster_min_size is not None and cluster_min_size != self._cluster_min_size:
            self._cluster_min_size = cluster_min_size
            changed = True
        if changed:
            self._clusters = None
        return changed

    # ---------------------------------------------------------------------------
    #
    def color_plddt(self, log_command = False):
        m = self.structure
        if m is None:
            return

        cmd = f'color bfactor #{m.id_string} palette {self._plddt_palette}'
        if not log_command:
            cmd += ' log false'
        from chimerax.core.commands import run
        run(m.session, cmd, log = log_command)

# -----------------------------------------------------------------------------
#
def read_pae_matrix(path):
    if path.endswith('.json'):
        return read_json_pae_matrix(path)
    elif path.endswith('.pkl'):
        return read_pickle_pae_matrix(path)
    else:
        from chimerax.core.errors import UserError
        raise UserError(f'AlphaFold predicted aligned error (PAE) files must be in JSON (*.json) or pickle (*.pkl) format, {path} unrecognized format')

# -----------------------------------------------------------------------------
#
def read_json_pae_matrix(path):
    '''Open AlphaFold database distance error PAE JSON file returning a numpy matrix.'''
    f = open(path, 'r')
    import json
    j = json.load(f)
    f.close()

    if isinstance(j, dict) and 'pae' in j:
        # ColabFold 1.3 produces a JSON file different from AlphaFold database.
        from numpy import array, float32
        pae = array(j['pae'], float32)
        return pae
    
    if not isinstance(j, list):
        from chimerax.core.errors import UserError
        raise UserError(f'JSON file "{path}" is not AlphaFold predicted aligned error data, expected a top level list')
    d = j[0]

    if not isinstance(d, dict):
        from chimerax.core.errors import UserError
        raise UserError(f'JSON file "{path}" is not AlphaFold predicted aligned error data, expected a top level list containing a dictionary')
        
    if 'residue1' in d and 'residue2' in d and 'distance' in d:
        # AlphaFold Database versions 1 and 2 use this format
        # Read PAE into numpy array
        from numpy import array, zeros, float32, int32
        r1 = array(d['residue1'], dtype=int32)
        r2 = array(d['residue2'], dtype=int32)
        ea = array(d['distance'], dtype=float32)
        # me = d['max_predicted_aligned_error']
        n = r1.max()
        pae = zeros((n,n), float32)
        pae[r1-1,r2-1] = ea
        return pae
        
    if 'predicted_aligned_error' in d:
        # AlphaFold Database version 3 uses this format.
        from numpy import array, float32
        pae = array(d['predicted_aligned_error'], dtype=float32)
        return pae
    
    keys = ', '.join(str(k) for k in d.keys())
    from chimerax.core.errors import UserError
    raise UserError(f'JSON file "{path}" is not AlphaFold predicted aligned error data, expected a dictionary with keys "predicted_aligned_error" or "residue1", "residue2" and "distance", got keys {keys}')

# -----------------------------------------------------------------------------
#
def read_pickle_pae_matrix(path):
    f = open(path, 'rb')
    import pickle
    try:
        p = pickle.load(f)
    except ModuleNotFoundError as e:
        if 'jax' in str(e):
            _fix_alphafold_pickle_jax_dependency()
            p = pickle.load(f)
        else:
            raise
            
    f.close()
    if isinstance(p, dict) and 'predicted_aligned_error' in p:
        return p['predicted_aligned_error']

    from chimerax.core.errors import UserError
    raise UserError(f'File {path} does not contain AlphaFold predicted aligned error (PAE) data. The AlphaFold "monomer" preset does not compute PAE.  Run AlphaFold with the "monomer_ptm" or "multimer" presets to get PAE values.')
    
# -----------------------------------------------------------------------------
#
def _fix_alphafold_pickle_jax_dependency():
    '''
    In AlphaFold 2.2.4 the pickle files written out have a dependency on the jax
    module which prevents unpickling them, ChimeraX bug #8032.
    Work around this by adding a fake jax module.
    '''
    import sys
    from types import ModuleType
    sys.modules['jax'] = ModuleType('dummy_jax_for_pickle')
    sys.modules['jax._src.device_array'] = m = ModuleType('dummy_jax_device_array')
    def dummy_jax_reconstruct_device_array(*args, **kw):
        return None
    m.reconstruct_device_array = dummy_jax_reconstruct_device_array

# -----------------------------------------------------------------------------
#
def pae_rgb(pae_matrix, colormap):
    rgb_flat = colormap.interpolated_rgba8(pae_matrix.ravel())[:,:3]
    n = pae_matrix.shape[0]
    rgb = rgb_flat.reshape((n,n,3)).copy()
    return rgb

# -----------------------------------------------------------------------------
#
def _pae_colormap(max = 30, step = 5):
    colors = []
    values = []
    from math import sqrt
    for p in range(0,max+1,step):
        f = p/max
        r = b = (30 + 225*f*f)/255
        g = (70 + 185*sqrt(f))/255
        a = 1
        values.append(p)
        colors.append((r,g,b,a))
    print(tuple(values))
    print('(' + ', '.join('(%.3f,%.3f,%.3f,%.0f)'%c for c in colors) + ')')

# -----------------------------------------------------------------------------
#
def pae_pixmap(rgb):
    # Save image to a PNG file
    from Qt.QtGui import QImage, QPixmap
    h, w = rgb.shape[:2]
    im = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(im)
    return pixmap

# -----------------------------------------------------------------------------
#
def pae_image(rgb):
    from PIL import Image
    pi = Image.fromarray(rgb)
    #pi.save('test.png')      # Save image to a PNG file
    return pi

# -----------------------------------------------------------------------------
# Code take from Tristan Croll, ChimeraX ticket #4966.
#
# https://github.com/tristanic/isolde/blob/master/isolde/src/reference_model/alphafold/find_domains.py
#
def pae_domains(pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=0.5,
                min_size = None):
    # PAE matrix is not strictly symmetric.
    # Prediction for error in residue i when aligned on residue j may be different from 
    # error in j when aligned on i. Take the smallest error estimate for each pair.
    import numpy
    pae_matrix = numpy.minimum(pae_matrix, pae_matrix.T)
    pae_matrix = numpy.maximum(pae_matrix, 0.2)	# AlphaFold Database version 3 has 0 values.
    weights = 1/pae_matrix**pae_power if pae_power != 1 else 1/pae_matrix

    import networkx as nx
    g = nx.Graph()
    size = weights.shape[0]
    g.add_nodes_from(range(size))
    edges = numpy.argwhere(pae_matrix < pae_cutoff)
    # Limit to bottom triangle of matrix
    edges = edges[edges[:,0]<edges[:,1]]
    sel_weights = weights[edges.T[0], edges.T[1]]
    wedges = [(i,j,w) for (i,j),w in zip(edges,sel_weights)]
    g.add_weighted_edges_from(wedges)

    from networkx.algorithms.community import greedy_modularity_communities
    clusters = greedy_modularity_communities(g, weight='weight', resolution=graph_resolution)
    if min_size:
        clusters = [c for c in clusters if len(c) >= min_size]
    return clusters

# -----------------------------------------------------------------------------
#
def color_by_pae_domain(residues, clusters, colors = None):
    if colors is None:
        from chimerax.core.colors import random_colors
        colors = random_colors(len(clusters), seed=0)

    from numpy import array, int32
    for c, color in zip(clusters, colors):
        cresidues = residues[array(list(c),int32)]
        cresidues.ribbon_colors = color
        cresidues.atoms.colors = color
    
# -----------------------------------------------------------------------------
#
def set_pae_domain_residue_attribute(residues, clusters):
    if len(residues) > 0:
        # Register attribute so it is saved in sessions.
        session = residues[0].structure.session
        from chimerax.atomic import Residue
        Residue.register_attr(session, 'pae_domain', "AlphaFold", attr_type = int)

    cnum = {}
    for i, cluster in enumerate(clusters):
        for ri in cluster:
            cnum[ri] = i+1
    for ri,r in enumerate(residues):
        r.pae_domain = cnum.get(ri)
    
# -----------------------------------------------------------------------------
#
def alphafold_pae(session, structure = None, file = None, uniprot_id = None,
                  palette = None, range = None, plot = None, divider_lines = None,
                  color_domains = False, connect_max_pae = 5, cluster = 0.5, min_size = 10,
                  version = None):
    '''Load AlphaFold predicted aligned error file and show plot or color domains.'''

    if uniprot_id:
        from .database import alphafold_pae_url
        pae_url = alphafold_pae_url(session, uniprot_id, database_version = version)
        file_name = pae_url.split('/')[-1]
        from chimerax.core.fetch import fetch_file
        file = fetch_file(session, pae_url, 'AlphaFold PAE %s' % uniprot_id,
                          file_name, 'AlphaFold', error_status = False)
        
    if file:
        pae = AlphaFoldPAE(file, structure)
        if structure:
            if not pae.reduce_matrix_to_residues_in_structure():
                from chimerax.core.errors import UserError
                raise UserError('Number of residues in structure "%s" is %d which does not match PAE matrix size %d.'
                                % (str(structure), structure.num_residues, pae.matrix_size) +
                                '\n\nThis can happen if chains were deleted from the AlphaFold model or if the PAE data was applied to a structure that was not the one predicted by AlphaFold.  Use the full-length AlphaFold model to show predicted aligned error.')
            structure.alphafold_pae = pae
    elif structure is None:
        from chimerax.core.errors import UserError
        raise UserError('No structure or PAE file specified.')
    else:
        pae = getattr(structure, 'alphafold_pae', None)
        if pae is None:
            from chimerax.core.errors import UserError
            raise UserError('No predicted aligned error (PAE) data opened for structure #%s'
                            % structure.id_string)

    if plot is None:
        plot = not color_domains	# Plot by default if not coloring domains.
        
    if plot:
        from chimerax.core.colors import colormap_with_range
        colormap = colormap_with_range(palette, range, default_colormap_name = 'pae',
                                       full_range = (0,30))
        p = getattr(structure, '_alphafold_pae_plot', None)
        if p is None or p.closed():
            dividers = True if divider_lines is None else divider_lines
            p = AlphaFoldPAEPlot(session, 'AlphaFold Predicted Aligned Error', pae,
                                 colormap=colormap, divider_lines=dividers)
            if structure:
                structure._alphafold_pae_plot = p
        else:
            p.display(True)
            if palette is not None or range is not None:
                p.set_colormap(colormap)
            if divider_lines is not None:
                p.show_chain_dividers(divider_lines)

    pae.set_default_domain_clustering(connect_max_pae, cluster)
    if color_domains:
        if structure is None:
            from chimerax.core.errors import UserError
            raise UserError('Must specify structure to color domains.')
        pae.color_domains(connect_max_pae, cluster, min_size)

# -----------------------------------------------------------------------------
#
def register_alphafold_pae_command(logger):
    from chimerax.core.commands import CmdDesc, register, OpenFileNameArg, ColormapArg, ColormapRangeArg, BoolArg, FloatArg, IntArg
    from chimerax.atomic import AtomicStructureArg, UniProtIdArg
    desc = CmdDesc(
        optional = [('structure', AtomicStructureArg)],
        keyword = [('file', OpenFileNameArg),
                   ('uniprot_id', UniProtIdArg),
                   ('palette', ColormapArg),
                   ('range', ColormapRangeArg),
                   ('plot', BoolArg),
                   ('divider_lines', BoolArg),
                   ('color_domains', BoolArg),
                   ('connect_max_pae', FloatArg),
                   ('cluster', FloatArg),
                   ('min_size', IntArg),
                   ('version', IntArg)],
        synopsis = 'Show AlphaFold predicted aligned error'
    )
    
    register('alphafold pae', desc, alphafold_pae, logger=logger)
