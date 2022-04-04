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
    help = 'help:user/tools/alphafold.html'

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
        run(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

# -----------------------------------------------------------------------------
#
def _matching_pae_file(structure_path):
    from os.path import split, isdir, join
    dir, filename = split(structure_path)
    if not isdir(dir):
        return None

    from os import listdir
    dfiles = listdir(dir)
    pfiles = [f for f in dfiles if f.endswith('.json') or f.endswith('.pkl')]
    if len(pfiles) == 1:
        path = join(dir, pfiles[0])
        return path
    elif len(pfiles) > 1:
        # Find longest matching prefix
        m = [(len(matching_prefix(pf,filename)),pf) for pf in pfiles]
        m.sort(reverse = True)
        min_match_length = 6
        if m[0][0] > min_match_length and m[0][0] > m[1][0]:
            path = join(dir, m[0][1])
            return path

    return None

# ---------------------------------------------------------------------------
#
def _guess_uniprot_id(structure_path):
    from os.path import split
    filename = split(structure_path)[1]
    from .database import uniprot_id_from_filename
    uniprot_id = uniprot_id_from_filename(filename)
    return uniprot_id

# -----------------------------------------------------------------------------
#
def matching_prefix(s1, s2):
    for i in range(len(s1)):
        if s2[i] != s1[i]:
            break
    return s1[:i]

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
    help = 'help:user/tools/alphafold.html'

    def __init__(self, session, tool_name, pae, colormap = None):

        self._pae = pae		# AlphaFoldPAE instance
        
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
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

        self._pae_view = gv = PAEView(parent, self._rectangle_select, self._rectangle_clear)
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
        
        layout.addStretch(1)    # Extra space at end

        self.set_colormap(colormap)

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
    def set_colormap(self, colormap):
        if colormap is None:
            from chimerax.core.colors import BuiltinColormaps
            colormap = BuiltinColormaps['pae']
        self._pae_view._make_image(self._pae._pae_matrix, colormap)
   
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
        self._pae.color_domains()
         
    # ---------------------------------------------------------------------------
    #
    def _color_plddt(self):
        self._pae.color_plddt()

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

    # ---------------------------------------------------------------------------
    #
    def _rectangle_select(self, xy1, xy2):
        x1,y1 = xy1
        x2,y2 = xy2
        r1, r2 = int(min(x1,x2)), int(max(x1,x2))
        r3, r4 = int(min(y1,y2)), int(max(y1,y2))
        if r2 < r3 or r4 < r1:
            # Use two colors
            self._color_residues(r1, r2, 'lime', 'gray')
            self._color_residues(r3, r4, 'magenta')
        else:
            # Use single color
            self._color_residues(min(r1,r3), max(r2,r4), 'lime', 'gray')

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
        
    # ---------------------------------------------------------------------------
    #
    def _rectangle_clear(self):
        pass

from Qt.QtWidgets import QGraphicsView
class PAEView(QGraphicsView):
    def __init__(self, parent, rectangle_select_cb=None, rectangle_clear_cb=None):
        QGraphicsView.__init__(self, parent)
        self.click_callbacks = []
        self.drag_callbacks = []
        self.mouse_down = False
        self._drag_box = None
        self._down_xy = None
        self._rectangle_select_callback = rectangle_select_cb
        self._rectangle_clear_callback = rectangle_clear_cb
        self._pixmap_item = None

    def sizeHint(self):
        from Qt.QtCore import QSize
        return QSize(500,500)

    def resizeEvent(self, event):
        # Rescale histogram when window resizes
        self.fitInView(self.sceneRect())
        QGraphicsView.resizeEvent(self, event)

    def mousePressEvent(self, event):
        self.mouse_down = True
        for cb in self.click_callbacks:
            cb(event)
        self._down_xy = self._scene_position(event)
        self._clear_drag_box()
        if self._rectangle_clear_callback:
            self._rectangle_clear_callback()

    def mouseMoveEvent(self, event):
        self._drag(event)

    def _drag(self, event):
        # Only process mouse move once per graphics frame.
        for cb in self.drag_callbacks:
            cb(event)
        self._draw_drag_box(event)

    def mouseReleaseEvent(self, event):
        self.mouse_down = False
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

    def _make_image(self, pae_matrix, colormap):
        scene = self.scene()
        pi = self._pixmap_item
        if pi is not None:
            scene.removeItem(pi)

        rgb = pae_rgb(pae_matrix, colormap)
        pixmap = pae_pixmap(rgb)
        self._pixmap_item = scene.addPixmap(pixmap)
        scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

# -----------------------------------------------------------------------------
#
class AlphaFoldPAE:
    def __init__(self, pae_path, structure):
        self._pae_matrix = read_pae_matrix(pae_path)
        self.structure = structure
        self._cluster_max_pae = 5
        self._cluster_clumping = 0.5
        self._clusters = None		# Cache computed clusters
        self._cluster_colors = None

    # ---------------------------------------------------------------------------
    #
    def color_domains(self, cluster_max_pae = None, cluster_clumping = None):
        m = self.structure
        if m is None:
            return

        self.set_default_domain_clustering(cluster_max_pae, cluster_clumping)

        if self._clusters is None:
            self._clusters = pae_domains(self._pae_matrix,
                                         pae_cutoff = self._cluster_max_pae,
                                         graph_resolution = self._cluster_clumping)
            from chimerax.core.colors import random_colors
            self._cluster_colors = random_colors(len(self._clusters), seed=0)

        color_by_pae_domain(m.residues, self._clusters, colors=self._cluster_colors)

    # ---------------------------------------------------------------------------
    #
    def set_default_domain_clustering(self, cluster_max_pae = None, cluster_clumping = None):
        changed = False
        if cluster_max_pae is not None and cluster_max_pae != self._cluster_max_pae:
            self._cluster_max_pae = cluster_max_pae
            changed = True
        if cluster_clumping is not None and cluster_clumping != self._cluster_clumping:
            self._cluster_clumping = cluster_clumping
            changed = True
        if changed:
            self._clusters = None
        return changed

    # ---------------------------------------------------------------------------
    #
    def color_plddt(self):
        m = self.structure
        if m is None:
            return

        cmd = 'color bfactor #%s palette alphafold log false' % m.id_string
        from chimerax.core.commands import run
        run(m.session, cmd, log = False)

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
    d = j[0]

    # Read distance errors into numpy array
    from numpy import array, zeros, float32, int32
    r1 = array(d['residue1'], dtype=int32)
    r2 = array(d['residue2'], dtype=int32)
    ea = array(d['distance'], dtype=float32)
    me = d['max_predicted_aligned_error']
    n = r1.max()
    pae = zeros((n,n), float32)
    pae[r1-1,r2-1] = ea

    return pae

# -----------------------------------------------------------------------------
#
def read_pickle_pae_matrix(path):
    f = open(path, 'rb')
    import pickle
    p = pickle.load(f)
    f.close()
    if isinstance(p, dict) and 'predicted_aligned_error' in p:
        return p['predicted_aligned_error']

    from chimerax.core.errors import UserError
    raise UserError(f'File {path} does not contain AlphaFold predicted aligned error (PAE) data')

# -----------------------------------------------------------------------------
#
def pae_rgb(pae_matrix, colormap):

    rgb_flat = colormap.interpolated_rgba8(pae_matrix.flat)[:,:3]
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
def pae_domains(pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=0.5):
    # PAE matrix is not strictly symmetric.
    # Prediction for error in residue i when aligned on residue j may be different from 
    # error in j when aligned on i. Take the smallest error estimate for each pair.
    import numpy
    pae_matrix = numpy.minimum(pae_matrix, pae_matrix.T)
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
    return clusters

# -----------------------------------------------------------------------------
#
def color_by_pae_domain(residues, clusters, colors = None, min_cluster_size=10):
    if colors is None:
        from chimerax.core.colors import random_colors
        colors = random_colors(len(clusters), seed=0)

    from numpy import array, int32
    for c, color in zip(clusters, colors):
        if len(c) >= min_cluster_size:
            cresidues = residues[array(list(c),int32)]
            cresidues.ribbon_colors = color
            cresidues.atoms.colors = color

# -----------------------------------------------------------------------------
#
def alphafold_pae(session, structure = None, file = None, uniprot_id = None,
                  palette = None, range = None, plot = None,
                  color_domains = False, connect_max_pae = 5, cluster = 0.5):
    '''Load AlphaFold predicted aligned error file and show plot or color domains.'''

    if uniprot_id:
        from .database import alphafold_pae_url
        pae_url = alphafold_pae_url(session, uniprot_id)
        file_name = pae_url.split('/')[-1]
        from chimerax.core.fetch import fetch_file
        file = fetch_file(session, pae_url, 'AlphaFold PAE %s' % uniprot_id,
                          file_name, 'AlphaFold', error_status = False)
        
    if file:
        pae = AlphaFoldPAE(file, structure)
        if structure:
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
            p = AlphaFoldPAEPlot(session, 'AlphaFold Predicted Aligned Error', pae,
                                 colormap=colormap)
            if structure:
                structure._alphafold_pae_plot = p
        else:
            p.display(True)
            if palette is not None or range is not None:
                p.set_colormap(colormap)

    pae.set_default_domain_clustering(connect_max_pae, cluster)
    if color_domains:
        if structure is None:
            from chimerax.core.errors import UserError
            raise UserError('Must specify structure to color domains.')
        pae.color_domains(connect_max_pae, cluster)

# -----------------------------------------------------------------------------
#
def register_alphafold_pae_command(logger):
    from chimerax.core.commands import CmdDesc, register, OpenFileNameArg, ColormapArg, ColormapRangeArg, BoolArg, FloatArg
    from chimerax.atomic import AtomicStructureArg, UniProtIdArg
    desc = CmdDesc(
        optional = [('structure', AtomicStructureArg)],
        keyword = [('file', OpenFileNameArg),
                   ('uniprot_id', UniProtIdArg),
                   ('palette', ColormapArg),
                   ('range', ColormapRangeArg),
                   ('plot', BoolArg),
                   ('color_domains', BoolArg),
                   ('connect_max_pae', FloatArg),
                   ('cluster', FloatArg)],
        synopsis = 'Show AlphaFold predicted aligned error'
    )
    
    register('alphafold pae', desc, alphafold_pae, logger=logger)
