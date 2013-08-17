keyboard_shortcuts = None

def register_shortcuts(viewer):

    global keyboard_shortcuts
    if keyboard_shortcuts is None:
        keyboard_shortcuts = Keyboard_Shortcuts(viewer)

    ks = keyboard_shortcuts
    v = viewer

    map_shortcuts = (
      ('me', show_mesh, 'Show mesh'),
      ('sf', show_surface, 'Show surface'),
      ('gs', show_grayscale, 'Show grayscale'),
      ('ob', toggle_outline_box, 'Toggle outline box'),
      ('pl', show_one_plane, 'Show one plane'),
      ('pa', show_all_planes, 'Show all planes'),
      ('o3', toggle_orthoplanes, 'Show 3 orthogonal planes'),
      ('bx', toggle_box_faces, 'Show box faces'),
      ('t5', show_map_transparent, 'Make map transparent'),
      ('fr', show_map_full_resolution, 'Show map at full resolution'),
      )
    mapcat = 'Map Display'
    for k,f,d in map_shortcuts:
      ks.add_shortcut(k, f, d, category = mapcat, each_map = True)

    mol_shortcuts = (
        ('bu', show_biological_unit, 'Show biological unit'),
        ('c1', color_one_color, 'Color molecule one color'),
        ('ce', color_by_element, 'Color atoms by element'),
        ('cc', color_by_chain, 'Color chains'),
        ('ms', lambda m,v=v: show_molecular_surface(m,v), 'Show molecular surface'),
        ('mb', molecule_bonds, 'Compute molecule bonds using templates'),
        ('da', show_atoms, 'Display molecule atoms'),
        ('ha', hide_atoms, 'Undisplay molecule atoms'),
        ('bs', show_ball_and_stick, 'Display atoms in ball and stick'),
        ('sp', show_sphere, 'Display atoms in sphere style'),
        ('st', show_stick, 'Display atoms in stick style'),
        ('rb', show_ribbon, 'Show molecule ribbon'),
        ('hr', hide_ribbon, 'Undisplay molecule ribbon'),
        ('la', show_ligands, 'Show ligand atoms'),
        ('r+', fat_ribbons, 'Fat ribbons'),
        ('r-', thin_ribbons, 'Thin ribbons'),
    )
    molcat = 'Molecule Display'
    for k,f,d in mol_shortcuts:
      ks.add_shortcut(k, f, d, category = molcat, each_molecule = True)

    ocat = 'Open, Save, Close'   # shortcut documentation category
    gcat = 'General Controls'
    from ..file_io import session, history, opensave
    view_shortcuts = (
        ('op', opensave.show_open_file_dialog, 'Open file', ocat),
        ('sv', opensave.save_session_as, 'Save session as...', ocat),
        ('Sv', opensave.save_session, 'Save session', ocat),
        ('si', opensave.save_image, 'Save image', ocat),
        ('oi', opensave.open_image, 'Open image', ocat),
        ('Ca', close_all_models, 'Close all models', ocat),
        ('mp', enable_move_planes_mouse_mode, 'Move planes mouse mode', mapcat),
        ('ct', enable_contour_mouse_mode, 'Adjust contour level mouse mode', mapcat),
        ('mm', enable_move_molecules_mouse_mode, 'Move molecules mouse mode', molcat),
        ('rm', enable_rotate_molecules_mouse_mode, 'Rotate molecules mouse mode', molcat),
        ('ft', fit_molecule_in_map, 'Fit molecule in map', mapcat),
        ('sh', tile_models, 'Show or hide models', gcat),
        ('bk', set_background_black, 'Black background', gcat),
        ('wb', set_background_white, 'White background', gcat),
        ('gb', set_background_gray, 'Gray background', gcat),
        ('sl', selection_mouse_mode, 'Select models mouse mode', gcat),
        ('Ds', delete_selected_models, 'Delete selected models', ocat),
        ('lp', leap_position_mode, 'Enable leap motion input device', gcat),
        ('lx', leap_chopsticks_mode, 'Enable leap motion chopstick mode', gcat),
        ('lv', leap_velocity_mode, 'Enable leap motion velocity mode', gcat),
        ('lf', leap_focus, 'Check if app has leap focus', gcat),
        ('lq', leap_quit, 'Quit using leap motion input device', gcat),
    )
    for k,f,d,cat in view_shortcuts:
      ks.add_shortcut(k, f, d, category = cat, view_arg = True)

    from .gui import show_log
    misc_shortcuts = (
        ('rv', v.initial_camera_view, 'Reset view', gcat),
        ('va', v.view_all, 'View all', gcat),
        ('rs', history.show_history_thumbnails, 'Show recent sessions', ocat),
        ('cs', v.clear_selection, 'Clear selection', gcat),
        ('Qt', v.quit, 'Quit', ocat),
        ('cl', command_line, 'Enter command', gcat),
        ('ks', list_keyboard_shortcuts, 'List keyboard shortcuts', gcat),
        ('mn', show_manual, 'Show manual', gcat),
        ('lg', show_log, 'Show command log', gcat),
        ('ch', show_command_history, 'Show command history', gcat),
        ('rt', show_stats, 'Show model statistics', gcat),
        ('bm', matrix_profile, 'matrix profiling', gcat),
        )
    for k,f,d,cat in misc_shortcuts:
      ks.add_shortcut(k, f, d, category = cat)

    ks.category_columns = ((ocat,molcat), (mapcat,), (gcat,))

    return ks

class Keyboard_Shortcuts:

  def __init__(self, viewer):

    # Keyboard shortcuts
    self.shortcuts = {}
    self.keys = ''
    self.viewer = viewer

  def add_shortcut(self, key_seq, func, description = '', category = None,
                   each_map = False, each_molecule = False, view_arg = False):

    v = self.viewer
    if each_map:
        def f(v=v, func=func):
            for m in shortcut_maps(v):
                func(m)
    elif each_molecule:
        def f(v=v, func=func):
            for m in shortcut_molecules(v):
                func(m)
    elif view_arg:
        def f(v=v, func=func):
            func(v)
    else:
        f = func
    self.shortcuts[key_seq] = (f, description, category)

  def key_pressed(self, event):

    c = str(event.text())
    self.keys += c
    k = self.keys
    s = self.shortcuts
    if k in s:
      keys = self.keys
      self.keys = ''
      self.run_shortcut(keys)
      return
    
    is_prefix = False
    for ks in s.keys():
        if ks.startswith(k):
            is_prefix = True
            break
    msg = self.keys if is_prefix else '%s - unknown shortcut' % self.keys
    if not is_prefix:
        self.keys = ''

    from .gui import show_status
    show_status(msg)

  def run_shortcut(self, keys):
      fdc = self.shortcuts.get(keys)
      if fdc is None:
        return
      f,d,c = fdc
      msg = '%s - %s' % (keys, d)
      from .gui import show_status, show_info
      show_status(msg)
      show_info(msg, color = '#808000')
      f()

def shortcut_maps(v):
  mlist = [m for m in v.maps() if m.selected]
  if len(mlist) == 0:
    mlist = [m for m in v.maps() if m.display]
  return mlist

def shortcut_molecules(v):
  mlist = [m for m in v.molecules() if m.selected]
  if len(mlist) == 0:
    mlist = [m for m in v.molecules() if m.display]
  return mlist

def close_all_models(viewer):
    viewer.close_all_models()
    from ..file_io import history
    history.show_history_thumbnails()

def show_mesh(m):
  m.set_representation('mesh')
  m.show()

def show_surface(m):
  m.set_representation('surface')
  m.show()

def show_grayscale(m):
  m.set_representation('solid')
  m.show()

def toggle_outline_box(m):
  ro = m.rendering_options
  ro.show_outline_box = not ro.show_outline_box
  m.show()

def show_one_plane(m):
  ijk_step = (1,1,1)
  ijk_min, ijk_max = [list(b) for b in m.region[:2]]
  ijk_min[2] = ijk_max[2] = (ijk_min[2] + ijk_max[2])//2
  m.set_parameters(orthoplanes_shown = (False, False, False),
                   box_faces = False)
  m.new_region(ijk_min, ijk_max, ijk_step, adjust_step = False)
  m.show('solid')
        
def show_all_planes(m):
  ijk_min = (0,0,0)
  ijk_max = tuple(s-1 for s in m.data.size)
  m.new_region(ijk_min, ijk_max)

def toggle_orthoplanes(m):
  s = False in m.rendering_options.orthoplanes_shown
  p = tuple(s/2 for s in m.data.size)
  m.set_parameters(orthoplanes_shown = (s,s,s),
                   orthoplane_positions = p,
                   color_mode = 'l8' if s else 'auto8',
                   box_faces = False)
  m.show('solid')

def toggle_box_faces(m):
  s = not m.rendering_options.box_faces
  m.set_parameters(box_faces = s,
                   color_mode = 'l8' if s else 'auto8',
                   orthoplanes_shown = (False, False, False))
  m.show('solid')

def enable_move_planes_mouse_mode(viewer, button = 'right'):

  from ..VolumeViewer.moveplanes import planes_mouse_mode as pmm
  viewer.bind_mouse_mode(button,
                         lambda e,v=viewer: pmm.mouse_down(v,e),
                         lambda e,v=viewer: pmm.mouse_drag(v,e),
                         lambda e,v=viewer: pmm.mouse_up(v,e))

def enable_contour_mouse_mode(viewer, button = 'right'):
  v = viewer
  v.bind_mouse_mode(button, v.mouse_down, v.mouse_contour_level, v.mouse_up)

def enable_move_molecules_mouse_mode(viewer, button = 'right'):
  v = viewer
  v.bind_mouse_mode(button, v.mouse_down, v.mouse_translate_molecules, v.mouse_up)

def enable_rotate_molecules_mouse_mode(viewer, button = 'right'):
  v = viewer
  v.bind_mouse_mode(button, v.mouse_down, v.mouse_rotate_molecules, v.mouse_up)

def fit_molecule_in_map(viewer):
    mols, maps = viewer.molecules(), viewer.maps()
    if len(mols) != 1 or len(maps) != 1:
        print('ft: Fit molecule in map requires exactly one open molecule and one open map.')
        return

    mol, map = mols[0], maps[0]
    points = mol.xyz
    point_weights = None        # Equal weight for each atom
    data_array = map.full_matrix()
    xyz_to_ijk_transform = map.data.xyz_to_ijk_transform * map.place.inverse() * mol.place
    from .. import FitMap
    move_tf, stats = FitMap.locate_maximum(points, point_weights, data_array, xyz_to_ijk_transform)
    mol.place = mol.place * move_tf
    for k,v in stats.items():
        print(k,v)

def show_biological_unit(m):

    if hasattr(m, 'pdb_text'):
        from ..file_io import biomt
        matrices = biomt.pdb_biomt_matrices(m.pdb_text)
        print (m.path, 'biomt', len(matrices))
        if matrices:
            m.copies = matrices
            m.update_level_of_detail()

def show_map_transparent(m):
    m.surface_colors = tuple((r,g,b,0.5 if a == 1 else 1) for r,g,b,a in m.surface_colors)
    m.show()

def tile_models(viewer):
    viewer.tile_models = not viewer.tile_models
    viewer.bind_mouse_mode('left', lambda e,v=viewer: hide_show_mouse_mode(e,v))
  
def hide_show_mouse_mode(event, viewer):
    if not viewer.tile_models:
        viewer.bind_standard_mouse_modes(['left'])
        return
    w, h = viewer.window_size
    x,y = event.x(), event.y()
    y = (h-1)-y   # OpenGL origin is lower left corner, Qt is upper left corner
    tiles = viewer.tiles()
    t = None
    for i,(tx,ty,tw,th) in enumerate(tiles):
        if x >= tx and y >= ty and x < tx+tw and y < ty+th:
            t = i
            break
    if t == 0 or t is None:
        viewer.tile_models = False
        viewer.bind_standard_mouse_modes(['left'])
    else:
        models = viewer.models
        if i <= len(models):
            m = models[i-1]
            m.display = not m.display

def set_background_color(viewer, color):
    viewer.background_color = color
def set_background_black(viewer):
    set_background_color(viewer, (0,0,0,1))
def set_background_gray(viewer):
    set_background_color(viewer, (0.5,0.5,0.5,1))
def set_background_white(viewer):
    set_background_color(viewer, (1,1,1,1))

def selection_mouse_mode(viewer):
    def mouse_down(event, v=viewer):
        x,y = event.x(), event.y()
        p, s = v.first_intercept(x,y)
        from .gui import show_status
        if s is None:
            v.clear_selection()
            show_status('cleared selection')
        else:
            for m in s.models():
                m.selected = not m.selected
                if m.selected:
                    from .qt import QtCore
                    if not (event.modifiers() & QtCore.Qt.ShiftModifier):
                        v.clear_selection()
                        v.select_model(m)
                        show_status('Selected %s' % m.name)
                else:
                    v.unselect_model(m)
    viewer.bind_mouse_mode('right', mouse_down)

def command_line():
  from .gui import main_window
  main_window.focus_on_command_line()
#  from .qt import QtCore
#  QtCore.QTimer.singleShot(1000, main_window.focus_on_command_line)

def delete_selected_models(viewer):
  viewer.close_models(tuple(viewer.selected))

def show_map_full_resolution(m):
  m.new_region(ijk_step = (1,1,1), adjust_step = False)

def show_molecular_surface(m, viewer, res = 3.0, grid = 0.5):
  if hasattr(m, 'molsurf') and m.molsurf in viewer.models:
    m.molsurf.display = True
  else:
    from .. import molecule, molmap
    atoms = molecule.Atom_Set()
    atoms.add_molecules([m])
    s = molmap.molecule_map(atoms, res, grid)
    s.new_region(ijk_step = (1,1,1), adjust_step = False)
    s.show()
    m.molsurf = s

def color_by_element(m):
  m.set_color_mode('by element')
def color_by_chain(m):
  m.set_color_mode('by chain')
def color_one_color(m):
  m.set_color_mode('single')

def show_atoms(m):
  m.set_atom_display(True)
def hide_atoms(m):
  m.set_atom_display(False)
def show_sphere(m):
  m.set_atom_style('sphere')
def show_stick(m):
  m.set_atom_style('stick')
def show_ball_and_stick(m):
  m.set_atom_style('ballstick')
def show_ribbon(m):
  m.set_ribbon_display(True)
def hide_ribbon(m):
  m.set_ribbon_display(False)
def fat_ribbons(m):
    m.set_ribbon_radius(1.0)
def thin_ribbons(m):
    m.set_ribbon_radius(0.5)
def show_ligands(m):
    m.show_nonribbon_atoms()
def molecule_bonds(m):
    from ..molecule import connect
    connect.create_molecule_bonds(m)
    if not m.bonds is None:
        msg = 'Created %d bonds for %s using templates' % (len(m.bonds), m.name)
        from .gui import show_status, show_info
        show_status(msg)
        show_info(msg)

def list_keyboard_shortcuts():
  from .gui import main_window as m
  if m.showing_text() and m.text_id == 'keyboard shortcuts':
    m.show_graphics()
  else:
    m.show_text(shortcut_descriptions(html = True), html = True, id = "keyboard shortcuts")

def shortcut_descriptions(html = False):
  global keyboard_shortcuts
  if keyboard_shortcuts is None:
    return 'No keyboard shortcuts registered'
  ks = keyboard_shortcuts
  ksc = {}
  for k, (f,d,c) in ks.shortcuts.items():
    ksc.setdefault(c,[]).append((k,d))
  cats = list(ksc.keys())
  cats.sort()
  for cat in cats:
    ksc[cat].sort(key = lambda a: str.lower(a[0]))
  if html:
    cols = ks.category_columns
    lines = ['<html>', '<body>']
    # Multi-column table
    lines.extend(['<table style="background-color:lightgray;">'
                  '<tr><th align=left colspan=%d><h1>Keyboard Shortcuts</h1>' % len(cols),
                  '<tr>']),
    for colcats in cols:
      lines.append('<td>')
      for cat in colcats:
        lines.extend(['<table>', '<tr><th colspan=2 align=left>%s' % cat])
        lines.extend(['<tr><td width=40>%s<td>%s' % (k,d) for k,d in ksc[cat]])
        lines.append('</table>')
    lines.append('</table>') # Multi-column table
  else:
    lines = ['Keyboard shortcuts']
    for cat in cats:
      lines.extend(['', cat])
      lines.extend(['%s - %s' % (k,d) for k,d in ksc[cat]])
  descrip = '\n'.join(lines)
  return descrip

def show_manual():
  from .gui import main_window as m
  if m.showing_text() and m.text_id == 'manual':
    m.show_graphics()
    m.show_back_forward_buttons(False)
  else:
    from os.path import join, dirname
    path = join(dirname(dirname(__file__)), 'docs', 'index.html')
#    f = open(path, 'r')
#    text = f.read()
#    f.close()
#    m.show_text(text, html = True, id = "manual", open_links = True)
    url = 'file:/%s' % path
    m.show_text(open_links = True, id = 'manual')
    from .qt import QtCore
    m.text.setSource(QtCore.QUrl(url))

def show_command_history():
    from . import commands
    commands.show_command_history()

def show_stats():
    from .gui import main_window as mw, show_status
    v = mw.view
    na = v.atoms_shown
    r = 1.0/v.last_draw_duration
    n = len(v.models)
    show_status('%d models, %d atoms, %.1f frames/sec' % (n, na, r))

def matrix_profile():
    from ..geometry.place import identity
    m = identity()
    import numpy
    n = 10000
    import time
    t0 = time.clock()
    mi = [m.inverse() for i in range(n)]
    t1 = time.clock()
    print('%.0f matrix inverse per second' % (n / (t1-t0),))
    t0 = time.clock()
#    from ..geometry.matrix import multiply_matrices_numpy
#    mi = [multiply_matrices_numpy(mn,mn) for i in range(n)]
    mi = [m*m for i in range(n)]
    t1 = time.clock()
    print('%.0f matrix multiplies per second' % (n / (t1-t0),))

def leap_chopsticks_mode(viewer):
    from . import c2leap
    c2leap.leap_mode('chopsticks', viewer)

def leap_position_mode(viewer):
    from . import c2leap
    c2leap.leap_mode('position', viewer)

def leap_velocity_mode(viewer):
    from . import c2leap
    c2leap.leap_mode('velocity', viewer)

def leap_focus(viewer):
    from . import c2leap
    c2leap.report_leap_focus(viewer)

def leap_quit(viewer):
    from . import c2leap
    c2leap.quit_leap(viewer)
