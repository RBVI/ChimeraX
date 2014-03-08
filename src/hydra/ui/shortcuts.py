def register_shortcuts(keyboard_shortcuts):
    '''Register the standard keyboard shortcuts.'''

    ks = keyboard_shortcuts
    s = ks.session
    v = s.main_window.view

    map_shortcuts = (
      ('me', show_mesh, 'Show mesh'),
      ('sf', show_surface, 'Show surface'),
      ('gs', show_grayscale, 'Show grayscale'),
      ('ob', toggle_outline_box, 'Toggle outline box'),
      ('pl', show_one_plane, 'Show one plane'),
      ('pa', show_all_planes, 'Show all planes'),
      ('o3', toggle_orthoplanes, 'Show 3 orthogonal planes'),
      ('bx', toggle_box_faces, 'Show box faces'),
      ('fr', show_map_full_resolution, 'Show map at full resolution'),
      )
    mapcat = 'Map Display'
    for k,f,d in map_shortcuts:
      ks.add_shortcut(k, f, d, category = mapcat, each_map = True)

    mol_shortcuts = (
        ('bu', lambda m,s=s: show_biological_unit(m,s), 'Show biological unit'),
        ('as', lambda m,s=s: show_asymmetric_unit(m,s), 'Show asymmetric unit'),
        ('c1', color_one_color, 'Color molecule one color'),
        ('ce', color_by_element, 'Color atoms by element'),
        ('cc', color_by_chain, 'Color chains'),
        ('ms', lambda m,s=s: show_molecular_surface(m,s), 'Show molecular surface'),
        ('mb', lambda m,s=s: molecule_bonds(m,s), 'Compute molecule bonds using templates'),
        ('da', show_atoms, 'Display molecule atoms'),
        ('ha', hide_atoms, 'Undisplay molecule atoms'),
        ('bs', show_ball_and_stick, 'Display atoms in ball and stick'),
        ('sp', show_sphere, 'Display atoms in sphere style'),
        ('st', show_stick, 'Display atoms in stick style'),
        ('rb', show_ribbon, 'Show molecule ribbon'),
        ('hr', hide_ribbon, 'Undisplay molecule ribbon'),
        ('la', show_ligands, 'Show ligand atoms'),
        ('sw', show_waters, 'Show water atoms'),
        ('hw', hide_waters, 'Hide water atoms'),
        ('r+', fat_ribbons, 'Fat ribbons'),
        ('r-', thin_ribbons, 'Thin ribbons'),
        ('sa', lambda m,s=s: accessible_surface_area(m,s), 'Compute solvent accesible surface area'),
    )
    molcat = 'Molecule Display'
    for k,f,d in mol_shortcuts:
      ks.add_shortcut(k, f, d, category = molcat, each_molecule = True)

    surf_shortcuts = (
        ('t5', show_surface_transparent, 'Make surface transparent'),
    )
    for k,f,d in surf_shortcuts:
      ks.add_shortcut(k, f, d, category = mapcat, each_surface = True)

    ocat = 'Open, Save, Close'   # shortcut documentation category
    gcat = 'General Controls'
    view_shortcuts = (
        ('Mp', enable_move_planes_mouse_mode, 'Move planes mouse mode', mapcat),
        ('ct', enable_contour_mouse_mode, 'Adjust contour level mouse mode', mapcat),
        ('mo', enable_move_selected_mouse_mode, 'Move selected mouse mode', gcat),
        ('bk', set_background_black, 'Black background', gcat),
        ('wb', set_background_white, 'White background', gcat),
        ('gb', set_background_gray, 'Gray background', gcat),
        ('dq', depth_cue, 'Toggle depth cue', gcat),
        ('bl', motion_blur, 'Toggle motion blur', gcat),
        ('Mo', mono_mode, 'Set mono camera mode', gcat),
        ('So', stereo_mode, 'Set sequential stereo mode', gcat),
        ('Oc', oculus_mode, 'Set Oculus Rift stereo mode', gcat),
    )
    for k,f,d,cat in view_shortcuts:
      ks.add_shortcut(k, f, d, category = cat, view_arg = True)

    misc_shortcuts = (
        ('dv', v.initial_camera_view, 'Default view', gcat),
        ('va', v.view_all, 'View all', gcat),
        ('cs', s.clear_selection, 'Clear selection', gcat),
        )
    for k,f,d,cat in misc_shortcuts:
      ks.add_shortcut(k, f, d, category = cat)

    from ..file_io import opensave
    from .modelpanel import show_model_panel
    session_shortcuts = (
        ('op', opensave.show_open_file_dialog, 'Open file', ocat),
        ('sv', opensave.save_session_as, 'Save session as...', ocat),
        ('Sv', opensave.save_session, 'Save session', ocat),
        ('si', lambda s: opensave.save_image(None,s), 'Save image', ocat),
        ('oi', opensave.open_image, 'Open image', ocat),
        ('Ca', close_all_models, 'Close all models', ocat),
        ('mp', show_model_panel, 'Show/hide model panel', ocat),
        ('Ds', delete_selected_models, 'Delete selected models', ocat),
        ('ks', list_keyboard_shortcuts, 'List keyboard shortcuts', gcat),
        ('rs', show_file_history, 'Show recent sessions', ocat),
        ('gr', show_graphics_window, 'Show graphics window', gcat),
        ('mn', show_manual, 'Show manual', gcat),
        ('ch', show_command_history, 'Show command history', gcat),
        ('sc', show_scenes, 'Show scene thumbnails', gcat),
        ('rt', show_stats, 'Show model statistics', gcat),
        ('lg', show_log, 'Show command log', gcat),
        ('sl', selection_mouse_mode, 'Select models mouse mode', gcat),
        ('ft', fit_molecule_in_map, 'Fit molecule in map', mapcat),
        ('cl', command_line, 'Enter command', gcat),
        ('sn', toggle_space_navigator, 'Toggle use of space navigator', gcat),
        ('nf', toggle_space_navigator_fly_mode, 'Toggle space navigator fly mode', gcat),
        ('nc', space_navigator_collisions, 'Toggle space navigator collision avoidance', gcat),
        ('oc', start_oculus, 'Start Oculus Rift stereo', gcat),
        ('ow', oculus_warp, 'Toggle Oculus Rift lens correction', gcat),
        ('lp', leap_position_mode, 'Enable leap motion input device', gcat),
        ('lx', leap_chopsticks_mode, 'Enable leap motion chopstick mode', gcat),
        ('lv', leap_velocity_mode, 'Enable leap motion velocity mode', gcat),
        ('lf', leap_focus, 'Check if app has leap focus', gcat),
        ('lq', leap_quit, 'Quit using leap motion input device', gcat),
        ('Qt', quit, 'Quit', ocat),
        )
    for k,f,d,cat in session_shortcuts:
        ks.add_shortcut(k, f, d, category = cat, session_arg = True)

    ks.category_columns = ((ocat,mapcat), (molcat,), (gcat,))

    return ks

class Keyboard_Shortcuts:
  '''
  Maintain a list of multi-key keyboard shortcuts and run them in response to key presses.
  '''
  def __init__(self, session):

    # Keyboard shortcuts
    self.shortcuts = {}
    self.keys = ''
    self.session = session

  def add_shortcut(self, key_seq, func, description = '', key_name = None, category = None,
                   each_map = False, each_molecule = False,
                   each_surface = False, view_arg = False, session_arg = False):
    '''
    Add a keyboard shortcut with a given key sequence and function to call when
    that key sequence is entered.  Shortcuts are put in categories and have
    textual descriptions for automatically creating documentation.  A shortcut
    function can take no arguments or it can take a map, molecule, surface or
    view argument.
    '''

    s = self.session
    if each_map:
        def f(s=s, func=func):
            for m in shortcut_maps(s):
                func(m)
    elif each_molecule:
        def f(s=s, func=func):
            for m in shortcut_molecules(s):
                func(m)
    elif each_surface:
        def f(s=s, func=func):
            for m in shortcut_surfaces(s):
                func(m)
    elif view_arg:
        v = s.main_window.view
        def f(v=v, func=func):
            func(v)
    elif session_arg:
        def f(s=s, func=func):
            func(s)
    else:
        f = func
    kn = key_seq if key_name is None else key_name
    self.shortcuts[key_seq] = (f, description, kn, category)

  def key_pressed(self, event):

#    t = event.text()
#    print(event, t, type(t), len(t), str(t), t.encode(), '%x' % event.key(), '%x' % int(event.modifiers()), event.count())

    c = str(event.text())
    self.keys += c
    self.try_shortcut()

  def try_shortcut(self, keys = None):

    if not keys is None:
        self.keys = keys
    k = self.keys
    s = self.shortcuts
    if k in s:
      keys = self.keys
      self.keys = ''
      self.run_shortcut(keys)
      return True
    
    is_prefix = False
    for ks in s.keys():
        if ks.startswith(k):
            is_prefix = True
            break
    msg = self.keys if is_prefix else '%s - unknown shortcut' % self.keys
    if not is_prefix:
        self.keys = ''

    self.session.show_status(msg)
    return not is_prefix

  def run_shortcut(self, keys):
      fdnc = self.shortcuts.get(keys)
      if fdnc is None:
        return
      f,d,n,c = fdnc
      msg = '%s - %s' % (n, d)
      s = self.session
      s.show_status(msg)
      s.show_info(msg, color = '#808000')
      f()

def shortcut_maps(session):
  mlist = [m for m in session.maps() if m.selected]
  if len(mlist) == 0:
    mlist = [m for m in session.maps() if m.display]
  return mlist

def shortcut_molecules(session):
  mlist = [m for m in session.molecules() if m.selected]
  if len(mlist) == 0:
    mlist = [m for m in session.molecules() if m.display]
  return mlist

def shortcut_surfaces(session):
  mlist = [m for m in session.surfaces() if m.selected]
  if len(mlist) == 0:
    mlist = [m for m in session.surfaces() if m.display]
  return mlist

def close_all_models(session):
    session.close_all_models()
    session.scenes.delete_all_scenes()
    session.file_history.show_thumbnails()

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
  p = tuple(s//2 for s in m.data.size)
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

  from ..map.moveplanes import planes_mouse_mode as pmm
  viewer.mouse_modes.bind_mouse_mode(button,
                         lambda e,v=viewer: pmm.mouse_down(v,e),
                         lambda e,v=viewer: pmm.mouse_drag(v,e),
                         lambda e,v=viewer: pmm.mouse_up(v,e))

def enable_contour_mouse_mode(viewer, button = 'right'):
  m = viewer.mouse_modes
  m.bind_mouse_mode(button, m.mouse_down, m.mouse_contour_level, m.mouse_up)

def enable_move_selected_mouse_mode(viewer, button = 'right'):
  m = viewer.mouse_modes
  m.move_selected = not m.move_selected

def fit_molecule_in_map(session):
    mols, maps = session.molecules(), session.maps()
    if len(mols) != 1 or len(maps) != 1:
        print('ft: Fit molecule in map requires exactly one open molecule and one open map.')
        return

    mol, map = mols[0], maps[0]
    points = mol.xyz
    point_weights = None        # Equal weight for each atom
    data_array = map.full_matrix()
    xyz_to_ijk_transform = map.data.xyz_to_ijk_transform * map.place.inverse() * mol.place
    from ..map import fit
    move_tf, stats = fit.locate_maximum(points, point_weights, data_array, xyz_to_ijk_transform)
    mol.place = mol.place * move_tf
    for k,v in stats.items():
        print(k,v)

def show_biological_unit(m, session):

    if hasattr(m, 'pdb_text'):
        from ..file_io import biomt
        matrices = biomt.pdb_biomt_matrices(m.pdb_text)
        print (m.path, 'biomt', len(matrices))
        if matrices:
            m.copies = matrices
            m.redraw_needed = True
            m.update_level_of_detail(session.view)

def show_asymmetric_unit(m, session):

    if len(m.copies) > 0:
        m.copies = []
        m.redraw_needed = True
        m.update_level_of_detail(session.view)

def show_surface_transparent(m):
    from ..map import Volume
    from ..surface import Surface
    if isinstance(m, Volume):
        m.surface_colors = tuple((r,g,b,(0.5 if a == 1 else 1)) for r,g,b,a in m.surface_colors)
        m.show()
    elif isinstance(m, Surface):
        for p in m.surface_pieces():
            r,g,b,a = p.color
            p.color = (r,g,b, (0.5 if a == 1 else 1))

def set_background_color(viewer, color):
    viewer.background_color = color
def set_background_black(viewer):
    set_background_color(viewer, (0,0,0,1))
def set_background_gray(viewer):
    set_background_color(viewer, (0.5,0.5,0.5,1))
def set_background_white(viewer):
    set_background_color(viewer, (1,1,1,1))

def depth_cue(viewer):
    r = viewer.render
    c = r.default_capabilities
    if r.SHADER_DEPTH_CUE in c:
        c.remove(r.SHADER_DEPTH_CUE)
    else:
        c.add(r.SHADER_DEPTH_CUE)
    viewer.redraw_needed = True
    
def selection_mouse_mode(session):
    v = session.view
    v.mouse_modes.bind_mouse_mode('right', v.mouse_modes.mouse_select)

def command_line(session):
    session.main_window.focus_on_command_line()
#  from .qt import QtCore
#  QtCore.QTimer.singleShot(1000, main_window.focus_on_command_line)

def delete_selected_models(session):
  session.close_models(tuple(session.selected))

def show_map_full_resolution(m):
  m.new_region(ijk_step = (1,1,1), adjust_step = False)

def show_molecular_surface(m, session, res = 3.0, grid = 0.5):
  if hasattr(m, 'molsurf') and m.molsurf in session.model_list():
    m.molsurf.display = True
  else:
    from ..surface.gridsurf import surface
    m.molsurf = surface(m.atoms(), session)

def color_by_element(m):
  m.set_color_mode('by element')
def color_by_chain(m):
  m.set_color_mode('by chain')
def color_one_color(m):
  m.set_color_mode('single')

def show_atoms(m):
  m.show_all_atoms()
def hide_atoms(m):
  m.hide_all_atoms()
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
    m.show_ligand_atoms()
def show_waters(m):
    m.show_solvent()
def hide_waters(m):
    m.hide_solvent()
def molecule_bonds(m, session):
    if m.bonds is None:
        from ..molecule import connect
        m.bonds, missing = connect.molecule_bonds(m, session)
        msg = 'Created %d bonds for %s using templates' % (len(m.bonds), m.name)
        session.show_status(msg)
        session.show_info(msg)
        if missing:
            session.show_info('Missing %d templates: %s' % (len(missing), ', '.join(missing)))
def accessible_surface_area(m, session):
    from .. import surface
    a = surface.accessible_surface_area(m)
    msg = 'Accessible surface area of %s = %.5g' % (m.name, a.sum())
    session.show_status(msg)
    session.show_info(msg)

def list_keyboard_shortcuts(session):
    m = session.main_window
    if m.showing_text() and m.text_id == 'keyboard shortcuts':
        m.show_graphics()
    else:
        t = shortcut_descriptions(session.keyboard_shortcuts, html = True)
        m.show_text(t, html = True, id = "keyboard shortcuts")

def shortcut_descriptions(ks, html = False):
  ksc = {}
  for k, (f,d,n,c) in ks.shortcuts.items():
    ksc.setdefault(c,[]).append((n,d))
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
        lines.extend(['<tr><td width=40>%s<td>%s' % (n,d) for n,d in ksc[cat]])
        lines.append('</table>')
    lines.append('</table>') # Multi-column table
  else:
    lines = ['Keyboard shortcuts']
    for cat in cats:
      lines.extend(['', cat])
      lines.extend(['%s - %s' % (n,d) for n,d in ksc[cat]])
  descrip = '\n'.join(lines)
  return descrip

def show_graphics_window(session):
    m = session.main_window
    m.show_graphics()
    m.show_back_forward_buttons(False)

def show_log(session):
  session.log.show()

def show_manual(session):
  m = session.main_window
  if m.showing_text() and m.text_id == 'manual':
    m.show_graphics()
    m.show_back_forward_buttons(False)
  else:
    from os.path import join, dirname
    path = join(dirname(dirname(__file__)), 'docs', 'index.html')
# Can't use back button to initial page if html text provided instead of file url.
#    f = open(path, 'r')
#    text = f.read()
#    f.close()
#    m.show_text(text, html = True, id = "manual", open_links = True)
    url = 'file:/%s' % path
    m.show_text(open_links = True, id = 'manual')
    from .qt import QtCore
    m.text.setSource(QtCore.QUrl(url))

def show_file_history(session):
    session.file_history.show_thumbnails()

def show_command_history(session):
    session.commands.history.show_command_history()

def show_scenes(session):
    session.scenes.show_thumbnails(toggle = True)

def show_stats(session):
    v = session.view
    na = v.atoms_shown
    r = 1.0/v.last_draw_duration
    n = session.model_count()
    session.show_status('%d models, %d atoms, %.1f frames/sec' % (n, na, r))

def leap_chopsticks_mode(session):
    from . import c2leap
    c2leap.leap_mode('chopsticks', session)

def leap_position_mode(session):
    from . import c2leap
    c2leap.leap_mode('position', session)

def leap_velocity_mode(session):
    from . import c2leap
    c2leap.leap_mode('velocity', session)

def leap_focus(session):
    from . import c2leap
    c2leap.report_leap_focus(session)

def leap_quit(session):
    from . import c2leap
    c2leap.quit_leap(session)

def motion_blur(viewer):
    from .crossfade import Motion_Blur
    mb = [o for o in viewer.overlays if isinstance(o, Motion_Blur)]
    if mb:
        viewer.remove_overlays(mb)
    else:
        Motion_Blur(viewer)

def mono_mode(viewer):
    viewer.set_camera_mode('mono')
def stereo_mode(viewer):
    viewer.set_camera_mode('stereo')
def oculus_mode(viewer):
    viewer.set_camera_mode('oculus')
def start_oculus(session):
    from . import oculus
    if oculus.oculus_on(session):
        oculus.stop_oculus(session)
    else:
        oculus.start_oculus(session)
def oculus_warp(session):
    from . import oculus
    oculus.toggle_warping(session)

def toggle_space_navigator(session):
    from . import spacenavigator
    spacenavigator.toggle_space_navigator(session)

def toggle_space_navigator_fly_mode(session):
    from . import spacenavigator
    spacenavigator.toggle_fly_mode(session)

def space_navigator_collisions(session):
    from . import spacenavigator
    spacenavigator.avoid_collisions(session)

def quit(session):
    import sys
    sys.exit(0)
