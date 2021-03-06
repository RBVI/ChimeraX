def register_shortcuts(keyboard_shortcuts):
    '''Register the standard keyboard shortcuts.'''

    ks = keyboard_shortcuts
    scuts, catcols = standard_shortcuts(ks.session)
    for sc in scuts:
        ks.add_shortcut(sc)

    ks.category_columns = catcols

    return ks

def standard_shortcuts(session):

    # Shortcut documentation categories
    mapcat = 'Map Display'
    molcat = 'Molecule Display'
    surfcat = 'Surfaces'
    gcat = 'General Controls'
    ocat = 'Open, Save, Close'
    catcols = ((ocat,mapcat,surfcat), (molcat,), (gcat,))

    maparg = {'each_map':True}
    molarg = {'each_molecule':True}
    atomsarg = {'atoms_arg': True}
    surfarg = {'each_surface':True}
    viewarg = {'view_arg':True}
    mmarg = {'mouse_modes_arg':True}
    noarg = {}
    sesarg = {'session_arg':True}

    fmenu = 'File'
    smenu = 'Scene'
    mmenu = 'Map'
    mlmenu = 'Molecule'
    sfmenu = 'Surface'
    pmenu = 'Panes'
    msmenu = 'Mouse'
    dmenu = 'Device'
    hmenu = 'Help'
    s = session

    sep = True  # Add menu separator after entry.
    from ..files import opensave
    from .. import ui
    shortcuts = [
        # Sessions
        ('op', ui.show_open_file_dialog, 'Open file', ocat, sesarg, fmenu),
        ('rf', show_file_history, 'Show recent files', ocat, sesarg, fmenu),
        ('Sv', opensave.save_session, 'Save session', ocat, sesarg, fmenu),
        ('sv', ui.save_session_dialog, 'Save session as...', ocat, sesarg, fmenu),
        ('si', lambda s: opensave.save_image(None,s), 'Save image', ocat, sesarg, fmenu),
        ('Ca', close_all_models, 'Close all models', ocat, sesarg, fmenu),
        ('Qt', quit, 'Quit', ocat, sesarg, fmenu),

        # Scene
        ('va', view_all, 'View all', gcat, viewarg, smenu),
        ('dv', default_view, 'Default orientation', gcat, viewarg, smenu),
        ('Sp', save_position, 'Save position, restore it with pp', gcat, sesarg, smenu),
        ('pp', restore_position, 'Restore previous position saved with Sp', gcat, sesarg, smenu, sep),

        ('dA', display_all_positions, 'Display all copies', gcat, sesarg, smenu),
        ('dm', display_selected_models, 'Display selected models', ocat, sesarg, smenu),
        ('hm', hide_selected_models, 'Hide selected models', ocat, sesarg, smenu),
        ('Ds', delete_selected_models, 'Delete selected models', ocat, sesarg, smenu, sep),
        ('cs', s.clear_selection, 'Clear selection', gcat, noarg, smenu),

        ('bk', set_background_black, 'Black background', gcat, viewarg, smenu),
        ('wb', set_background_white, 'White background', gcat, viewarg, smenu),
        ('gb', set_background_gray, 'Gray background', gcat, viewarg, smenu, sep),

        ('dq', depth_cue, 'Toggle depth cue', gcat, viewarg, smenu),
        ('bl', motion_blur, 'Toggle motion blur', gcat, viewarg, smenu, sep),

        ('sh', toggle_shadows, 'Toggle shadows', gcat, viewarg, smenu),
        ('se', toggle_silhouettes, 'Toggle silhouette edges', gcat, viewarg, smenu),

        ('Mo', mono_mode, 'Set mono camera mode', gcat, viewarg, smenu),
        ('So', stereo_mode, 'Set sequential stereo mode', gcat, viewarg, smenu, sep),

        ('uh', undisplay_half, 'Undisplay z > 0', gcat, sesarg, smenu),
        ('rt', show_framerate, 'Show framerate', gcat, sesarg, smenu),
        ('nt', show_triangle_count, 'Show scene triangle count', gcat, sesarg, smenu),

        # Maps
        ('ft', fit_molecule_in_map, 'Fit molecule in map', mapcat, sesarg, mmenu),
        ('fr', show_map_full_resolution, 'Show map at full resolution', mapcat, maparg, mmenu),
        ('ob', toggle_outline_box, 'Toggle outline box', mapcat, maparg, mmenu, sep),

        ('sf', show_surface, 'Show surface', mapcat, maparg, mmenu),
        ('me', show_mesh, 'Show mesh', mapcat, maparg, mmenu),
        ('gs', show_grayscale, 'Show grayscale', mapcat, maparg, mmenu, sep),

        ('pl', show_one_plane, 'Show one plane', mapcat, maparg, mmenu),
        ('pa', show_all_planes, 'Show all planes', mapcat, maparg, mmenu),
        ('o3', toggle_orthoplanes, 'Show 3 orthogonal planes', mapcat, maparg, mmenu),
        ('bx', toggle_box_faces, 'Show box faces', mapcat, maparg, mmenu),

        # Molecules
        ('da', show_atoms, 'Display atoms', molcat, atomsarg, mlmenu),
        ('ha', hide_atoms, 'Undisplay atoms', molcat, atomsarg, mlmenu, sep),

        ('bs', show_ball_and_stick, 'Display atoms in ball and stick', molcat, atomsarg, mlmenu),
        ('sp', show_sphere, 'Display atoms in sphere style', molcat, atomsarg, mlmenu),
        ('st', show_stick, 'Display atoms in stick style', molcat, atomsarg, mlmenu, sep),

        ('rb', show_ribbon, 'Display ribbon', molcat, atomsarg, mlmenu),
        ('hr', hide_ribbon, 'Undisplay ribbon', molcat, atomsarg, mlmenu),
        ('r+', fatter_ribbons, 'Thicker ribbons', molcat, molarg, mlmenu),
        ('r-', thinner_ribbons, 'Thinner ribbons', molcat, molarg, mlmenu, sep),

        ('la', show_ligands, 'Show ligand atoms', molcat, molarg, mlmenu),
        ('sw', show_waters, 'Show water atoms', molcat, molarg, mlmenu),
        ('hw', hide_waters, 'Hide water atoms', molcat, molarg, mlmenu, sep),

        ('c1', color_one_color, 'Color molecule one color', molcat, molarg, mlmenu),
        ('ce', color_by_element, 'Color atoms by element', molcat, molarg, mlmenu),
        ('cc', color_by_chain, 'Color chains', molcat, molarg, mlmenu, sep),
        ('ao', ambient_occlusion_coloring, 'Ambient occlusion', molcat, sesarg, mlmenu, sep),

        ('ms', lambda m,s=s: show_molecular_surface(m,s), 'Show molecular surface', molcat, molarg, mlmenu),
        ('sa', lambda m,s=s: accessible_surface_area(m,s), 'Compute solvent accesible surface area', molcat, molarg, mlmenu, sep),

        ('bu', lambda m,s=s: show_biological_unit(m,s), 'Show biological unit', molcat, molarg, mlmenu),
        ('au', lambda m,s=s: show_asymmetric_unit(m,s), 'Show asymmetric unit', molcat, molarg, mlmenu),

        ('mb', lambda m,s=s: molecule_bonds(m,s), 'Compute molecule bonds using templates', molcat, molarg),

        # Surfaces
        ('ds', display_surface, 'Display surface', surfcat, sesarg, sfmenu),
        ('hs', hide_surface, 'Hide surface', surfcat, sesarg, sfmenu),
        ('tt', toggle_surface_transparency, 'Toggle surface transparency', surfcat, sesarg, sfmenu),
        ('t5', show_surface_transparent, 'Make surface transparent', surfcat, sesarg, sfmenu),
        ('t0', show_surface_opaque, 'Make surface opaque', surfcat, sesarg, sfmenu),

        # Pane
        ('mp', ui.show_model_panel, 'Show model panel', ocat, sesarg, pmenu),
        ('lg', show_log, 'Show command log', gcat, sesarg, pmenu),
        ('gr', show_graphics_window, 'Show graphics window', gcat, sesarg, pmenu),
        ('sc', show_scenes, 'Show scene thumbnails', gcat, sesarg, pmenu),
        ('ch', show_command_history, 'Show command history', gcat, sesarg, pmenu),
        ('cl', command_line, 'Enter command', gcat, sesarg),

        # Mouse
        ('mv', enable_move_mouse_mode, 'Movement mouse mode', gcat, mmarg, msmenu),
        ('mo', enable_move_selected_mouse_mode, 'Move selected mouse mode', gcat, mmarg, msmenu),
        ('Mp', enable_move_planes_mouse_mode, 'Move planes mouse mode', mapcat, mmarg, msmenu),
        ('ct', enable_contour_mouse_mode, 'Adjust contour level mouse mode', mapcat, mmarg, msmenu),
        ('vs', enable_map_series_mouse_mode, 'Map series mouse mode', mapcat, mmarg, msmenu),
        ('sl', selection_mouse_mode, 'Select models mouse mode', gcat, sesarg),

        # Devices
        ('sn', toggle_space_navigator, 'Toggle use of space navigator', gcat, sesarg, dmenu),
        ('nf', toggle_space_navigator_fly_mode, 'Toggle space navigator fly mode', gcat, sesarg, dmenu, sep),
        ('nc', space_navigator_collisions, 'Toggle space navigator collision avoidance', gcat, sesarg),

        ('oc', start_oculus, 'Start Oculus Rift stereo', gcat, sesarg, dmenu),
        ('om', oculus_move, 'Move Oculus window to primary display', gcat, sesarg, dmenu),

        ('lp', toggle_leap, 'Toggle leap motion input device', gcat, sesarg, dmenu),
        ('lP', leap_position_mode, 'Enable leap motion position mode', gcat, sesarg),
        ('lx', leap_chopsticks_mode, 'Enable leap motion chopstick mode', gcat, sesarg),
        ('lv', leap_velocity_mode, 'Enable leap motion velocity mode', gcat, sesarg),
        ('lf', leap_focus, 'Check if app has leap focus', gcat, sesarg),
        ('lq', leap_quit, 'Quit using leap motion input device', gcat, sesarg),

        # Help
        ('mn', show_manual, 'Show manual', gcat, sesarg, hmenu),
        ('ks', list_keyboard_shortcuts, 'List keyboard shortcuts', gcat, sesarg, hmenu),
        ]

    from ..molecule.blastpdb import blast_shortcuts
    shortcuts.extend(blast_shortcuts())

    scuts = []
    for sc in shortcuts:
        k,f,d,cat,argskw = sc[:5]
        menu = sc[5] if len(sc) >= 6 else None
        sep = sc[6] if len(sc) >= 7 else False
        sc = Shortcut(k, f, s, d, category = cat, menu = menu, menu_separator = sep, **argskw)
        scuts.append(sc)

    return scuts, catcols

class Shortcut:

    def __init__(self, key_seq, func, session, description = '', key_name = None, category = None,
                 menu = None, menu_separator = False, each_map = False, each_molecule = False,
                 each_surface = False, atoms_arg = False, view_arg = False, mouse_modes_arg = False,
                 session_arg = False):
        '''
        A keyboard shortcut is a key sequence and function to call when
        that key sequence is entered.  Shortcuts are put in categories and have
        textual descriptions for automatically creating documentation.  A shortcut
        function can take no arguments or it can take a map, molecule, surface or
        view argument.
        '''
        self.key_seq = key_seq
        self.key_name = key_seq if key_name is None else key_name
        self.func = func
        self.description = description
        self.category = category
        self.menu = menu
        self.menu_separator = menu_separator

        self.each_map = each_map
        self.each_molecule = each_molecule
        self.each_surface = each_surface
        self.atoms_arg = atoms_arg
        self.view_arg = view_arg
        self.mouse_modes_arg = mouse_modes_arg
        self.session_arg = session_arg
        
    def run(self, session):
        f = self.func
        s = session
        if self.atoms_arg:
            f(shortcut_atoms(s))
        elif self.each_map:
            for m in shortcut_maps(s):
                f(m)
        elif self.each_molecule:
            for m in shortcut_molecules(s):
                f(m)
        elif self.each_surface:
            for m in shortcut_surfaces(s):
                f(m)
        elif self.view_arg:
            v = s.view
            f(v)
        elif self.mouse_modes_arg:
            m = s.main_window.graphics_window.mouse_modes
            f(m)
        elif self.session_arg:
            f(s)
        else:
            f()
    
class Keyboard_Shortcuts:
  '''
  Maintain a list of multi-key keyboard shortcuts and run them in response to key presses.
  '''
  def __init__(self, session):

    # Keyboard shortcuts
    self.shortcuts = {}
    self.keys = ''
    self.session = session

  def add_shortcut(self, sc):
    self.shortcuts[sc.key_seq] = sc

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
      sc = self.shortcuts.get(keys)
      if sc is None:
        return
      msg = '%s - %s' % (sc.key_name, sc.description)
      s = self.session
      s.show_status(msg)
      s.show_info(msg, color = '#808000')
      sc.run(s)

def shortcut_maps(session):
    maps = set(session.maps())
    sel = session.selected_models()
    mlist = [m for m in sel if m in maps]
    if len(mlist) == 0:
        mlist = [m for m in maps if m.display]
    return mlist

def shortcut_molecules(session):
    mols = session.molecules()
    sel = session.selected_models()
    mlist = [m for m in sel if m in mols]
    if len(mlist) == 0:
        mlist = [m for m in mols if m.display]
    return mlist

def shortcut_atoms(session):
    a = session.selected_atoms()
    if a.count() == 0 and not session.selected_models():
        a = session.all_atoms()
    return a

def shortcut_selection(session):
  from .parse import Selection
  sel = Selection()
  sel.add_atoms(shortcut_atoms(session))
  sel.add_models(shortcut_surfaces(session))
  sel.add_models(shortcut_maps(session))
  return sel

def shortcut_surfaces(session):
    surfs = session.surfaces()
    sel = session.selected_models()
    mlist = [m for m in sel if m in surfs]
    if len(mlist) == 0:
        mlist = [m for m in surfs if m.display]
    return mlist

def shortcut_surfaces_and_maps(session):
    som = set(session.surfaces())
    som.update(session.maps())
    sel = session.selected_models()
    mlist = [m for m in sel if m in som]
    if len(mlist) == 0:
        mlist = [m for m in som if m.display]
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

def enable_move_planes_mouse_mode(mouse_modes, button = 'right'):
    m = mouse_modes
    from ..map import PlanesMouseMode
    m.bind_mouse_mode(button, PlanesMouseMode(m.session))

def enable_contour_mouse_mode(mouse_modes, button = 'right'):
    m = mouse_modes
    from ..map import ContourLevelMouseMode
    m.bind_mouse_mode(button, ContourLevelMouseMode(m.session))

def enable_map_series_mouse_mode(mouse_modes, button = 'right'):
    m = mouse_modes
    from ..map import series
    mode = series.map_series_mouse_mode(m.session)
    if mode:
        m.bind_mouse_mode(button, mode)

def enable_move_selected_mouse_mode(mouse_modes, button = 'right'):
    from .. import  ui
    m = mouse_modes
    m.bind_mouse_mode('left', ui.RotateSelectedMouseMode(m.session))
    tm = ui.TranslateSelectedMouseMode(m.session)
    m.bind_mouse_mode('middle', tm)
    m.bind_mouse_mode('right', tm)

def enable_move_mouse_mode(mouse_modes, button = 'right'):
    from .. import  ui
    m = mouse_modes
    m.bind_mouse_mode('left', ui.RotateMouseMode(m.session))
    tm = ui.TranslateMouseMode(m.session)
    m.bind_mouse_mode('middle', tm)
    m.bind_mouse_mode('right', tm)

def fit_molecule_in_map(session):
    mols, maps = session.molecules(), session.maps()
    if len(mols) != 1 or len(maps) != 1:
        print('ft: Fit molecule in map requires exactly one open molecule and one open map.')
        return

    mol, map = mols[0], maps[0]
    points = mol.xyz
    point_weights = None        # Equal weight for each atom
    data_array = map.full_matrix()
    xyz_to_ijk_transform = map.data.xyz_to_ijk_transform * map.position.inverse() * mol.position
    from ..map import fit
    move_tf, stats = fit.locate_maximum(points, point_weights, data_array, xyz_to_ijk_transform)
    mol.position = mol.position * move_tf
    for k,v in stats.items():
        print(k,v)

def show_biological_unit(m, session):

    if hasattr(m, 'pdb_text'):
        from ..molecule import biomt
        places = biomt.pdb_biomt_matrices(m.pdb_text)
        print (m.path, 'biomt', len(places))
        if places:
            m.positions = places

def show_asymmetric_unit(m, session):

    if len(m.positions) > 1:
        from ..geometry.place import Places
        m.positions = Places([m.positions[0]])

def display_surface(session):
    for m in shortcut_surfaces(session):
        sp = m.selected_positions
        if sp is None or sp.sum() == len(sp):
            m.display = True
        else:
            dp = m.display_positions
            if dp is None:
                m.display_positions = sp
            else:
                from numpy import logical_or
                m.display_positions = logical_or(dp,sp)

def hide_surface(session):
    for m in shortcut_surfaces(session):
        sp = m.selected_positions
        if sp is None or sp.sum() == len(sp):
            m.display = False
        else:
            dp = m.display_positions
            from numpy import logical_and, logical_not
            if dp is None:
                m.display_positions = logical_not(sp)
            else:
                m.display_positions = logical_and(dp,logical_not(sp))

def toggle_surface_transparency(session):
    from ..map import Volume
    from ..graphics import Drawing
    for m in shortcut_surfaces_and_maps(session):
        if isinstance(m, Volume):
            m.surface_colors = tuple((r,g,b,(0.5 if a == 1 else 1)) for r,g,b,a in m.surface_colors)
            m.show()
        elif isinstance(m, Drawing):
            for d in m.all_drawings():
                c = d.colors
                opaque = (c[:,3] == 255)
                c[:,3] = 255                # Make transparent opaque
                c[opaque,3] = 128           # and opaque transparent.
                d.colors = c.copy()         # TODO: Need copy or opengl color buffer does not update.

def show_surface_transparent(session, alpha = 0.5):
    from ..map import Volume
    from ..graphics import Drawing
    for m in shortcut_surfaces_and_maps(session):
        if isinstance(m, Volume):
            m.surface_colors = tuple((r,g,b,alpha) for r,g,b,a in m.surface_colors)
            m.show()
        elif isinstance(m, Drawing):
            for d in m.all_drawings():
                c = d.colors
                c[:,3] = int(255*alpha)
                d.colors = c.copy()         # TODO: Need copy or opengl color buffer does not update.

def show_surface_opaque(session):
    show_surface_transparent(session, alpha = 1)

def set_background_color(viewer, color):
    viewer.background_color = color
def set_background_black(viewer):
    set_background_color(viewer, (0,0,0,1))
def set_background_gray(viewer):
    set_background_color(viewer, (0.5,0.5,0.5,1))
def set_background_white(viewer):
    set_background_color(viewer, (1,1,1,1))

def toggle_shadows(viewer):
    viewer.shadows = not viewer.shadows

def toggle_silhouettes(viewer):
    viewer.silhouettes = not viewer.silhouettes
    viewer.redraw_needed = True

def depth_cue(viewer):
    viewer.enable_depth_cue(not viewer.depth_cue_enabled())
    
def selection_mouse_mode(session):
    mm = session.main_window.graphcs_window.mouse_modes
    mm.mouse_modes.bind_mouse_mode('right', mm.mouse_select)

def command_line(session):
    session.main_window.enable_shortcuts(False)

def display_selected_models(session):
  session.display_models(session.selected_models())

def hide_selected_models(session):
  session.hide_models(session.selected_models())

def delete_selected_models(session):
  session.close_models(session.selected_models())

def show_map_full_resolution(m):
  m.new_region(ijk_step = (1,1,1), adjust_step = False)

def show_molecular_surface(m, session, res = 3.0, grid = 0.5):
  if hasattr(m, 'molsurf') and m.molsurf in session.model_list():
    m.molsurf.display = True
  else:
    from ..molecule.surface import surface
    m.molsurf = surface(m.atoms(), session)

def color_by_element(m):
  m.color_by_element()
def color_by_chain(m):
  m.color_by_chain()
def color_one_color(m):
  m.single_color()
def ambient_occlusion_coloring(session):
  from ..molecule import ambient
  ambient.ambient_occlusion_coloring(shortcut_selection(session))

def show_atoms(a):
  a.show_atoms()
def hide_atoms(a):
  a.hide_atoms()
def show_sphere(atoms):
  atoms.set_atom_style(atoms.SPHERE_STYLE)
def show_stick(atoms):
  atoms.set_atom_style(atoms.STICK_STYLE)
def show_ball_and_stick(atoms):
  atoms.set_atom_style(atoms.BALL_STICK_STYLE)
def show_ribbon(a):
  a.show_ribbon()
def hide_ribbon(a):
  a.hide_ribbon()
def fatter_ribbons(m):
    m.set_ribbon_radius(2*m.ribbon_radius)
def thinner_ribbons(m):
    m.set_ribbon_radius(0.5*m.ribbon_radius)
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
    from .. import molecule
    a = molecule.accessible_surface_area(m)
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
  for k, sc in ks.shortcuts.items():
    ksc.setdefault(sc.category,[]).append((sc.key_name,sc.description))
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
    url = 'file:/%s' % path
    m.show_text(url = url, open_links = True, id = 'manual')

def show_file_history(session):
    session.file_history.show_thumbnails()

def show_command_history(session):
    session.commands.history.show_command_history()

def show_scenes(session):
    session.scenes.show_thumbnails(toggle = True)

def show_framerate(session):
    session.view.report_framerate()

def show_triangle_count(session):
    mols = session.molecules()
    na = sum(m.shown_atom_count() for m in mols) if mols else 0
    nt = sum(m.shown_atom_count() * m.triangles_per_sphere for m in mols) if mols else 0
    n = session.model_count()
    msg = '%d models, %d atoms, %d atom triangles' % (n, na, nt)
    session.show_status(msg)
    session.show_info(msg)

def default_view(view):
    view.initial_camera_view()

def view_all(view):
    view.view_all()

def toggle_leap(session):
    from ..devices import c2leap
    c2leap.toggle_leap(session)

def leap_chopsticks_mode(session):
    from ..devices import c2leap
    c2leap.leap_mode('chopsticks', session)

def leap_position_mode(session):
    from ..devices import c2leap
    c2leap.leap_mode('position', session)

def leap_velocity_mode(session):
    from ..devices import c2leap
    c2leap.leap_mode('velocity', session)

def leap_focus(session):
    from ..devices import c2leap
    c2leap.report_leap_focus(session)

def leap_quit(session):
    from ..devices import c2leap
    c2leap.quit_leap(session)

def motion_blur(viewer):
    from ..graphics import MotionBlur
    mb = [o for o in viewer.overlays() if isinstance(o, MotionBlur)]
    if mb:
        viewer.remove_overlays(mb)
    else:
        MotionBlur(viewer)

def mono_mode(viewer):
    from .. import graphics
    viewer.camera.mode = graphics.mono_camera_mode
def stereo_mode(viewer):
    from .. import graphics
    viewer.camera.mode = graphics.stereo_camera_mode
def start_oculus(session):
    from ..devices import oculus
    if session.view.camera.mode.name() == 'oculus':
        oculus.stop_oculus(session)
    else:
        oculus.start_oculus(session)
def oculus_move(session):
    oc = session.oculus
    if oc:
        if hasattr(oc, 'on_primary') and oc.on_primary:
            oc.oculus_full_screen()
            oc.on_primary = False
        else:
            oc.window.move_window_to_primary_screen()
            oc.on_primary = True

def toggle_space_navigator(session):
    from ..devices import spacenavigator
    spacenavigator.toggle_space_navigator(session)

def toggle_space_navigator_fly_mode(session):
    from ..devices import spacenavigator
    spacenavigator.toggle_fly_mode(session)

def space_navigator_collisions(session):
    from ..devices import spacenavigator
    spacenavigator.avoid_collisions(session)

def quit(session):
    import sys
    sys.exit(0)

def undisplay_half(session):
    for m in session.models_list():
        undisplay_half_model(m)

def undisplay_half_model(m):
    if not m.empty_drawing():
        mp = m.positions
        va = m.vertices
        c = 0.5*(va.min(axis=0) + va.max(axis=0))
        if len(mp) == 1:
            if (mp[0]*c)[2] > 0:
                m.display = False
        else:
            from numpy import array, bool
            pmask = array([(pl*c)[2] <= 0 for pl in mp], bool)
            m.display_positions = pmask
            print('uh', m.name, pmask.sum())
    for c in m.child_drawings():
        undisplay_half_model(c)

def display_all_positions(session):
    for m in session.model_list():
        for c in m.all_drawings():
            if c.display:
                c.display_positions = None

def save_position(session):
    c = session.view.camera
    session._saved_camera_view = c.position

def restore_position(session):
    if hasattr(session, '_saved_camera_view'):
        c = session.view.camera
        c.position = session._saved_camera_view
