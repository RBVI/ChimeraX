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
# User interface for sending scenes to Quest VR headset for viewing with
# Lookie app
#
from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class ToQuest(ToolInstance):
    SESSION_ENDURING = True

    # help = 'help:user/tools/markerplacement.html'

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys=False)
        self.tool_window = tw
        parent = tw.ui_area
        
        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))
        
        # Triangles settings
        tf = self._create_triangles_gui(parent)
        layout.addWidget(tf)
        
        # Send to Quest, Options, and Help buttons
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        # Options panel
        options = self._create_options_gui(parent)
        layout.addWidget(options)

        layout.addStretch(1)    # Extra space at end
        
        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, ToQuest, 'Send to Quest', create=create)
        
    # ---------------------------------------------------------------------------
    #
    def _create_triangles_gui(self, parent):
        from Qt.QtWidgets import QFrame
        f = QFrame(parent)
        from chimerax.ui.widgets import vertical_layout, EntriesRow
        layout = vertical_layout(f, margins = (5,0,0,0))
        tc = EntriesRow(f, '#', 'scene triangles',
                        ('Update', self._report_triangle_count),
                        '    ', True, 'Maximum', 900000)
        self._triangle_count = tcount = tc.labels[0]
        self._use_max_triangles, self._max_triangles = um, mt = tc.values
        mt.pixel_width = 70

        abt = EntriesRow(f, 'Simplify atoms', 100,
                         ('4', lambda: self._set_atom_triangles(4)),
                         ('10', lambda: self._set_atom_triangles(10)),
                         ('30', lambda: self._set_atom_triangles(30)),
                         ('100', lambda: self._set_atom_triangles(100)),
                         '   bonds', 60,
                         ('12', lambda: self._set_bond_triangles(12)),
                         ('20', lambda: self._set_bond_triangles(20)),
                         ('60', lambda: self._set_bond_triangles(20)))
        self._atom_triangles, self._bond_triangles = at,bt = abt.values
        at.pixel_width, bt.pixel_width = 40,20
        at.return_pressed.connect(lambda *unused: self._set_atom_triangles())
        bt.return_pressed.connect(lambda *unused: self._set_bond_triangles())
        abt.frame.setStyleSheet('QPushButton {padding: 5px}') # Smaller buttons
        
        rt = EntriesRow(f, '          ribbon sides', 12,
                        ('4', lambda: self._set_ribbon_sides(4)),
                        ('8', lambda: self._set_ribbon_sides(8)),
                        ('12', lambda: self._set_ribbon_sides(12)),
                        '   divisions', 20,
                        ('2', lambda: self._set_ribbon_divisions(2)),
                        ('5', lambda: self._set_ribbon_divisions(5)),
                        ('10', lambda: self._set_ribbon_divisions(10)),
                        ('20', lambda: self._set_ribbon_divisions(20)))
        self._ribbon_sides, self._ribbon_divisions = rs,rd = rt.values
        rs.pixel_width = rd.pixel_width = 20
        rt.frame.setStyleSheet('QPushButton {padding: 5px}') # Smaller buttons
        rs.return_pressed.connect(lambda *unused: self._set_ribbon_sides())
        rd.return_pressed.connect(lambda *unused: self._set_ribbon_divisions())
        
        self._report_triangle_count()
        
        return f
    
    # ---------------------------------------------------------------------------
    #
    def _set_atom_triangles(self, tri = None):
        if tri is None:
            tri = self._atom_triangles.value
        from chimerax.core.commands import run
        run(self.session, f'graphics quality atomTriangles {tri}')
        self._atom_triangles.value = tri
        self._report_triangle_count()
        
    # ---------------------------------------------------------------------------
    #
    def _set_bond_triangles(self, tri = None):
        if tri is None:
            tri = self._bond_triangles.value
        from chimerax.core.commands import run
        run(self.session, f'graphics quality bondTriangles {tri}')
        self._bond_triangles.value = tri
        self._report_triangle_count()
        
    # ---------------------------------------------------------------------------
    #
    def _set_ribbon_sides(self, sides = None):
        if sides is None:
            sides = self._ribbon_sides.value
        from chimerax.core.commands import run
        run(self.session, f'graphics quality ribbonSides {sides}')
        self._ribbon_sides.value = sides
        self._report_triangle_count()
        
    # ---------------------------------------------------------------------------
    #
    def _set_ribbon_divisions(self, div = None):
        if div is None:
            div = self._ribbon_divisions.value
        from chimerax.core.commands import run
        run(self.session, f'graphics quality ribbonDivisions {div}')
        self._ribbon_divisions.value = div
        self._report_triangle_count()
    
    # ---------------------------------------------------------------------------
    #
    def _report_quality(self):
        from chimerax.atomic import Structure
        structs = [m for m in self.session.models.list(type = Structure) if m.visible]
        atri = max([len(s._atoms_drawing.triangles) for s in structs if s._atoms_drawing],
                   default = None)
        btri = max([len(s._bonds_drawing.triangles) for s in structs if s._bonds_drawing],
                   default = None)

        STYLE_ROUND = 1
        rside = max([s.ribbon_xs_mgr.params[STYLE_ROUND]['sides'] for s in structs],
                    default = None)
        from chimerax.atomic import structure_graphics_updater
        gu = structure_graphics_updater(self.session)
        lod = gu.level_of_detail
        rdiv = max([lod.ribbon_divisions(s.num_residues) for s in structs],
                   default = None)
        if atri:
            self._atom_triangles.value = atri
        if btri:
            self._bond_triangles.value = btri
        if rside:
            self._ribbon_sides.value = rside
        if rdiv:
            self._ribbon_divisions.value = rdiv
    
    # ---------------------------------------------------------------------------
    #
    def _report_triangle_count(self):
        tcount = self._scene_triangles()
        tc = self._triangle_count
        tc.setText('%d' % tcount)
        tlimit = self._triangle_limit
        color = '' if tlimit is None or tcount <= tlimit else 'QLabel { color : red; }'
        tc.setStyleSheet(color);
        self._report_quality()
        return tcount
    
    # ---------------------------------------------------------------------------
    #
    def _scene_triangles(self):
        self.session.update_loop.update_graphics_now()  # Need to draw to get current ribbon
        models = [m for m in self.session.models.list() if len(m.id) == 1 and m.display]
        lines = []
        from chimerax.std_commands.graphics import _drawing_triangles
        tri = _drawing_triangles(models, lines)
        return tri

    # ---------------------------------------------------------------------------
    #
    @property
    def _triangle_limit(self):
        if self._use_max_triangles.enabled:
            try:
                return self._max_triangles.value
            except:
                return None
        return None

    # ---------------------------------------------------------------------------
    #
    def _too_many_triangles(self, warn = True):
        tlimit = self._triangle_limit
        tcount = self._report_triangle_count()
        too_many = tlimit is not None and tcount > tlimit
        if warn and too_many:
            self.session.logger.error('Too many triangles to send to Quest, '
                                      f'{tcount} > {tlimit}')
        return too_many
    
    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import EntriesRow
        r = EntriesRow(parent,
                       ('Send to Quest', self._send_to_quest),
                       'filename', '',
                       ('Options', self._show_or_hide_options),
                       ('Help', self._help))
        self._filename = fn = r.values[0]
        fn.value = '    scene'
        return r.frame
    
    # ---------------------------------------------------------------------------
    #
    @property
    def _scene_filename(self):
        fn = self._filename.value.strip()
        if not fn:
            fn = 'scene.glb'
        elif not fn.endswith('.glb'):
            fn += '.glb'
        return fn
    
    # ---------------------------------------------------------------------------
    #
    def _send_to_quest(self):
        if self._too_many_triangles():
            return
        
        # Save current scene.
        from os.path import expanduser, sep
        path = expanduser(f'~/Desktop/{self._scene_filename}').replace(sep, '/')
        from chimerax.core.commands import run
        run(self.session, f'save {path}')

        # Transfer scene file to Quest using adb
        adb = self._adb_path.value
        app = 'Lookie' if self._send_to_lookie.enabled else 'LookieAR'
        lookie_dir = f'/sdcard/Android/data/com.UCSF.{app}/files'
        cmd = f'"{adb}" push {path} {lookie_dir}'
        self.session.logger.info(f'Running command: {cmd}')

        # all output is on stderr, but Windows needs all standard I/O to
        # be redirected if one is, so stdout is a pipe too
        args = [adb, "push", path, lookie_dir]
        from subprocess import Popen, PIPE, DEVNULL
        p = Popen(args, stdin=DEVNULL, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        exit_code = p.returncode

        if exit_code != 0:
            output = '\n'.join(['stdout:', out.decode('utf-8'),
                                'stderr:', err.decode('utf-8')])
            self.session.logger.info(output)

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()
        
    # ---------------------------------------------------------------------------
    #
    def _create_options_gui(self, parent):
        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None)
        f = p.content_area

        from chimerax.ui.widgets import EntriesRow, radio_buttons

        # Results directory
        ac = EntriesRow(f, 'adb executable', '', ('Browse', self._choose_adb_path))
        self._adb_path = adb = ac.values[0]
        adb.pixel_width = 350
        from sys import platform
        adb_path = '/opt/homebrew/bin/adb' if platform == 'darwin' else adb
# Windows vive.cgl.ucsf.edu:        
#        adb_path = 'C:/Program Files/Unity/Hub/Editor/2022.2.5f1/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb.exe'
        adb.value = adb_path

        # Use PDB structure templates option for prediction
        ut = EntriesRow(f, 'Send to Quest application', True, 'Lookie', False, 'LookieAR')
        self._send_to_lookie, self._send_to_lookie_ar = ut.values
        radio_buttons(*ut.values)
        
        return p
        
    # ---------------------------------------------------------------------------
    #
    def _choose_adb_path(self):
        parent = self.tool_window.ui_area
        from Qt.QtWidgets import QFileDialog
        path, ftype  = QFileDialog.getOpenFileName(parent, caption = f'adb executable')
        if path:
            self._adb_path.value = path

    # ---------------------------------------------------------------------------
    #
    def _help(self):
      from os.path import dirname, join
      help_url = 'file://' + join(dirname(__file__), 'help.html')
      from chimerax.help_viewer import show_url
      show_url(self.session, help_url)
        
def to_quest_panel(session, create = True):
  return ToQuest.get_singleton(session, create)
