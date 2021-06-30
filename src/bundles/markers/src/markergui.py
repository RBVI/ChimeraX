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
# User interface for building cages.
#
from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class MarkerModeSettings(ToolInstance):
    SESSION_ENDURING = True

    help = 'help:user/tools/markerplacement.html'

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self.display_name = 'Marker Placement'

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys=False)
        self.tool_window = tw
        parent = tw.ui_area
        
        from Qt.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QSizePolicy, QCheckBox

        playout = QVBoxLayout(parent)
        playout.setContentsMargins(0,0,0,0)
        playout.setSpacing(0)
        parent.setLayout(playout)

        f = QFrame(parent)
        f.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        playout.addWidget(f)
        layout = QVBoxLayout(f)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        f.setLayout(layout)
        
        # Marker and link color and radius
        mf = QFrame(f)
        mf.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(mf)
        mm_layout = QHBoxLayout(mf)
        mm_layout.setContentsMargins(0,0,0,0)
        mm_layout.setSpacing(5)
        mf.setLayout(mm_layout)

        ml = QLabel('Marker color', mf)
        mm_layout.addWidget(ml)
        from chimerax.ui.widgets import ColorButton
        self._marker_color = mc = ColorButton(mf, max_size = (16,16))
        mc.color_changed.connect(self._marker_color_changed)
        mm_layout.addWidget(mc)
        rl = QLabel(' radius', mf)
        mm_layout.addWidget(rl)
        self._marker_radius = mr = QLineEdit('', mf)
        mr.setMaximumWidth(40)
        mr.returnPressed.connect(self._marker_radius_changed)
        mm_layout.addWidget(mr)

        mm_layout.addSpacing(20)
        ml = QLabel('Link color', mf)
        mm_layout.addWidget(ml)
        from chimerax.ui.widgets import ColorButton
        self._link_color = lc = ColorButton(mf, max_size = (16,16))
        lc.color_changed.connect(self._link_color_changed)
        mm_layout.addWidget(lc)
        rl = QLabel(' radius', mf)
        mm_layout.addWidget(rl)
        self._link_radius = lr = QLineEdit('', mf)
        lr.setMaximumWidth(40)
        lr.returnPressed.connect(self._link_radius_changed)
        mm_layout.addWidget(lr)

        mm_layout.addStretch(1)    # Extra space at end

        # Link consecutive markers checkbutton
        self.link_new_button = lm = QCheckBox('Link new marker to selected marker', f)
        lm.stateChanged.connect(self.link_new_cb)
        layout.addWidget(lm)

        layout.addSpacing(5)
        hl = QLabel('Place markers using mouse modes in the Markers toolbar')
        layout.addWidget(hl)

        self.update_settings()
        
        tw.manage(placement="side")

    @property
    def _settings(self):
        from .mouse import _mouse_marker_settings
        mms = _mouse_marker_settings(self.session)
        return mms
    
    def _marker_color_changed(self, color):
        self._settings['marker color'] = color
        from . import selected_markers
        markers = selected_markers(self.session)
        markers.colors = color
        _log_markers_recolor(self.session, markers, color)

    def _marker_radius_changed(self):
        try:
            r = float(self._marker_radius.text())
        except ValueError:
            self.session.logger.status('Marker radius is not an number')
            return
        if r <= 0:
            self.session.logger.status('Marker radius must be > 0')
            return
        self._settings['marker radius'] = r
        from . import selected_markers
        markers = selected_markers(self.session)
        markers.radii = r
        _log_markers_resize(self.session, markers, r)
    
    def _link_color_changed(self, color):
        self._settings['link color'] = color
        from . import selected_links
        links = selected_links(self.session)
        links.colors = color
        _log_links_recolor(self.session, links, color)

    def _link_radius_changed(self):
        try:
            r = float(self._link_radius.text())
        except ValueError:
            self.session.logger.status('Link radius is not an number')
            return
        if r <= 0:
            self.session.logger.status('Link radius must be > 0')
            return
        self._settings['link radius'] = r
        from . import selected_links
        links = selected_links(self.session)
        links.radii = r
        _log_links_resize(self.session, links, r)
        
    def update_settings(self):
        s = self._settings
        self._marker_color.set_color(s['marker color'])
        self._marker_radius.setText('%.3g' % s['marker radius'])
        self._link_color.set_color(self._settings['link color'])
        self._link_radius.setText('%.3g' % s['link radius'])
        from Qt.QtCore import Qt
        self.link_new_button.setChecked(Qt.Checked if s['link_new_markers'] else Qt.Unchecked)
        
    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def link_new_cb(self, link):
        from . import mouse
        s = mouse._mouse_marker_settings(self.session)
        s['link_new_markers'] = link

def _log_markers_recolor(session, markers, color):
    if len(markers) == 0:
        return
    mspec = _markers_spec(markers)
    lf = 'links false' if len(markers.intra_bonds) > 0 else ''
    from chimerax.core.colors import color_name
    cmd = 'marker change %s color %s %s' % (mspec, color_name(color), lf)
    from chimerax.core.commands import log_equivalent_command
    log_equivalent_command(session, cmd)

def _log_markers_resize(session, markers, radius):
    if len(markers) == 0:
        return
    mspec = _markers_spec(markers)
    lf = 'links false' if len(markers.intra_bonds) > 0 else ''
    cmd = 'marker change %s radius %.4g %s' % (mspec, radius, lf)
    from chimerax.core.commands import log_equivalent_command
    log_equivalent_command(session, cmd)

def _markers_spec(markers):
    specs = []
    for mset,ms in markers.by_structure:
        if len(ms) == mset.num_atoms:
            spec = mset.atomspec
        else:
            spec = '%s:%s' % (mset.atomspec, _range_spec(m.residue.number for m in ms))
        specs.append(spec)
    mspec = ''.join(specs)
    return mspec

def _range_spec(integers):
    isort = list(set(integers))
    if len(isort) == 0:
        return ''
    isort.sort()
    ranges = []
    istart = iend = None
    for i in isort:
        if istart is None:
            istart = iend = i
        elif i == iend + 1:
            iend = i
        else:
            ranges.append(('%d-%d' % (istart,iend)) if iend > istart else ('%d' % istart))
            istart = iend = i
    ranges.append(('%d-%d' % (istart,iend)) if iend > istart else ('%d' % istart))
    return ','.join(ranges)
            
def _log_links_recolor(session, links, color):
    if len(links) == 0:
        return
    from chimerax.core.commands import log_equivalent_command
    from chimerax.core.colors import color_name
    cname = color_name(color)
    for mspec in _link_specs(links):
        cmd = 'marker change %s color %s markers false' % (mspec, cname)
        log_equivalent_command(session, cmd)
        
def _log_links_resize(session, links, radius):
    if len(links) == 0:
        return
    from chimerax.core.commands import log_equivalent_command
    for mspec in _link_specs(links):
        cmd = 'marker change %s radius %.4g markers false' % (mspec, radius)
        log_equivalent_command(session, cmd)

def _link_specs(links):
    m1,m2 = links.atoms
    from chimerax.atomic import concatenate
    markers = concatenate((m1,m2))
    mlinks = markers.intra_bonds
    if len(mlinks) == len(links):
        lspecs = [_markers_spec(markers)]
    else:
        lspecs = [_markers_spec(Atoms(mpair)) for mpair in zip(m1,m2)]
    return lspecs

def marker_panel(session, tool_name):
  cb = getattr(session, '_markers_gui', None)
  if cb is None:
    session._markers_gui = cb = MarkerModeSettings(session, tool_name)
  return cb
