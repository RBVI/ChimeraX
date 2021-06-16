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
# Panel for coloring surfaces by volume data value or spatial position.
#
from chimerax.core.tools import ToolInstance
class SurfaceColorGUI(ToolInstance):

    help = 'help:user/tools/surfacecolor.html'

    def __init__(self, session, tool_name):

        self._mouse_handler = None
        self._last_mouse_xy = None
        
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        tw.shown_changed = self._shown_changed
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Make menus to choose surface and coloring method
        self._surface_menu, self._method_menu, mf = self._create_surface_method_menus(parent)
        layout.addWidget(mf)

        # Center and axis for geometric coloring
        self._center_axis = caf = self._create_center_axis_entries(parent)
        layout.addWidget(caf)

        # Map menu for coloring by volume data value
        self._map = mm = self._create_map_menu(parent)
        layout.addWidget(mm.frame)
        mm.frame.setVisible(False)

        # Color palette
        self._colors = ce = self._create_color_entries(parent)
        layout.addWidget(ce.frame)

        # Create buttons
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        # Options panel
        options = self._create_options_pane(parent)
        layout.addWidget(options)

        layout.addStretch(1)    # Extra space at end

        # Set color range for surface
        self._surface_chosen()

        # Show and hide gui widgets for specific methods
        self._method_chosen()

        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, SurfaceColorGUI, 'Surface Color',
                                   create=create)
    
    # ---------------------------------------------------------------------------
    #
    def _create_surface_method_menus(self, parent):

        from chimerax.ui.widgets import row_frame, EntriesRow
        f, layout = row_frame(parent, spacing = 10)
        
        from chimerax.core.models import Surface
        from chimerax.map import Volume, VolumeSurface
        def _not_volume_surface(m):
            return not isinstance(m, VolumeSurface)
        from chimerax.ui.widgets import ModelMenu
        smenu = ModelMenu(self.session, f, label = 'Color surface',
                          model_types = [Volume, Surface],
                          model_filter = _not_volume_surface,
                          model_chosen_cb = self._surface_chosen)
        layout.addWidget(smenu.frame)

        from chimerax.ui.widgets import EntriesRow
        me = EntriesRow(f, 'by', self._method_names)
        meth = me.values[0]
        meth.widget.menu().triggered.connect(self._method_chosen)

        layout.addStretch(1)
        
        return smenu, meth, f

    # ---------------------------------------------------------------------------
    #
    @property
    def _surfaces(self):
        m = self._surface_menu.value
        if m is None:
            return []
        from chimerax.map import Volume
        surfs = m.surfaces if isinstance(m, Volume) else [m]
        if self._color_clip_cap_only.enabled:
            surfs = _clip_cap_models(surfs)
        return surfs

    # ---------------------------------------------------------------------------
    #
    def _create_center_axis_entries(self, parent):
        from chimerax.ui.widgets import row_frame, EntriesRow
        f, layout = row_frame(parent)
        ce = EntriesRow(f, 'using origin', '', ('Center', self._set_center))
        self._center_entry = center = ce.values[0]
        center.value = '0 0 0'
        center.pixel_width = 150
        ax = EntriesRow(f, 'axis', '')
        self._axis_entry = axis = ax.values[0]
        axis.value = '0 0 1'
        self._axis_widget = ax.frame
        layout.addStretch(1)
        return f

    # ---------------------------------------------------------------------------
    #
    def _create_map_menu(self, parent):
        from chimerax.map import Volume
        from chimerax.ui.widgets import ModelMenu
        m = ModelMenu(self.session, parent, label = 'using map',
                      model_types = [Volume])
        return m

    # ---------------------------------------------------------------------------
    #
    def _create_color_entries(self, parent):

        p = PaletteWidget(parent)
        from chimerax.core.colors import BuiltinColormaps
        p.set_colormap(BuiltinColormaps['redblue'], set_values = False)
        return p

    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Color', self._color),
                        ('Uncolor', self._uncolor),
                        ('Key', self._create_color_key),
                        ('Options', self._show_or_hide_options),
                        ('Help', self._show_help)],
                       spacing = 10)
        return f
    
    # ---------------------------------------------------------------------------
    #
    def _create_options_pane(self, parent):

        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None)
        f = p.content_area

        palette_names = ('rainbow', 'red-white-blue', 'cyan-white-maroon',
                         'blue-white-red', 'grayscale')
        num_colors = tuple(str(count) for count in range(2,9))
        from chimerax.ui.widgets import EntriesRow, ColorButton
        po = EntriesRow(f,
                        'Colors', num_colors,
                        'Palette', palette_names,
                        ('Reverse', self._reverse_palette))
        self._num_colors, self._palette_name = ncol, pal = po.values
        ncol.value = '3'
        ncol.widget.menu().triggered.connect(self._num_colors_chosen)
        pal.value = 'red-white-blue'
        pal.widget.menu().triggered.connect(self._palette_chosen)

        EntriesRow(f, ('Set', self._set_full_range), 'full range of surface values')

        so = EntriesRow(f, 'Surface offset', 1.4)
        self._surface_offset = so.values[0]
        self._surface_offset_widget = so.frame

        co = EntriesRow(f, 'Color outside volume', ColorButton)
        self._color_outside = cov = co.values[0]
        cov.color = 'gray'
        self._color_outside_widget = co.frame

        cc = EntriesRow(f, False, 'Only color clipped surface face')
        self._color_clip_cap_only = cc.values[0]

        vam = EntriesRow(f, False, 'Report value at mouse position')
        self._value_at_mouse = rv = vam.values[0]
        rv.changed.connect(self._mouse_report)

        return p

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    # ---------------------------------------------------------------------------
    #
    _method_names = ('radius', 'cylinder radius', 'height',
                     'electrostatic potential', 'volume data value',
                     'volume data gradient norm')
    _origin_methods = ('radius', 'cylinder radius', 'height')
    _axis_methods = ('cylinder radius', 'height')
    _volume_methods = ('electrostatic potential', 'volume data value',
                       'volume data gradient norm')
    _offset_methods = ('electrostatic potential',)
    _method_subcommand = {'radius':'radial',
                          'cylinder radius':'cylindrical',
                          'height':'height',
                          'electrostatic potential':'electrostatic',
                          'volume data value':'sample',
                          'volume data gradient norm':'gradient'}
    
    # ---------------------------------------------------------------------------
    #
    def _method_chosen(self):
        method = self._method_menu.value
        self._map.frame.setVisible(method in self._volume_methods)
        self._center_axis.setVisible(method in self._origin_methods)
        self._surface_offset_widget.setVisible(method in self._offset_methods)
        self._color_outside_widget.setVisible(method in self._volume_methods)
        self._axis_widget.setVisible(method in self._axis_methods)
        if self._options_panel.shown:
            self._options_panel.resize_panel()
        if method == 'electrostatic potential' and not self._colors.has_values():
            self._colors.set_value_range(-10,10)

    # ---------------------------------------------------------------------------
    #
    def _method_name(self, color_updater):
        from .colorgeom import RadialColor, CylinderColor, HeightColor
        from .colorvol import GradientColor, VolumeColor
        if isinstance(color_updater, RadialColor):
            name = 'radius'
        elif isinstance(color_updater, CylinderColor):
            name = 'cylinder radius'
        elif isinstance(color_updater, HeightColor):
            name = 'height'
        elif isinstance(color_updater, GradientColor):
            name = 'volume data gradient norm'
        elif isinstance(color_updater, VolumeColor):
            if color_updater.offset == 0:
                name = 'volume data value'
            else:
                name = 'electrostatic potential'
        return name
    
    # ---------------------------------------------------------------------------
    #
    @property
    def _center(self):
        cstring = self._center_entry.value
        try:
            center = tuple(float(x) for x in cstring.split())
        except ValueError:
            center = None
        if center is None or len(center) != 3:
            center = (0,0,0)
            self.warn('Surface Color: could not parse center "%s" as 3 numbers' % cstring)
        return center

    # ---------------------------------------------------------------------------
    #
    def _set_center(self):
        surface = self._surface_menu.value
        if surface is None:
            self.warn('Surface Color: no surface chosen')
            return

        b = surface.bounds()
        if b is None:
            self.warn('Surface Color: surface has no bounding box')
            return
    
        self._center_entry.value = '%.5g %.5g %.5g' % tuple(b.center())
        
    # ---------------------------------------------------------------------------
    #
    @property
    def _axis(self):
        astring = self._axis_entry.value
        try:
            axis = tuple(float(x) for x in astring.split())
        except ValueError:
            axis = None
        if axis is None or len(axis) != 3:
            axis = (0,0,1)
            self.warn('Surface Color: could not parse axis "%s" as 3 numbers' % astring)
        return axis
        
    # ---------------------------------------------------------------------------
    #
    @property
    def _offset(self):
        try:
            offset = self._surface_offset.value
        except ValueError:
            offset = 1.4
            self.warn('Surface Color: could not parse offset "%s" as a numbers'
                      % self._surface_offset.widget.text())
        return offset

    # ---------------------------------------------------------------------------
    #
    def _num_colors_chosen(self):
        ncolors = int(self._num_colors.value)
        self._colors.color_count = ncolors

    # ---------------------------------------------------------------------------
    #
    def _palette_chosen(self):
        cmap = _named_colormap(self._palette_name.value, self._colors.color_count)
        self._colors.set_colormap(cmap, update_num_colors = False, set_values = False)

    # ---------------------------------------------------------------------------
    #
    def _reverse_palette(self):
        self._colors.reverse_colors()

    # ---------------------------------------------------------------------------
    #
    @property
    def _palette_spec(self):
        return self._colors.palette_specifier
    
    # ---------------------------------------------------------------------------
    #
    def _create_color_key(self):
        cmap = self._colors.colormap()
        from chimerax.color_key import show_key
        show_key(self.session, cmap)

    # ---------------------------------------------------------------------------
    #
    def _set_full_range(self):
        self._colors.clear_values()
        self._color()
    
    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

    # ---------------------------------------------------------------------------
    #
    def _surface_chosen(self):
        sc = self._surface_coloring
        if sc is None:
            return
        self._update_color_levels()
        self._method_menu.value = method_name = self._method_name(sc)
        self._method_chosen()
        if method_name in self._origin_methods:
            self._center_entry.value = '%.5g %.5g %.5g' % tuple(sc.origin)
        if method_name in self._axis_methods:
            self._axis_entry.value = '%.5g %.5g %.5g' % tuple(sc.axis)
        if method_name in self._offset_methods:
            self._surface_offset.value = sc.offset
        if method_name in self._volume_methods:
            self._map.value = sc.volume

    # ---------------------------------------------------------------------------
    #
    def _update_color_levels(self):
        sc = self._surface_coloring
        if sc is None:
            return
        self._colors.set_colormap(sc.colormap)
    
    # ---------------------------------------------------------------------------
    #
    @property
    def _surface_coloring(self):
        # If volume model chosen and multiple surfaces, just return first coloring.
        for s in self._surfaces:
            sc = _surface_color_updater(s)
            if sc:
                return sc
        return None
        
    # ---------------------------------------------------------------------------
    #
    def _color(self, *, log_command = True, log_command_only = False):

        surfs = self._surfaces
        if len(surfs) == 0:
            if log_command:
                self.warn('No surface chosen for coloring')
            return

        surf_spec = ''.join('#%s' % s.id_string for s in surfs)
        method = self._method_menu.value
        subcmd = self._method_subcommand[method]
        palette = self._palette_spec
        
        if method in self._origin_methods:
            cmd = 'color %s %s palette %s' % (subcmd, surf_spec, palette)
            c = self._center
            if c != (0,0,0):
                cmd += ' center %.5g,%.5g,%.5g' % c
            if method in self._axis_methods:
                a = self._axis
                if a != (0,0,1):
                    cmd += ' axis %.5g,%.5g,%.5g' % a
        elif method in self._volume_methods:
            map = self._map.value
            if map is None:
                self.warn('No map chosen for coloring')
                return
            cmd = ('color %s %s map #%s palette %s'
                   % (subcmd, surf_spec, map.id_string, palette))
            if method in self._offset_methods:
                o = self._offset
                if o != 1.4:
                    cmd += ' offset %.4g' % o

        if log_command_only:
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, cmd)
        else:
            from chimerax.core.commands import run
            run(self.session, cmd, log = log_command)

        if not self._colors.has_values():
            self._update_color_levels()
            
    # ---------------------------------------------------------------------------
    #
    def _uncolor(self):

        surfs = self._surfaces
        if len(surfs) == 0:
            return

        surf_spec = ''.join('#%s' % s.id_string for s in surfs)
        cmd = 'color single %s' % surf_spec
        from chimerax.core.commands import run
        run(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def _shown_changed(self, shown):
        # Turn off reporting mouse surface values if panel is hidden.
        if not shown and self._value_at_mouse.enabled:
            self._value_at_mouse.enabled = False
            
    # ---------------------------------------------------------------------------
    #
    def _mouse_report(self):
        enable = self._value_at_mouse.enabled
        h = self._mouse_handler
        if enable and h is None:
            h = self.session.triggers.add_handler('new frame',
                                                  self._report_surface_value_at_mouse)
            self._mouse_handler = h
        elif not enable and h:
            self.session.triggers.remove_handler(h)
            self._mouse_handler = None
                
    # ---------------------------------------------------------------------------
    #
    def _report_surface_value_at_mouse(self, *unused_args):

        from Qt.QtGui import QCursor
        gw = self.session.ui.main_window.graphics_window
        mouse_pos = gw.mapFromGlobal(QCursor.pos())
        win_xy = win_x, win_y = mouse_pos.x(), mouse_pos.y()
        if win_xy == self._last_mouse_xy:
            return
        self._last_mouse_xy = win_xy
        _report_surface_value(self.session, win_x, win_y)
        
    # ---------------------------------------------------------------------------
    #
    def warn(self, message):
        log = self.session.logger
        log.warning(message)
        log.status(message, color='red')

# -----------------------------------------------------------------------------
#
def _surface_color_updater(surface):
    from .colorgeom import geometry_coloring
    gc = geometry_coloring(surface)
    if gc:
        return gc
    from .colorvol import volume_coloring
    vc = volume_coloring(surface)
    if vc:
        return vc
    return None

# ---------------------------------------------------------------------------
#
def _report_surface_value(session, win_x, win_y):

    pick = session.main_view.picked_object(win_x, win_y)
    if pick is None:
        return
    # For PickedMap surface pick is pick.triangle_pick
    surf_pick = pick.triangle_pick if hasattr(pick, 'triangle_pick') else pick
    if not hasattr(surf_pick, 'drawing'):
        return
    surf = surf_pick.drawing()
    sc = _surface_color_updater(surf)
    p = pick.position
    from .colorgeom import GeometryColor
    from .colorvol import VolumeColor
    if isinstance(sc, GeometryColor):
        v = sc.values(p.reshape((1,3)))
    elif isinstance(sc, VolumeColor):
        from chimerax.geometry import Place
        transform_to_scene_coords = Place()
        v, outside = sc.vertex_values(p.reshape((1,3)), transform_to_scene_coords)
        if len(outside) > 0:
            return
    else:
        return
    value = v[0]
    m = pick.drawing()
    mname = '%s (#%s)' % (m.name, m.id_string)
    msg = 'Value %.5g at %.4g,%.4g,%.4g of %s' % (value, p[0],p[1],p[2], mname)
    session.logger.status(msg)

# -----------------------------------------------------------------------------
#
class PaletteWidget:
    def __init__(self, parent, max_colors = 8):
        from Qt.QtWidgets import QFrame, QGridLayout, QLineEdit
        self.frame = cf = QFrame(parent)
        layout = QGridLayout(cf)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)

        self._buttons = buttons = []
        row = 0
        from chimerax.ui.widgets import ColorButton
        from Qt.QtCore import Qt
        for col in range(max_colors):
            cb = ColorButton(cf, max_size = (24,24))
            cb.color = (128,128,128,255)
            buttons.append(cb)
            layout.addWidget(cb, row, col, Qt.AlignHCenter)
        layout.setColumnStretch(max_colors, 1)

        self._values = values = []
        row += 1
        for col in range(max_colors):
            le = QLineEdit(cf)
            le.setMaximumWidth(65)
            values.append(le)
            layout.addWidget(le, row, col)

        self._num_shown = max_colors
        
    @property
    def palette_specifier(self):
        clist,vlist = self._colors_and_values()
        from chimerax.ui.widgets import hex_color_name
        cnames = [hex_color_name(c) for c in clist]
        if vlist is None:
            spec = ':'.join(cnames)
        else:
            spec = ':'.join('%.5g,%s' % (v,c) for v,c in zip(vlist, cnames))
        return spec

    def _colors_and_values(self):
        clist = []
        vlist = []
        all_colors = []
        ns = self._num_shown
        for b,v in zip(self._buttons[:ns], self._values[:ns]):
            color = b.color
            all_colors.append(color)
            try:
                value = float(v.text())
            except ValueError:
                continue
            vlist.append(value)
            clist.append(color)
        return (clist, vlist) if vlist else (all_colors, None)
        
    def colormap(self):
        clist, vlist = self._colors_and_values()
        if vlist is None:
            from numpy import linspace
            vlist = linspace(0.0, 1.0, len(clist))
        from chimerax.core.colors import Colormap, rgba8_to_rgba
        cmap = Colormap(vlist, [rgba8_to_rgba(c) for c in clist])
        return cmap

    def set_colormap(self, colormap, update_num_colors = True, set_values = True):
        if update_num_colors:
            self._set_num_colors(len(colormap.colors))
        ns = self._num_shown
        for i,(c,v) in enumerate(zip(colormap.colors[:ns], colormap.data_values[:ns])):
               self._buttons[i].color = c
               if set_values:
                   self._values[i].setText('%.4g' % v)

    def reverse_colors(self):
        ns = self._num_shown
        colors = [self._buttons[i].color for i in range(ns)]
        colors.reverse()
        for b, color in zip(self._buttons[:ns], colors):
            b.color = color
        
    def _get_color_count(self):
        return self._num_shown
    def _set_color_count(self, nc):
        self._set_num_colors(min(nc, len(self._buttons)))
    color_count = property(_get_color_count, _set_color_count)
    
    def _set_num_colors(self, nc):
        if nc == self._num_shown:
            return
        for i,(b,v) in enumerate(zip(self._buttons, self._values)):
            vis = i < nc
            b.setVisible(vis)
            v.setVisible(vis)
        self._num_shown = nc
        
    def has_values(self):
        for v in self._values:
            try:
                value = float(v.text())
                return True
            except ValueError:
                continue
        return False

    def clear_values(self):
        for v in self._values:
            v.setText('')

    def set_value_range(self, min_value, max_value):
        n = self.color_count
        step = (max_value - min_value) / (n-1) if n > 1 else 1
        for i,v in enumerate(self._values):
            value = min_value + i * step
            v.setText('%.4g' % value)
        
# -----------------------------------------------------------------------------
#
def _named_colormap(name, color_count):
    from chimerax.core.colors import BuiltinColormaps, Colormap
    cmap = BuiltinColormaps[name]
    if len(cmap.colors) == color_count:
        return cmap
    # Use requested number of colors
    from numpy import linspace
    values = linspace(0.0, 1.0, color_count)
    colors = cmap.interpolated_rgba8(values)
    cmap = Colormap(values, colors,
                    cmap.color_above_value_range,
                    cmap.color_below_value_range,
                    cmap.color_no_value)
    return cmap
    
# -----------------------------------------------------------------------------
#
def _clip_cap_models(surfaces):
    caps = []
    for s in surfaces:
        for c in s.all_models():
            if getattr(c, 'is_clip_cap', False):
                caps.append(c)
    return caps
    
# -----------------------------------------------------------------------------
#
def surface_color_panel(session, create = False):
    return SurfaceColorGUI.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_surface_color_panel(session):
    return surface_color_panel(session, create = True)

