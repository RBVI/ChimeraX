# vim: set expandtab ts=4 sw=4:

from chimerax.core.tools import ToolInstance


# ------------------------------------------------------------------------------
#
class VolumeViewer(ToolInstance):

    SESSION_SKIP = True

    def __init__(self, session, bundle_info, *, volume=None):
        ToolInstance.__init__(self, session, bundle_info)

        self.volume = volume

        vname = volume.name_with_id()
        self.display_name = "Map %s" % vname

        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QMenu, QLineEdit
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(6)

        d = volume.data
        sx,sy,sz = d.size
        size = '%d<sup>3</sup>' % (sx,) if sx == sy and sy == sz else '%d,%d,%d' % (sx,sy,sz)
        sl = QLabel('%s %s' % (vname, size))
        layout.addWidget(sl)
        st = QLabel('step')
        layout.addWidget(st)
        stx,sty,stz = volume.region[2]
        step = '%d'%stx if stx == sty and sty == stz else '%d,%d,%d' % (stx,sty,stz)
        self.step = sb = QPushButton(step)
        sm = QMenu()
        for step in (1,2,4,8,16):
            sm.addAction('%d' % step, lambda s=step: self.set_step_cb(s))
        sb.setMenu(sm)
        layout.addWidget(sb)
        ll = QLabel('level')
        layout.addWidget(ll)
        self.level = lev = QLineEdit('%.3g' % volume.surface_levels[0])
        lev.setMaximumWidth(30)
        lev.returnPressed.connect(self.set_level_cb)
        layout.addWidget(lev)
        layout.addStretch(1)	# Extra space at end of button row.
        parent.setLayout(layout)

        tw.manage(placement="right")

        from chimerax.core.models import REMOVE_MODELS
        self.model_close_handler = session.triggers.add_handler(REMOVE_MODELS, self.models_closed_cb)
        
    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def set_step_cb(self, step):
        s = step
        self.volume.new_region(ijk_step = (s,s,s), adjust_step = False)
        self.step.setText('%d' % s)

    def set_level_cb(self):
        level = float(self.level.text())
        v = self.volume
        v.set_parameters(surface_levels = [level])
        v.show()

    def models_closed_cb(self, name, models):
        if self.volume in models:
            self.delete()

    # Override ToolInstance method
    def delete(self):
        s = self.session
        s.triggers.delete_handler(self.model_close_handler)
        super().delete()

def show_viewer_on_open(session):
    # Register callback to show volume viewer when a map is opened
    if not hasattr(session, '_registered_volume_viewer'):
        session._registered_volume_viewer = True
        from chimerax.core.models import ADD_MODELS
        session.triggers.add_handler(ADD_MODELS, lambda name, m, s=session: models_added_cb(m, s))

def models_added_cb(models, session):
    # Show volume viewer when a map is opened.
    from chimerax.core.map import Volume
    vlist = [m for m in models if isinstance(m, Volume)]
    if vlist:
        for v in vlist:
            bundle_info = session.toolshed.find_bundle('volume_viewer')
            vv = VolumeViewer(session, bundle_info, volume = v)
            vv.show()
