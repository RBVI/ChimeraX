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
# Dialog for creating or joining meetings
#
from chimerax.core.tools import ToolInstance
class MeetingTool(ToolInstance):

  help = 'help:user/tools/meeting.html'
  SESSION_ENDURING = True	# Don't remove tool when meeting is joined
  
  def __init__(self, session, tool_name):

    self._face_image = None
    from os.path import join, dirname
    self._default_face_image = join(dirname(__file__), 'face.png')

    ToolInstance.__init__(self, session, tool_name)

    from chimerax.ui import MainToolWindow
    tw = MainToolWindow(self)
    self.tool_window = tw
    parent = tw.ui_area

    from Qt.QtWidgets import QVBoxLayout, QLabel
    layout = QVBoxLayout(parent)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(0)
    parent.setLayout(layout)

    from .meeting import _meeting_settings
    settings = _meeting_settings(session)

    # Entry field for meeting name, participant name, color, photo
    nf = self._create_name_gui(parent, settings)
    layout.addWidget(nf)

    # Create, Join, Leave, Option, Help buttons
    bf = self._create_buttons(parent)
    layout.addWidget(bf)

    # Proxy and name server options panel
    self._options_panel = options = self._create_options_gui(parent, settings)
    layout.addWidget(options)
        
    layout.addStretch(1)    # Extra space at end

    tw.manage(placement="side")
    
  # ---------------------------------------------------------------------------
  #
  def _create_name_gui(self, parent, settings):

    from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton
    from Qt.QtGui import QIcon
    from Qt.QtCore import QSize
        
    f = QFrame(parent)
    layout = QHBoxLayout(f)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(8)

    mnl = QLabel('Meeting name', f)
    layout.addWidget(mnl)

    self._meeting_name = mn = QLineEdit('', f)
    mn.setMaximumWidth(60)
    mn.setMinimumWidth(60)
    layout.addWidget(mn)
        
    pnl = QLabel('Participant name', f)
    layout.addWidget(pnl)

    self._participant_name = pn = QLineEdit('', f)
    pn.setMaximumWidth(60)
    pn.setMinimumWidth(60)
    layout.addWidget(pn)
    if settings.name != 'Remote':
      pn.setText(settings.name)

    pcl = QLabel('Pointer color', f)
    layout.addWidget(pcl)

    from chimerax.ui.widgets import ColorButton
    cl = ColorButton(f, max_size = (16,16))
    self._color_button = cl
    cl.color = settings.color
#    cl.color_changed.connect(self._color_chosen)
    layout.addWidget(cl)    

    ppl = QLabel('Photo', f)
    layout.addWidget(ppl)

    self._photo_button = pb = QPushButton(f)
    pb.setStyleSheet("* { padding: 0; margin: 0; border: 0; }")
    from os.path import isfile
    photo_path = settings.face_image
    if photo_path is None or not isfile(photo_path):
      photo_path = self._default_face_image
    else:
      self._face_image = photo_path
    icon = QIcon(photo_path)
    pb.setIcon(icon)
    pb.setIconSize(QSize(20,20))
    pb.clicked.connect(self._choose_photo)
    layout.addWidget(pb)

    layout.addStretch(1)    # Extra space at end
    
    return f

  # ---------------------------------------------------------------------------
  #
  def _choose_photo(self):
    parent = self.tool_window.ui_area
    from Qt.QtWidgets import QFileDialog
    path, ftype  = QFileDialog.getOpenFileName(parent, caption = 'Face Image',
                                               filter = 'Images (*.png *.jpg *.tif)')
    if path:
      self._face_image = path
      from Qt.QtGui import QIcon
      icon = QIcon(path)
      self._photo_button.setIcon(icon)
  
  # ---------------------------------------------------------------------------
  #
  def _create_options_gui(self, parent, settings):

    from chimerax.ui.widgets import CollapsiblePanel
    p = CollapsiblePanel(parent, title = None, margins = (30,0,30,10))
    f = p.content_area
    layout = f.layout()

    from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QMenu

    mf = QFrame(f)
    mlayout = QHBoxLayout(mf)
    mlayout.setContentsMargins(0,0,0,0)
    mlayout.setSpacing(8)
    sl = QLabel('Server', mf)
    mlayout.addWidget(sl)
    self._server_button = sb = QPushButton(mf)
    sb.setText(settings.server)
    sm = QMenu(mf)
    for name in settings.servers.keys():
        sm.addAction(name, lambda n=name: self._set_server_menu(n))
    sb.setMenu(sm)
    mlayout.addWidget(sb)
    mlayout.addStretch(1)
    layout.addWidget(mf)

    msg = 'The server option is used when starting a meeting.  Direct means participants connect directly to the computer that started the meeting.  If that computer is behind a firewall and cannot be reached by participants then a public computer chimeraxmeeting.net can be used as the server.'
    self._server_label = el = QLabel(msg, f)
    # Size hint does not update when window made narrower and word wrape enabled
    # causing wrong panel height.
    el.setWordWrap(True)
    layout.addWidget(el)

    layout.addStretch(1)
    
    return p

  # ---------------------------------------------------------------------------
  #
  def _set_server_menu(self, name):
    self._server_button.setText(name)

  # ---------------------------------------------------------------------------
  #
  def _create_buttons(self, parent):
    
    from Qt.QtWidgets import QFrame, QHBoxLayout, QPushButton
    f = QFrame(parent)
    layout = QHBoxLayout(f)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(10)

    for name, callback in (('Create', self._start_meeting),
                           ('Join', self._join_meeting),
                           ('Leave', self._leave_meeting),
                           ('Server...', self._toggle_options),
                           ('Help', self._show_help)):
      b = QPushButton(name, f)
      b.clicked.connect(callback)
      layout.addWidget(b)
        
    layout.addStretch(1)    # Extra space at end

    return f

  # ---------------------------------------------------------------------------
  #
  def _start_meeting(self):
    cmd = self._command(start = True)
    if cmd:
      from chimerax.core.commands import run
      run(self.session, cmd)

  # ---------------------------------------------------------------------------
  #
  def _command(self, start = False):
    cmd = 'meeting start' if start else 'meeting join'
    
    meeting_name = self._meeting_name.text().strip()
    if not meeting_name:
      self.session.logger.error('Must specify a meeting name')
      return None

    from chimerax.core.commands import quote_if_necessary
    cmd += ' ' + quote_if_necessary(meeting_name)
    
    from .meeting import _meeting_settings
    settings = _meeting_settings(self.session)
    opts = (self._partipant_options(settings) +
            (self._server_options(settings) if start else []))
    if opts:
      cmd += ' ' + ' '.join(opts)

    return cmd

  # ---------------------------------------------------------------------------
  #
  def _partipant_options(self, settings):
    opts = []
    from chimerax.core.commands import quote_if_necessary

    name = self._participant_name.text().strip()
    if name != settings.name:
      opts.append('name %s' % quote_if_necessary(name))

    color = tuple(self._color_button.color)
    if color != settings.color:
      from chimerax.core.colors import hex_color
      opts.append('color %s' % hex_color(color))

    face_image = self._face_image
    if face_image is not None and face_image != settings.face_image:
      opts.append('faceImage %s' % quote_if_necessary(face_image))

    return opts

  # ---------------------------------------------------------------------------
  #
  def _server_options(self, settings):

    opts = []
    server = self._server_button.text()
    if server != settings.server:
      from chimerax.core.commands import quote_if_necessary
      opts.append('server %s' % quote_if_necessary(server))
    return opts
    
  # ---------------------------------------------------------------------------
  #
  def _join_meeting(self):
    cmd = self._command()
    if cmd:
      from chimerax.core.commands import run
      run(self.session, cmd)

  # ---------------------------------------------------------------------------
  #
  def _leave_meeting(self):
    from chimerax.core.commands import run
    run(self.session, 'meeting close')

  # ---------------------------------------------------------------------------
  #
  def _toggle_options(self):
    self._options_panel.toggle_panel_display()

  # ---------------------------------------------------------------------------
  #
  def _show_help(self):
    from chimerax.core.commands import run
    run(self.session, 'help %s' % self.help)

  # ---------------------------------------------------------------------------
  #
  @classmethod
  def get_singleton(self, session, create=True):
    from chimerax.core import tools
    return tools.get_singleton(session, MeetingTool, 'Meeting', create=create)

# -----------------------------------------------------------------------------
#
def meeting_gui(session, create = False):

  return MeetingTool.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_meeting_gui(session):

  return meeting_gui(session, create = True)
