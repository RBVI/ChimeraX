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

BUG_HOST = "www.rbvi.ucsf.edu"
BUG_SELECTOR = "/chimerax/cgi-bin/chimerax_bug_report.py"
BUG_URL = "http://" + BUG_HOST + BUG_SELECTOR

# -----------------------------------------------------------------------------
# User interface for bug reporter.
#
from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class BugReporter(ToolInstance):

    def __init__(self, session, tool_name):

        self._ses = session
        
        ToolInstance.__init__(self, session, tool_name)

        from .settings import BugReporterSettings
        self.settings = BugReporterSettings(session, 'Bug Reporter')

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area
        parent.setMinimumWidth(600)

        from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton, QLineEdit, QTextEdit
        from PyQt5.QtWidgets import QWidget, QHBoxLayout, QCheckBox
        from PyQt5.QtCore import Qt
        
        layout = QGridLayout(parent)
        layout.setContentsMargins(3,3,3,3)
        layout.setHorizontalSpacing(3)
        layout.setVerticalSpacing(3)
        parent.setLayout(layout)

        row = 1
        
        intro = '''
        <center><h1>Report a Bug</h1></center>
        <p>Thank you for using our feedback system.
	  Feedback is greatly appreciated and plays a crucial role
	  in the development of ChimeraX.</p>
	  <p><b>Note</b>:
          We do not automatically collect any personal information or the data
          you were working with when the problem occurred.  Providing your e-mail address is optional,
          but will allow us to inform you of a fix or to ask questions, if needed.
          Attaching data may also be helpful.  However, any information or data
          you wish to keep confidential should be sent separately (not using this form).</p>
        '''
        il = QLabel(intro)
        il.setWordWrap(True)
        layout.addWidget(il, row, 1, 1, 2)
        row += 1
        
        cnl = QLabel('Contact Name:')
        cnl.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        layout.addWidget(cnl, row, 1)
        self.contact_name = cn = QLineEdit(self.settings.contact_name)
        layout.addWidget(cn, row, 2)
        row += 1

        eml = QLabel('Email Address:')
        eml.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        layout.addWidget(eml, row, 1)
        self.email_address = em = QLineEdit(self.settings.email_address)
        layout.addWidget(em, row, 2)
        row += 1

        class TextEdit(QTextEdit):
            def __init__(self, text, initial_line_height):
                self._lines = initial_line_height
                QTextEdit.__init__(self, text)
            def sizeHint(self):
                from PyQt5.QtCore import QSize
                fm = self.fontMetrics()
                h = self._lines * fm.lineSpacing() + fm.ascent()
                size = QSize(-1, h)
                return size
            def minimumSizeHint(self):
                from PyQt5.QtCore import QSize
                return QSize(1,1)

        dl = QLabel('Description:')
        dl.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        layout.addWidget(dl, row, 1)
        self.description = d = TextEdit('', 3)
        d.setText('<font color=blue>(Describe the actions that caused this problem to occur here)</font>')
        layout.addWidget(d, row, 2)
        row += 1

        gil = QLabel('Gathered Information:')
        gil.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        layout.addWidget(gil, row, 1)
        self.gathered_info = gi = TextEdit('', 3)
        gi.setText(self.opengl_info())
        layout.addWidget(gi, row, 2)
        row += 1

        fal = QLabel('File Attachment:')
        fal.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        layout.addWidget(fal, row, 1)
        fb = QWidget()
        layout.addWidget(fb, row, 2)
        fbl = QHBoxLayout(fb)
        fbl.setSpacing(3)
        fbl.setContentsMargins(0,0,0,0)
        self.attachment = fa = QLineEdit('')
        fbl.addWidget(fa)
        fab = QPushButton('Browse')
        fab.clicked.connect(lambda e: self.file_browse())
        fbl.addWidget(fab)
        
        row += 1
        
        pl = QLabel('Platform:')
        pl.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        layout.addWidget(pl, row, 1)
        from platform import platform
        self.platform = p = QLineEdit(platform())
        p.setReadOnly(True)
        layout.addWidget(p, row, 2)
        row += 1
        
        vl = QLabel('ChimeraX Version:')
        vl.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        layout.addWidget(vl, row, 1)
        self.version = v = QLineEdit(self.chimerax_version())
        v.setReadOnly(True)
        layout.addWidget(v, row, 2)
        row += 1
        
        il = QWidget()
        layout.addWidget(il, row, 2)
        ilayout = QHBoxLayout(il)
        ilayout.setContentsMargins(0,0,0,0)
        self.include_log = ilc = QCheckBox()
        ilc.setChecked(True)
        ilayout.addWidget(ilc)
        ill = QLabel('Include log contents in bug report')
        ill.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
        ilayout.addWidget(ill)
        ilayout.addStretch(1)
        row += 1

        self.result = rl = QLabel('')
        rl.setWordWrap(True)
        layout.addWidget(rl, row, 1, 1, 2)
        row += 1

        # Button row
        brow = QWidget()
        blayout = QHBoxLayout()
        blayout.setContentsMargins(0,0,0,0)
        blayout.setSpacing(2)
        brow.setLayout(blayout)
        layout.addWidget(brow, row, 1, 1, 2)
        row += 1

        blayout.addStretch(1)    # Extra space at start of button row.
        
        self.submit_button = sb = QPushButton('Submit', brow)
        sb.clicked.connect(lambda e: self.submit())
        blayout.addWidget(sb)
        
        self.cancel_button = cb = QPushButton('Cancel', brow)
        cb.clicked.connect(lambda e: self.cancel())
        blayout.addWidget(cb)

        tw.manage(placement=None)  # Don't dock it to main window.

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def set_description(self, text):
        self.description.setText(text)

    def submit(self):

        entry_values = self.entry_values()

	# Include log contents in description
        if self.include_log.isChecked():
            from chimerax.log.cmd import get_singleton
            log = get_singleton(self._ses)
            if log:
                log_text = "\n\nLog:\n%s\n" % log.plain_text()
                entry_values['description'] += log_text

        # Include info field in description
        info = entry_values['info']
        if info:
            entry_values['description'] += "\n\n" + info
        del entry_values['info']

        # Set form data
        from .form_fields import hidden_attrs
        my_attrs = hidden_attrs.copy()
        for hk, hv in my_attrs.items():
            for k, v in entry_values.items():
                if k.upper() + "_VAL" == hv:
                    my_attrs[hk] = v

        # Add attachment to form data.
        file_list = self.read_attachment(entry_values['filename'])
        fields = [(k, None, v) for k,v in my_attrs.items()]
        fields.extend(file_list)

        # Post form data.
        self.status("Contacting CGL....", color="blue")
        from .post_form import post_multipart_formdata
        try:
            errcode, errmsg, headers, body = post_multipart_formdata(BUG_HOST, BUG_SELECTOR, fields)
        except Exception:
            self._ses.logger.warning('Failed to send bug report. Error while sending follows:')
            import traceback
            traceback.print_exc()	# Log detailed exception info
            self.report_failure()
            return

        # Report success or error.
        if int(errcode) == 200:
            self.report_success()
            self.cancel_button.setText("Close")
            self.submit_button.deleteLater()	# Prevent second submission
            s = self.settings
            s.contact_name = self.contact_name.text()
            s.email_address = self.email_address.text()
        else:
            self.report_failure()

    def read_attachment(self, file_path):
        if file_path:
            import os, os.path
            if not os.path.isfile(file_path):
                self.status("Couldn't locate file '%s'."
                            "  Please choose a valid file."
                            % os.path.split(file_path)[1], color='red')
                return
            try:
                file_content = open(file_path, 'rb').read()
            except IOError as what:
                error_msg = "Couldn't read file '%s' (%s)" % (
                    os.path.split(file_path)[1],
                    os.strerror(what.errno))
                self.status(error_msg, color='red')
                return
            from .form_fields import FILE_FIELD
            file_list = [(FILE_FIELD, os.path.split(file_path)[1], file_content)]
        else:
            file_list = []
        return file_list

    def status(self, text, color = None):
        if color is not None:
            text = '<font color="%s">%s</font>' % (color, text)
        self.result.setText(text)
        
    def report_success(self):
        thanks = ("<font color=blue><h3>Thank you for your report.</h3></font>"
                "<p>Your report will be evaluated by a Chimera developer"
                " and if you provided an e-mail address,"
                " then you will be contacted with a report status.")
        self.result.setText(thanks)

    def report_failure(self):
        oops = ("<font color=red><h3>Error while submitting feedback.</h3></font>"
                "<p>An error occured when trying to submit your feedback."
                "  No information was received by the Computer Graphics Lab."
                "  This could be due to network problems, but more likely,"
                " there is a problem with Computer Graphics Lab's server."
                "  Please report this problem by sending email to"
                " <font color=\"blue\">chimerax-bugs@cgl.ucsf.edu</font>"
                " and paste a copy of the ChimeraX log into the email.</p>"
                "<p>We apologize for any inconvenience, and do appreciate"
                " you taking the time to provide us with valuable feedback.")
        self.result.setText(oops)

    def cancel(self):
        self.delete()

    def file_browse(self):
        from PyQt5.QtWidgets import QFileDialog
        path,type = QFileDialog.getOpenFileName()
        if path:
            self.attachment.setText(path)

    def opengl_info(self):
        r = self._ses.main_view.render
        try:
            r.make_current()
            lines = ['OpenGL version: ' + r.opengl_version(),
                     'OpenGL renderer: ' + r.opengl_renderer(),
                     'OpenGL vendor: ' + r.opengl_vendor()]
            r.done_current()
        except:
            lines = ['OpenGL version: unknown',
                     'Could not make opengl context current']
        return '\n'.join(lines)

    def chimerax_version(self):
        from chimerax.core import buildinfo
        from chimerax import app_dirs as ad
        return '%s (%s)' % (ad.version, buildinfo.date.split()[0])

    def entry_values(self):
        values = {
            'name': self.contact_name.text(),
            'email': self.email_address.text(),
            'description': self.description.toPlainText(),
            'info': self.gathered_info.toPlainText(),
            'filename': self.attachment.text(),
            'platform': self.platform.text(),
            'version': self.version.text()
        }
        return values

def show_bug_reporter(session):
    tool_name = 'Bug Reporter'
    tool = BugReporter(session, tool_name)
    return tool

def add_help_menu_entry(session):
    ui = session.ui
    if ui.is_gui:
        def main_window_created(tname, tdata):
            mw = ui.main_window
            mw.add_menu_entry(['Help'], 'Report a Bug', lambda: show_bug_reporter(session),
                insertion_point = "Contact Us")
            return "delete handler"
    
        ui.triggers.add_handler('ready', main_window_created)
