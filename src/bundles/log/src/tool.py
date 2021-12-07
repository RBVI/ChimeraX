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

from chimerax.core.tools import ToolInstance
from chimerax.core.logger import HtmlLog

cxcmd_css = """
.cxcmd {
    display: block;
    font-weight: bold;
    margin-top: .5em;
    background-color: #ddd;
}
a.no_underline {
    text-decoration: none;
}
"""
cxcmd_as_doc_css = """
.cxcmd_as_doc {
    display: inline;
}
.cxcmd_as_cmd {
    display: none;
}
"""
cxcmd_as_cmd_css = """
.cxcmd_as_doc {
    display: none;
}
.cxcmd_as_cmd {
    display: inline;
}
"""

context_menu_html = """
<nav id="context-menu" class="context-menu">
    <ul class="context-menu-items">
        <li class="context-menu-item">
            <a href="log:image" class="context-menu-link"> Insert image </a>
        </li>
        <li class="context-menu-item">
            <a href="log:save" class="context-menu-link"> Save </a>
        </li>
        <li class="context-menu-item">
            <a href="log:clear" class="context-menu-link"> Clear </a>
        </li>
        <li>
        <li class="context-menu-item">
            <a href="log:copy" class="context-menu-link"> Copy selection </a>
        </li>
        <li class="context-menu-item">
            <a href="log:select-all" class="context-menu-link"> Select all </a>
        </li>
        <hr style="margin:0;">
        <li class="context-menu-item">
            <a href="log:help" class="context-menu-link"> Help </a>
        </li>
    </ul>
</nav>
"""

context_menu_script = """
<script>
function init_menus() {
    "use strict";

    var context_menu = document.querySelector(".context-menu");
    var context_menu_shown = false;
    var active_css = "context-menu-active";

    function show_context_menu() {
        if (!context_menu_shown) {
            context_menu_shown = true;
            context_menu.classList.add(active_css);
        }
    }

    function hide_context_menu() {
        if (context_menu_shown) {
            context_menu_shown = false;
            context_menu.classList.remove(active_css);
        }
    }

    function position_menu(menu, e) {
        var x = e.pageX;
        var y = e.pageY;

        menu.style.left = x + "px";
        menu.style.top = y + "px";
    }

    function init() {
        document.addEventListener("contextmenu", function (e) {
                e.preventDefault();
                show_context_menu();
                position_menu(context_menu, e);
        });

        document.addEventListener("click", function (e) {
                var button = e.which;
                if (button === 1)	// left button used
                        hide_context_menu();
        });

        context_menu.addEventListener("mouseleave", hide_context_menu);
        window.scrollTo(0, document.body.scrollHeight);
    }

    init();
}
</script>
"""


class Log(ToolInstance, HtmlLog):

    SESSION_ENDURING = True
    SESSION_SAVE = True
    help = "help:user/tools/log.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from .settings import settings
        self.settings = settings
        self.suppress_scroll = False
        self._log_file = None
        from chimerax.ui import MainToolWindow
        class LogToolWindow(MainToolWindow):
            def fill_context_menu(self, menu, x, y, session=session):
                def save_image(ses=session):
                    from chimerax.core.commands import run
                    run(ses, "log thumbnail")
                menu.addAction("Insert Image", save_image)
                log_window = self.tool_instance.log_window
                menu.addAction("Save As...", log_window.cm_save)
                menu.addAction("Clear", self.tool_instance.clear)
                menu.addAction("Copy Selection", lambda:
                    self.session.ui.clipboard().setText(log_window.selectedText()))
                menu.addAction("Select All", lambda:
                    log_window.page().triggerAction(log_window.page().SelectAll))
                from Qt.QtWidgets import QAction
                link_action = QAction("Executable Command Links", menu)
                link_action.setCheckable(True)
                link_action.setChecked(self.tool_instance.settings.exec_cmd_links)
                link_action.triggered.connect(self.tool_instance.cm_set_cmd_links)
                menu.addAction(link_action)
        self.tool_window = LogToolWindow(self, close_destroys = False)

        parent = self.tool_window.ui_area
        from chimerax.ui.widgets import ChimeraXHtmlView

        from Qt.QtWebEngineWidgets import QWebEnginePage
        class MyPage(QWebEnginePage):

            def acceptNavigationRequest(self, qurl, nav_type, is_main_frame):
                if qurl.scheme() in ('http', 'https'):
                    session = self.view().session
                    def show_url(url):
                        from chimerax.help_viewer import show_url
                        show_url(session, url)
                    session.ui.thread_safe(show_url, qurl.toString())
                    return False
                return True

        class HtmlWindow(ChimeraXHtmlView):

            def __init__(self, session, parent, log):
                from chimerax.ui.widgets.htmlview import create_chimerax_profile
                profile = create_chimerax_profile(parent, interceptor=self.link_intercept)
                super().__init__(session, parent, size_hint=(575, 500), tool_window=log.tool_window, profile=profile)
                page = MyPage(self._profile, self)
                self.setPage(page)
                s = page.settings()
                s.setAttribute(s.LocalStorageEnabled, True)
                self.log = log
                ## The below three lines shoule be sufficent to allow the ui_area
                ## to Handle the context menu, but apparently not for QWebView widgets,
                ## so we define contextMenuEvent as a workaround.
                # defer context menu to parent
                #from Qt.QtCore import Qt
                #self.setContextMenuPolicy(Qt.NoContextMenu)

            def link_intercept(self, request_info, *args, **kw):
                # for #2289, don't scroll log when a link in it is clicked
                qurl = request_info.requestUrl()
                scheme = qurl.scheme()
                if scheme == 'cxcmd':
                    from Qt.QtCore import QUrl
                    no_formatting = QUrl.FormattingOptions(QUrl.None_)
                    cmd = qurl.toString(no_formatting)[6:].lstrip()  # skip cxcmd:
                    self.log.suppress_scroll = cmd and (
                            cmd.split(maxsplit=1)[0] not in ('log', 'echo'))
                from chimerax.ui.widgets.htmlview import chimerax_intercept
                chimerax_intercept(request_info, *args, session=self.session, view=self, **kw)

            def cm_save(self):
                from chimerax.ui.open_save import SaveDialog
                fmt = session.data_formats["HTML"]
                save_dialog = SaveDialog(session, self, "Save Log", data_formats=[fmt])
                if not save_dialog.exec():
                    return
                filename = save_dialog.selectedFiles()[0]
                if not filename:
                    from chimerax.core.errors import UserError
                    raise UserError("No file specified for save log contents")
                self.log.save(filename)

        self.log_window = lw = HtmlWindow(session, parent, self)
        from Qt.QtCore import QTimer
        self.regulating_timer = QTimer()
        self.regulating_timer.timeout.connect(self._actually_show)

        from Qt.QtWidgets import QGridLayout, QErrorMessage
        class BiggerErrorDialog(QErrorMessage):
            def sizeHint(self):
                from Qt.QtCore import QSize
                return QSize(600, 300)
        self.error_dialog = BiggerErrorDialog(parent)
        self._add_report_bug_button()
        layout = QGridLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.log_window, 0, 0)
        parent.setLayout(layout)
        #self.log_window.EnableHistory(False)
        self.page_source = ""
        self.tool_window.manage(placement="side")
        session.logger.add_log(self)
        # Don't record html history as log changes.
        def clear_history(okay, lw=lw):
            lw.history().clear()
        lw.loadFinished.connect(clear_history)
        self.show_page_source()

    def cm_set_cmd_links(self, checked):
        self.settings.exec_cmd_links = checked
        self._show()

    def _add_report_bug_button(self):
        '''
        Add "Report a Bug" button to the error dialog.
        Unfortunately the QErrorMessage dialog being used has no API to add a button.
        So this code uses that implementation details of QErrorMessage to add the button.
        We could instead emulate QErrorMessage with a QMessageBox but it would require
        adding the queueing and "Show this message again" checkbox.
        '''
        ed = self.error_dialog
        el = ed.layout()
        from Qt.QtWidgets import QGridLayout, QPushButton, QHBoxLayout
        if isinstance(el, QGridLayout):
            i = 0
            while True:
                item = el.itemAt(i)
                if item is None or isinstance(item.widget(), QPushButton):
                    break
                i += 1
            if item is not None:
                row, col, rowspan, colspan = el.getItemPosition(i)
                w = item.widget()
                el.removeWidget(w)
                brow = QHBoxLayout()
                brow.addStretch(1)
                brow.addWidget(w)
                rb = QPushButton('Report Bug')
                rb.clicked.connect(self._report_a_bug)
                brow.addWidget(rb)
                ed.report_bug_button = rb
                el.addLayout(brow, row, col, rowspan, colspan)

    def _report_a_bug(self):
        '''Show the bug report tool.'''
        from chimerax.bug_reporter import show_bug_reporter
        show_bug_reporter(self.session)
        self.error_dialog.done(0)

    #
    # Implement logging
    #
    def log(self, level, msg, image_info, is_html):
        """Log a message

        Parameters documented in HtmlLog base class
        """

        start_len = len(self.page_source)
        if image_info[0] is not None:
            from chimerax.core.logger import image_info_to_html
            self._append_message(image_info_to_html(msg, image_info))
        else:
            from .settings import settings
            if ((level >= self.LEVEL_ERROR and settings.errors_raise_dialog) or
                    (level == self.LEVEL_WARNING and settings.warnings_raise_dialog)):
                if not is_html:
                    dlg_msg = "<br>".join(msg.split("\n"))
                else:
                    # error dialog doesn't actually handle anchor links, so they
                    # look misleadingly clickable; strip them...
                    search_text = msg
                    dlg_msg = ""
                    while '<a href=' in search_text:
                        before, partial = search_text.split('<a href=', 1)
                        dlg_msg += before
                        html, text_plus = partial.split(">", 1)
                        if '</a>' not in text_plus:
                            # can't parse link, just use original message
                            dlg_msg = ""
                            search_text = msg
                            break
                        link, search_text = text_plus.split('</a>', 1)
                        dlg_msg += link
                    dlg_msg += search_text
                if level == self.LEVEL_BUG:
                    f = lambda dlg=self.error_dialog, msg=dlg_msg: (dlg.report_bug_button.show(),
                        dlg.showMessage(msg))
                else:
                    f = lambda dlg=self.error_dialog, msg=dlg_msg: (dlg.report_bug_button.hide(),
                        dlg.showMessage(msg))
                self.session.ui.thread_safe(f)
            if not is_html:
                from html import escape
                msg = escape(msg)
                msg = msg.replace("\n", "<br>\n")

            if level == self.LEVEL_ERROR:
                from chimerax.core.logger import error_text_format
                msg = error_text_format % msg
            elif level == self.LEVEL_WARNING:
                msg = '<p style="color:darkorange">' + msg + '</p>'

            # compact repeated output, e.g. ISOLDE's 'stepto' command
            #
            # printing to stderr can produce extremely piecemeal logging (e.g. consecutive loggings of
            # a space character), so only try to compact if "<br>" or "<div" is in msg
            if ("<br>" in msg and not msg.replace("<br>", " ").isspace()) or "<div" in msg:
                compaction_start_text = "[Repeated "
                compaction_end_text = " time(s)]"
                compaction_start = None
                if self.page_source.endswith(compaction_end_text):
                    # possibly already compacting some output
                    try:
                        bracket_start = self.page_source.rindex(compaction_start_text)
                    except ValueError:
                        pass
                    else:
                        if self.page_source[bracket_start:].startswith(compaction_start_text):
                            try:
                                compaction_number = int(self.page_source[
                                    bracket_start+len(compaction_start_text):-len(compaction_end_text)])
                            except ValueError:
                                pass
                            else:
                                compaction_start = bracket_start
                if compaction_start is None:
                    test_text = self.page_source
                else:
                    test_text = self.page_source[:compaction_start]
                if test_text.endswith(msg):
                    if compaction_start is None:
                        self.page_source += compaction_start_text + "1" + compaction_end_text
                    else:
                        self.page_source = self.page_source[:compaction_start] + "%s%d%s" % (
                            compaction_start_text, compaction_number+1, compaction_end_text)
                else:
                    self._append_message(msg)
            else:
                self._append_message(msg)
        self.show_page_source()
        return True

    def _append_message(self, msg):
        self.page_source += msg
        self._log_to_file(msg)

    def _log_to_file(self, msg):
        file = self._log_file
        if file is not None:
            try:
                file.write(msg)
                file.flush()
            except Exception:
                pass

    def record_to_file(self, file):
        '''Save messages to a file for crash reporting.'''
        self._log_file = file
        self._log_to_file(self.page_source)
        
    def show_page_source(self):
        self.session.ui.thread_safe(self._show)

    def _show(self):
        # start() is documented to stop the timer if it is running
        self.regulating_timer.start(100)

    def _actually_show(self):
        self.regulating_timer.stop()
        if self.suppress_scroll:
            sp = self.log_window.page().scrollPosition()
            height = str(sp.y())
            self.suppress_scroll = False
        else:
            height = 'document.body.scrollHeight'
        html = "<style>%s%s</style>\n<body onload=\"window.scrollTo(0, %s);\">%s</body>" % (
            cxcmd_css,
            cxcmd_as_cmd_css if self.settings.exec_cmd_links else cxcmd_as_doc_css,
            height,
            self.page_source
        )
        lw = self.log_window
        lw.setHtml(html)

    def plain_text(self):
        """Convert HTML to plain text"""
        return log_html_to_plain_text(self.page_source)

    def clear(self):
        self.page_source = ""
        self.show_page_source()

    def save(self, path, *, executable_links=None):
        if executable_links is None:
            executable_links = self.settings.exec_cmd_links
        from os.path import expanduser
        path = expanduser(path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write("<html>\n"
                    "<head>\n"
                    "<meta charset='utf-8'>\n"
                    "<title> ChimeraX Log </title>\n"
                    '<script type="text/javascript">\n'
                    "%s"
                    "</script>\n"
                    "</head>\n"
                    "<h1> ChimeraX Log </h1>\n"
                    "<style>\n"
                    "%s"
                    "%s"
                    "</style>\n" % (
                        self._get_cxcmd_script(), cxcmd_css,
                        cxcmd_as_cmd_css if executable_links else cxcmd_as_doc_css,
                    )
            )
            f.write(self.page_source)
            f.write("</body>\n"
                    "</html>\n")

    def _get_cxcmd_script(self):
        try:
            return self._cxcmd_script
        except AttributeError:
            import chimerax, os.path
            fname = os.path.join(chimerax.app_data_dir, "docs", "js",
                                 "cxlinks.js")
            with open(fname) as f:
                self._cxcmd_script = f.read()
            return self._cxcmd_script

    #
    # Override ToolInstance methods
    #
    def delete(self):
        self.regulating_timer.stop()
        self.session.logger.remove_log(self)
        super().delete()

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, Log, 'Log')

    def set_state_from_snapshot(self, session, data):
        super().set_state_from_snapshot(session, data)
        log_data = data['log data']
        # don't blindly trust HTML in log_data
        date = log_data['date']
        if '<' in date:
            session.logger.warning("Potentially malicious date found in session file's log data ignored")
            date = "unknown"
        from lxml import html
        contents = log_data['contents']
        tmp = html.fromstring(contents)
        script_elements = tmp.findall(".//script")
        if script_elements:
            session.logger.warning("Potentially malicious HTML script elements found in session file's log data ignored")
            for e in reversed(script_elements):
                e.getparent().remove(e)
            contents = html.tostring(tmp)
        prev_ses_html = '<details style="background-color: #ebf5fb"><summary>Log from %s</summary>%s<p>&mdash;&mdash;&mdash; End of log from %s &mdash;&mdash;&mdash;</p></details>' % (date, contents, date)
        if self.settings.session_restore_clears and session.restore_options['clear log']:
            def clear_log_unless_error(trig_name, session, *, self=self, prev_ses_html=prev_ses_html):
                if session.restore_options['error encountered']:
                    # if the session did't restore successfully, don't clear the log
                    self._append_message(prev_ses_html)
                else:
                    # "retain" version info
                    class FakeLogger:
                        def __init__(self):
                            self.msgs = []
                        def info(self, msg):
                            self.msgs.append(msg)
                    fl = FakeLogger()
                    from chimerax.core.logger import log_version
                    log_version(fl)
                    version = "<br>".join(fl.msgs)

                    # look for the command that restored the session and include it
                    index = self.page_source.rfind('<div class="cxcmd">')
                    if index == -1:
                        retain = version
                    else:
                        retain = version + '<br>' + self.page_source[index:]
                    self.page_source = retain + prev_ses_html
                from chimerax.core.triggerset import DEREGISTER
                return DEREGISTER
            session.triggers.add_handler('end restore session', clear_log_unless_error)
        else:
            self._append_message(prev_ses_html)
        #self.show_page_source()

    def take_snapshot(self, session, flags):
        from datetime import datetime
        data = super().take_snapshot(session, flags)
        data['log data'] = {
            'version': 1,
            'date': datetime.now().ctime(),
            'contents': self.page_source,
        }
        return data

def log_html_to_plain_text(log_html_text):
    import html2text
    h = html2text.HTML2Text()
    h.unicode_snob = True
    h.ignore_links = True
    h.ignore_emphasis = True
    # html2text doesn't understand css style display:None
    # so remove "duplicate" of command and add leading '> '
    import lxml.html
    html = lxml.html.fromstring(log_html_text)
    for node in html.find_class("cxcmd"):
        cxcmd_as_doc = False
        for child in node:
            cls = child.attrib.get('class', None)
            if cls == 'cxcmd_as_doc':
                cxcmd_as_doc = True
                node.remove(child)
                continue
            if cls == 'cxcmd_as_cmd':
                child.text = '> '
                break
        if not cxcmd_as_doc:
            node.text = '> '
    src = lxml.html.tostring(html, encoding='unicode')
    log_plain_text = h.handle(src)
    return log_plain_text
