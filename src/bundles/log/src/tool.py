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
    help = "help:user/tools/log.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from .settings import settings
        self.settings = settings
        self.suppress_scroll = False
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
                    log_window.page().triggerAction(log_window.page().Copy))
                menu.addAction("Select All", lambda:
                    log_window.page().triggerAction(log_window.page().SelectAll))
                from PyQt5.QtWidgets import QAction
                link_action = QAction("Executable Command Links", menu)
                link_action.setCheckable(True)
                link_action.setChecked(self.tool_instance.settings.exec_cmd_links)
                link_action.triggered.connect(self.tool_instance.cm_set_cmd_links)
                menu.addAction(link_action)
        self.tool_window = LogToolWindow(self, close_destroys = False)

        parent = self.tool_window.ui_area
        from chimerax.ui.widgets import ChimeraXHtmlView

        from PyQt5.QtWebEngineWidgets import QWebEnginePage
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
                super().__init__(session, parent, size_hint=(575, 500), tool_window=log.tool_window)
                page = MyPage(self._profile, self)
                self.setPage(page)
                s = page.settings()
                s.setAttribute(s.LocalStorageEnabled, True)
                self.log = log
                # as of Qt 5.6.0, the keyboard shortcut for copying text
                # from the QWebEngineView did nothing on Mac, the below
                # gets it to work
                import sys
                if sys.platform == "darwin":
                    from PyQt5.QtGui import QKeySequence
                    from PyQt5.QtWidgets import QShortcut
                    self.copy_sc = QShortcut(QKeySequence.Copy, self)
                    self.copy_sc.activated.connect(
                        lambda: self.page().triggerAction(self.page().Copy))
                ## The below three lines shoule be sufficent to allow the ui_area
                ## to Handle the context menu, but apparently not for QWebView widgets,
                ## so we define contextMenuEvent as a workaround.
                # defer context menu to parent
                #from PyQt5.QtCore import Qt
                #self.setContextMenuPolicy(Qt.NoContextMenu)

            def link_clicked(self, request_info, *args, **kw):
                # for #2289, don't scroll log when a link in it is clicked
                qurl = request_info.requestUrl()
                scheme = qurl.scheme()
                if scheme == 'data':
                    # fix #2303, spurious link_clicked
                    return
                if scheme == 'cxcmd':
                    cmd = qurl.url(qurl.None_)[6:].lstrip()  # skip cxcmd:
                    self.log.suppress_scroll = cmd and (
                            cmd.split(maxsplit=1)[0] not in ('log', 'echo'))
                super().link_clicked(request_info, *args, **kw)
                if not self.log.suppress_scroll:
                    return
                def defer(log_tool):
                    log_tool.suppress_scroll = False
                # clicked link is executed via thread_safe, so add another
                # that is executed after that one
                self.session.ui.thread_safe(defer, self.log)


            ## Moved into ui/widgets/htmlview.py
            ## def contextMenuEvent(self, event):
            ##     # kludge to allow QWebView to show our context menu (see comment above)
            ##     self.log.tool_window._show_context_menu(event)

            def cm_save(self):
                from chimerax.ui.open_save import export_file_filter, SaveDialog
                from chimerax.core.io import format_from_name
                fmt = format_from_name("HTML")
                ext = fmt.extensions[0]
                save_dialog = SaveDialog(self, "Save Log",
                                         name_filter=export_file_filter(format_name="HTML"),
                                         add_extension=ext)
                if not save_dialog.exec():
                    return
                filename = save_dialog.selectedFiles()[0]
                if not filename:
                    from chimerax.core.errors import UserError
                    raise UserError("No file specified for save log contents")
                self.log.save(filename)

        self.log_window = lw = HtmlWindow(session, parent, self)
        from PyQt5.QtWidgets import QGridLayout, QErrorMessage
        class BiggerErrorDialog(QErrorMessage):
            def sizeHint(self):
                from PyQt5.QtCore import QSize
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
        from PyQt5.QtWidgets import QGridLayout, QPushButton, QHBoxLayout
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
            self.page_source += image_info_to_html(msg, image_info)
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

            self.page_source += msg
        self.show_page_source()
        return True

    def show_page_source(self):
        self.session.ui.thread_safe(self._show)

    def _show(self):
        if self.suppress_scroll:
            sp = self.log_window.page().scrollPosition()
            height = str(sp.y())
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
        import html2text
        h = html2text.HTML2Text()
        h.unicode_snob = True
        h.ignore_links = True
        h.ignore_emphasis = True
        # html2text doesn't understand css style display:None
        # so remove "duplicate" of command and add leading '> '
        import lxml.html
        html = lxml.html.fromstring(self.page_source)
        for node in html.find_class("cxcmd"):
            for child in node:
                if (child.tag != 'div' or child.attrib.get('class', None)
                        not in (None, 'cxcmd_as_cmd')):
                    node.remove(child)
                    continue
                child.text = '> '
                break
        src = lxml.html.tostring(html, encoding='unicode')
        return h.handle(src)

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
        self.session.logger.remove_log(self)
        super().delete()

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, Log, 'Log')
