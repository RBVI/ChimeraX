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

context_menu_css = """
.context-menu {
    display: none;
    position: absolute;
    z-index: 100;
    border: solid 1px #000000;
    box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5);
    cursor: pointer;
}
.context-menu-items {
    list-style-type: none;
    padding: 0;
    margin: 0;
}
.context-menu-item a:link, a:visited {
    display: block;
    color: #000;
    background-color: #fff;
    opacity: 1;
    text-decoration: none;
    padding: 4px 8px;
}
.context-menu-item a:hover, a:active {
    background-color: #cccccc;
}
.context-menu-active {
    display: block;
}
"""

cxcmd_css = """
.cxcmd {
    display: block;
    font-weight: bold;
    margin-top: .5em;
    background-color: #ddd;
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

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)
        self.warning_shows_dialog = False
        self.error_shows_dialog = True
        from chimerax.core.ui.gui import MainToolWindow

        class LogWindow(MainToolWindow):
            close_destroys = False
        self.tool_window = LogWindow(self)
        parent = self.tool_window.ui_area
        from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
        class HtmlWindow(QWebEngineView):
            def __init__(self, parent, log):
                super().__init__(parent)
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

            def sizeHint(self):
                from PyQt5.QtCore import QSize
                return QSize(575, 500)

            def contextMenuEvent(self, event):
                event.accept()
                cm = getattr(self, 'context_menu', None)
                if cm is None:
                    from PyQt5.QtWidgets import QMenu
                    cm = self.context_menu = QMenu()
                    def save_image(self=self):
                        from .cmd import log
                        log(self.log.session, thumbnail=True)
                    cm.addAction("Insert image", save_image)
                    cm.addAction("Save", self.cm_save)
                    cm.addAction("Clear", self.log.clear)
                    cm.addAction("Copy selection",
                        lambda: self.page().triggerAction(self.page().Copy))
                    cm.addAction("Select all",
                        lambda: self.page().triggerAction(self.page().SelectAll))
                    cm.addAction("Help", self.log.display_help)
                cm.popup(event.globalPos())

            def cm_save(self):
                from chimerax.core.ui.open_save import export_file_filter, SaveDialog
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
        self.log_window = lw = HtmlWindow(parent, self)
        from PyQt5.QtWidgets import QGridLayout, QErrorMessage
        self.error_dialog = QErrorMessage(parent)
        layout = QGridLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.log_window, 0, 0)
        parent.setLayout(layout)
        #self.log_window.EnableHistory(False)
        self.page_source = ""
        self.tool_window.manage(placement="side")
        session.logger.add_log(self)
        #self.log_window.contextMenuEvent = self.contextMenuEvent
        #from PyQt5.QtCore import Qt
        #self.log_window.setContextMenuPolicy(Qt.CustomContextMenu)
        #self.log_window.customContextMenuRequested.connect(self.contextMenu)
        #import sys
        #print("context menu policy:", self.log_window.contextMenuPolicy(), file=sys.__stderr__)
        def link_clicked(qurl, nav_type, is_main_frame):
            self.navigate(qurl)
            return False
        lw.page().acceptNavigationRequest = link_clicked
        #self.log_window.Bind(html2.EVT_WEBVIEW_NAVIGATING, self.on_navigating,
        #                     id=self.log_window.GetId())
        # Don't record html history as log changes.
        def clear_history(okay, lw=lw):
            lw.history().clear()
        lw.loadFinished.connect(clear_history)
        self.show_page_source()

    #
    # Implement logging
    #
    def log(self, level, msg, image_info, is_html):
        """Log a message

        Parameters documented in HtmlLog base class
        """

        image, image_break = image_info
        if image:
            import io
            img_io = io.BytesIO()
            image.save(img_io, format='PNG')
            png_data = img_io.getvalue()
            import codecs
            bitmap = codecs.encode(png_data, 'base64')
            width, height = image.size
            img_src = '<img src="data:image/png;base64,%s" width=%d height=%d style="vertical-align:middle">' % (bitmap.decode('utf-8'), width, height)
            self.page_source += img_src
            if image_break:
                self.page_source += "<br>\n"
        else:
            if ((level == self.LEVEL_ERROR and self.error_shows_dialog) or
                    (level == self.LEVEL_WARNING and self.warning_shows_dialog)):
                if not is_html:
                    dlg_msg = "<br>".join(msg.split("\n"))
                self.error_dialog.showMessage(dlg_msg)
            if not is_html:
                from html import escape
                msg = escape(msg)
                msg = msg.replace("\n", "<br>\n")

            if level == self.LEVEL_ERROR:
                msg = '<p style="color:crimson;font-weight:bold">' + msg + '</p>'
            elif level == self.LEVEL_WARNING:
                msg = '<p style="color:darkorange">' + msg + '</p>'

            self.page_source += msg
        self.show_page_source()
        return True

    def show_page_source(self):
        css = context_menu_css + cxcmd_css
        html = "<style>%s</style>\n<body onload=\"window.scrollTo(0, document.body.scrollHeight);\">%s</body>" % (cxcmd_css, self.page_source)
        lw = self.log_window
        # Disable and reenable to avoid QWebEngineView taking focus, QTBUG-52999 in Qt 5.7
        lw.setEnabled(False)
        # HACK ALERT: to get around a QWebEngineView bug where HTML
        # source is converted into a "data:" link and runs into the
        # URL length limit.
        if len(html) < 1000000:
            lw.setHtml(html)
        else:
            try:
                tf = open(self._tf_name, "wb")
            except AttributeError:
                import tempfile, atexit
                tf = tempfile.NamedTemporaryFile(prefix="chtmp", suffix=".html",
                                                 delete=False, mode="wb")
                self._tf_name = tf.name
                def clean(filename):
                    import os
                    try:
                        os.remove(filename)
                    except OSError:
                        pass
                atexit.register(clean, tf.name)
            from PyQt5.QtCore import QUrl
            tf.write(bytes(html, "utf-8"))
            # On Windows, we have to close the temp file before
            # trying to open it again (like loading HTML from it).
            tf.close()
            lw.load(QUrl.fromLocalFile(self._tf_name))
        lw.setEnabled(True)

    def navigate(self, data):
        session = self.session
        # Handle event
        # data is QUrl
        url = data.toString()
        link_handled = lambda: False
        from urllib.parse import unquote
        url = unquote(url)
        link_handled()
        if url.startswith("log:"):
            link_handled()
            cmd = url.split(':', 1)[1]
            if cmd == 'help':
                self.display_help()
            elif cmd == 'clear':
                self.clear()
            elif cmd == 'copy':
                page = self.log_window.page()
                page.triggerAction(page.Copy)
            elif cmd == 'select-all':
                page = self.log_window.page()
                page.triggerAction(page.SelectAll)
            elif cmd == 'save':
                from chimerax.core.ui.open_save import export_file_filter, SaveDialog
                from chimerax.core.io import format_from_name
                fmt = format_from_name("HTML")
                ext = fmt.extensions[0]
                save_dialog = SaveDialog(self.log_window, "Save Log",
                                         name_filter=export_file_filter(format_name="HTML"),
                                         add_extension=ext)
                if not save_dialog.exec():
                    return
                filename = save_dialog.selectedFiles()[0]
                if not filename:
                    from chimerax.core.errors import UserError
                    raise UserError("No file specified for save log contents")
                self.save(filename)
            elif cmd == 'image':
                from .cmd import log
                log(self.session, thumbnail=True)
            return
        from urllib.parse import urlparse
        parts = urlparse(url)
        if parts.scheme in ('', 'cxcmd', 'help', 'file', 'http'):
            from chimerax.core.commands import run
            run(session, "help %s" % url, log=False)
            return
        # unknown scheme
        session.logger.error("Unknown URL scheme: '%s'" % parts.scheme)

    def clear(self):
        self.page_source = ""
        self.show_page_source()

    def save(self, path):
        from os.path import expanduser
        path = expanduser(path)
        f = open(path, 'w')
        f.write("<!DOCTYPE html>\n"
                "<html>\n"
                "<head>\n"
                "<title> ChimeraX Log </title>\n"
                "</head>\n"
                "<body>\n"
                "<h1> ChimeraX Log </h1>\n"
                "<style>\n"
                "%s"
                "</style>\n" % cxcmd_css)
        f.write(self.page_source)
        f.write("</body>\n"
                "</html>\n")
        f.close()

    #
    # Override ToolInstance methods
    #
    def delete(self):
        self.session.logger.remove_log(self)
        super().delete()

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, Log, 'log')
