# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.tools import ToolInstance


class SampleTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SKIP = True         # No session saving for now
    CUSTOM_SCHEME = "sample"    # HTML scheme for custom links
    display_name = "Sample Tool"

    def __init__(self, session, tool_name):
        # Standard template stuff for intializing tool
        super().__init__(session, tool_name)
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self.tool_window.manage(placement="side")
        parent = self.tool_window.ui_area

        # Create an HTML viewer for our user interface.
        # We can include other Qt widgets if we want to.
        from PyQt5.QtWidgets import QGridLayout
        from chimerax.core.ui.widgets import HtmlView
        layout = QGridLayout()
        self.html_view = HtmlView(parent, size_hint=(575, 200),
                                  interceptor=self._navigate,
                                  schemes=[self.CUSTOM_SCHEME])
        layout.addWidget(self.html_view, 0, 0)  # row 0, column 0
        parent.setLayout(layout)

        # Register for model addition/removal so we can update model list
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        t = session.triggers
        self._add_handler = t.add_handler(ADD_MODELS, self._update_models)
        self._remove_handler = t.add_handler(REMOVE_MODELS, self._update_models)

        # Go!
        self._update_models()

    def _update_models(self, trigger=None, trigger_data=None):
        # Called to update page with current list of models
        from chimerax.core.atomic import AtomicStructure
        html = ["<h2>Sample Tool</h2>", "<ul>"]
        from urllib.parse import quote
        for m in self.session.models.list(type=AtomicStructure):
            html.append("<li><a href=\"%s:%s\">%s - %s</li>" %
                        (self.CUSTOM_SCHEME, quote(m.atomspec()),
                         m.id_string(), m.name))
        html.append("</ul>")
        self.html_view.setHtml('\n'.join(html))

    def _navigate(self, info):
        # Called when link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        url = info.requestUrl()
        scheme = url.scheme()
        if scheme == self.CUSTOM_SCHEME:
            # Intercept our custom scheme.
            # Method may be invoked in a different thread than
            # the main thread where Qt calls may be made.
            self.session.ui.thread_safe(self._run, url.path())

    def _run(self, atomspec):
        # Execute "sample count" command for given atomspec
        from chimerax.core.commands import run
        run(self.session, "sample count " + atomspec)
