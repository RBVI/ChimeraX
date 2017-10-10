# vim: set expandtab shiftwidth=4 softtabstop=4:

from ..tools import ToolInstance


class HtmlToolInstance(ToolInstance):

    def __init__(self, session, tool_name, size_hint=None):
        from PyQt5.QtWidgets import QGridLayout
        from ..models import ADD_MODELS, REMOVE_MODELS
        from .gui import MainToolWindow
        from .widgets import HtmlView

        # ChimeraX tool instance setup
        super().__init__(session, tool_name)
        self.tool_window = MainToolWindow(self)
        self.tool_window.manage(placement="side")
        parent = self.tool_window.ui_area

        # Check if class wants to handle custom scheme
        kw = {}
        if size_hint is not None:
            kw["size_hint"] = size_hint
        try:
            scheme = getattr(self, "CUSTOM_SCHEME")
            handle_scheme = getattr(self, "handle_scheme")
        except AttributeError:
            pass
        else:
            kw["interceptor"] = self._navigate
            self.__schemes = scheme if isinstance(scheme, list) else [scheme]
            kw["schemes"] = self.__schemes

        # GUI (Qt) setup
        layout = QGridLayout()
        self.html_view = HtmlView(parent, **kw)
        layout.addWidget(self.html_view, 0, 0)
        parent.setLayout(layout)

        # If class implements "update_models", register for
        # model addition/removal
        try:
            update_models = getattr(self, "update_models")
        except AttributeError:
            self._add_handler = None
            self._remove_handler = None
        else:
            t = session.triggers
            self._add_handler = t.add_handler(ADD_MODELS, update_models)
            self._remove_handler = t.add_handler(REMOVE_MODELS, update_models)

    def delete(self):
        t = self.session.triggers
        if self._add_handler:
            t.remove_handler(self._add_handler)
            self._add_handler = None
        if self._remove_handler:
            t.remove_handler(self._remove_handler)
            self._remove_handler = None
        super().delete()

    def _navigate(self, info):
        # Called when link is clicked
        # "info" is an instance of QWebEngineUrlRequestInfo
        url = info.requestUrl()
        if url.scheme() in self.__schemes:
            # Intercept our custom schemes and call the handler
            # method in the main thread where it can make UI calls.
            self.session.ui.thread_safe(self.handle_scheme, url)
