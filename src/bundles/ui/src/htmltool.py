# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.tools import ToolInstance


class HtmlToolInstance(ToolInstance):
    """Base class for creating a ChimeraX tool using HTML
    as its main user interface.  `HtmlToolInstance` takes care of
    creating the ChimeraX tool instance, main window, and HTML
    widget.  Derived classes can also define methods that are
    called when hyperlinks in the HTML widget are clicked, or
    when ChimeraX models are added or removed.

    The :py:attr:`tool_window` instance attribute refers to the
    :py:class:`~chimerax.ui.gui.MainToolWindow` instance
    for the tool.

    The :py:attr:`html_view` instance attribute refers to the
    :py:class:`~chimerax.ui.widgets.htmlview.HtmlView` instance
    for managing HTML content and link actions.

    To facilitate customizing the HTML view, if the derived class
    has an attribute :py:attr:`CUSTOM_SCHEME` and a method
    :py:meth:`handle_scheme`, then the `HtmlView` instance will
    be configured to support the custom scheme.

    If the `HtmlToolInstance` has a method :py:meth:`update_models`,
    then it will be called as a handler to model addition and
    removal events.  :py:meth:`update_models` should take three
    arguments: `self`, `trigger_name` and `trigger_data`.
    `trigger_name` is a string and `trigger_data` is a list of
    models added or removed.

    Parameters
    ----------
    session : a :py:class:`~chimerax.core.session.Session` instance
        The session in which the tool is created.
    tool_name : a string
        The name of the tool being created.
    size_hint : a 2-tuple of integers, or ''None''
        The suggested initial widget size in pixels.
    """
    PLACEMENT = "side"

    def __init__(self, session, tool_name, size_hint=None,
                 show_http_in_help=True, log_errors=False):
        from Qt.QtWidgets import QGridLayout
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        from . import MainToolWindow
        from .widgets import HtmlView

        # ChimeraX tool instance setup
        super().__init__(session, tool_name)
        self.tool_window = MainToolWindow(self)
        self.tool_window.manage(placement=self.PLACEMENT)
        parent = self.tool_window.ui_area

        # Check if class wants to handle custom scheme
        kw = {"tool_window": self.tool_window,
              "log_errors": log_errors}
        if size_hint is not None:
            kw["size_hint"] = size_hint
        self.__show_http_in_help = show_http_in_help
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
        layout.setContentsMargins(0, 0, 0, 0)
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
        """Supported API. Delete this HtmlToolInstance."""
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
        scheme = url.scheme()
        if scheme in self.__schemes:
            # Intercept our custom schemes and call the handler
            # method in the main thread where it can make UI calls.
            self.session.ui.thread_safe(self.handle_scheme, url)
        if self.__show_http_in_help and scheme in ["http", "https"]:
            self.html_view.stop()
            from chimerax.help_viewer import show_url
            self.session.ui.thread_safe(show_url, self.session,
                                        url.toString(), new_tab=True)
