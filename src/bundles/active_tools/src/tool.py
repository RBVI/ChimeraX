# vim: set expandtab shiftwidth=4 softtabstop=4:
from chimerax.ui import HtmlToolInstance


class ActiveToolsTool(HtmlToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False

    CUSTOM_SCHEME = "active"

    name = "Active Tools"
    # help = "help:user/tools/active_tools.html"

    def __init__(self, session, tool_name, log_errors=True):
        super().__init__(session, tool_name, size_hint=(575,400),
                         log_errors=log_errors)
        self._html_state = None
        self._loaded_page = False
        self._handlers = None
        self.setup_page("active_tools.html")

        from chimerax.core.tools import ADD_TOOL_INSTANCE, REMOVE_TOOL_INSTANCE
        triggers = session.triggers
        t1 = triggers.add_handler(ADD_TOOL_INSTANCE, self.update_tools)
        t2 = triggers.add_handler(REMOVE_TOOL_INSTANCE, self.update_tools)
        self._handlers = (t1, t2)

    def setup_page(self, html_file):
        import os.path
        dir_path = os.path.dirname(__file__)
        template_path = os.path.join(os.path.dirname(__file__), html_file)
        with open(template_path, "r") as f:
            template = f.read()
        from PyQt5.QtCore import QUrl
        qurl = QUrl.fromLocalFile(template_path)
        output = template.replace("URLBASE", qurl.url())
        self.html_view.setHtml(output, qurl)
        self.html_view.loadFinished.connect(self._load_finished)

    def _load_finished(self, success):
        # First time through, we need to wait for the page to load
        # before trying to update data.  Afterwards, we don't care.
        if success:
            self._loaded_page = True
            self.update_tools()
            self.html_view.loadFinished.disconnect(self._load_finished)

    def delete(self):
        if self._handlers:
            from chimerax.core import triggers
            for h in self._handlers:
                triggers.remove_handler(h)
            self._handlers = None
        super().delete()

    def update_tools(self, trigger=None, trigger_data=None):
        data = []
        for tool in self.session.tools.list():
            data.append({"name": tool.display_name,
                         "id": tool.id,
                         "killable": not tool.SESSION_ENDURING})
        import json
        tools_data = json.dumps(data)
        js = "%s.update_tools(%s);" % (self.CUSTOM_SCHEME, tools_data)
        self.html_view.runJavaScript(js)

    def handle_scheme(self, url):
        # Called when custom link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        from urllib.parse import parse_qs
        method = getattr(self, "_cb_" + url.path())
        query = parse_qs(url.query())
        method(query)

    def _cb_show_hide_close(self, query):
        """Show/hide/close tool"""
        # print("cb_show_hide", query)
        try:
            tool_id = int(query["tool"][0])
        except ValueError:
            tool_id = query["tool"][0]
        tool = self.session.tools.find_by_id(tool_id)
        if tool is None:
            print(self.session.tools._tool_instances.keys())
            raise KeyError("cannot find tool with id: %s" % tool_id)
        action = query["action"][0]
        if action == "show":
            tool.display(True)
        elif action == "hide":
            tool.display(False)
        elif action == "close":
            tool.delete()
