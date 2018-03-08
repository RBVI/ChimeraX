# vim: set expandtab shiftwidth=4 softtabstop=4:
from chimerax.ui import HtmlToolInstance


class TargetsTool(HtmlToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True

    CUSTOM_SCHEME = "tgttable"

    name = "User-defined Specifier Targets"
    help = "help:user/tools/targets.html"

    def __init__(self, session, tool_name, all, log_errors=False):
        super().__init__(session, tool_name, size_hint=(575,400),
                         log_errors=log_errors)
        self._show_all = all
        self._html_state = None
        self._loaded_page = False
        self._handlers = None
        self.setup_page("targets.html")

    def setup(self, html_state=None):
        self._html_state = html_state
        try:
            self._setup()
        except ValueError as e:
            self.delete()
            raise

    def _setup(self):
        #
        # Register for updates of target register/deregister events
        #
        session = self.session
        from chimerax.core import triggers
        from chimerax.core.commands import (ATOMSPEC_TARGET_REGISTERED,
                                            ATOMSPEC_TARGET_DEREGISTERED)
        t1 = triggers.add_handler(ATOMSPEC_TARGET_REGISTERED,
                                  self.update_targets)
        t2 = triggers.add_handler(ATOMSPEC_TARGET_DEREGISTERED,
                                  self.update_targets)
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
            self.update_targets()
            self.update_models()
            self._set_html_state()
            self.html_view.loadFinished.disconnect(self._load_finished)

    def delete(self):
        if self._handlers:
            from chimerax.core import triggers
            for h in self._handlers:
                triggers.remove_handler(h)
            self._handlers = None
        super().delete()

    def update_models(self, trigger=None, trigger_data=None):
        models = []
        from chimerax.core.models import Model
        from chimerax.core.atomic import AtomicStructure
        model_components = {}
        composite_models = {}
        for m in self.session.models.list():
            if isinstance(m, AtomicStructure):
                model_components[m] = getattr(m, "chains")
            else:
                composite_models[m] = m.child_models()
        data = []
        for m in sorted(model_components.keys()):
            chains = model_components[m]
            if len(chains) == 0:
                data.append({"type":"Model", "atomspec":m.atomspec()})
            else:
                for c in chains:
                    data.append({"type":"Chain", "atomspec":c.atomspec()})
        import json
        model_data = json.dumps(data)
        js = "%s.update_components(%s);" % (self.CUSTOM_SCHEME, model_data)
        self.html_view.runJavaScript(js)

    def update_targets(self, trigger=None, trigger_data=None):
        from .cmd import target_list
        targets = target_list(self.session, all=self._show_all, log=False)
        data = []
        for name in sorted(targets.keys()):
            data.append({"name": name, "info": targets[name]})
        import json
        target_data = json.dumps(data)
        js = "%s.update_targets(%s);" % (self.CUSTOM_SCHEME, target_data)
        self.html_view.runJavaScript(js)

    def handle_scheme(self, url):
        # Called when custom link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        from urllib.parse import parse_qs
        method = getattr(self, "_cb_" + url.path())
        query = parse_qs(url.query())
        method(query)

    def _cb_show_hide(self, query):
        """Shows or hides target"""
        # print("cb_show_hide", query)
        action = query["action"][0];
        selector = query["selector"][0];
        cmd = "%s %s" % (action, selector)
        from chimerax.core.commands import run
        run(self.session, cmd);

    def _cb_color(self, query):
        """Colors target"""
        # print("cb_color", query)
        color = query["color"][0];
        target = query["target"][0];
        selector = query["selector"][0];
        cmd = "color %s %s target %s" % (selector, color, target)
        from chimerax.core.commands import run
        run(self.session, cmd);

    def _cb_select(self, query):
        """Select target"""
        # print("cb_select", query)
        selector = query["selector"][0];
        cmd = "select %s" % selector
        from chimerax.core.commands import run
        run(self.session, cmd);

    def _cb_builtin(self, query):
        """shows only selected structure"""
        self._show_all = query["show"][0] == "true";
        self.update_targets()

    # Session stuff

    html_state = "_html_state"

    def take_snapshot(self, session, flags):
        data = {
            "_super": super().take_snapshot(session, flags),
            "_show_all": self._show_all,
        }
        self.add_webview_state(data)
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data["_super"])
        inst._show_all = data["_show_all"]
        inst.setup(data.get(cls.html_state, None))
        return inst

    def add_webview_state(self, data):
        # Add webview state to data dictionary, synchronously.
        #
        # You have got to be kidding me - Johnny Mac
        # JavaScript callbacks are executed asynchronously,
        # and it looks like (in Qt 5.9) it is handled as
        # part of event processing.  So we cannot simply
        # use a semaphore and wait for the callback to
        # happen, since it will never happen because we
        # are not processing events.  So we use a busy
        # wait until the data we expect to get shows up.
        # Using a semaphore is overkill, since we can just
        # check for the presence of the key to be added,
        # but it does generalize if we want to call other
        # JS functions and get the value back synchronously.
        from PyQt5.QtCore import QEventLoop
        from threading import Semaphore
        event_loop = QEventLoop()
        js = "%s.get_state();" % self.CUSTOM_SCHEME
        def add(state):
            data[self.html_state] = state
            event_loop.quit()
        self.html_view.runJavaScript(js, add)
        while self.html_state not in data:
            event_loop.exec_()

    def _set_html_state(self):
        if self._html_state:
            import json
            js = "%s.set_state(%s);" % (self.CUSTOM_SCHEME,
                                        json.dumps(self._html_state))
            self.html_view.runJavaScript(js)
            self._html_state = None
