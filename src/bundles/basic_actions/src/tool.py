# vim: set expandtab shiftwidth=4 softtabstop=4:
from chimerax.ui import HtmlToolInstance


class BasicActionsTool(HtmlToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True

    CUSTOM_SCHEME = "basicactions"

    name = "Basic Actions"
    help = "help:user/tools/basicactions.html"

    def __init__(self, session, tool_name, log_errors=True):
        super().__init__(session, tool_name, size_hint=(575,400),
                         log_errors=log_errors)
        self._show_all = False
        self._hide_nonmatching = False
        self._nonmatching = {}
        self._html_state = None
        self._loaded_page = False
        self._handlers = None
        self._updating_names = False
        self.setup_page("basic_actions.html")

    def setup(self, html_state=None):
        self._html_state = html_state
        try:
            self._setup()
        except ValueError as e:
            self.delete()
            raise

    def _setup(self):
        #
        # Register for updates of name register/deregister events
        #
        session = self.session
        from chimerax.core.toolshed import get_toolshed
        ts = get_toolshed()
        t1 = ts.triggers.add_handler("selector registered", self.update_names)
        t2 = ts.triggers.add_handler("selector deregistered", self.update_names)
        self._handlers = (t1, t2)

    def setup_page(self, html_file):
        import os.path
        dir_path = os.path.dirname(__file__)
        template_path = os.path.join(os.path.dirname(__file__), html_file)
        with open(template_path, "r") as f:
            template = f.read()
        from Qt.QtCore import QUrl
        qurl = QUrl.fromLocalFile(template_path)
        output = template.replace("URLBASE", qurl.url())
        self.html_view.setHtml(output, qurl)
        self.html_view.loadFinished.connect(self._load_finished)

    def _load_finished(self, success):
        # First time through, we need to wait for the page to load
        # before trying to update data.  Afterwards, we don't care.
        if success:
            self._loaded_page = True
            self.update_models(force=True)
            self._set_html_state()
            self.html_view.loadFinished.disconnect(self._load_finished)

    def delete(self):
        if self._handlers:
            from chimerax.core.toolshed import get_toolshed
            ts = get_toolshed()
            if ts:
                for h in self._handlers:
                    ts.triggers.remove_handler(h)
            self._handlers = None
        super().delete()

    def update_models(self, trigger=None, trigger_data=None, force=False):
        if not self._loaded_page:
            return
        self._nonmatching = {}
        models = []
        from chimerax.core.models import Model
        from chimerax.atomic import AtomicStructure
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
                data.append({"type":"Model", "atomspec":m.atomspec})
            else:
                for c in chains:
                    data.append({"type":"Chain", "atomspec":c.atomspec})
        import json
        model_data = json.dumps(data)
        js = "%s.update_components(%s);" % (self.CUSTOM_SCHEME, model_data)
        self.html_view.runJavaScript(js)
        self.update_names()

    def update_names(self, trigger=None, trigger_data=None):
        if not self._loaded_page:
            return
        if trigger is not None and self._updating_names:
            # We are in the middle of an update of specifier names,
            # so this must be called when the first time a
            # selector was used and is being reregistered with
            # the real function.
            # Just ignore and let the update continue
            return
        self._updating_names = True
        from .cmd import name_list
        from chimerax.core.commands import (is_selector_atomic,
                                            is_selector_user_defined)
        names = name_list(self.session, builtins=self._show_all, log=False)
        data = []
        for name in sorted(names.keys()):
            if not is_selector_atomic(name):
                continue
            if self._hide_nonmatching and self._is_nonmatching(name):
                continue
            data.append({"name": name,
                         "info": names[name],
                         "builtin": not is_selector_user_defined(name)})
        import json
        name_data = json.dumps(data)
        js = "%s.update_names(%s);" % (self.CUSTOM_SCHEME, name_data)
        self.html_view.runJavaScript(js)
        self._updating_names = False

    def _is_nonmatching(self, name):
        try:
            return self._nonmatching[name]
        except KeyError:
            from chimerax.core.commands import get_selector, is_selector_atomic
            from chimerax.core.objects import Objects
            sel = get_selector(name)
            if callable(sel):
                objs = Objects()
                sel(self.session, self.session.models.list(), objs)
                nonmatching = objs.num_atoms == 0 and objs.num_pseudobonds == 0
            elif isinstance(sel, Objects):
                nonmatching = sel.num_atoms == 0 and sel.num_pseudobonds == 0
            else:
                # Do not know, so assume it matches something
                nonmatching = is_selector_atomic(name)
            self._nonmatching[name] = nonmatching
            return nonmatching

    def handle_scheme(self, url):
        # Called when custom link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        from urllib.parse import parse_qs
        method = getattr(self, "_cb_" + url.path())
        query = parse_qs(url.query())
        method(query)

    def _cb_show_hide(self, query):
        """Shows or hides names"""
        # print("cb_show_hide", query)
        action = query["action"][0]
        target = query["target"][0]
        try:
            selector = query["selector"][0]
        except KeyError:
            # Happens when chain id is blank
            selector = ""
        cmd = "%s %s target %s" % (action, selector, target)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _cb_color(self, query):
        """Colors names"""
        # print("cb_color", query)
        color = query["color"][0]
        target = query["target"][0]
        try:
            selector = query["selector"][0]
        except KeyError:
            # Happens when chain id is blank
            selector = ""
        cmd = "color %s %s target %s" % (selector, color, target)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _cb_select(self, query):
        """Select names"""
        # print("cb_select", query)
        try:
            selector = query["selector"][0]
        except KeyError:
            # Happens when chain id is blank
            selector = ""
        self.session.ui.main_window.select_by_mode(selector)
        """
        cmd = "select %s" % selector
        from chimerax.core.commands import run
        run(self.session, cmd)
        """

    def _cb_builtin(self, query):
        """shows builtin names"""
        self._show_all = query["show"][0] == "true"
        self.update_names()

    def _cb_nonmatching(self, query):
        """hide names with no matching items"""
        self._hide_nonmatching = query["hide"][0] == "true"
        self.update_names()

    # Session stuff

    html_state = "_html_state"

    def take_snapshot(self, session, flags):
        data = {
            "_super": super().take_snapshot(session, flags),
            "_show_all": self._show_all,
            "_hide_nonmatching": self._hide_nonmatching,
        }
        self.add_webview_state(data)
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data["_super"])
        inst._show_all = data["_show_all"]
        inst._hide_nonmatching = data["_hide_nonmatching"]
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
        from Qt.QtCore import QEventLoop
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
