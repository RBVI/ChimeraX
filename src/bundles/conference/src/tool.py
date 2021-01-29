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

from chimerax.ui import HtmlToolInstance
import sys


_conf_tool = None


def find(session):
    global _conf_tool
    if _conf_tool is None:
        _conf_tool = ConferenceUI(session)
    return _conf_tool


class ConferenceUI(HtmlToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    CUSTOM_SCHEME = "conference"

    help = "help:user/tools/conference.html"

    def __init__(self, session, tool_name="Conference Call"):
        # ``session`` - ``chimerax.core.session.Session`` instance
        from . import conference
        conference.setup_triggers(session)

        # Initialize base class.  ``size_hint`` is the suggested
        # initial tool size in pixels.  For debugging, add
        # "log_errors=True" to get Javascript errors logged
        # to the ChimeraX log window.
        super().__init__(session, tool_name, size_hint=(600, 430),
                         log_errors=True)
        self._initialized = False
        self._handlers = None
        self._build_ui()

    def delete(self):
        if self._handlers is not None:
            triggers = self.session.triggers
            for h in self._handlers:
                triggers.remove_handler(h)
            self._handlers = None
        super().delete()

    def _build_ui(self):
        # Fill in html viewer with initial page in the module
        import os.path
        html_file = os.path.join(os.path.dirname(__file__), "gui.html")
        import pathlib
        self.html_view.setUrl(pathlib.Path(html_file).as_uri())

    def handle_scheme(self, url):
        # Called when GUI sets browser URL location.
        # ``url`` - ``Qt.QtCore.QUrl`` instance

        # First check that the path is a real command
        command = url.path()
        if command == "initialize":
            self.initialize()
        elif command == "action":
            from urllib.parse import parse_qs
            query = parse_qs(url.query())
            action = query["action"][0]
            if action == "join":
                self.action_join(query)
            elif action == "start":
                self.action_start(query)
            elif action == "host":
                self.action_host(query)
            elif action == "send":
                self.action_send(query)
            elif action == "leave":
                self.action_leave(query)
        else:
            from chimerax.core.errors import UserError
            raise UserError("unknown conference action: %s" % command)

    #
    # Initialize after GUI is ready
    #

    def initialize(self):
        self._initialized = True
        self.update_parameters()
        from .conference import (TRIGGER_JOINED, TRIGGER_DEPARTED,
                                 TRIGGER_CONNECTED, TRIGGER_DISCONNECTED,
                                 TRIGGER_BEFORE_RESTORE, TRIGGER_AFTER_RESTORE)
        triggers = self.session.triggers
        handlers = []
        handlers.append(triggers.add_handler(TRIGGER_CONNECTED,
                                             self._connected_cb))
        handlers.append(triggers.add_handler(TRIGGER_DISCONNECTED,
                                             self._disconnected_cb))
        handlers.append(triggers.add_handler(TRIGGER_JOINED,
                                             self._joined_cb))
        handlers.append(triggers.add_handler(TRIGGER_DEPARTED,
                                             self._departed_cb))
        handlers.append(triggers.add_handler(TRIGGER_BEFORE_RESTORE,
                                             self._before_restore_cb))
        handlers.append(triggers.add_handler(TRIGGER_AFTER_RESTORE,
                                             self._after_restore_cb))
        self._handlers = handlers

    def update_parameters(self):
        from . import conference
        server = conference.conference_server(self.session)
        data = self._get_params(server)
        import json
        js = "update_params(%s);" % json.dumps(data)
        self.html_view.runJavaScript(js)

    def _get_params(self, server):
        params = None
        if server is not None:
            params = server.parameters()
        if params is None:
            # No active conference
            from . import conference
            prefs = conference.settings(self.session)
            params = (prefs.host_name, prefs.port,
                      prefs.conf_name, prefs.user_name)
        data = {}
        data["host"] = params[0]
        data["port"] = str(params[1])
        if data["port"] == "0":
            data["port"] = ""
        data["conf_name"] = params[2]
        if data["conf_name"] == "unnamed":
            data["conf_name"] = ""
        data["user_name"] = params[3]
        return data

    def update_participants(self, trigger=None, trigger_data=None):
        if not self._initialized:
            return
        from . import conference
        server = conference.conference_server(self.session)
        if server is None:
            # No active conference
            return
        def cb(status, data):
            from . import mux
            if status != mux.Resp.Success:
                raise RuntimeError(data)
            plist = []
            for ident in data:
                # ident is a name-source 2-tuple
                plist.append("%s [%s]" % ident)
            msg = "Conference participants: %s" % ", ".join(plist)
            import json
            js = "message(%s);" % json.dumps(msg)
            self.html_view.runJavaScript(js)
        server.get_participants(cb)

    def _joined_cb(self, trigger, data):
        # data is a 2-tuple of name and source for participant
        msg = "\"%s\" [%s] joined conference" % data
        import json
        js = "message(%s);" % json.dumps(msg)
        self.html_view.runJavaScript(js)

    def _departed_cb(self, trigger, data):
        # data is a 2-tuple of name and source for participant
        msg = "\"%s\" [%s] left conference" % data
        import json
        js = "message(%s);" % json.dumps(msg)
        self.html_view.runJavaScript(js)

    def update_status(self, active, server):
        data = {"active":active}
        if active:
            data.update(self._get_params(server))
            data["location"] = server.location()
        import json
        js = "update_status(%s);" % json.dumps(data)
        self.html_view.runJavaScript(js)

    def _connected_cb(self, trigger, server):
        self.update_status(True, server)
        self.update_participants()

    def _disconnected_cb(self, trigger, server):
        self.update_status(False, server)

    def _before_restore_cb(self, trigger, server):
        self.session.tools.remove([self])

    def _after_restore_cb(self, trigger, server):
        self.session.tools.add([self])

    #
    # Code for running commands
    #

    def action_join(self, query):
        # Collect the optional parameters from URL query parameters
        # and construct a command to execute
        host, port, conf_name, user_name = self._get_args(query)
        cmd = self._build_action("join", host, port, conf_name, user_name)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def action_start(self, query):
        # Collect the optional parameters from URL query parameters
        # and construct a command to execute
        host, port, conf_name, user_name = self._get_args(query)
        cmd = self._build_action("start", host, port, conf_name, user_name)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def action_host(self, query):
        # Collect the optional parameters from URL query parameters
        # and construct a command to execute
        host, port, conf_name, user_name = self._get_args(query)
        cmd = self._build_action("host", None, None, None, user_name)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def action_send(self, query):
        cmd = self._build_action("send", None, None, None, None)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def action_leave(self, query):
        cmd = self._build_action("close", None, None, None, None)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _get_args(self, query):
        host = self._get_argument(query, "host")
        port = self._get_argument(query, "port")
        conf_name = self._get_argument(query, "conf_name")
        user_name = self._get_argument(query, "user_name")
        return host, port, conf_name, user_name

    def _get_argument(self, query, arg_name):
        try:
            arg_list = query[arg_name]
        except KeyError:
            return None
        if len(arg_list) > 1:
            from chimerax.core.errors import UserError
            raise UserError("Too many values for \"%s\"" % arg_name)
        elif len(arg_list) == 0:
            return None
        else:
            return arg_list[0]

    def _build_action(self, action, host, port, conf_name, user_name):
        cmd_text = ["conference", action]
        if host or port or conf_name:
            if not host:
                from chimerax.core.errors import UserError
                raise UserError("Host name must be specified")
            location = host
            if port:
                location += ':' + port
            if conf_name:
                location += '/' + conf_name
            cmd_text.append(location)
        if user_name:
            cmd_text.extend(["name", user_name])
        return ' '.join(cmd_text)
