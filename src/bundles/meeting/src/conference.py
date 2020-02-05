# vim: set expandtab ts=4 sw=4:

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


from chimerax.core.errors import UserError
from chimerax.core.settings import Settings
from .tracking import MouseTracking, VRTracking
from . import mux


class _ConferenceSettings(Settings):
    AUTO_SAVE = {
        "conf_name": True,
        "user_name": True,
        "host_name": True,
        "port": True,
    }


_settings = None


def settings(session):
    global _settings
    if _settings is None:
        _settings = _ConferenceSettings(session, "Conference")
    return _settings


TRIGGER_CONNECTED = "connected to conference"
TRIGGER_DISCONNECTED = "disconnected from conference"
TRIGGER_JOINED = "participant joined conference"
TRIGGER_DEPARTED = "participant left conference"
TRIGGER_BEFORE_RESTORE = "before restoring conference session"
TRIGGER_AFTER_RESTORE = "after restoring conference session"


def setup_triggers(session):
    try:
        session.triggers.add_trigger(TRIGGER_CONNECTED)
        session.triggers.add_trigger(TRIGGER_DISCONNECTED)
        session.triggers.add_trigger(TRIGGER_JOINED)
        session.triggers.add_trigger(TRIGGER_DEPARTED)
        session.triggers.add_trigger(TRIGGER_BEFORE_RESTORE)
        session.triggers.add_trigger(TRIGGER_AFTER_RESTORE)
    except KeyError:
        # someone must have added them already
        pass


def conference(session, action, location=None, name=None, **kw):
    '''Similar to "meeting", but uses a hub that all participants
    can reach to exchange messages.

    Parameters
    ----------
    action : enumeration
      One of "start" or "join".
    location : string of form host[:port]/conference_name
      String to identify the conference location.
      'host' is the name or IP address of the host to connect to.
      'port' is the TCP port number, default 443, to connect to.
      'conference_name' is the name of the conference to start or join.
    name : string
      A unique handle for this participant in the conference.
    color : Color
      Color for my mouse pointer shown on other machines.
    face_image : string
      Path to PNG or JPG image file for image to use for VR face depiction.
    copy_scene : bool
      Whether to copy the open models from the ChimeraX that started the
      conference to other ChimeraX instances when they join the conference.
    relay_commands : bool
      Whether to have every command you run sent to the other participants
      and their commands sent to you and automatically run so changes to
      the scene are mirrored for all participants.
      Default true.
    update_interval : int
      How often VR hand and head model positions are sent for this
      ChimeraX instance in frames.  Value of 1 updates every frame.
      Default 9.
    '''
    conference = conference_server(session, create=False)
    if conference is not None:
        raise UserError("ChimeraX conference already in progress")
    conference = conference_server(session, create=True)
    try:
        conference.connect(action, location, name, **kw)
    except ConnectionRefusedError as e:
        conference.close()
        raise UserError(str(e))


def conference_set(session, color=None, face_image=None, copy_scene=None,
                   relay_commands=None, update_interval=None, announce=False):
    '''Change options for the currently active conference.

    Parameters
    ----------
    Same as the keyword arguments as ``conference``.
    '''
    mux.logger.debug("conference_set")
    conference = conference_server(session, create=False)
    if conference is None:
        raise UserError("No ChimeraX conference started")
    warnings = []
    acted = False
    if color is not None:
        conference.set_color(color.uint8x4())
        acted = True
    if face_image is not None:
        from os.path import isfile
        if isfile(face_image):
            conference.vr_tracker.new_face_image(face_image)
        else:
            warnings.append("Face image file \"%s\" does not exist" %
                            face_image)
        acted = True
    if copy_scene is not None:
        conference.copy_scene(copy_scene)
        acted = True
    if relay_commands is not None:
        if conference.connected:
            conference.relay_commands(relay_commands)
        acted = True
    if update_interval is not None:
        conference.vr_tracker.update_interval = update_interval
        acted = True
    mux.logger.debug("conference_set: warnings: %s", warnings)
    log = session.logger
    if warnings:
        for msg in warnings:
            log.warning(msg)
        log.status(msg, color="red")
    if announce or not acted:
        log.status("In conference %s as \"%s\"" % (conference.location(),
                                                   conference.name), log=True)


def conference_close(session):
    '''Close connection to conference.'''
    conference = conference_server(session, create=False)
    if conference is None:
        raise UserError("No ChimeraX conference started")
    conference.close()


def conference_send(session):
    '''Send my scene to all participants in the conference.'''
    conference = conference_server(session, create=False)
    if conference is None:
        raise UserError("No ChimeraX conference started")
    conference.send_scene()


def conference_server(session, create=False):
    try:
        return session._conference_server
    except AttributeError:
        if create:
            session._conference_server = ConferenceServer(session)
            return session._conference_server
        else:
            return None


class ConferenceServer:
    # We want to use the same variables and methods for managing
    # VR and commands, but use a different connection mechanism.
    # So we create the meeting server but override everything
    # related to connections.

    def __init__(self, session):
        setup_triggers(session)
        self._session = session
        self._name = 'Remote'
        self._color = (0,255,0,255)	# Tracking model color

        self._trackers = []
        self._mouse_tracker = None
        self._vr_tracker = None
        self._copy_scene = False

        self._command_relay_handler = None	# Send commands to peers
        self._running_received_command = False

        self._status_report_interval = 0.5
        self._status_start_time = None
        self._last_status_time = None
        self._last_status_bytes = None

        self._mux_node = None
        self._mux_addresses = None

    @property
    def name(self):
        return self._name

    #
    # Connection stuff
    #

    @property
    def connected(self):
        return self._mux_node is not None

    def _setup_cb(self, status, data):
        import threading
        if status != mux.Resp.Success:
            self.close()
            raise RuntimeError("conference connection failed: %s" % data)
        conf_name, identity, addrs = data
        self._name = identity
        self._mux_node.set_identity(identity)
        self._mux_addresses = addrs
        self._parameters = None
        conference_set(self._session, announce=True, **self._setup_kw)
        self._session.triggers.activate_trigger(TRIGGER_CONNECTED, self)
        del self._setup_kw

    def connect(self, action, location, identity, **kw):
        prefs = settings(self._session)
        if identity is None:
            identity = prefs.user_name
        else:
            prefs.user_name = identity
        if action == "start":
            if location is None:
                raise("conference location must be specified for \"start\"")
            host, port, conf_name = self._parse_location(location, True)
            self.copy_scene(True)
            self.set_color = (255,255,0,255)
            self._setup_kw = kw
            # Note that _mux_node assignment must be done /before/
            # starting the node or else the callback may reference
            # a non-existent attribute
            self._mux_node = MuxNetworkNode(host, port, conf_name, identity,
                                            True, server=self)
            self._mux_node.start(callback=self._setup_cb)
        elif action == "join":
            if location is None:
                raise("conference location must be specified for \"join\"")
            host, port, conf_name = self._parse_location(location, False)
            self._setup_kw = kw
            self._mux_node = MuxNetworkNode(host, port, conf_name, identity,
                                            False, server=self)
            self._mux_node.start(callback=self._setup_cb)
        elif action == "host":
            if location is None:
                host = ""
                port = 0
                conf_name = "unnamed"
            else:
                host, port, conf_name = self._parse_location(location, False)
            self.copy_scene(True)
            self.set_color = (255,255,0,255)
            self._setup_kw = kw
            self._mux_node = MuxLoopbackNode(conf_name, identity, server=self)
            self._mux_node.start(callback=self._setup_cb)
        else:
            raise UserError("unknown conference mode: \"%s\"" % action)
        # Cache parameters for GUI
        self._parameters = (host, port, conf_name, self._name)
        self.relay_commands()

    def _parse_location(self, location, need_name):
        prefs = settings(self._session)
        try:
            parts = location.split('/', 1)
            if len(parts) == 1:
                if need_name:
                    if prefs.conf_name is None:
                        raise UserError("conference name missing from location")
                    else:
                        addr = location
                        conf_name = prefs.conf_name
                else:
                    addr = location
                    conf_name = "unnamed"
            else:
                addr, conf_name = parts
                prefs.conf_name = conf_name
            parts = addr.split(':', 1)
            if len(parts) != 2:
                host = addr.strip()
                if not host:
                    if prefs.host_name is None:
                        raise UserError("host name is missing")
                    else:
                        host = prefs.host_name
                if prefs.port is None:
                    port = 443
                else:
                    port = prefs.port
            else:
                host = parts[0]
                try:
                    port = int(parts[1])
                except ValueError:
                    raise UserError("port number must be an integer")
                prefs.host_name = host
                prefs.port = port
        except ValueError:
            raise UserError("bad conference location: %s" % location)
        return host, port, conf_name

    def location(self):
        mux.logger.debug("conference.location: %s", self._mux_addresses)
        if self._mux_addresses is None:
            loc = "unknown"
        else:
            if self._mux_node.conf_name == "unnamed":
                loc = ", ".join(self._mux_addresses)
            else:
                loc = ", ".join(["%s/%s" % (addr, self._mux_node.conf_name)
                                 for addr in self._mux_addresses])
        mux.logger.debug("conference.location: loc %s", loc)
        return loc

    def parameters(self):
        # custom method for GUI
        return self._parameters

    def close(self):
        # Have to override because we both delete trackers
        # and close connections
        if self._trackers:
            for t in self._trackers:
                t.delete()
            self._trackers = []
        if self._mux_node:
            self._mux_node.close()
            self._mux_node = None
            self._session.triggers.activate_trigger(TRIGGER_DISCONNECTED, self)
        if self._session._conference_server is self:
            del self._session._conference_server

    #
    # Scene/session stuff
    #

    def copy_scene(self, copy):
        self._copy_scene = copy

    def send_scene(self, identities=None):
        if identities is not None and len(identities) == 0:
            return
        if self._session.models.empty():
            return
        msg = {'scene': self._encode_session()}
        self._send_message(msg, identities=identities)

    def _encode_session(self):
        from io import BytesIO
        stream = BytesIO()
        self._session.save(stream, version=3, include_maps=True)
        from base64 import b64encode
        sbytes = b64encode(stream.getbuffer())
        return sbytes

    def _restore_session(self, base64_sbytes):
        ses = self._session
        ses.logger.status('Opening scene (%.1f Mbytes)' %
                          (len(base64_sbytes)/2**20,))
        from time import time
        t1 = time()
        from base64 import b64decode
        sbytes = b64decode(base64_sbytes)
        from io import BytesIO
        stream = BytesIO(sbytes)
        restore_camera = (ses.main_view.camera.name != 'vr')
        self._session.triggers.activate_trigger(TRIGGER_BEFORE_RESTORE, self)
        ses.restore(stream, resize_window=False, restore_camera=restore_camera)
        self._session.triggers.activate_trigger(TRIGGER_AFTER_RESTORE, self)
        t2 = time()
        ses.logger.status('Opened scene %.1f Mbytes, %.1f seconds'
                          % (len(sbytes)/2**20, (t2-t1)))

    #
    # Command stuff
    #
    def relay_commands(self, relay=True):
        h = self._command_relay_handler
        triggers = self._session.triggers
        if relay and h is None:
            h = triggers.add_handler('command finished', self._ran_command)
            self._command_relay_handler = h
        elif not relay and h:
            triggers.remove_handler(h)
            self._command_relay_handler = None

    def _ran_command(self, trigger_name, command):
        if self._running_received_command:
            return
        cmd = command.lstrip()
        if cmd.startswith('conference'):
            return
        msg = {'command': command}  # Have peers run the command we just ran
        self._send_message(msg)

    def _run_command(self, msg):
        if self._command_relay_handler is None:
            return	# Don't run commands from others if we are not relaying commands.
        command = msg['command']
        self._running_received_command = True
        from chimerax.core.commands import run
        try:
            run(self._session, command)
        finally:
            self._running_received_command = False

    #
    # Tracking stuff
    #

    def set_color(self, color):
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def vr_tracker(self):
        self._initiate_tracking()
        return self._vr_tracker

    def _initiate_tracking(self):
        if not self._trackers:
            s = self._session
            self._mouse_tracker = mt = MouseTracking(s, self)
            self._vr_tracker = vrt = VRTracking(s, self)
            self._trackers = [mt, vrt]

    def _send_room_coords(self, sockets = None):
        rts = self.vr_tracker.last_room_to_scene
        if rts is not None:
            # Tell peer the current vr room coordinates.
            msg = {'vr coords':_place_matrix(rts)}
            self._send_message(msg, sockets=sockets)

    #
    # Callbacks from hub
    #

    def add_participant(self, identity):
        # "identity" is a 2-tuple of (name, address)
        logger = self._session.logger
        logger.info("\"%s\" [%s] joined conference" % identity)
        self._initiate_tracking()
        name = identity[0]
        if self._copy_scene:
            logger.info("copying scene to \"%s\"" % name)
            self.send_scene([name])
        self._send_room_coords([name])
        self._session.triggers.activate_trigger(TRIGGER_JOINED, identity)

    def drop_participant(self, identity):
        # The hub handles all that for us, so we do not need to do anything
        # "identity" is a 2-tuple of (name, address) but address
        # is typically uninformative since the participant already left
        logger = self._session.logger
        logger.info("\"%s\" left conference" % identity[0])
        self._session.triggers.activate_trigger(TRIGGER_DEPARTED, identity)

    def get_participants(self, cb):
        self._mux_node.get_participants(callback=cb)

    def _send_message(self, msg, identities=None):
        self._mux_node.send(msg, receivers=identities)

    def _message_execute(self, q, sender, msg):
        mux.logger.debug("_message_execute")
        if 'id' not in msg:
            msg['id'] = sender
        if 'scene' in msg:
            self._restore_session(msg['scene'])
        if 'command' in msg:
            self._run_command(msg)
        for t in self._trackers:
            t.update_model(msg)
        mux.logger.debug("_message_execute returning")
        q.put((mux.Resp.Success, None))

    def _notification_execute(self, q, ntype, data):
        if ntype == mux.Notify.Joined:
            # data is just the name of the participant that joined
            self.add_participant(data)
        elif ntype == mux.Notify.Departed:
            # data is just the name of the participant that joined
            self.drop_participant(data)
        else:
            raise RuntimeError("unsupported mux notification: %s" % ntype)
        q.put((mux.Resp.Success, None))


class BaseMuxNode:

    def handle_packet(self, ptype, serial, packet):
        mux.logger.debug("handle_packet: %s %s", mux.PacketType.name(ptype), serial)
        from queue import SimpleQueue
        q = SimpleQueue()
        session = self._server._session
        session.ui.thread_safe(self._handle_pkt, q, ptype, serial, packet)
        mux.logger.debug("handle_packet: waiting for queue")
        return q.get()

    def _handle_pkt(self, q, ptype, serial, packet):
        mux.logger.debug("_handle_pkt: %s %s", mux.PacketType.name(ptype), serial)
        try:
            retval = super().handle_packet(ptype, serial, packet)
        except RuntimeError as e:
            session = self._server._session
            session.logger.error(str(e))
            retval = None
        mux.logger.debug("_handle_pkt: waiting for queue")
        q.put(retval)

    def handle_message(self, msg, sender):
        #logger = self._server._session.logger
        #logger.info("Packet from %s: %s" % (sender, packet))
        # packet should contain two keys: "receivers" and "data"
        # for now, we just ignore who the message was sent to
        mux.logger.debug("handle_message: %s %s", msg, sender)
        from queue import SimpleQueue
        q = SimpleQueue()
        session = self._server._session
        session.ui.thread_safe(self._server._message_execute,
                               q, sender, msg["data"])
        mux.logger.debug("handle_message: waiting for queue")
        return q.get()

    def handle_notification(self, ntype, data):
        #logger = self._server._session.logger
        #logger.info("Notification from %s: %s" % (ntype, data))
        from queue import SimpleQueue
        q = SimpleQueue()
        session = self._server._session
        session.ui.thread_safe(self._server._notification_execute,
                               q, ntype, data)
        return q.get()


class MuxNetworkNode(BaseMuxNode, mux.NetworkNode):

    def __init__(self, hostname, port, conf_name, ident, create,
                 server=None, **kw):
        self._server = server
        super().__init__(hostname, port, conf_name, ident, create, **kw)


class MuxLoopbackNode(BaseMuxNode, mux.LoopbackNode):

    def __init__(self, conf_name, ident, server=None):
        self._server = server
        super().__init__(conf_name, ident)
