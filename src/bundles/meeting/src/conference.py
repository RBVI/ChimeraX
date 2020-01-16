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
from .tracking import MouseTracking, VRTracking
from . import mux


def conference(session, action, location, **kw):
    '''Similar to "meeting", but uses a hub that all participants
    can reach to exchange messages.

    Parameters
    ----------
    action : enumeration
      One of "start" or "join".
    location : string of form [identity@]host[:port]/session
      String to identify the session location.  Optional 'identity'
      provides a unique handle for this participant in the conference.
      'host' is the name or IP address of the host to connect to.
      'port' is the TCP port number, default 443, to connect to.
      'session' is the name of the session to start or join.
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
    host, port, ses_name, identity = _parse_location(location)
    conference = conference_server(session, create=True)
    try:
        conference.connect(action, host, port, ses_name, identity)
    except RuntimeError as e:
        conference.close()
        raise UserError(str(e))
    else:
        conference_set(session, announce=True, **kw)


def _parse_location(location):
    parts = location.split('@', 1)
    if len(parts) == 1:
        identity = None
        loc = location
    else:
        identity = parts[0]
        loc = parts[1]
    try:
        addr, ses_name = loc.split('/', 1)
        parts = addr.split(':', 1)
        if len(parts) != 2:
            host = addr
            port = 443
        else:
            host, port = parts
    except ValueError:
        raise UserError("bad conference location: %s" % location)
    return host, port, ses_name, identity


def conference_set(session, color=None, face_image=None, copy_scene=None,
                   relay_commands=None, update_interval=None, announce=False):
    '''Change options for the currently active conference.

    Parameters
    ----------
    Same as the keyword arguments as ``conference``.
    '''
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
            conference.relay_commands(realy_commands)
        acted = True
    if update_interval is not None:
        conference.vr_tracker.update_interval = update_interval
        acted = True
    log = session.logger
    if warnings:
        for msg in warnings:
            log.warning(msg)
        log.status(msg, color="red")
    if announce or not acted:
        log.status("Conference at %s" % conference.location(False), log=True)


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
            session._conference_server = ConferenceClient(session)
            return session._conference_server
        else:
            return None


class ConferenceClient:
    # We want to use the same variables and methods for managing
    # VR and commands, but use a different connection mechanism.
    # So we create the meeting server but override everything
    # related to connections.

    def __init__(self, session):
        self._session = session
        self._name = 'Remote'
        self._color = (0,255,0,255)	# Tracking model color

        self._trackers = []
        self._mouse_tracker = None
        self._vr_tracker = None
        self._copy_scene = False

        self._command_relay_handler = None	# Send commands to peers
        self._running_received_command = False
        self.relay_commands()

        self._status_report_interval = 0.5
        self._status_start_time = None
        self._last_status_time = None
        self._last_status_bytes = None

        self._hub = None

    @property
    def session(self):
        return self._session

    #
    # Connection stuff
    #

    @property
    def connected(self):
        return self._hub is not None

    def location(self, show_identity):
        if self._hub is None:
            return "not connected to conference"
        addr, session = self._hub.get_session_info()
        location = "%s/%s" % (addr, session)
        if show_identity:
            identity = self._hub.identity
            if identity is None:
                identity = "[anonymous]"
            location = identity + '@' + location
        return location

    def connect(self, action, host, port, ses_name, ident):
        if action == "start":
            self._hub = MuxClient(host, port, ses_name, ident,
                                  True, server=self)
            self.copy_scene(True)
            self.set_color = (255,255,0,255)
        elif action == "join":
            self._hub = MuxClient(host, port, ses_name, ident,
                                  False, server=self)
        else:
            raise UserError("unknown conference mode: \"%s\"" % action)

    def close(self):
        # Have to override because we both delete trackers
        # and close connections
        if self._trackers:
            for t in self._trackers:
                t.delete()
            self._trackers = []
        if self._hub:
            self._hub.close()
            self._hub = None
        if self.session._conference_server is self:
            del self.session._conference_server

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
        ses.restore(stream, resize_window=False, restore_camera=restore_camera)
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

    def add_client(self, identity):
        logger = self.session.logger
        logger.info("\"%s\" joined conference" % identity)
        self._initiate_tracking()
        if self._copy_scene:
            self.send_scene([identity])
        self._send_room_coords([identity])

    def drop_client(self, identity):
        # The hub handles all that for us, so we do not need to do anything
        logger = self.session.logger
        logger.info("\"%s\" left conference" % identity)

    def _send_message(self, msg, identities=None):
        self._hub.send(msg, receivers=identities)

    def _message_received(self, sender, msg):
        # Note that this method is typically called from the "run()"
        # thread of the ConferenceClient, but we need to do the actual
        # work in the main thread.  So we use ui.thread_safe to
        # call the function, and use a Queue instance for synchronization
        #logger = self.session.logger
        #logger.info("Message from %s: %s" % (sender, msg))
        from queue import SimpleQueue
        q = SimpleQueue()
        self.session.ui.thread_safe(self._message_execute, q, sender, msg)
        return q.get()

    def _message_execute(self, q, sender, msg):
        if 'id' not in msg:
            msg['id'] = sender
        if 'scene' in msg:
            self._restore_session(msg['scene'])
        if 'command' in msg:
            self._run_command(msg)
        for t in self._trackers:
            t.update_model(msg)
        q.put(None)


class MuxClient(mux.Client):

    def __init__(self, *args, server=None, **kw):
        self._server = server
        super().__init__(*args, **kw)
        self.start()

    def handle_message(self, packet, sender):
        #logger = self._server.session.logger
        #logger.info("Packet from %s: %s" % (sender, packet))
        # packet should contain two keys: "receivers" and "data"
        # for now, we just ignore who the message was sent to
        retval = self._server._message_received(sender, packet["data"])
        return mux.Resp.Success, retval

    def handle_notification(self, ntype, data):
        #logger = self._server.session.logger
        #logger.info("Notification from %s: %s" % (ntype, data))
        if ntype == mux.Notify.Joined:
            # data is just the name of the client that joined
            self._server.add_client(data)
        elif ntype == mux.Notify.Departed:
            # data is just the name of the client that joined
            self._server.drop_client(data)
        else:
            raise RuntimeError("unsupported mux notification: %s" % ntype)
        return mux.Resp.Success, None
