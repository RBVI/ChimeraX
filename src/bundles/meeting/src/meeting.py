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

# -----------------------------------------------------------------------------
#
def meeting(session, host = None, port = 52194, name = None, color = None,
            face_image = None, copy_scene = None, relay_commands = None,
            update_interval = None):
    '''Allow two or more ChimeraX instances to show each others' VR hand-controller
    and headset positions or mouse positions.

    Parameters
    ----------
    host : string
      If the value is "start" then a shared session is started that other ChimeraX instances can join.
      The log will output the host name or IP address that other ChimeraX users should connect to.
      To connect to a meeting started by another ChimeraX the host value should be the host name or
      IP address of the ChimeraX that started the meeting, for example, "descartes.cgl.ucsf.edu" or "169.230.21.39".
      One ChimeraX specifies this as "start" which starts listening for connections
      If no address is specified, the current meeting connections are reported to the log.
    port : int
      Optional port number.  Can be omitted in which case default port 52194 is used.
    name : string
      Name to identify this ChimeraX on remote machines.
    color : r,g,b,a (range 0-255)
      Color for my mouse pointer shown on other machines
    face_image : string
      Path to PNG or JPG image file for image to use for VR face depiction.
    copy_scene : bool
      Whether to copy the open models from the ChimeraX that started the meeting to other ChimeraX instances
      when they join the meeting.
    relay_commands : bool
      Whether to have every command you run sent to the other participants and their commands
      sent to you and automatically run so changes to the scene are mirrored for all participants.
      Default true.
    update_interval : int
      How often VR hand and head model positions are sent for this ChimeraX instance in frames.
      Value of 1 updates every frame.  Default 1.
    '''

    if host is None:
        p = _meeting_participant(session)
        if p is None:
            session.logger.status('No ChimeraX meeting started', log = True)
            return
        _report_connection(session, p)
    elif host == 'start':
        p = _start_meeting(session, port, copy_scene)
    else:
        p = _join_meeting(session, host, port)

    _set_appearance(session, p, name, color, face_image)

    if copy_scene is not None and p.hub:
        p.hub.copy_scene(copy_scene)

    if relay_commands is not None:
        p.send_and_receive_commands(relay_commands)

    if update_interval is not None:
        p.vr_tracker.update_interval = update_interval

# -----------------------------------------------------------------------------
#
def _start_meeting(session, port, copy_scene):
    p = _meeting_participant(session, create = True, start_hub = True)
    h = p.hub
    if h is None:
        from chimerax.core.errors import UserError
        raise UserError('To start a meeting you must exit'
                        ' the meeting you are currently in using'
                        ' command "meeting close"')
    if copy_scene is None:
        h.copy_scene(True)
    h.listen(port)
    msg = "Meeting at %s" % h.listen_host_info()
    session.logger.status(msg, log = True)
    return p

# -----------------------------------------------------------------------------
#
def _join_meeting(session, host, port):
    p = _meeting_participant(session, create = True)
    if p.connected:
        from chimerax.core.errors import UserError
        raise UserError('To join another meeting you must exit'
                        ' the meeting you are currently in using'
                        ' command "meeting close"')
    p.connect(host, port)
    return p

# -----------------------------------------------------------------------------
#
def _report_connection(session, participant):
    h = participant.hub
    lines = []
    if h:
        if h.listening:
            lines.append("Meeting at %s" % h.listen_host_info())
            clist = h.connected_ip_port_list()
            lines.extend(["Connected to %s port %d" % (ip,port)
                          for (ip,port) in clist])
    else:
        ms = participant._message_stream
        if ms:
            lines.append('Connected to meeting at %s port %d'
                         % ms.host_and_port())
    msg = '\n'.join(lines) if lines else "No ChimeraX meeting started"
    session.logger.status(msg, log = True)

# -----------------------------------------------------------------------------
#
def _set_appearance(session, participant, name, color, face_image):
    p = participant
    settings = _meeting_settings(session)
    save_settings = False
    
    if name is not None:
        p.name = settings.name = name
        save_settings = True
    else:
        p.name = settings.name

    if color is not None:
        p.color = settings.color = color
        save_settings = True
    else:
        p.color = settings.color

    from os.path import isfile
    if face_image is None:
        fimage = settings.face_image
        if fimage and isfile(fimage):
            p.vr_tracker.new_face_image(fimage)
    elif isfile(face_image):
        settings.face_image = face_image
        save_settings = True
        p.vr_tracker.new_face_image(face_image)
    else:
        msg = 'Face image file "%s" does not exist' % face_image
        log = session.logger
        log.warning(msg)
        log.status(msg, color = 'red')

    if save_settings:
        settings.save()

# -----------------------------------------------------------------------------
#
def _meeting_settings(session):
    settings = getattr(session, '_meeting_settings', None)
    if settings is None:
        from chimerax.core.settings import Settings
        class _MeetingSettings(Settings):
            EXPLICIT_SAVE = {
                'name': 'Remote',	# Name seen by other participants
                'color': (0,255,0,255),	# Hand color seen by others
                'face_image': None,	# Path to image file
            }
        settings = _MeetingSettings(session, "meeting")
        session._meeting_settings = settings
    return settings

# -----------------------------------------------------------------------------
#
def meeting_close(session):
    '''Close all connection shared pointers.'''
    p = _meeting_participant(session)
    if p:
        p.close()

# -----------------------------------------------------------------------------
#
def meeting_send(session):
    '''Send my scene to all participants in the meeting.'''
    p = _meeting_participant(session)
    if p:
        p.send_scene()
        
# -----------------------------------------------------------------------------
# Register the connect command for ChimeraX.
#
def register_meeting_command(logger):
    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import StringArg, IntArg, Color8TupleArg, OpenFileNameArg, BoolArg
    desc = CmdDesc(optional = [('host', StringArg)],
                   keyword = [('port', IntArg),
                              ('name', StringArg),
                              ('color', Color8TupleArg),
                              ('face_image', OpenFileNameArg),
                              ('copy_scene', BoolArg),
                              ('relay_commands', BoolArg),
                              ('update_interval', IntArg)],
                   synopsis = 'Show synchronized mouse or VR hand controllers between two ChimeraX instances')
    register('meeting', desc, meeting, logger=logger)
    desc = CmdDesc(synopsis = 'Close meeting')
    register('meeting close', desc, meeting_close, logger=logger)
    desc = CmdDesc(synopsis = 'Copy my scene to all other meeting participants')
    register('meeting send', desc, meeting_send, logger=logger)

# -----------------------------------------------------------------------------
#
def _meeting_participant(session, create = False, start_hub = False):
    p = getattr(session, '_meeting_participant', None)
    if p and p.closed:
        session._meeting_participant = p = None
    if p is None and create:
        p = MeetingParticipant(session, start_hub = start_hub)
        session._meeting_participant = p
    return p

# -----------------------------------------------------------------------------
#
class MeetingParticipant:
    def __init__(self, session, start_hub = False):
        self._session = session
        self._name = 'Remote'
        self._color = (0,255,0,255)
	# Tracking model color
        self._message_stream = None	# MessageStream for communicating with hub
        self._hub = None		# MeetingHub if we are hosting the meeting
        self._trackers = []
        self._mouse_tracker = None
        self._vr_tracker = None
        self._copy_scene = False

        self._command_handlers = []	# Trigger handlers to capture executed commands
        self._running_received_command = False
        self.send_and_receive_commands(True)

        if start_hub:
            self._hub = h = MeetingHub(session, self)
            self._message_stream = MessageStreamLocal(h._message_received)
            

    def _get_name(self):
        return self._name
    def _set_name(self, name):
        self._name = name
    name = property(_get_name, _set_name)
    
    def _get_color(self):
        return self._color
    def _set_color(self, color):
        self._color = color
    color = property(_get_color, _set_color)
    
    @property
    def connected(self):
        return self._message_stream is not None
    
    def connect(self, host, port):
        if self._hub:
            raise RuntimeError('Cannot join a meeting when currently hosting a meeting.')
        from PyQt5.QtNetwork import QTcpSocket
        socket = QTcpSocket()
        msg_stream = MessageStream(socket, self._message_received,
                                   self._disconnected, self._session.logger)
        self._message_stream = msg_stream
        socket.connectToHost(host, port)
        self._session.logger.status('Waiting for scene data from meeting host')

        self._initiate_tracking()
        
    @property
    def hub(self):
        return self._hub

    @property
    def closed(self):
        return self._message_stream is None
    
    def close(self):
        self.send_and_receive_commands(False)
        
        self._close_trackers()

        msg_stream = self._message_stream
        if msg_stream:
            msg_stream.close()
            self._message_stream = None

        if self._hub:
            self._hub.close()
            self._hub = None
            
    def _close_trackers(self):
        for t in self._trackers:
            t.delete()
        self._trackers = []

    def send_scene(self):
        if self._session.models.empty():
            return
        msg = {'scene': self._encode_session()}
        self._send_message(msg)
            
    def _encode_session(self):
        from io import BytesIO
        stream = BytesIO()
        self._session.save(stream, version=3, include_maps=True)
        from base64 import b64encode
        sbytes = b64encode(stream.getbuffer())
        return sbytes

    def _restore_session(self, base64_sbytes):
        ses = self._session
        ses.logger.status('Opening scene (%.1f Mbytes)' % (len(base64_sbytes)/2**20,))
        from time import time
        t1 = time()
        from base64 import b64decode
        sbytes = b64decode(base64_sbytes)
        from io import BytesIO
        stream = BytesIO(sbytes)
        restore_camera = (ses.main_view.camera.name != 'vr')
        ses.restore(stream, resize_window = False, restore_camera = restore_camera,
                    clear_log = False)
        t2 = time()
        ses.logger.status('Opened scene %.1f Mbytes, %.1f seconds'
                          % (len(sbytes)/2**20, (t2-t1)))

    def send_and_receive_commands(self, enable=True): 
        h = self._command_handlers
        ses = self._session
        triggers = ses.triggers
        from chimerax.core.commands import enable_motion_commands
        if enable and not h:
            enable_motion_commands(ses, True, frame_skip = 0)
            h = [triggers.add_handler('command finished', self._ran_command),
                 triggers.add_handler('motion command', self._motion_command)]
            self._command_handlers = h
        elif not enable and h:
            enable_motion_commands(ses, False)
            for handler in h:
                triggers.remove_handler(handler)
            self._command_handlers.clear()

    def _ran_command(self, trigger_name, command, motion = False):
        if self._running_received_command:
            return
        if command.lstrip().startswith('meeting'):
            return
        msg = {
            'command': command,   # Send command to other participants
            'motion': motion,	  # Others will not log motion commands
        }
        self._send_message(msg)

    def _motion_command(self, trigger_name, command):
        self._ran_command(trigger_name, command, motion = True)
        
    def _run_command(self, msg):
        if not self._command_handlers:
            return	# Don't run commands from others if we are not relaying commands.
        command = msg['command']
        log_cmd = not msg.get('motion')
        self._running_received_command = True
        from chimerax.core.commands import run
        try:
            run(self._session, command, log=log_cmd)
        finally:
            self._running_received_command = False

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

    def _message_received(self, msg, msg_stream):
        if 'scene' in msg:
            self._restore_session(msg['scene'])
        if 'command' in msg:
            self._run_command(msg)
        for t in self._trackers:
            t.update_model(msg)
        if 'disconnected' in msg:
            self._participant_left(msg)

    def _send_message(self, msg):
        msg_bytes = MessageStream.message_as_bytes(msg)
        self._message_stream.send_message_bytes(msg_bytes)

    def _participant_left(self, msg):
        participant_id = msg['id']
        for t in self._trackers:
            t.remove_model(participant_id)

    def _disconnected(self, msg_stream):
        self.close()

# -----------------------------------------------------------------------------
#
class MeetingHub:
    def __init__(self, session, host_participant):
        self._session = session
        self._server = None		# QTcpServer listens for connections
        msg_stream = MessageStreamLocal(host_participant._message_received)
        self._connections = [msg_stream] # List of MessageStream for each participant
        self._next_participant_id = 1
        self._host = host_participant	# MeetingParticipant that provides session for new participants.
        self._copy_scene = True		# Whether new participants get copy of scene

    def close(self):
        for msg_stream in tuple(self._connections):
            msg_stream.close()
        self._connections = []
        
        self._server.close()
        self._server = None

        self._host = None
        
    @property
    def listening(self):
        s = self._server
        return s is not None and s.isListening()

    @property
    def started_server(self):
        return self._server is not None

    def listen(self, port):
        if self._server:
            return
        from PyQt5.QtNetwork import QTcpServer, QHostAddress
        self._server = s = QTcpServer()
        aa = self._available_server_ipv4_addresses()
        a = aa[0] if aa else QHostAddress.Any
        if not s.listen(a, port):
            self._session.logger.warning('QTcpServer.listen() failed for address %s, port %d: %s'
                                         % (a.toString(), port, s.errorString()))
        else:
            s.newConnection.connect(self._new_connection)
    
    def listen_host_info(self):
        if not self.listening:
            return None
        
        s = self._server
        hi = '%s port %d' % (s.serverAddress().toString(), s.serverPort())
        from PyQt5.QtNetwork import QHostInfo
        host = QHostInfo.localHostName()
        if host:
            hi = '%s or %s' % (host, hi)
        return hi
    
    def connected_ip_port_list(self):
        return [c.host_and_port() for c in self._connections
                if not isinstance(c, MessageStreamLocal)]
    
    def _available_server_ipv4_addresses(self):
        from PyQt5.QtNetwork import QNetworkInterface, QAbstractSocket
        a = []
        for ni in QNetworkInterface.allInterfaces():
            flags = ni.flags()
            if (flags & QNetworkInterface.IsUp) and not (flags & QNetworkInterface.IsLoopBack):
                for ae in ni.addressEntries():
                    ha = ae.ip()
                    if (ha.protocol() == QAbstractSocket.IPv4Protocol
                        and not ha.isLoopback()
                        and not ha.isNull()
                        and not ha.toString().startswith('169.254')): # Exclude link-local addresses
                        a.append(ha)
        return a

    def copy_scene(self, copy):
        self._copy_scene = copy
        
    def _new_connection(self):
        s = self._server
        while s.hasPendingConnections():
            socket = s.nextPendingConnection()
            msg_stream = MessageStream(socket, self._message_received,
                                       self._disconnected, self._session.logger)
            self._add_connection(msg_stream)
            host, port = (socket.peerAddress().toString(), socket.peerPort())
            self._session.logger.info('Connection accepted from %s port %d' % (host, port))

    def _add_connection(self, msg_stream):
        msg_stream.participant_id = self._next_participant_id
        self._next_participant_id += 1
            
        self._connections.append(msg_stream)

        # Send new participant the initial scene
        if self._copy_scene:
            self._copy_scene_to_participant(msg_stream)

        # Send new participant position of scene in VR room
        self._send_room_coords(msg_stream)

    def _copy_scene_to_participant(self, message_stream):
        if self._session.models.empty():
            return
        msg = {'scene': self._host._encode_session()}
        self._send_message(msg, message_streams=[message_stream])

    def _send_room_coords(self, message_stream):
        rts = self._host.vr_tracker.last_room_to_scene
        if rts is not None:
            # Tell peer the current vr room coordinates.
            msg = {'vr coords': _place_matrix(rts)}
            self._send_message(msg, message_streams=[message_stream])

    def _message_received(self, msg, msg_stream):
        if 'id' not in msg:
            msg['id'] = msg_stream.participant_id
        self._send_message(msg)

    def _send_message(self, msg, message_streams = None):
        '''Message will not be sent to participant with same id as message.'''
        if 'id' not in msg:
            msg['id'] = 0	# Message from host.

        if message_streams is None:
            message_streams = self._connections
        exclude_id = msg['id']
        message_streams = [ms for ms in message_streams
                           if ms.participant_id != exclude_id]

        if message_streams:
            msg_bytes = MessageStream.message_as_bytes(msg)
            for msg_stream in message_streams:
                msg_stream.send_message_bytes(msg_bytes)

    def _disconnected(self, msg_stream):
        self._connections.remove(msg_stream)

        # Sendmessage to other participants that this participant has left
        # so they can remove the head/hand models.
        participant_id = msg_stream.participant_id
        msg = {'id': participant_id, 'disconnected': True}
        self._send_message(msg)

class MessageStream:
    def __init__(self, socket, message_received_cb, disconnected_cb, log):
        self._socket = socket
        self._message_received_cb = message_received_cb
        self._disconnected_cb = disconnected_cb
        
        self._log = log
        self._bytes_list = []	# blocks of bytes read so far.
        self._byte_count = 0
        self._msg_length = None

        self._status_report_interval = 0.5
        self._status_start_time = None
        self._last_status_time = None
        self._last_status_bytes = None

        socket.error.connect(self._socket_error)
        socket.disconnected.connect(self._socket_disconnected)

        # Register callback called when data available to read on socket.
        socket.readyRead.connect(self._data_available)

    def close(self):
        # Calling close crashes if it is in disconnect callback.
#        self._socket.close()
        s = self._socket
        s.deleteLater()
        self._socket = None
#        self._keep_alive = s

    def host_and_port(self):
        s = self._socket
        return (s.peerAddress().toString(), s.peerPort())
    
    def send_message_bytes(self, msg_bytes):
        from PyQt5.QtCore import QByteArray
        qbytes = QByteArray(msg_bytes)
        self._socket.write(qbytes)

    @staticmethod
    def message_as_bytes(message):
        '''
        message is a dictionary which is sent as a string
        prepended by 4 bytes giving the length of the dictionary string.
        We include the length in order to know where a dictionary ends
        when transmitted over a socket.
        '''
        msg_bytes = repr(message).encode('utf-8')
        if b'\0' in msg_bytes:
            raise ValueError('Null byte in message:\n%s' % msg_bytes)
        from numpy import array, uint32
        msg_len = array([len(msg_bytes)], uint32).tobytes()
        return msg_len + msg_bytes

    def _data_available(self):
        while True:
            msg = self._read_message()
            if msg is None:
                break	# Don't have full message yet.
            else:
                self._message_received_cb(msg, self)
        
    def _read_message(self):
        socket = self._socket
#        bavail = socket.bytesAvailable()
#        rbytes = socket.read(bavail)
        rbytes = socket.readAll()
        self._bytes_list.append(rbytes)
        self._byte_count += len(rbytes)
        if self._have_full_message():
            self._report_message_received()
            msg = self._assemble_message()
        else:
            self._report_message_progress()
            msg = None
        self._check_for_null_bytes(rbytes)  # Debug code. Ticket #3784
        return msg

    def _check_for_null_bytes(self, rbytes):
        '''Debug code. Ticket #3784. Getting null bytes in messages.'''
        if self._msg_length is None:
            return
        last_block = self._bytes_list[-1]
        null_count = last_block.count(b'\0')
        if null_count > 0:
            msg = ('Error: Got %d null bytes in last block of size %d, last socket read of %d bytes:/n%s'
                   % (null_count, len(last_block), len(rbytes), rbytes))
            self._log.info(msg)

    def _have_full_message(self):
        msg_len = self._message_length
        return msg_len is not None and self._byte_count >= msg_len

    @property
    def _message_length(self):
        msg_len = self._msg_length
        if msg_len is not None:
            return msg_len
        if self._byte_count < 4:
            return None
        bytes = self._take_bytes(4)
        from numpy import frombuffer, uint32
        msg_len = frombuffer(bytes, uint32)[0]
        self._msg_length = msg_len
        return msg_len

    def _assemble_message(self):
        bytes = self._take_bytes(self._message_length)
        self._msg_length = None
        msg = _decode_message_bytes(bytes)
        return msg

    def _take_bytes(self, nbytes):
        if nbytes > self._byte_count:
            raise ValueError('Tried to take %d bytes when only %d available'
                             % (nbytes, self._byte_count))
        all_bytes = b''.join(self._bytes_list)
        if nbytes == self._byte_count:
            self._bytes_list = []
            self._byte_count = 0
        else:
            self._bytes_list = [all_bytes[nbytes:]]
            self._byte_count = len(all_bytes) - nbytes
        return all_bytes[:nbytes]

    def _report_message_received(self):
        '''Report receiving message from a peer.'''
        lt = self._last_status_time
        self._last_status_time = None
        if lt is None or lt <= self._status_start_time:
            return
        from time import time
        t = time()
        msg = ('Received %.1f Mbytes in %.1f seconds'
               % (self._message_length / 2**20, t - self._status_start_time))
        self._log.status(msg)

    def _report_message_progress(self):
        '''Report progress receiving message.'''
        msg_len = self._msg_length
        if msg_len is None or msg_len == 0:
            return
        bytes_received = self._byte_count
        from time import time
        t = time()
        lt = self._last_status_time
        if lt is None:
            self._status_start_time = t
            self._last_status_time = t
            self._last_status_bytes = bytes_received
        elif t - lt > self._status_report_interval:
            percent = 100 * bytes_received/msg_len
            rate = ((bytes_received - self._last_status_bytes) / 2**20) / (t - lt)
            msg = ('Receiving data %.0f%% of %.1f Mbytes, (%.1f Mbytes/sec)'
                   % (percent, msg_len / 2**20, rate))
            self._log.status(msg)
            self._last_status_time = t
            self._last_status_bytes = bytes_received
            
    def _socket_error(self, error_type):
        self._log.info('Socket error %s' % self._socket.errorString())
#        self._socket_disconnected()

    def _socket_disconnected(self):
        socket = self._socket
        if socket is None:
            return
        import sip
        if sip.isdeleted(socket):
            return	# Happens when exiting ChimeraX
        host, port = (socket.peerAddress().toString(), socket.peerPort())
        msg = 'Disconnected from %s:%d' % (host, port)
        self._log.info(msg)
        # Closing or deallocating the socket in this socket callback causes a crash.
        # So close routine must use deleteLater()
        # self.close()
        self._disconnected_cb(self)

def _decode_message_bytes(bytes):
    msg_string = bytes.decode('utf-8')
    import ast
    try:
        msg_data = ast.literal_eval(msg_string)
    except ValueError as e:
        raise ValueError('ChimeraX meeting message could not be parsed, length %d, content "%s"' % (len(bytes), bytes)) from e
    return msg_data

class MessageStreamLocal:
    def __init__(self, send_message_cb):
        self._send_message_cb = send_message_cb
        self.participant_id = 0
    def send_message_bytes(self, msg_bytes):
        msg = _decode_message_bytes(msg_bytes[4:])
        self._send_message_cb(msg, self)
    def close(self):
        pass

class PointerModels:
    '''Manage mouse or VR pointer models for all connected hosts.'''
    def __init__(self, session):
        self._session = session
        self._pointer_models = {}  # Map participant id to MousePointerModel or VRPointerModel

    def delete(self):
        for participant_id in tuple(self._pointer_models.keys()):
            self.remove_model(participant_id)

    def pointer_model(self, participant_id = None):
        pm = self._pointer_models
        if participant_id in pm:
            m = pm[participant_id]
            if not m.deleted:
                return m

        m = self.make_pointer_model(self._session)
        models = self._session.models
        models.add([m], minimum_id = 100)
        pm[participant_id] = m
        return m

    @property
    def pointer_models(self):
        return tuple(self._pointer_models.items())
    
    def make_pointer_model(self, session):
        # Must be defined by subclass.
        pass

    def update_model(self, msg):
        m = self.pointer_model(msg.get('id'))
        if not m.deleted:
            m.update_pointer(msg)

    def remove_model(self, participant_id):
        pm = self._pointer_models
        if participant_id in pm:
            m = pm[participant_id]
            del pm[participant_id]
            if not m.deleted:
                self._session.models.close([m])

class MouseTracking(PointerModels):
    def __init__(self, session, participant):
        PointerModels.__init__(self, session)
        self._participant = participant		# MeetingParticipant instance

        t = session.triggers
        self._mouse_hover_handler = t.add_handler('mouse hover', self._mouse_hover_cb)

    def delete(self):
        t = self._session.triggers
        t.remove_handler(self._mouse_hover_handler)
        self._mouse_hover_handler = None

        PointerModels.delete(self)

    def update_model(self, msg):
        if 'mouse' in msg:
            PointerModels.update_model(self, msg)

    def make_pointer_model(self, session):
        return MousePointerModel(self._session, 'my pointer')

    def _mouse_hover_cb(self, trigger_name, pick):
        if _vr_camera(self._session):
            return

        xyz = getattr(pick, 'position', None)
        if xyz is None:
            return
        c = self._session.main_view.camera
        axis = c.view_direction()
        msg = {'name': self._participant._name,
               'color': tuple(self._participant._color),
               'mouse': (tuple(xyz), tuple(axis)),
               }

        # Update my own mouse pointer position
        self.update_model(msg)

        # Tell other participants my new mouse pointer position.
        self._participant._send_message(msg)

from chimerax.core.models import Model
class MousePointerModel(Model):
    SESSION_SAVE = False
    
    def __init__(self, session, name, radius = 1, height = 3, color = (0,255,0,255)):
        Model.__init__(self, name, session)
        from chimerax.surface import cone_geometry
        va, na, ta = cone_geometry(radius = radius, height = height)
        va[:,2] -= 0.5*height	# Place tip of cone at origin
        self.set_geometry(va, na, ta)
        self.color = color
        self.pickable = False

    def update_pointer(self, msg):
        if 'name' in msg:
            if 'id' in msg:  # If id not in msg leave name as "my pointer".
                self.name = '%s pointer' % msg['name']
        if 'color' in msg:
            self.color = msg['color']
        if 'mouse' in msg:
            xyz, axis = msg['mouse']
            from chimerax.geometry import vector_rotation, translation
            p = translation(xyz) * vector_rotation((0,0,1), axis)
            self.position = p

class VRTracking(PointerModels):
    def __init__(self, session, participant, sync_coords = True, update_interval = 1):
        PointerModels.__init__(self, session)
        self._participant = participant		# MeetingParticipant instance
        self._sync_coords = sync_coords

        t = session.triggers
        self._vr_tracking_handler = t.add_handler('vr update', self._vr_tracking_cb)
        self._update_interval = update_interval	# Send vr position every N frames.
        self._last_vr_camera = c = _vr_camera(self._session)
        self._last_room_to_scene = c.room_to_scene if c else None
        self._new_face_image = None	# Path to image file
        self._face_image = None		# Encoded image
        self._send_face_image = False
        self._gui_state = {'shown':False, 'size':(0,0), 'room position':None, 'image':None}

    def delete(self):
        t = self._session.triggers
        t.remove_handler(self._vr_tracking_handler)
        self._vr_tracking_handler = None

        PointerModels.delete(self)

    @property
    def last_room_to_scene(self):
        return self._last_room_to_scene
    
    def _get_update_interval(self):
        return self._update_interval
    def _set_update_interval(self, update_interval):
        self._update_interval = update_interval
    update_interval = property(_get_update_interval, _set_update_interval)
    
    def update_model(self, msg):
        if 'vr coords' in msg and self._sync_coords:
            matrix = msg['vr coords']
            rts = self._last_room_to_scene = _matrix_place(matrix)
            c = _vr_camera(self._session)
            if c:
                c.room_to_scene = rts
                self._reposition_vr_head_and_hands(c)

        if 'vr head' in msg:
            PointerModels.update_model(self, msg)

    def make_pointer_model(self, session):
        # Make sure new meeting participant gets my head image and button modes.
        self._send_face_image = True
        self._send_button_modes()
        
        pm = VRPointerModel(self._session, 'VR', self._last_room_to_scene)
        return pm

    def new_face_image(self, path):
        self._new_face_image = path

    def _send_button_modes(self):
        c = _vr_camera(self._session)
        if c:
            for h in c._hand_controllers:
                h._meeting_button_modes = {}
            
    def _vr_tracking_cb(self, trigger_name, camera):
        c = camera
        
        if c is not self._last_vr_camera:
            # VR just turned on so use current meeting room coordinates
            self._last_vr_camera = c
            rts = self._last_room_to_scene
            if rts is not None:
                c.room_to_scene = rts
                
        scene_moved = (c.room_to_scene is not self._last_room_to_scene)
        if scene_moved:
            self._reposition_vr_head_and_hands(c)

        v = self._session.main_view
        if v.frame_number % self.update_interval != 0:
            return

        # Report VR hand and head motions.
        msg = {'name': self._participant._name,
               'color': tuple(self._participant._color),
               'vr head': self._head_position(c),	# In room coordinates
               'vr hands': self._hand_positions(c),	# In room coordinates
               }

        fi = self._face_image_update()
        if fi:
            msg['vr head image'] = fi
            
        hb = self._hand_buttons_update(c)
        if hb:
            msg['vr hand buttons'] = hb
            
        # Report scene moved in room
        if scene_moved:
            msg['vr coords'] = _place_matrix(c.room_to_scene)
            self._last_room_to_scene = c.room_to_scene

        # Report changes in VR GUI panel
        gu = self._gui_updates(c)
        if gu:
            msg.update(gu)

        # Tell other participants my new vr state
        self._participant._send_message(msg)

    def _head_position(self, vr_camera):
        from chimerax.geometry import scale
        return _place_matrix(vr_camera.room_position * scale(1/vr_camera.scene_scale))

    def _face_image_update(self):
        # Report VR face image change.
        fi = None
        if self._new_face_image:
            image = _encode_face_image(self._new_face_image)
            self._face_image = image
            fi = image
            self._new_face_image = None
            self._send_face_image = False
        elif self._send_face_image:
            self._send_face_image = False
            if self._face_image is not None:
                fi = self._face_image
        return fi
    
    def _hand_positions(self, vr_camera):
        # Hand controller room position includes scaling from room to scene coordinates
        return [_place_matrix(h.room_position)
                for h in vr_camera._hand_controllers if h.on]

    def _hand_buttons_update(self, vr_camera):
        bu = []
        update = False
        hc = vr_camera._hand_controllers
        for h in hc:
            last_mode = getattr(h, '_meeting_button_modes', None)
            if last_mode is None:
                h._meeting_button_modes = last_mode = {}
            hm = []
            for button, mode in h.button_modes.items():
                if mode != last_mode.get(button):
                    last_mode[button] = mode
                    hm.append((button, mode.name))
                    update = True
            bu.append(hm)
        return bu if update else None

    def _gui_updates(self, vr_camera):
        msg = {}
        c = vr_camera
        ui = c.user_interface
        shown = ui.shown()
        gui_state = self._gui_state
        shown_changed = (shown != gui_state['shown'])
        if shown_changed:
            msg['vr gui shown'] = gui_state['shown'] = shown
        if shown:
            rpos = ui.model.room_position
            if rpos is not gui_state['room position'] or shown_changed:
                gui_state['room position'] = rpos
                msg['vr gui position'] = _place_matrix(rpos)

            # Notify about changes in panel size, position or image.
            pchanges = []	# GUI panel changes
            for panel in ui.panels:
                name, size, pos = panel.name, panel.size, panel.drawing.position
                rgba = panel.panel_image_rgba()
                if rgba is None:
                    continue  # Panel has not yet been drawn.
                pstate = gui_state.setdefault(('panel', name), {})
                pchange = {}
                if 'size' not in pstate or size != pstate['size'] or shown_changed:
                    pchange['size'] = pstate['size'] = size
                if 'position' not in pstate or pos != pstate['position'] or shown_changed:
                    pstate['position'] = pos
                    pchange['position'] = _place_matrix(pos)
                if 'image' not in pstate or rgba is not pstate['image'] or shown_changed:
                    pstate['image'] = rgba
                    pchange['image'] = _encode_numpy_array(rgba)
                if pchange:
                    pchange['name'] = panel.name
                    pchanges.append(pchange)

            # Notify about removed panels
            panel_names = set(panel.name for panel in ui.panels)
            for pname in tuple(gui_state.keys()):
                if (isinstance(pname, tuple) and len(pname) == 2 and pname[0] == 'panel'
                    and pname[1] not in panel_names):
                    pchanges.append({'name':pname[1], 'closed':True})
                    del gui_state[pname]

            if pchanges:
                msg['vr gui panels'] = pchanges

        return msg
    
    def _reposition_vr_head_and_hands(self, camera):
        '''
        If my room to scene coordinates change correct the VR head and hand model
        scene positions so they maintain the same apparent position in the room.
        '''
        c = camera
        rts = c.room_to_scene
        for participant_id, vrm in self.pointer_models:
            if isinstance(vrm, VRPointerModel):
                for m in vrm.child_models():
                    if m.room_position is not None:
                        m.scene_position = rts*m.room_position

            
from chimerax.core.models import Model
class VRPointerModel(Model):
    casts_shadows = False
    pickable = False
    skip_bounds = True
    SESSION_SAVE = False
    model_panel_show_expanded = False
    
    def __init__(self, session, name, room_to_scene, color = (0,255,0,255)):
        Model.__init__(self, name, session)
        self._head = h = VRHeadModel(session)
        self.add([h])
        self._hands = []
        self._color = color
        self._gui = None

	# Last room to scene transformation for this participant.
        # Used if we are not using VR camera so have no room coordinates.
        self._room_to_scene = room_to_scene

    def _hand_models(self, nhands):
        from chimerax.vive.vr import HandModel
        new_hands = [HandModel(self.session, 'Hand %d' % (i+1), color=self._color)
                     for i in range(len(self._hands), nhands)]
        if new_hands:
            self.add(new_hands)
            self._hands.extend(new_hands)
        return self._hands[:nhands]

    @property
    def _gui_panel(self):
        g = self._gui
        if g is None:
            self._gui = g = VRGUIModel(self.session)
            self.add([g])
        return g
    
    def _get_room_to_scene(self):
        c = _vr_camera(self.session)
        return c.room_to_scene if c else self._room_to_scene
    def _set_room_to_scene(self, rts):
        self._room_to_scene = rts
    room_to_scene = property(_get_room_to_scene, _set_room_to_scene)

    def update_pointer(self, msg):
        if 'name' in msg:
            if 'id' in msg:  # If id not in msg leave name as "my pointer".
                self.name = '%s VR' % msg['name']
        if 'color' in msg:
            for h in self._hands:
                h.set_cone_color(msg['color'])
        if 'vr coords' in msg:
            self.room_to_scene = _matrix_place(msg['vr coords'])
        if 'vr head' in msg:
            h = self._head
            h.room_position = rp = _matrix_place(msg['vr head'])
            h.position = self.room_to_scene * rp
        if 'vr head image' in msg:
            self._head.update_image(msg['vr head image'])
        if 'vr hands' in msg:
            hpos = msg['vr hands']
            rts = self.room_to_scene
            for h,hm in zip(self._hand_models(len(hpos)), hpos):
                h.room_position = rp = _matrix_place(hm)
                h.position = rts * rp
        if 'vr hand buttons' in msg:
            hbut = msg['vr hand buttons']
            from chimerax.vive.vr import hand_mode_icon_path
            for h,hb in zip(self._hand_models(len(hbut)), hbut):
                for button, mode_name in hb:
                    path = hand_mode_icon_path(self.session, mode_name)
                    if path:
                        h._set_button_icon(button, path)
        if 'vr gui shown' in msg:
            self._gui_panel.display = msg['vr gui shown']
        if 'vr gui position' in msg:
            g = self._gui_panel
            g.room_position = rp = _matrix_place(msg['vr gui position'])
            g.position = self.room_to_scene * rp
        if 'vr gui panels' in msg:
            for pchanges in msg['vr gui panels']:
                self._gui_panel.update_panel(pchanges)

class VRHeadModel(Model):
    '''Size in meters.'''
    casts_shadows = False
    pickable = False
    skip_bounds = True
    SESSION_SAVE = False
    default_face_file = 'face.png'
    def __init__(self, session, name = 'Head', size = 0.3, image_file = None):
        self.room_position = None

        Model.__init__(self, name, session)

        # Avoid head disappearing when models are zoomed small.
        self.allow_depth_cue = False
        
        r = size / 2
        from chimerax.surface import box_geometry
        va, na, ta = box_geometry((-r,-r,-0.1*r), (r,r,0.1*r))

        if image_file is None:
            from os.path import join, dirname
            image_file = join(dirname(__file__), self.default_face_file)
        from PyQt5.QtGui import QImage
        qi = QImage(image_file)
        aspect = qi.width() / qi.height()
        va[:,0] *= aspect
        from chimerax.graphics import qimage_to_numpy, Texture
        rgba = qimage_to_numpy(qi)
        from numpy import zeros, float32
        tc = zeros((24,2), float32)
        tc[:] = 0.5
        tc[8:12,:] = ((0,0), (1,0), (0,1), (1,1))

        self.set_geometry(va, na, ta)
        self.color = (255,255,255,255)
        self.texture = Texture(rgba)
        self.texture_coordinates = tc

    def update_image(self, base64_image_bytes):
        image_bytes = _decode_face_image(base64_image_bytes)
        from PyQt5.QtGui import QImage
        qi = QImage()
        qi.loadFromData(image_bytes)
        aspect = qi.width() / qi.height()
        va = self.vertices
        caspect = va[:,0].max() / va[:,1].max()
        va[:,0] *= aspect / caspect
        self.set_geometry(va, self.normals, self.triangles)
        from chimerax.graphics import qimage_to_numpy, Texture
        rgba = qimage_to_numpy(qi)
        r = self.session.main_view.render
        r.make_current()
        self.texture.delete_texture()
        self.texture = Texture(rgba)

class VRGUIModel(Model):
    '''Size in meters.'''
    SESSION_SAVE = False

    def __init__(self, session, name = 'GUI Panel'):
        Model.__init__(self, name, session)
        self.room_position = None
        self._panels = {}		# Maps panel name to VRGUIPanel
        self.casts_shadows = False
        self.pickable = False
        self.skip_bounds = True		# Panels should not effect view all command.
        self.allow_depth_cue = False	# Avoid panels fading out far from models.

    def update_panel(self, panel_changes):
        name = panel_changes['name']
        panels = self._panels

        if 'closed' in panel_changes:
            if name in panels:
                self.remove_drawing(panels[name])
                del panels[name]
            return
        
        if name in panels:
            p = panels[name]
        else:
            panels[name] = p = VRGUIPanel(name)
            self.add_drawing(p)
            
        if 'size' in panel_changes:
            p.set_size(panel_changes['size'])
        if 'position' in panel_changes:
            p.position = _matrix_place(panel_changes['position'])
        if 'image' in panel_changes:
            p.update_image(panel_changes['image'], self.session)

from chimerax.graphics import Drawing
class VRGUIPanel(Drawing):
    casts_shadows = False
    pickable = False
    skip_bounds = True

    def __init__(self, name):
        Drawing.__init__(self, name)
        self._size = (1,1)		# Meters

    def set_size(self, size):
        self._size = (rw, rh) = size
        from chimerax.graphics.drawing import position_rgba_drawing
        position_rgba_drawing(self, pos = (-0.5*rw,-0.5*rh), size = (rw,rh))
        
    def update_image(self, encoded_rgba, session):
        rw, rh = self._size
        rgba = _decode_numpy_array(encoded_rgba)
        r = session.main_view.render
        r.make_current() # Required for deleting previous texture in rgba_drawing()
        from chimerax.graphics.drawing import rgba_drawing
        rgba_drawing(self, rgba, pos = (-0.5*rw,-0.5*rh), size = (rw,rh))

def _vr_camera(session):
    c = session.main_view.camera
    from chimerax.vive.vr import SteamVRCamera
    return c if isinstance(c, SteamVRCamera) else None

def _place_matrix(p):
    '''Encode Place as tuple for sending over socket.'''
    return tuple(tuple(row) for row in p.matrix)

def _matrix_place(m):
    from chimerax.geometry import Place
    return Place(matrix = m)

def _encode_face_image(path):
    from base64 import b64encode
    hf = open(path, 'rb')
    he = b64encode(hf.read())
    hf.close()
    return he

def _decode_face_image(bytes):
    from base64 import b64decode
    image_bytes = b64decode(bytes)
    return image_bytes

def _encode_numpy_array(array):
    from base64 import b64encode
    from zlib import compress
    data = b64encode(compress(array.tobytes(), level = 1))
    data = {
        'shape': tuple(array.shape),
        'dtype': array.dtype.str,
        'data': data
    }
    return data

def _decode_numpy_array(array_data):
    shape = array_data['shape']
    dtype = array_data['dtype']
    from base64 import b64decode
    from zlib import decompress
    data = decompress(b64decode(array_data['data']))
    import numpy
    a = numpy.frombuffer(data, dtype).reshape(shape)
    return a
