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
    color : Color
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
      Value of 1 updates every frame.  Default 9.
    '''

    if host is None:
        s = meeting_server(session)
        if s is None:
            msg = "No ChimeraX meeting started"
        else:
            lines = []
            if s.listening:
                lines.append("Meeting at %s" % s.listen_host_info())
            clist = s.connected_ip_port_list()
            if clist:
                lines.extend(["Connected to %s port %d" % (ip,port) for (ip,port) in clist])
            msg = '\n'.join(lines) if lines else "No ChimeraX meeting started"
        session.logger.status(msg, log = True)
    elif host == 'start':
        s = meeting_server(session, create = True)
        if copy_scene is None:
            s.copy_scene(True)
        s.listen(port)
        msg = "Meeting at %s" % s.listen_host_info()
        session.logger.status(msg, log = True)
    else:
        s = meeting_server(session, create = True)
        s.connect(host, port)

    if name is not None:
        s = meeting_server(session)
        if s:
            s._name = name

    if color is not None:
        s = meeting_server(session)
        if s:
            s.set_color(color.uint8x4())

    if face_image is not None:
        from os.path import isfile
        if isfile(face_image):
            s = meeting_server(session)
            if s:
                s.vr_tracker.new_face_image(face_image)
        else:
            msg = 'Face image file "%s" does not exist' % face_image
            log = session.logger
            log.warning(msg)
            log.status(msg, color = 'red')

    if copy_scene is not None:
        s = meeting_server(session)
        if s:
            s.copy_scene(copy_scene)

    if relay_commands is not None:
        s = meeting_server(session)
        if s:
            s.relay_commands(relay_commands)

    if update_interval is not None:
        s = meeting_server(session)
        if s:
            s.vr_tracker.update_interval = update_interval

# -----------------------------------------------------------------------------
#
def meeting_close(session):
    '''Close all connection shared pointers.'''
    s = meeting_server(session)
    if s:
        s.close_all_connections()

# -----------------------------------------------------------------------------
#
def meeting_send(session):
    '''Send my scene to all participants in the meeting.'''
    s = meeting_server(session)
    if s:
        s.copy_scene_to_peers()
        
# -----------------------------------------------------------------------------
# Register the connect command for ChimeraX.
#
def register_meeting_command(logger):
    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import StringArg, IntArg, ColorArg, OpenFileNameArg, BoolArg
    desc = CmdDesc(optional = [('host', StringArg)],
                   keyword = [('port', IntArg),
                              ('name', StringArg),
                              ('color', ColorArg),
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
def meeting_server(session, create = False):
    if hasattr(session, '_meeting_server'):
        s =session._meeting_server
    elif create:
        session._meeting_server = s = MeetingServer(session)
    else:
        s = None
    return s

# -----------------------------------------------------------------------------
#
class MeetingServer:
    def __init__(self, session):
        self._session = session
        self._name = 'Remote'
        self._color = (0,255,0,255)	# Tracking model color
        self._server = None
        self._connections = []		# List of QTcpSocket
        self._message_buffer = {}	# Map of socket to bytearray of buffered data
        self._peer_ids = {}		# Number associated with each ChimeraX instance passed in messages from server.
        self._next_peer_id = 1
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

    def set_color(self, color):
        self._color = color
        
    @property
    def connected(self):
        c = self._connections
        return len(c) > 0
    
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
        return [(c.peerAddress().toString(), c.peerPort()) for c in self._connections]

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
        self._color = (255,255,0,255)

    @property
    def listening(self):
        s = self._server
        return s is not None and s.isListening()

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
        
    def connect(self, host, port):
        if self._server:
            raise RuntimeError('MeetingServer: Must call either listen, or connect, not both')
        from PyQt5.QtNetwork import QTcpSocket
        socket = QTcpSocket()
        def socket_error(error_type, self=self, socket=socket):
            self._socket_error(error_type, socket)
        socket.error.connect(socket_error)
        socket.connectToHost(host, port)
        self._add_connection(socket)
        self._session.logger.status('Waiting for scene data from meeting host')

    def close_all_connections(self):
        if self._trackers:
            for t in self._trackers:
                t.delete()
            self._trackers = []

        con = tuple(self._connections)
        for socket in con:
            socket.close()
        self._connections = []
        self._peer_ids = {}
        
        if self._server:
            self._server.close()
            self._server = None
        
    def _new_connection(self):
        s = self._server
        while s.hasPendingConnections():
            socket = s.nextPendingConnection()
            self._add_connection(socket)
            self._session.logger.info('Connection accepted from %s port %d'
                                      % (socket.peerAddress().toString(), socket.peerPort()))

    def _add_connection(self, socket):
        socket.disconnected.connect(self._disconnected)
        self._connections.append(socket)

        # Register callback called when data available to read on socket.
        def read_socket(self=self, socket=socket):
            while self._message_received(socket):
                pass
        socket.readyRead.connect(read_socket)

        self._initiate_tracking()

        if self._copy_scene:
            self.copy_scene_to_peers([socket])

        self._send_room_coords([socket])

    def copy_scene_to_peers(self, sockets = None):
        if sockets is None:
            sockets = self._connections
        if self._session.models.empty() or len(sockets) == 0:
            return
        msg = {'scene': self._encode_session()}
        self._send_message(msg, sockets=sockets)

    def _send_room_coords(self, sockets = None):
        rts = self.vr_tracker.last_room_to_scene
        if rts is not None:
            msg = {'vr coords': _place_matrix(rts)}  # Tell peer the current vr room coordinates.
            self._send_message(msg, sockets=sockets)
            
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
        ses.restore(stream, resize_window = False, restore_camera = restore_camera)
        t2 = time()
        ses.logger.status('Opened scene %.1f Mbytes, %.1f seconds'
                          % (len(sbytes)/2**20, (t2-t1)))

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
        if command.lstrip().startswith('meeting'):
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

    def _message_received(self, socket):
        msg = self._decode_socket_message(socket)
        if msg is None:
            return  False # Did not get full message yet.
        if 'id' not in msg:
            msg['id'] = self._peer_id(socket)
        if 'scene' in msg:
            self._restore_session(msg['scene'])
        if 'command' in msg:
            self._run_command(msg)
        for t in self._trackers:
            t.update_model(msg)
        self._relay_message(msg)
        return True

    def _send_message(self, msg, sockets = None):
        if 'id' not in msg and self._server:
            msg['id'] = 0
        ba = self._encode_message_data(msg)
        if sockets is None:
            sockets = self._connections
        for socket in sockets:
            socket.write(ba)

    def _encode_message_data(self, msg_data):
        '''
        msg_data is a dictionary which is sent as a string
        prepended by 4 bytes giving the length of the dictionary string.
        We include the length in order to know where a dictionary ends
        when transmitted over a socket.
        '''
        msg = repr(msg_data).encode('utf-8')
        from numpy import array, uint32
        msg_len = array([len(msg)], uint32).tobytes()
        from PyQt5.QtCore import QByteArray
        ba = QByteArray(msg_len + msg)
        return ba

    def _decode_socket_message(self, socket):
        '''
        Return dictionary decoded from socket stream.
        If message is incomplete buffer bytes and return None.
        '''
        rbytes = socket.readAll()

#        lmsg = ('Received %d bytes from %s:%d, xyz = %s'
#                % (len(bytes), socket.peerAddress().toString(), socket.peerPort(), bytes))
#        self._session.logger.info(lmsg)

        mbuf = self._message_buffer
        if socket not in mbuf:
            mbuf[socket] = bytearray()
        msg_bytes = mbuf[socket]
        msg_bytes.extend(rbytes)

        if len(msg_bytes) < 4:
            return None

        from numpy import frombuffer, uint32
        msg_len = frombuffer(msg_bytes[:4], uint32)[0]
        self._report_message_status(len(msg_bytes), msg_len+4)
        if len(msg_bytes) < msg_len + 4:
            return None

        msg = msg_bytes[4:4+msg_len].decode('utf-8')
        mbuf[socket] = msg_bytes[4+msg_len:]
        import ast
        msg_data = ast.literal_eval(msg)
        return msg_data

    def _report_message_status(self, bytes_received, message_bytes):
        '''Report progress receiving session from a peer.'''
        if bytes_received >= message_bytes:
            lt = self._last_status_time
            if lt is not None and lt > self._status_start_time:
                from time import time
                t = time()
                msg = ('Received %.1f Mbytes in %.1f seconds'
                       % (message_bytes / 2**20, t - self._status_start_time))
                self._session.logger.status(msg)
            self._last_status_time = None
            return
        from time import time
        t = time()
        lt = self._last_status_time
        if lt is None:
            self._status_start_time = t
            self._last_status_time = t
            self._last_status_bytes = bytes_received
        elif t - lt > self._status_report_interval:
            percent = 100 * bytes_received/message_bytes
            rate = ((bytes_received - self._last_status_bytes) / 2**20) / (t - lt)
            msg = ('Receiving data %.0f%% of %.1f Mbytes, (%.1f Mbytes/sec)'
                   % (percent, message_bytes / 2**20, rate))
            self._session.logger.status(msg)
            self._last_status_time = t
            self._last_status_bytes = bytes_received
                                    
    def _relay_message(self, msg):
        if len(self._connections) <= 1 or self._server is None:
            return
        emsg = self._encode_message_data(msg)
        for socket, pid in self._peer_ids.items():
            if pid != msg['id']:
                socket.write(emsg)

    def _peer_id(self, socket):
        pids = self._peer_ids
        if socket in pids:
            pid = pids[socket]
        else:
            pids[socket] = pid = self._next_peer_id
            self._next_peer_id += 1
        return pid
            
    def _socket_error(self, error_type, socket):
        self._session.logger.info('Socket error %s' % socket.errorString())
        if self._server is None:
            self.close_all_connections()
        else:
            self._socket_closed(socket)

    def _disconnected(self):
        con = []
        from PyQt5.QtNetwork import QAbstractSocket
        for c in self._connections:
            if c.state() == QAbstractSocket.ConnectedState:
                con.append(c)
            else:
                self._socket_closed(c)
        self._connections = con

    def _socket_closed(self, socket):
        if socket in self._peer_ids:
            peer_id = self._peer_id(socket)
            del self._peer_ids[socket]
            for t in self._trackers:
                t.remove_model(peer_id)
        if socket in self._message_buffer:
            del self._message_buffer[socket]
        self._session.logger.status('Disconnected from %s:%d'
                                    % (socket.peerAddress().toString(), socket.peerPort()))

class PointerModels:
    '''Manage mouse or VR pointer models for all connected hosts.'''
    def __init__(self, session):
        self._session = session
        self._pointer_models = {}	# Map peer id to MousePointerModel or VRPointerModel

    def delete(self):
        for peer_id in tuple(self._pointer_models.keys()):
            self.remove_model(peer_id)

    def pointer_model(self, peer_id = None):
        pm = self._pointer_models
        if peer_id in pm:
            m = pm[peer_id]
            if not m.deleted:
                return m

        m = self.make_pointer_model(self._session)
        models = self._session.models
        models.add([m], minimum_id = 100)
        pm[peer_id] = m
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

    def remove_model(self, peer_id):
        pm = self._pointer_models
        if peer_id in pm:
            m = pm[peer_id]
            del pm[peer_id]
            if not m.deleted:
                self._session.models.close([m])

class MouseTracking(PointerModels):
    def __init__(self, session, meeting):
        PointerModels.__init__(self, session)
        self._meeting = meeting		# MeetingServer instance

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

    def _mouse_hover_cb(self, trigger_name, xyz):
        if _vr_camera(self._session):
            return

        c = self._session.main_view.camera
        axis = c.view_direction()
        msg = {'name': self._meeting._name,
               'color': tuple(self._meeting._color),
               'mouse': (tuple(xyz), tuple(axis)),
               }

        # Update my own mouse pointer position
        self.update_model(msg)

        # Tell connected peers my new mouse pointer position.
        self._meeting._send_message(msg)

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

    def update_pointer(self, msg):
        if 'name' in msg:
            if 'id' in msg:  # If id not in msg leave name as "my pointer".
                self.name = '%s pointer' % msg['name']
        if 'color' in msg:
            self.color = msg['color']
        if 'mouse' in msg:
            xyz, axis = msg['mouse']
            from chimerax.core.geometry import vector_rotation, translation
            p = translation(xyz) * vector_rotation((0,0,1), axis)
            self.position = p

class VRTracking(PointerModels):
    def __init__(self, session, meeting, sync_coords = True, update_interval = 1):
        PointerModels.__init__(self, session)
        self._meeting = meeting		# MeetingServer instance
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
        # Make sure new meeting participant gets my head image.
        self._send_face_image = True
        
        pm = VRPointerModel(self._session, 'VR', self._last_room_to_scene)
        return pm

    def new_face_image(self, path):
        self._new_face_image = path
        
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
        msg = {'name': self._meeting._name,
               'color': tuple(self._meeting._color),
               'vr head': self._head_position(c),	# In room coordinates
               'vr hands': self._hand_positions(c),	# In room coordinates
               }

        # Report scene moved in room
        if scene_moved:
            msg['vr coords'] = _place_matrix(c.room_to_scene)
            self._last_room_to_scene = c.room_to_scene

        # Report VR face image change.
        if self._new_face_image:
            image = _encode_face_image(self._new_face_image)
            self._face_image = image
            msg['vr head image'] = image
            self._new_face_image = None
            self._send_face_image = False
        elif self._send_face_image:
            self._send_face_image = False
            if self._face_image is not None:
                msg['vr head image'] = self._face_image

        # Report changes in VR GUI panel
        ui = c.user_interface
        shown = ui.shown()
        gui_state = self._gui_state
        shown_changed = (shown != gui_state['shown'])
        if shown_changed:
            msg['vr gui shown'] = gui_state['shown'] = shown
        if shown:
            size, rpos, rgba = ui.size(), ui.drawing.room_position, ui.panel_image_rgba()
            if size != gui_state['size'] or shown_changed:
                msg['vr gui size'] = gui_state['size'] = size
            if rpos is not gui_state['room position'] or shown_changed:
                gui_state['room position'] = rpos
                msg['vr gui position'] = _place_matrix(rpos)
            if rgba is not gui_state['image'] or shown_changed:
                gui_state['image'] = rgba
                msg['vr gui image'] = _encode_numpy_array(rgba)
                
        # Tell connected peers my new vr state
        self._meeting._send_message(msg)

    def _head_position(self, vr_camera):
        from chimerax.core.geometry import scale
        return _place_matrix(vr_camera.room_position * scale(1/vr_camera.scene_scale))

    def _hand_positions(self, vr_camera):
        # Hand controller room position includes scaling from room to scene coordinates
        return [_place_matrix(h.room_position) for h in vr_camera._controller_models]

    def _reposition_vr_head_and_hands(self, camera):
        '''
        If my room to scene coordinates change correct the VR head and hand model
        scene positions so they maintain the same apparent position in the room.
        '''
        c = camera
        rts = c.room_to_scene
        for peer_id, vrm in self.pointer_models:
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

	# Last room to scene transformation for this peer.
        # Used if we are not using VR camera so have no room coordinates.
        self._room_to_scene = room_to_scene

    def _hand_models(self, nhands):
        new_hands = [VRHandModel(self.session, 'Hand %d' % (i+1), color=self._color)
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
                h.color = msg['color']
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
        if 'vr gui shown' in msg:
            self._gui_panel.display = msg['vr gui shown']
        if 'vr gui size' in msg:
            self._gui_panel.set_size(msg['vr gui size'])
        if 'vr gui position' in msg:
            g = self._gui_panel
            g.room_position = rp = _matrix_place(msg['vr gui position'])
            g.position = self.room_to_scene * rp
        if 'vr gui image' in msg:
            self._gui_panel.update_image(msg['vr gui image'])

class VRHandModel(Model):
    '''Radius and height in meters.'''
    casts_shadows = False
    pickable = False
    skip_bounds = True
    SESSION_SAVE = False
    
    def __init__(self, session, name, radius = 0.04, height = 0.2, color = (0,255,0,255)):
        Model.__init__(self, name, session)
        self.room_position = None
        
        from chimerax.surface import cone_geometry
        va, na, ta = cone_geometry(radius = radius, height = height, points_up = False)
        va[:,2] += 0.5*height	# Place tip of cone at origin
        self.set_geometry(va, na, ta)
        self.color = color

class VRHeadModel(Model):
    '''Size in meters.'''
    casts_shadows = False
    pickable = False
    skip_bounds = True
    SESSION_SAVE = False
    default_face_file = 'face.png'
    def __init__(self, session, name = 'Head', size = 0.3, image_file = None):
        Model.__init__(self, name, session)
        self.room_position = None
        
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
        from chimerax.core.graphics import qimage_to_numpy, Texture
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
        from chimerax.core.graphics import qimage_to_numpy, Texture
        rgba = qimage_to_numpy(qi)
        r = self.session.main_view.render
        r.make_current()
        self.texture.delete_texture()
        self.texture = Texture(rgba)

class VRGUIModel(Model):
    '''Size in meters.'''
    casts_shadows = False
    pickable = False
    skip_bounds = True
    SESSION_SAVE = False

    def __init__(self, session, name = 'GUI Panel'):
        Model.__init__(self, name, session)
        self.room_position = None
        self._size = (1,1)		# Meters

    def set_size(self, size):
        self._size = (rw, rh) = size
        from chimerax.core.graphics.drawing import position_rgba_drawing
        position_rgba_drawing(self, pos = (-0.5*rw,-0.5*rh), size = (rw,rh))
        
    def update_image(self, encoded_rgba):
        rw, rh = self._size
        rgba = _decode_numpy_array(encoded_rgba)
        r = self.session.main_view.render
        r.make_current() # Required for deleting previous texture in rgba_drawing()
        from chimerax.core.graphics.drawing import rgba_drawing
        rgba_drawing(self, rgba, pos = (-0.5*rw,-0.5*rh), size = (rw,rh))

def _vr_camera(session):
    c = session.main_view.camera
    from chimerax.vive.vr import SteamVRCamera
    return c if isinstance(c, SteamVRCamera) else None

def _place_matrix(p):
    '''Encode Place as tuple for sending over socket.'''
    return tuple(tuple(row) for row in p.matrix)

def _matrix_place(m):
    from chimerax.core.geometry import Place
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
    data = {
        'shape': tuple(array.shape),
        'dtype': array.dtype.str,
        'data': b64encode(array.tobytes())
    }
    return data

def _decode_numpy_array(array_data):
    shape = array_data['shape']
    dtype = array_data['dtype']
    from base64 import b64decode
    data = b64decode(array_data['data'])
    import numpy
    a = numpy.frombuffer(data, dtype).reshape(shape)
    return a
