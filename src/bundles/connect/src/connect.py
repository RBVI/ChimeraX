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
def connect(session, ip_address = None, port = 52194, name = None, color = None):
    '''Allow two ChimeraX instances to show each others' mouse positions or
    VR handcontroller and headset positions.

    Parameters
    ----------
    ip_address : string
      IP address of the other ChimeraX to sync with, for example, "169.230.21.39".
      One ChimeraX specifies this as "accept" which starts listening for connections
      on the host machines IP address which is reported to the log.  Then the other ChimeraX
      specifies that server IP address.  If no address is specified, the current connections
      are reported to the log.  If "quit" is specified then connections are dropped and
      the server is closed.
    port : int
      Port number to connect to.  Can be omitted in which case default 52194 is used.
    name : string
      Name to identify this ChimeraX on remote machines.
    color : Color
      Color for my mouse pointer shown on other machines
    '''

    if ip_address is None:
        s = connection_server(session)
        if s is None:
            msg = "No current ChimeraX connections"
        elif not s.connected:
            msg = "Listening for ChimeraX connections at %s:%d" % (s.ip_address, s.port)
        else:
            msg = "Connected to %s:%d" % (s.connected_ip_address, s.connected_port)
        session.logger.status(msg, log = True)
    elif ip_address == 'quit':
        s = connection_server(session)
        if s:
            s.quit()
    elif ip_address == 'accept':
        s = connection_server(session, create = True)
        s.listen(port = port)
        msg = "Started listening for ChimeraX connections at %s:%d" % (s.ip_address, s.port)
        session.logger.status(msg, log = True)
    else:
        s = connection_server(session, create = True)
        s.connect(ip_address, port = port)
        msg = "Connected to %s:%d" % (s.connected_ip_address, s.connected_port)
        session.logger.status(msg, log = True)

    if name is not None:
        s = connection_server(session)
        if s:
            s._name = name

    if color is not None:
        s = connection_server(session)
        if s:
            s.set_color(color.uint8x4())

# -----------------------------------------------------------------------------
#
def connect_close(session):
    '''Close all connection shared pointers.'''
    s = connection_server(session)
    if s:
        s.close_all_connections()

        
# -----------------------------------------------------------------------------
# Register the connect command for ChimeraX.
#
def register_connect_command(logger):
    from chimerax.core.commands import CmdDesc, StringArg, IntArg, ColorArg
    from chimerax.core.commands import register, create_alias
    desc = CmdDesc(optional = [('ip_address', StringArg)],
                   keyword = [('port', IntArg),
                              ('name', StringArg),
                              ('color', ColorArg),],
                   synopsis = 'Show synchronized mouse or VR hand controllers between two ChimeraX instances')
    register('connect', desc, connect, logger=logger)
    desc = CmdDesc(synopsis = 'Close synchronized pointer connection')
    register('connect close', desc, connect_close, logger=logger)

# -----------------------------------------------------------------------------
#
def connection_server(session, create = False):
    if hasattr(session, '_connect_server'):
        s =session._connect_server
    elif create:
        session._connect_server = s = ConnectServer(session)
    else:
        s = None
    return s

# -----------------------------------------------------------------------------
#
class ConnectServer:
    def __init__(self, session):
        self._session = session
        self._name = 'remote'
        self._color = (0,255,0,255)	# Tracking model color
        self._server = None
        self._connections = []		# List of QTcpSocket
        self._message_buffer = {}	# Map of socket to bytearray of buffered data
        self._peer_ids = {}		# Number associated with each ChimeraX instance passed in messages from server.
        self._next_peer_id = 1
        self._trackers = []

    def set_color(self, color):
        self._color = color
        
    @property
    def connected(self):
        c = self._connections
        return len(c) > 0
    
    @property
    def ip_address(self):
        s = self._server
        if s and s.isListening():
            return s.serverAddress().toString()
        c = self._connections
        if c:
            return c[0].localAddress().toString()
        return None

    @property
    def port(self):
        s = self._server
        if s and s.isListening():
            return s.serverPort()
        c = self._connections
        if c:
            return c[0].localPort()
        return None
    
    @property
    def connected_ip_address(self):
        c = self._connections
        if c:
            return c[0].peerAddress().toString()
        return None

    @property
    def connected_port(self):
        c = self._connections
        if c:
            return c[0].peerPort()
        return None

    def listen(self, port):
        from PyQt5.QtNetwork import QTcpServer, QHostAddress
        self._server = s = QTcpServer()
        aa = self._available_server_ipv4_addresses()
        a = aa[0] if aa else QHostAddress.Any
        if not s.listen(a, port):
            self._session.logger.warning('QTcpServer.listen() failed: %s' % s.errorString())
        else:
            s.newConnection.connect(self._new_connection)
        self._color = (255,255,0,255)

    def _available_server_ipv4_addresses(self):
        from PyQt5.QtNetwork import QNetworkInterface, QAbstractSocket
        a = [ha for ha in QNetworkInterface.allAddresses()
             if not ha.isLoopback() and not ha.isNull() and
             ha.protocol() == QAbstractSocket.IPv4Protocol]
        return a
            
    def connect(self, ip_address, port = port):
        if self._server:
            raise RuntimeError('ConnectServer: Must call either listen, or connect, not both')
        from PyQt5.QtNetwork import QTcpSocket
        socket = QTcpSocket()
        def socket_error(error_type, self=self, socket=socket):
            self._socket_error(error_type, socket)
        socket.error.connect(socket_error)
        socket.connectToHost(ip_address, port)
        self._add_connection(socket)

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

    def _add_connection(self, socket):
        socket.disconnected.connect(self._disconnected)
        self._connections.append(socket)

        # Register callback called when data available to read on socket.
        def read_socket(self=self, socket=socket):
            self._message_received(socket)
        socket.readyRead.connect(read_socket)

        self._initiate_tracking()

    def _initiate_tracking(self):
        if not self._trackers:
            s = self._session
            self._trackers = [MouseTracking(s, self)]
#                              VRTracking(s, self)

    def _message_received(self, socket):
        msg = self._decode_socket_message(socket)
        if msg is None:
            return  # Did not get full message yet.
        if 'id' not in msg:
            msg['id'] = self._peer_id(socket)
        for t in self._trackers:
            t.update_model(msg)
        self._relay_message(msg)

    def _send_message(self, msg):
        if 'id' not in msg and self._server:
            msg['id'] = 0
        ba = self._encode_message_data(msg)
        for socket in self._connections:
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
        if len(msg_bytes) < msg_len + 4:
            return None

        msg = msg_bytes[4:4+msg_len].decode('utf-8')
        mbuf[socket] = msg_bytes[4+msg_len:]
        import ast
        msg_data = ast.literal_eval(msg)
        return msg_data

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
        self._session.logger.status('Disconnected connection from %s:%d'
                                    % (socket.peerAddress().toString(), socket.peerPort()))

class MouseTracking:
    def __init__(self, session, connect):
        self._session = session
        self._connect = connect		# ConnectServer instance
        t = session.triggers
        self._mouse_hover_handler = t.add_handler('mouse hover', self._mouse_hover_cb)
        self._mouse_pointer_models = {}	# Map peer id to pointer model

    def delete(self):
        t = self._session.triggers
        t.remove_handler(self._mouse_hover_handler)
        self._mouse_hover_handler = None

        for peer_id in tuple(self._mouse_pointer_models.keys()):
            self._remove_pointer_model(peer_id)

    def update_model(self, msg):
        self._update_mouse_pointer_model(msg)

    def remove_model(self, peer_id):
        self._remove_pointer_model(peer_id)

    def _mouse_hover_cb(self, trigger_name, xyz):
        c = self._session.main_view.camera
        from chimerax.vive.vr import SteamVRCamera
        if isinstance(c, SteamVRCamera):
            return

        axis = c.view_direction()
        msg = {'name': self._connect._name,
               'color': tuple(self._connect._color),
               'mouse': (tuple(xyz), tuple(axis)),
               }

        # Update my own mouse pointer position
        self._update_mouse_pointer_model(msg)

        # Tell connected peers my new mouse pointer position.
        self._connect._send_message(msg)

    def _update_mouse_pointer_model(self, msg):
        m = self._mouse_pointer_model(msg.get('id'))
        if 'name' in msg:
            if 'id' in msg:  # If id not in msg leave name as "my pointer".
                m.name = '%s pointer' % msg['name']
        if 'color' in msg:
            m.color = msg['color']
        if 'mouse' in msg:
            xyz, axis = msg['mouse']
            from chimerax.core.geometry import vector_rotation, translation
            p = translation(xyz) * vector_rotation((0,0,1), axis)
            m.position = p
            
    def _mouse_pointer_model(self, peer_id = None):
        mpm = self._mouse_pointer_models
        if peer_id in mpm:
            return mpm[peer_id]

        from chimerax.core.models import Model
        mpm[peer_id] = m = Model('my pointer', self._session)
        from chimerax.core.surface import cone_geometry
        h = 3
        va, na, ta = cone_geometry(radius = 1, height = h)
        va[:,2] -= 0.5*h	# Place tip of cone at origin
        m.vertices = va
        m.normals = na
        m.triangles = ta
        m.color = (0,255,0,255)
        self._session.models.add([m])

        return m

    def _remove_pointer_model(self, peer_id):
        mpm = self._mouse_pointer_models
        if peer_id in mpm:
            m = mpm[peer_id]
            del mpm[peer_id]
            self._session.models.close([m])

class VRTracking:
    def __init__(self, session, connect):
        self._session = session
        self._connect = connect		# ConnectServer instance
        t = session.triggers
        self._vr_tracking_handler = t.add_handler('new frame', self._vr_tracking_cb)
        self._vr_update_interval = 9	# Send vr position every N frames.

    def delete(self):
        t = self._session.triggers
        t.remove_handler(self._vr_tracking_handler)
        self._vr_tracking_handler = None

        for peer_id in tuple(self._vr_pointer_models.keys()):
            self._remove_pointer_model(peer_id)

    def update_model(self, msg):
        pass

    def remove_model(self, peer_id):
        pass

    def _vr_tracking_cb(self, trigger_name, frame):
        c = self._session.main_view.camera
        from chimerax.vive.vr import SteamVRCamera
        if not isinstance(c, SteamVRCamera):
            return
        if frame % self._vr_update_interval != 0:
            return

        msg = {'name': self._name,
               'color': tuple(self._color),
               'vr head': _place_matrix(c.position),
               'vr coords': _place_matrix(c.scene_to_room),
               }
        for i,h in enumerate(c._controller_models):
            msg['vr hand %d' % (i+1)] = _place_matrix(h.position)

        # Tell connected peers my new vr state
        self._send_message(msg)

def _place_matrix(p):
    '''Encode Place as tuple for sending over socket.'''
    return tuple(tuple(row) for row in p.matrix)
