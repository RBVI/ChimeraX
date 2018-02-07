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
def connect(session, ip_address = None, port = 52194, name = None, color = None,
            head_image = None, copy_scene = True):
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
    head_image : string
      Path to PNG or JPG image file for image to use for VR head depiction.
    copy_scene : bool
      Whether to copy the open models to the peer machines.
    '''

    if ip_address is None:
        s = connection_server(session)
        if s is None:
            msg = "No ChimeraX connections"
        else:
            lines = []
            lip = s.listen_ip_port()
            if lip:
                lines.append("Listening for ChimeraX connections at %s port %d" % lip)
            clist = s.connected_ip_port_list()
            if clist:
                lines.extend(["Connected to %s port %d" % (ip,port) for (ip,port) in clist])
            msg = '\n'.join(lines) if lines else "No ChimeraX connections"
        session.logger.status(msg, log = True)
    elif ip_address == 'accept':
        s = connection_server(session, create = True)
        s.listen(port)
        msg = "Listening for ChimeraX connections at %s port %d" % s.listen_ip_port()
        session.logger.status(msg, log = True)
    else:
        s = connection_server(session, create = True)
        s.connect(ip_address, port)

    if name is not None:
        s = connection_server(session)
        if s:
            s._name = name

    if color is not None:
        s = connection_server(session)
        if s:
            s.set_color(color.uint8x4())

    if head_image is not None:
        s = connection_server(session)
        if s:
            s.vr_tracker.new_head_image(head_image)

    if copy_scene:
        s = connection_server(session)
        if s:
            s.copy_scene(copy_scene)

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
    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import StringArg, IntArg, ColorArg, OpenFileNameArg, BoolArg
    desc = CmdDesc(optional = [('ip_address', StringArg)],
                   keyword = [('port', IntArg),
                              ('name', StringArg),
                              ('color', ColorArg),
                              ('head_image', OpenFileNameArg),
                              ('copy_scene', BoolArg)],
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
        self._mouse_tracker = None
        self._vr_tracker = None
        self._copy_scene = True

    def set_color(self, color):
        self._color = color
        
    @property
    def connected(self):
        c = self._connections
        return len(c) > 0
    
    def listen_ip_port(self):
        s = self._server
        if s and s.isListening():
            return (s.serverAddress().toString(), s.serverPort())
        return None
    
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
            self._session.logger.warning('QTcpServer.listen() failed: %s' % s.errorString())
        else:
            s.newConnection.connect(self._new_connection)
        self._color = (255,255,0,255)

    @property
    def listening(self):
        return self._server is not None

    def _available_server_ipv4_addresses(self):
        from PyQt5.QtNetwork import QNetworkInterface, QAbstractSocket
        a = [ha for ha in QNetworkInterface.allAddresses()
             if not ha.isLoopback()
             and not ha.isNull()
             and ha.protocol() == QAbstractSocket.IPv4Protocol
             and not ha.toString().startswith('169.254')] # Exclude link-local addresses
        return a

    def copy_scene(self, copy):
        self._copy_scene = copy
        if copy:
            self._copy_scene_to_peers(self._connections)
        
    def connect(self, ip_address, port):
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
            self._copy_scene_to_peers([socket])

    def _copy_scene_to_peers(self, sockets):
        if self._session.models.empty():
            return
        msg = {'scene': self._encode_session()}
        self._send_message(msg, sockets=sockets)

    def _encode_session(self):
        from io import BytesIO
        stream = BytesIO()
        self._session.save(stream, version=3)
        from base64 import b64encode
        sbytes = b64encode(stream.getbuffer())
        return sbytes

    def _restore_session(self, base64_sbytes):
        from base64 import b64decode
        sbytes = b64decode(base64_sbytes)
        from io import BytesIO
        stream = BytesIO(sbytes)
        self._session.restore(stream)
            
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

class PointerModels:
    '''Manage mouse or VR pointer models for all connected hosts.'''
    def __init__(self, session):
        self._session = session
        self._pointer_models = {}	# Map peer id to MousePointerModel

    def delete(self):
        for peer_id in tuple(self._pointer_models.keys()):
            self.remove_model(peer_id)

    def pointer_model(self, peer_id = None):
        pm = self._pointer_models
        if peer_id in pm:
            m = pm[peer_id]
        else:
            m = self.make_pointer_model(self._session)
            self._session.models.add([m])
            pm[peer_id] = m
        return m

    def make_pointer_model(self, session):
        # Must be defined by subclass.
        pass

    def update_model(self, msg):
        m = self.pointer_model(msg.get('id'))
        m.update_pointer(msg)

    def remove_model(self, peer_id):
        pm = self._pointer_models
        if peer_id in pm:
            m = pm[peer_id]
            del pm[peer_id]
            self._session.models.close([m])

class MouseTracking(PointerModels):
    def __init__(self, session, connect):
        PointerModels.__init__(self, session)
        self._connect = connect		# ConnectServer instance

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
        self.update_model(msg)

        # Tell connected peers my new mouse pointer position.
        self._connect._send_message(msg)

from chimerax.core.models import Model
class MousePointerModel(Model):
    SESSION_SAVE = False
    
    def __init__(self, session, name, radius = 1, height = 3, color = (0,255,0,255)):
        Model.__init__(self, name, session)
        from chimerax.core.surface import cone_geometry
        va, na, ta = cone_geometry(radius = radius, height = height)
        va[:,2] -= 0.5*height	# Place tip of cone at origin
        self.vertices = va
        self.normals = na
        self.triangles = ta
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
    def __init__(self, session, connect, sync_coords = True):
        PointerModels.__init__(self, session)
        self._connect = connect		# ConnectServer instance
        self._sync_coords = sync_coords

        t = session.triggers
        self._vr_tracking_handler = t.add_handler('new frame', self._vr_tracking_cb)
        self._vr_update_interval = 9	# Send vr position every N frames.
        self._last_room_to_scene = None
        self._new_head_image = None	# Path to image file

    def delete(self):
        t = self._session.triggers
        t.remove_handler(self._vr_tracking_handler)
        self._vr_tracking_handler = None

        PointerModels.delete(self)

    def update_model(self, msg):
        if 'vr coords' in msg and self._sync_coords:
            c = self._session.main_view.camera
            from chimerax.vive.vr import SteamVRCamera
            if isinstance(c, SteamVRCamera):
                from chimerax.core.geometry import Place
                c.room_to_scene = Place(matrix = msg['vr coords'])
                self._last_room_to_scene = c.room_to_scene
        if 'vr head' in msg:
            PointerModels.update_model(self, msg)

    def make_pointer_model(self, session):
        return VRPointerModel(self._session, 'vr head and hands')

    def new_head_image(self, path):
        self._new_head_image = path
        
    def _vr_tracking_cb(self, trigger_name, *unused):
        v = self._session.main_view
        c = v.camera
        from chimerax.vive.vr import SteamVRCamera
        if not isinstance(c, SteamVRCamera):
            return
        if v.frame_number % self._vr_update_interval != 0:
            return

        msg = {'name': self._connect._name,
               'color': tuple(self._connect._color),
               'vr head': self._head_position(c),
               'vr hands': self._hand_positions(c),
               }
        if c.room_to_scene is not self._last_room_to_scene:
            msg['vr coords'] = _place_matrix(c.room_to_scene)
            self._last_room_to_scene = c.room_to_scene
        if self._new_head_image:
            from base64 import b64encode
            hf = open(self._new_head_image, 'rb')
            msg['vr head image'] = b64encode(hf.read())
            hf.close()
            self._new_head_image = None
            
        # Tell connected peers my new vr state
        self._connect._send_message(msg)

    def _head_position(self, vr_camera):
        from chimerax.core.geometry import scale
        return _place_matrix(vr_camera.position * scale(1/vr_camera.scene_scale))

    def _hand_positions(self, vr_camera):
        return [_place_matrix(h.position) for h in vr_camera._controller_models]

from chimerax.core.models import Model
class VRPointerModel(Model):
    SESSION_SAVE = False
    
    def __init__(self, session, name, color = (0,255,0,255)):
        Model.__init__(self, name, session)
        self._head = h = VRHeadModel(session)
        self.add([h])
        self._hands = []
        self._color = color

    def _hand_models(self, nhands):
        new_hands = [VRHandModel(self.session, 'hand %d' % (i+1), color=self._color)
                     for i in range(len(self._hands), nhands)]
        if new_hands:
            self.add(new_hands)
            self._hands.extend(new_hands)
        return self._hands[:nhands]

    def update_pointer(self, msg):
        if 'name' in msg:
            if 'id' in msg:  # If id not in msg leave name as "my pointer".
                self.name = '%s vr head and hands' % msg['name']
        if 'color' in msg:
            for h in self._hands:
                h.color = msg['color']
        if 'vr head' in msg:
            from chimerax.core.geometry import Place
            self._head.position = Place(matrix = msg['vr head'])
        if 'vr head image' in msg:
            self._head.update_image(msg['vr head image'])
        if 'vr hands' in msg:
            hpos = msg['vr hands']
            from chimerax.core.geometry import Place
            for h,hm in zip(self._hand_models(len(hpos)), hpos):
                h.position = Place(matrix = hm)

class VRHandModel(Model):
    '''Radius and height in meters.'''
    SESSION_SAVE = False
    
    def __init__(self, session, name, radius = 0.04, height = 0.2, color = (0,255,0,255)):
        Model.__init__(self, name, session)
        from chimerax.core.surface import cone_geometry
        va, na, ta = cone_geometry(radius = radius, height = height, points_up = False)
        va[:,2] += 0.5*height	# Place tip of cone at origin
        self.vertices = va
        self.normals = na
        self.triangles = ta
        self.color = color

class VRHeadModel(Model):
    '''Size in meters.'''
    SESSION_SAVE = False
    default_face_file = 'face.png'
    def __init__(self, session, name = 'head', size = 0.3, image_file = None):
        Model.__init__(self, name, session)
        
        r = size / 2
        from chimerax.core.surface import box_geometry
        va, na, ta = box_geometry((-r,-r,-0.1*r), (r,r,0.1*r))
        self.vertices = va
        self.normals = na
        self.triangles = ta
        self.color = (255,255,255,255)

        if image_file is None:
            from os.path import join, dirname
            image_file = join(dirname(__file__), self.default_face_file)
        from PyQt5.QtGui import QImage
        qi = QImage(image_file)
        from chimerax.core.graphics import qimage_to_numpy, Texture
        rgba = qimage_to_numpy(qi)
        from numpy import zeros, float32
        tc = zeros((24,2), float32)
        tc[:] = 0.5
        tc[8:12,:] = ((0,0), (1,0), (0,1), (1,1))

        self.texture = Texture(rgba)
        self.texture_coordinates = tc

    def update_image(self, base64_image_bytes):
        from base64 import b64decode
        image_bytes = b64decode(base64_image_bytes)
        from PyQt5.QtGui import QImage
        qi = QImage()
        qi.loadFromData(image_bytes)
        from chimerax.core.graphics import qimage_to_numpy, Texture
        rgba = qimage_to_numpy(qi)
        r = self.session.main_view.render
        r.make_current()
        self.texture.delete_texture()
        self.texture = Texture(rgba)

def _place_matrix(p):
    '''Encode Place as tuple for sending over socket.'''
    return tuple(tuple(row) for row in p.matrix)
