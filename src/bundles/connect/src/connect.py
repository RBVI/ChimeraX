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
def connect(session, ip_address = None, port = 52194):
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
    '''

    if ip_address is None:
        s = connection_server(session)
        if s is None:
            msg = "No current ChimeraX connections"
        elif not s.connected():
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
        
# -----------------------------------------------------------------------------
# Register the connect command for ChimeraX.
#
def register_connect_command(logger):
    from chimerax.core.commands import CmdDesc, StringArg, IntArg
    from chimerax.core.commands import register, create_alias
    desc = CmdDesc(optional = [('ip_address', StringArg)],
                   keyword = [('port', IntArg)],
                   synopsis = 'Show synchronized mouse or VR hand controllers between two ChimeraX instances')
    register('connect', desc, connect, logger=logger)

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
        self._server = None
        self._connections = []		# List of QTcpSocket
        self._mouse_pointer_model = None

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
        socket.error.connect(lambda error_type, self=self, socket=socket: self._socket_error(error_type, socket))
        socket.connectToHost(ip_address, port)
        self._add_connection(socket)

    def _socket_error(self, error_type, socket):
        self._session.logger.info('Socket error %s' % socket.errorString())
        
    def _new_connection(self):
        s = self._server
        while s.hasPendingConnections():
            socket = s.nextPendingConnection()
            self._add_connection(socket)

    def _add_connection(self, socket):
        socket.disconnected.connect(self._disconnected)
        self._connections.append(socket)
        socket.readyRead.connect(lambda self=self, socket=socket: self._message_received(socket))
        self._session.triggers.add_handler('mouse hover', self._mouse_hover_cb)
        if self._server is None:
            pass
            # TODO: Set new frame trigger and send pointer position updates.
            # TODO: For non-vr send pointer position for mouse hover.

    def _disconnected(self):
        con = []
        from PyQt5.QtNetwork import QAbstractSocket
        for c in self._connections:
            if c.state() == QAbstractSocket.ConnectedState:
                con.append(c)
            else:
                self._session.logger.status('Disconnected connection from %s:%d'
                                            % (c.peerAddress().toString(), c.peerPort()))
        self._connections = con

    def _message_received(self, socket):
        bmsg = socket.readAll()
        msg = bytes(bmsg).decode('utf-8')
        lmsg = ('Received %d bytes from %s:%d, xyz = %s'
                % (len(bmsg), socket.peerAddress().toString(), socket.peerPort(), msg))
#        self._session.logger.info(lmsg)
        m = self._mouse_pointer_model
        if m is None:
            from chimerax.core.models import Model
            self._mouse_pointer_model = m = Model('mouse pointer', self._session)
            from chimerax.core.surface import cone_geometry
            h = 3
            va, na, ta = cone_geometry(radius = 1, height = h)
            va[:,2] -= 0.5*h	# Place tip of cone at origin
            m.vertices = va
            m.normals = na
            m.triangles = ta
            m.color = (0,255,0,255)
            self._session.models.add([m])
        values = [float(x) for x in msg.split(',')]
        xyz = values[:3]
        axis = values[3:]
        from chimerax.core.geometry import vector_rotation, translation
        p = translation(xyz) * vector_rotation((0,0,1), axis)
        m.position = p
        
        # TODO: Draw pointers at positions in message.
        # TODO: if we are server, reply with our position info.

    def _mouse_hover_cb(self, trigger_name, xyz):
        msg = '%.6g,%.6g,%.6g' % tuple(xyz)
        d = self._session.main_view.camera.view_direction()
        msg += ',%.6g,%.6g,%.6g' % tuple(d)
        from PyQt5.QtCore import QByteArray
        ba = QByteArray(msg.encode('utf-8'))
        for socket in self._connections:
            socket.write(ba)
        
