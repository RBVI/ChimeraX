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
def vr_sync(session, ip_address = None, port = 52194):
    '''Allow two ChimeraX instances running VR to show each others handcontroller
    and headset positions.

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
            msg = "No current VR connections"
        elif not s.connected():
            msg = "Listening for VR connections at %s:%d" % (s.ip_address, s.port)
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
        msg = "Started listening for VR connections at %s:%d" % (s.ip_address, s.port)
        session.logger.status(msg, log = True)
    else:
        s = connection_server(session, create = True)
        s.connect(ip_address, port = port)
        msg = "Connected to %s:%d" % (s.connected_ip_address, s.connected_port)
        session.logger.status(msg, log = True)
        
# -----------------------------------------------------------------------------
# Register the vr sync command for ChimeraX.
#
def register_vr_command(logger):
    from chimerax.core.commands import CmdDesc, BoolArg, FloatArg, PlaceArg, Or, EnumOf, NoArg
    from chimerax.core.commands import register, create_alias
    desc = CmdDesc(optional = [('ip_address', StringArg)],
                   keyword = [('port', IntArg)],
                   synopsis = 'Show synchronized VR hand controllers between two ChimeraX instances')
    register('vr_sync', desc, vr_sync, logger=logger)

# -----------------------------------------------------------------------------
#
def connection_server(session, create = False):
    if hasattr(session, '_vr_sync_server'):
        s =session._vr_sync_server
    elif create:
        session._vr_sync_server = s = VRSyncServer(session)
    else:
        s = None
    return s

# -----------------------------------------------------------------------------
#
class VRSyncServer:
    def __init__(self, session):
        self._session = session
        self._server = None
        self._connections = []		# List of QTcpSocket

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
        self._server = s = QTcpServer()
        if not s.listen(QHostAddress.Any, port):
            self._session.logger.warning('QTcpServer.listen() failed: %s' % s.errorString())
        else:
            s.newConnection.connect(self._new_connection)
            self.ip_address = s.serverAddress().toString()
            self.port = s.serverPort()

    def connect(self, ip_address, port = port):
        if self._server:
            raise RuntimeError('VRSyncServer: Must call either listen, or connect, not both')
        socket = QTcpSocket()
        socket.error.connect(lambda self=self, socket=socket: self._socket_error(socket))
        socket.connectToHost(ip_address, port)
        self._add_connection(socket)

    def _socket_error(self, socket):
        self._session.logger.info('Socket error %s' % socket.errorString())
        
    def _new_connection(self):
        s = self._server
        while s.hasPendingConnections():
            socket = s.nextPendingConnection()
            self._add_connection(socket)

    def _add_connection(self, socket):
        socket.disconnected.connected(self._disconnected)
        self._connections.append(socket)
        socket.readyRead.connect(lambda self=self, socket=socket: self._message_received(socket))
        if self._server is None:
            #TODO: Set new frame trigger and send position updates.
            pass
    def _disconnected(self):
        con = []
        for c in self._connections:
            if c.state() == QAbstractSocket.ConnectedState:
                con.append(c)
            else:
                self._session.logger.status('Disconnected VR sync to %s:%d'
                                            % (c.peerAddress().toString(), c.peerPort()))
        self._connections = con

    def _message_received(self, socket):
        bytes = socket.readAll()
        self._session.logger.info('Received %d bytes from %s:%d'
                                  % (len(bytes), socket.peerAddress().toString(), socket.peerPort()))
        # TODO: if we are server, reply with our position info.
