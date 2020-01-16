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

import socket, threading, struct, weakref, itertools, ssl
import sys, os.path, logging
try:
    import cPickle as pickle
except ImportError:
    import pickle


_use_ssl = True
_ctx_server = None
_ctx_client = None


def get_ctx_server():
    import os.path, ssl
    global _ctx_server
    if _ctx_server is None:
        cert = os.path.join(os.path.dirname(__file__), "server.pem")
        _ctx_server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        _ctx_server.load_cert_chain(cert)
    return _ctx_server


def make_server_socket(hostname, port, backlog=0):
    import socket
    # Pre-3.8 code
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    s.bind((hostname, port))
    s.listen(5)
    # Post-3.8 code
    # s = socket.create_server((hostname, port), backlog=backlog)
    if _use_ssl:
        s = get_ctx_server().wrap_socket(s, server_side=True)
    return s


def get_ctx_client():
    import os.path, ssl
    global _ctx_client
    if _ctx_client is None:
        # ssl in Python 3.7 on Windows has a problem with TLS 1.3.
        # It raises an error:
        # "ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
        # So explicitly ask for TLS 1.2.
        # This bug might already be fixed in Python 3.8.
        _ctx_client = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        # _ctx_client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        _ctx_client.check_hostname = False
        _ctx_client.verify_mode = ssl.CERT_NONE
    return _ctx_client


def make_client_socket(hostname, port):
    import socket
    s = socket.create_connection((hostname, port))
    if _use_ssl:
        s = get_ctx_client().wrap_socket(s, server_hostname=hostname)
    return s


class PacketType:
    _enum = itertools.count(1)
    Req = next(_enum)
    Resp = next(_enum)

class Req:
    _enum = itertools.count(1)
    # debugging requests
    Echo = next(_enum)
    # admin requests
    Exit = next(_enum)
    GetSessions = next(_enum)
    # Notifications
    Joined = next(_enum)
    Departed = next(_enum)
    # client requests
    CreateSession = next(_enum)
    JoinSession = next(_enum)
    GetSessionInfo = next(_enum)
    GetIdentities = next(_enum)
    Message = next(_enum)

ReqNames = {getattr(Req, a):a for a in dir(Req)
            if isinstance(getattr(Req, a), int)}

class Resp:
    _enum = itertools.count(1)
    Failure = next(_enum)
    Success = next(_enum)

class Notify:
    Joined = "joined"
    Departed = "departed"


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("WARNING")


def req_name(n):
    try:
        return ReqNames[n]
    except KeyError:
        return str(n)


class EndPoint:

    HeaderFormat = "III"
    HeaderSize = struct.calcsize(HeaderFormat)

    def __init__(self, s):
        self._socket = s
        self._lock = threading.Lock()

    def get(self):
        # Only one thread will read /from/ the socket so no thread-safety
        try:
            header_data = self._socket.recv(self.HeaderSize)
            if len(header_data) == 0:
                raise ConnectionError("recv")
        except ConnectionError:
            self.close()
            return None, None, None
        # logger.debug("get: %s: received %d bytes", self, len(header_data))
        try:
            s = header_data.decode().lower()
        except:
            pass
        else:
            if s.startswith("get") or s.startswith("post"):
                self._send(b"HTTP/2.0 418 I'm a teapot\n\n")
                self.close()
                return None, None, None
        dtype, serial, count = struct.unpack(self.HeaderFormat, header_data)
        sdata = bytes()
        while len(sdata) < count:
            new_data = self._socket.recv(count - len(sdata))
            if len(new_data) == 0:
                self._socket.close()
                return None, None, None
            sdata += new_data
        data = pickle.loads(sdata) if len(sdata) > 0 else None
        # logger.debug("get: %s %s %s %s", self, dtype, serial, data)
        return dtype, serial, data

    def put(self, dtype, serial, data):
        # Make "put" thread-safe since multiple clients might want
        # to send simultaneously
        # Serialization code copied from ActiveState recipe
        # logger.debug("put: %s %s %s %s", self, dtype, serial, data)
        if self._socket is None:
            logger.warning("put: socket closed already: %s %s %s %s",
                           self, dtype, serial, data)
            return
        d = pickle.dumps(data)
        sdata = struct.pack(self.HeaderFormat, dtype, serial, len(d)) + d
        with self._lock:
            self._send(sdata)

    def _send(self, *args, **kw):
        try:
            self._socket.send(*args, **kw)
        except ConnectionError:
            self.close()
        except:
            logger.exception("_send: %s", self._socket)
            raise

    def close(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class _BasicClient(threading.Thread):

    def __init__(self, s):
        super().__init__(daemon=True)
        self._endpoint = EndPoint(s)
        self._requests = {}
        self._serial = itertools.count(1)

    def _send_request(self, data, callback, action=Req.Message):
        logger.debug("_send_request: %s %s", action, data)
        serial = next(self._serial)
        self._requests[serial] = callback
        self._endpoint.put(PacketType.Req, serial, (action, data))

    def send_response(self, status, data, serial):
        self._endpoint.put(PacketType.Resp, serial, (status, data))

    def run(self):
        while True:
            dtype, serial, data = self._endpoint.get()
            if dtype is None:
                break
            try:
                self.handle_packet(dtype, serial, data)
            except:
                logger.error("message handler failed", exc_info=sys.exc_info())
        logger.info("run terminated: %s", self)

    def handle_packet(self, dtype, serial, packet):
        if dtype == PacketType.Req:
            action, request = packet
            status, response = self.process_request(action, request)
            self.send_response(status, response, serial)
        elif dtype == PacketType.Resp:
            status, data = packet
            self.process_response(serial, status, data)
        else:
            raise ValueError("unknown packet type: %s" % dtype)

    def process_request(self, action, data):
        raise NotImplementedError("process_request()")

    def process_response(self, serial, status, data):
        try:
            callback = self._requests.pop(serial)
        except KeyError:
            pass
        else:
            if callback is not None:
                callback(status, data)


class _SyncCall:

    def __init__(self, client, action, data):
        self.__event = threading.Event()
        client._send_request(data, self, action=action)
        self.__event.wait()
        if self.__status != Resp.Success:
            raise RuntimeError(self.__data)

    def __call__(self, status, data):
        self.__status = status
        self.__data = data
        self.__event.set()

    @property
    def response(self):
        return self.__data


class Client(_BasicClient):

    def __init__(self, hostname, port, session, identity, create, timeout=None):
        super().__init__(make_client_socket(hostname, port))
        self.session = session
        self.identity = identity
        self._create = create

    def __str__(self):
        return "%s/%s" % (self.session, self.identity)

    def close(self):
        self._endpoint.close()

    def start(self):
        super().start()
        req = Req.CreateSession if self._create else Req.JoinSession
        ev = threading.Event()
        self.identity = self.sync_request(req, (self.session, self.identity))

    def send(self, data, *, receivers=None, callback=None, action=Req.Message):
        packet = {"receivers":receivers, "data":data}
        self._send_request(packet, callback, action=action)

    def get_session_info(self):
        return self.sync_request(Req.GetSessionInfo, None)

    def get_identities(self):
        return self.sync_request(Req.GetIdentities, None)

    def sync_request(self, action, data):
        return _SyncCall(self, action, data).response

    def process_request(self, action, req_data):
        logger.info("process_request: %s", req_name(action))
        sender, data = req_data
        if action == Req.Message:
            return self.handle_message(data, sender)
        elif action == Req.Echo:
            return Resp.Success, data
        elif action == Req.Joined:
            return self.handle_notification(Notify.Joined, data)
        elif action == Req.Departed:
            return self.handle_notification(Notify.Departed, data)
        else:
            logger.error("unknown action: %s", action)
            return Resp.Failure, "unknown action %d" % action

    def handle_message(self, data, sender):
        # "data" comes from remote client
        # "sender" is session identity of requester
        raise NotImplementedError("handle_message()")

    def handle_notification(self, ntype, data):
        raise NotImplementedError("handle_notification()")


class Server(threading.Thread):

    def __init__(self, hostname, port, admin_word):
        super().__init__(daemon=True)
        self._socket = make_server_socket(hostname, port)
        self._handlers = set()
        self._sessions = {}
        self._stopped = False
        self._lock = threading.RLock()
        self._address = None
        self._admin_word = admin_word

    @property
    def service_address(self):
        if self._socket is None:
            return "not_in_service"
        if self._address is None:
            ip, port = self._socket.getsockname()
            try:
                host_info = socket.gethostbyaddr(ip)
            except socket.herror:
                host_name = ip
            else:
                host_name = host_info[0]
            self._address = "%s:%s" % (host_name, port)
        return self._address

    def run(self):
        # Overrides Thread method
        logger.debug("server accepting connections: %s" % self._socket)
        while not self._stopped:
            try:
                ns, addr = self._socket.accept()
                logger.info("server connection:: from %s", addr)
            except socket.timeout:
                pass
            except:
                logger.error("accept() failed", exc_info=sys.exc_info())
            else:
                _Handler(ns, self).start()

    def stop(self):
        with self._lock:
            self._stopped = True

    def register(self, rc):
        with self._lock:
            self._handlers.add(rc)

    def get_session_info(self, data, handler):
        if handler.session is None:
            return Resp.Failure, "not associated with session"
        with self._lock:
            if handler.session in self._sessions:
                return Resp.Success, (self.service_address, handler.session)
            else:
                return Resp.Failure, ("session \"%s\" terminated" %
                                      handler.session)

    def get_identities(self, data, handler):
        if handler.session is None:
            return Resp.Failure, "not associated with session"
        with self._lock:
            try:
                ses = self._sessions[handler.session]
            except KeyError:
                return Resp.Failure, ("session \"%s\" terminated" %
                                      handler.session)
            else:
                return Resp.Success, list(ses.keys())

    def get_sessions(self, data, handler):
        if data != self._admin_word:
            return Resp.Failure, "permission denied"
        with self._lock:
            data = {}
            for ses_name, ses in self._sessions.items():
                ses_data = []
                for ident, handler in ses.items():
                    client_data = {}
                    client_data["identity"] = handler.identity
                    client_data["address"] = handler.get_address()
                    ses_data.append(client_data)
                data[ses_name] = ses_data
            return Resp.Success, data

    def create_session(self, handler, session, identity):
        identity = self._make_identity(identity)
        with self._lock:
            try:
                ses = self._sessions[session]
            except KeyError:
                ses = self._sessions[session] = weakref.WeakValueDictionary()
            else:
                if len(ses) > 0:
                    return Resp.Failure, "session already in use: %s" % session
            handler.set_session_identity(session, identity)
            ses[handler.identity] = handler
            return Resp.Success, identity

    def join_session(self, handler, session, identity):
        identity = self._make_identity(identity)
        with self._lock:
            try:
                ses = self._sessions[session]
            except KeyError:
                return Resp.Failure, "no such session: %s" % session
            else:
                if identity in ses:
                    return Resp.Failure, "identity in use: %s" % identity
                handler.set_session_identity(session, identity)
                ses[handler.identity] = handler
                self.notify(handler, Req.Joined)
                return Resp.Success, identity

    def _make_identity(self, identity):
        if identity is not None:
            return identity
        else:
            return handler.make_identity()

    def process_request(self, handler, action, data):
        logger.info("server process_request [from %s]: %s",
                    str(handler), req_name(action))
        if action == Req.Message:
            return self.handle_message(data, handler)
        elif action == Req.GetIdentities:
            return self.get_identities(data, handler)
        elif action == Req.GetSessionInfo:
            return self.get_session_info(data, handler)
        elif action == Req.CreateSession:
            session, identity = data
            return self.create_session(handler, session, identity)
        elif action == Req.JoinSession:
            session, identity = data
            return self.join_session(handler, session, identity)
        elif action == Req.Exit:
            return self.terminate(data, handler)
        elif action == Req.GetSessions:
            return self.get_sessions(data, handler)
        elif action == Req.Echo:
            return Resp.Success, data
        else:
            return Resp.Failure, "unknown action %d" % action

    def handle_message(self, data, handler):
        with self._lock:
            # "data" comes from remote client,
            # "handler" is session handler for client
            if handler.session is None:
                return Resp.Failure, "not associated with session"
            receivers = data.get("receivers")
            ses = self._sessions[handler.session]
            if receivers is None:
                receivers = set(ses.keys())
                receivers.discard(handler.identity)
            unknowns = [r for r in receivers if r not in ses]
            if unknowns:
                return Resp.Failure, "not in session: %s" % ", ".join(unknowns)
            count = 0
            for r in receivers:
                ses[r].forward(handler, data)
                count += 1
            return Resp.Success, count

    def notify(self, handler, action):
        # Send joined or departed message to all others in session
        with self._lock:
            try:
                ses = self._sessions[handler.session]
            except KeyError:
                # Session must be gone.  Just ignore.
                pass
            else:
                logger.info("notify: %s %s", req_name(action), str(handler))
                if action == Req.Departed:
                    try:
                        del ses[handler.identity]
                    except KeyError:
                        pass
                for ident, hdlr in ses.items():
                    if ident != handler.identity:
                        logger.debug("notify request [to: %s]: %s %s",
                                     str(hdlr), req_name(action), str(handler))
                        hdlr.forward(None, handler.identity, action=action)
                if not ses:
                    logger.info("end session: %s", handler.session)
                    del self._sessions[handler.session]


class _Handler(_BasicClient):

    def __init__(self, s, server):
        super().__init__(s)
        self._server = server
        self.session = None
        self.identity = None
        server.register(self)

    def __str__(self):
        return "%s/%s" % (self.session, self.identity)

    def set_session_identity(self, session, identity):
        self.session = session
        self.identity = identity

    def get_address(self):
        return self._endpoint._socket.getpeername()

    def make_identity(self):
        addr, port = self._endpoint._socket.getpeername()
        return "%s_%s" % (addr, port)

    def process_request(self, action, data):
        return self._server.process_request(self, action, data)

    def run(self):
        super().run()
        self._server.notify(self, Req.Departed)

    def _send_request(self, data, callback, action=Req.Message):
        # Override to include our identity
        logger.info("handler send_request [to %s]: %s %s %s",
                    str(self), req_name(action), data)
        super()._send_request((self.identity, data), callback, action=action)

    def forward(self, sender, data, action=Req.Message):
        ident = None if sender is None else sender.identity
        logger.info("handler forward [to %s, from %s]: %s",
                    str(self), str(sender), req_name(action))
        super()._send_request((ident, data), None, action=action)


def noop_callback(status, data):
    pass


def _dump_ciphers(ctx, prefix):
    for cipher in ctx.get_ciphers():
        logger.info("%s: %s: %s", prefix, cipher["protocol"],
                    cipher["name"], flush=True)


def serve(hostname, port, admin_word):
    # Main program to act as hub
    server = Server(hostname, port, admin_word)
    logger.info("Starting server on %s:%s" % (hostname, port))
    server.start()
    logger.info("Waiting for server to finish")
    server.join()
    logger.info("Exiting")


if __name__ == "__main__":

    def test_basic(hostname, port, admin_word):
        finished = threading.Event()
        # If something goes wrong (e.g., exception thrown in client/server
        # code), make sure that we terminate anyway.
        def timeout_cb():
            logger.info("timeout expired")
            finished.set()
        timer = threading.Timer(20, timeout_cb)
        timer.start()

        logger.debug("create server")
        server = Server(hostname, port, admin_word)
        logger.info("server created: %s", server)
        server.start()
        logger.info("server running: %s", server)

        if True:
            class MyClient(Client):
                def handle_notification(self, ntype, data):
                    logger.info("notification: %s: %s %s",
                                self.identity, ntype, data)
                    return Resp.Success, None
            clients = []
            for i in range(2):
                logger.debug("create client %s", i)
                session = "session-%d" % i
                ident = "client-%d" % i
                c = MyClient(hostname, port, session, ident, True)
                logger.info("client created: %s", c)
                clients.append(c)
                ident = "client-%da" % i
                c = MyClient(hostname, port, session, ident, False)
                logger.info("client created: %s", c)
                clients.append(c)
            for c in clients:
                logger.debug("start client: %s", c)
                c.start()
                logger.info("client running: %s", c)

            class Counter:
                def __init__(self):
                    self._lock = threading.RLock()
                    self._count = 1

                def __str__(self):
                    return str(self._count)

                def increment(self):
                    with self._lock:
                        self._count += 1

                def decrement(self):
                    with self._lock:
                        self._count -= 1
                        if self._count == 0:
                            finished.set()
            count = Counter()

            for c in clients:
                msg = "hello %s" % c.identity
                def cb(status, data, msg=msg):
                    count.decrement()
                    logger.info("(%s) Response \"%s\": status=%d, data=%s",
                                count, msg, status, data)
                count.increment()
                logger.debug("[%s] Sending \"%s\"", count, msg)
                c._send_request(msg, cb, action=Req.Echo)
                logger.info("[%s] Request \"%s\"", count, msg)

            for c in clients:
                logger.info("identities [%s]: %s", c.identity,
                            c.sync_request(Req.GetIdentities, None))

            ac = Client(hostname, port, "admin", "admin", True)
            logger.debug("admin client created: %s", ac)
            ac.start()
            logger.info("admin client started: %s", ac)
            logger.info("sessions: %s", ac.sync_request(Req.GetSessions,
                                                        admin_word))
            count.decrement()

        # If something goes wrong (e.g., exception thrown in client/server
        # code, make sure that we terminate anyway.
        finished.wait()
        timer.cancel()


    def test_session(hostname, port, admin_word):
        finished = threading.Event()
        # If something goes wrong (e.g., exception thrown in client/server
        # code), make sure that we terminate anyway.
        def timeout_cb():
            logger.info("timeout expired: %s", finished)
            finished.set()
        timer = threading.Timer(20, timeout_cb)
        timer.start()

        logger.debug("create server")
        server = Server(hostname, port, admin_word)
        logger.info("server created: %s", server)
        server.start()
        logger.info("server running: %s", server)

        class MyClient(Client):
            def handle_message(self, data, sender):
                logger.info("handle_message: %s %s %s", self, sender, data)
                return Resp.Success, None
            def handle_notification(self, ntype, data):
                logger.info("notification: %s: %s %s",
                            self.identity, ntype, data)
                return Resp.Success, None

        # Create two sessions, each with three clients
        c00 = MyClient(hostname, port, "s0", "c0", True)
        c01 = MyClient(hostname, port, "s0", "c1", False)
        c02 = MyClient(hostname, port, "s0", "c2", False)
        # c10 = MyClient(hostname, port, "s1", "c0", True)
        # c11 = MyClient(hostname, port, "s1", "c1", False)
        # c12 = MyClient(hostname, port, "s1", "c2", False)
        clients = [c00, c01, c02]
        for c in clients:
            c.start()
        c01.close()
        logger.debug("closed c01")
        logger.info("session info: %s", c00.get_session_info())

        # Send a broadcast from c00 (to c01 and c02)
        def cb(status, data):
            logger.info("broadcast status: %s", status)
            logger.info("broadcast data: %s", data)
            finished.set()
        c00.send("junk", callback=cb)
        logger.info(c00.get_identities())

        logger.info("waiting for activity to finish: %s", finished)
        finished.wait()
        timer.cancel()
        logger.info("finished")


    def main():
        import sys, getopt
        opts, args = getopt.getopt(sys.argv[1:], "h:p:l:a:T:")
        hostname = "localhost"
        port = 8443
        admin_word = "chimeraxmux"
        run = "serve"
        log_level = "INFO"
        for opt, val in opts:
            if opt == "-h":
                hostname = val
            elif opt == "-p":
                port = int(val)
            elif opt == "-l":
                log_level = val
            elif opt == "-a":
                admin_word = val
            elif opt == "-T":
                run = val
        logger.setLevel(log_level)
        run_map = {
            "serve": serve,
            "basic": test_basic,
            "session": test_session
        }
        if run not in run_map:
            print("-T value must be one of %s" % ", ".join(run_map.keys()))
            raise SystemExit(1)
        run_map[run](hostname, port, admin_word)

    main()
