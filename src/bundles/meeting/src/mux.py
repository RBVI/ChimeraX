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

# Terminology:
#   Conference = Collection of nodes communicating with each other.
#       Each conference has a unique name.
#   Node = participant in a conference.
#       Each node has a unique name, aka "identity", within a conference.
#   Hub = server that manages multiple conferences and forwards messages

import socket, threading, struct, weakref, itertools, ssl
import sys, os.path, logging
try:
    import cPickle as pickle
except ImportError:
    import pickle


_use_ssl = True
_ctx_hub = None
_ctx_node = None


def get_ctx_hub():
    global _ctx_hub
    if _ctx_hub is None:
        cert = os.path.join(os.path.dirname(__file__), "server.pem")
        _ctx_hub = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        _ctx_hub.load_cert_chain(cert)
    return _ctx_hub


def make_hub_socket(hostname, port, backlog=0):
    # Pre-3.8 code
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    s.bind((hostname, port))
    s.listen(5)
    # Post-3.8 code
    # s = socket.create_hub((hostname, port), backlog=backlog)
    if _use_ssl:
        s = get_ctx_hub().wrap_socket(s, server_side=True)
    return s


def get_ctx_node():
    global _ctx_node
    if _ctx_node is None:
        # ssl in Python 3.7 on Windows has a problem with TLS 1.3.
        # It raises an error:
        # "ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
        # So explicitly ask for TLS 1.2.
        # This bug might already be fixed in Python 3.8.
        _ctx_node = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        # _ctx_node = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        _ctx_node.check_hostname = False
        _ctx_node.verify_mode = ssl.CERT_NONE
    return _ctx_node


def make_node_socket(hostname, port):
    s = socket.create_connection((hostname, port))
    if _use_ssl:
        s = get_ctx_node().wrap_socket(s, server_hostname=hostname)
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
    GetConferences = next(_enum)
    # Notifications
    Joined = next(_enum)
    Departed = next(_enum)
    # node requests
    CreateConference = next(_enum)
    JoinConference = next(_enum)
    GetConferenceInfo = next(_enum)
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
        if self._socket is None:
            return None, None, None
        try:
            header_data = self._socket.recv(self.HeaderSize)
            if len(header_data) == 0:
                raise ConnectionError("recv")
        except (OSError, ConnectionError):
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
        # Make "put" thread-safe since multiple nodes might want
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


class _BasicNode(threading.Thread):

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

    def __init__(self, node, action, data):
        self.__event = threading.Event()
        node._send_request(data, self, action=action)
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


class Node(_BasicNode):

    def __init__(self, hostname, port, conf_name, identity,
                 create, timeout=None):
        super().__init__(make_node_socket(hostname, port))
        self.conf_name = conf_name
        self.identity = identity
        self._create = create

    def __str__(self):
        return "%s/%s" % (self.conf_name, self.identity)

    def close(self):
        self._endpoint.close()

    def start(self):
        super().start()
        req = Req.CreateConference if self._create else Req.JoinConference
        ev = threading.Event()
        self.identity = self.sync_request(req, (self.conf_name, self.identity))

    def send(self, data, *, receivers=None, callback=None, action=Req.Message):
        packet = {"receivers":receivers, "data":data}
        self._send_request(packet, callback, action=action)

    def get_conference_info(self):
        return self.sync_request(Req.GetConferenceInfo, None)

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
        # "data" comes from remote node
        # "sender" is identity of requester
        raise NotImplementedError("handle_message()")

    def handle_notification(self, ntype, data):
        raise NotImplementedError("handle_notification()")


class Hub(threading.Thread):

    def __init__(self, hostname, port, admin_word):
        super().__init__(daemon=True)
        self._socket = make_hub_socket(hostname, port)
        self._handlers = set()
        self._conferences = {}
        self._stopped = False
        self._lock = threading.RLock()
        self._addresses = None
        self._admin_word = admin_word

    @property
    def service_addresses(self):
        if self._socket is None:
            return "not_in_service"
        if self._addresses is None:
            ip, port = self._socket.getsockname()
            if ip == "0.0.0.0":
                self._addresses = ["%s:%s" % (host_name, port)
                                   for host_name in self._get_hostnames()]
            else:
                try:
                    host_info = socket.gethostbyaddr(ip)
                except socket.herror:
                    host_name = ip
                else:
                    host_name = host_info[0]
                self._addresses = ["%s:%s" % (host_name, port)]
        return self._addresses

    def _get_hostnames(self):
        import netifaces
        gateways = netifaces.gateways().get(netifaces.AF_INET)
        if not gateways:
            interfaces = netifaces.interfaces()
        else:
            interfaces = [gw[1] for gw in gateways]
        ips = set()
        for intf in interfaces:
            addrs = netifaces.ifaddresses(intf).get(netifaces.AF_INET, [])
            for addr in addrs:
                ips.add(addr["addr"])
        hostnames = set()
        for ip in ips:
            try:
                host_info = socket.gethostbyaddr(ip)
            except socket.herror:
                host_name = ip
            else:
                host_name = host_info[0]
            hostnames.add(host_name)
        return hostnames

    def get_port(self):
        if self._socket is None:
            raise RuntimeError("no active conference hub")
        _, port = self._socket.getsockname()
        return port

    def run(self):
        # Overrides Thread method
        logger.debug("hub accepting connections: %s" % self._socket)
        while not self._stopped:
            try:
                ns, addr = self._socket.accept()
                logger.info("hub connection:: from %s", addr)
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

    def get_conference_info(self, data, handler):
        if handler.conf_name is None:
            return Resp.Failure, "not associated with conference"
        with self._lock:
            if handler.conf_name in self._conferences:
                return Resp.Success, (self.service_addresses, handler.conf_name)
            else:
                return Resp.Failure, ("conference \"%s\" terminated" %
                                      handler.conf_name)

    def get_identities(self, data, handler):
        if handler.conf_name is None:
            return Resp.Failure, "not associated with conference"
        with self._lock:
            try:
                ses = self._conferences[handler.conf_name]
            except KeyError:
                return Resp.Failure, ("conference \"%s\" terminated" %
                                      handler.conf_name)
            else:
                return Resp.Success, list(ses.keys())

    def get_conferences(self, data, handler):
        if data != self._admin_word:
            return Resp.Failure, "permission denied"
        with self._lock:
            data = {}
            for conf_name, ses in self._conferences.items():
                ses_data = []
                for ident, handler in ses.items():
                    node_data = {}
                    node_data["identity"] = handler.identity
                    node_data["address"] = handler.get_address()
                    ses_data.append(node_data)
                data[conf_name] = ses_data
            return Resp.Success, data

    def create_conference(self, handler, conf_name, identity):
        identity = self._make_identity(identity)
        with self._lock:
            try:
                ses = self._conferences[conf_name]
            except KeyError:
                ses = self._conferences[conf_name] = weakref.WeakValueDictionary()
            else:
                if len(ses) > 0:
                    return (Resp.Failure,
                            "conference name already in use: %s" % conf_name)
            handler.set_conference_identity(conf_name, identity)
            ses[handler.identity] = handler
            return Resp.Success, identity

    def join_conference(self, handler, conf_name, identity):
        identity = self._make_identity(identity)
        with self._lock:
            try:
                ses = self._conferences[conf_name]
            except KeyError:
                return Resp.Failure, "no such conference: %s" % conf_name
            else:
                if identity in ses:
                    return Resp.Failure, "identity in use: %s" % identity
                handler.set_conference_identity(conf_name, identity)
                ses[handler.identity] = handler
                self.notify(handler, Req.Joined)
                return Resp.Success, identity

    def _make_identity(self, identity):
        if identity is not None:
            return identity
        else:
            return handler.make_identity()

    def process_request(self, handler, action, data):
        logger.info("hub process_request [from %s]: %s",
                    str(handler), req_name(action))
        if action == Req.Message:
            return self.handle_message(data, handler)
        elif action == Req.GetIdentities:
            return self.get_identities(data, handler)
        elif action == Req.GetConferenceInfo:
            return self.get_conference_info(data, handler)
        elif action == Req.CreateConference:
            conf_name, identity = data
            return self.create_conference(handler, conf_name, identity)
        elif action == Req.JoinConference:
            conf_name, identity = data
            return self.join_conference(handler, conf_name, identity)
        elif action == Req.Exit:
            return self.terminate(data, handler)
        elif action == Req.GetConferences:
            return self.get_conferences(data, handler)
        elif action == Req.Echo:
            return Resp.Success, data
        else:
            return Resp.Failure, "unknown action %d" % action

    def handle_message(self, data, handler):
        with self._lock:
            # "data" comes from remote node,
            # "handler" is connection handler for node
            if handler.conf_name is None:
                return Resp.Failure, "not associated with conference"
            receivers = data.get("receivers")
            ses = self._conferences[handler.conf_name]
            if receivers is None:
                receivers = set(ses.keys())
                receivers.discard(handler.identity)
            unknowns = [r for r in receivers if r not in ses]
            if unknowns:
                return (Resp.Failure,
                        "not in conference: %s" % ", ".join(unknowns))
            count = 0
            for r in receivers:
                ses[r].forward(handler, data)
                count += 1
            return Resp.Success, count

    def notify(self, handler, action):
        # Send joined or departed message to all others in conference
        with self._lock:
            try:
                ses = self._conferences[handler.conf_name]
            except KeyError:
                # Conference must be gone.  Just ignore.
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
                    logger.info("end conference: %s", handler.conf_name)
                    del self._conferences[handler.conf_name]


class _Handler(_BasicNode):

    def __init__(self, s, hub):
        super().__init__(s)
        self._hub = hub
        self.conf_name = None
        self.identity = None
        hub.register(self)

    def __str__(self):
        return "%s/%s" % (self.conf_name, self.identity)

    def set_conference_identity(self, conf_name, identity):
        self.conf_name = conf_name
        self.identity = identity

    def get_address(self):
        return self._endpoint._socket.getpeername()

    def make_identity(self):
        addr, port = self._endpoint._socket.getpeername()
        return "%s_%s" % (addr, port)

    def process_request(self, action, data):
        return self._hub.process_request(self, action, data)

    def run(self):
        super().run()
        self._hub.notify(self, Req.Departed)

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
    hub = Hub(hostname, port, admin_word)
    logger.info("Starting hub on %s:%s" % (hostname, port))
    hub.start()
    logger.info("Waiting for hub to finish")
    hub.join()
    logger.info("Exiting")


if __name__ == "__main__":

    def test_basic(hostname, port, admin_word):
        finished = threading.Event()
        # If something goes wrong (e.g., exception thrown in node/hub
        # code), make sure that we terminate anyway.
        def timeout_cb():
            logger.info("timeout expired")
            finished.set()
        timer = threading.Timer(20, timeout_cb)
        timer.start()

        logger.debug("create hub")
        hub = Hub(hostname, port, admin_word)
        logger.info("hub created: %s", hub)
        hub.start()
        logger.info("hub running: %s", hub)

        if True:
            class MyNode(Node):
                def handle_notification(self, ntype, data):
                    logger.info("notification: %s: %s %s",
                                self.identity, ntype, data)
                    return Resp.Success, None
            nodes = []
            for i in range(2):
                logger.debug("create node %s", i)
                conf_name = "conf-%d" % i
                ident = "node-%d" % i
                n = MyNode(hostname, port, conf_name, ident, True)
                logger.info("node created: %s", n)
                nodes.append(n)
                ident = "node-%da" % i
                n = MyNode(hostname, port, conf_name, ident, False)
                logger.info("node created: %s", n)
                nodes.append(n)
            for n in nodes:
                logger.debug("start node: %s", n)
                n.start()
                logger.info("node running: %s", n)

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

            for n in nodes:
                msg = "hello %s" % n.identity
                def cb(status, data, msg=msg):
                    count.decrement()
                    logger.info("(%s) Response \"%s\": status=%d, data=%s",
                                count, msg, status, data)
                count.increment()
                logger.debug("[%s] Sending \"%s\"", count, msg)
                n._send_request(msg, cb, action=Req.Echo)
                logger.info("[%s] Request \"%s\"", count, msg)

            for n in nodes:
                logger.info("identities [%s]: %s", n.identity,
                            n.sync_request(Req.GetIdentities, None))

            ac = Node(hostname, port, "admin", "admin", True)
            logger.debug("admin node created: %s", ac)
            ac.start()
            logger.info("admin node started: %s", ac)
            logger.info("conferences: %s", ac.sync_request(Req.GetConferences,
                                                           admin_word))
            count.decrement()

        # If something goes wrong (e.g., exception thrown in node/hub
        # code, make sure that we terminate anyway.
        finished.wait()
        timer.cancel()


    def test_conference(hostname, port, admin_word):
        finished = threading.Event()
        # If something goes wrong (e.g., exception thrown in node/hub
        # code), make sure that we terminate anyway.
        def timeout_cb():
            logger.info("timeout expired: %s", finished)
            finished.set()
        timer = threading.Timer(20, timeout_cb)
        timer.start()

        logger.debug("create hub")
        hub = Hub(hostname, port, admin_word)
        logger.info("hub created: %s", hub)
        hub.start()
        logger.info("hub running: %s", hub)

        class MyNode(Node):
            def handle_message(self, data, sender):
                logger.info("handle_message: %s %s %s", self, sender, data)
                return Resp.Success, None
            def handle_notification(self, ntype, data):
                logger.info("notification: %s: %s %s",
                            self.identity, ntype, data)
                return Resp.Success, None

        # Create two conferences, each with three nodes
        n00 = MyNode(hostname, port, "s0", "n0", True)
        n01 = MyNode(hostname, port, "s0", "n1", False)
        n02 = MyNode(hostname, port, "s0", "n2", False)
        # n10 = MyNode(hostname, port, "s1", "n0", True)
        # n11 = MyNode(hostname, port, "s1", "n1", False)
        # n12 = MyNode(hostname, port, "s1", "n2", False)
        nodes = [n00, n01, n02]
        for n in nodes:
            n.start()
        n01.close()
        logger.debug("closed n01")
        logger.info("conference info: %s", n00.get_conference_info())

        # Send a broadcast from n00 (to n01 and n02)
        def cb(status, data):
            logger.info("broadcast status: %s", status)
            logger.info("broadcast data: %s", data)
            finished.set()
        n00.send("junk", callback=cb)
        logger.info(n00.get_identities())

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
            "conference": test_conference
        }
        if run not in run_map:
            print("-T value must be one of %s" % ", ".join(run_map.keys()))
            raise SystemExit(1)
        run_map[run](hostname, port, admin_word)

    main()
