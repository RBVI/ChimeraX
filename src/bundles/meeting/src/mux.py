#!/usr/bin/python3
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

import socket, threading, queue, struct, weakref, itertools, ssl
import sys, os.path, logging
try:
    import cPickle as pickle
except ImportError:
    import pickle

#
# Logging code
#

logger = logging.getLogger()

#
# SSL and connection code
#

_use_ssl = True
_ctx_hub = None
_ctx_node = None


def get_ctx_hub():
    global _ctx_hub
    if _ctx_hub is None:
        cert = os.path.join(os.path.dirname(__file__), "server.pem")
        if not os.path.exists(cert):
            cert = "/usr/local/etc/cxconference.pem"
            if not os.path.exists(cert):
                logger.error("no SSL certificate file found")
                raise SystemExit(1)
#        _ctx_hub = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        _ctx_hub = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        _ctx_hub.options |= (ssl.OP_NO_TLSv1|ssl.OP_NO_TLSv1_1)
        _ctx_hub.set_ciphers('HIGH:!aNULL:!eNULL')	# Avoid weak ciphers, ticket #
#        _ctx_hub.check_hostname = False
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


def _dump_ciphers(ctx, prefix):
    for cipher in ctx.get_ciphers():
        logger.info("%s: %s: %s", prefix, cipher["protocol"], cipher["name"], flush=True)


#
# Data exchange format and constants
#


class Enum:
    _enum = itertools.count(1)
    _enum_map = {}

    @classmethod
    def define(cls, name):
        value = next(cls._enum)
        cls._enum_map[value] = name
        setattr(cls, name, value)

    @classmethod
    def name(cls, value):
        try:
            return cls._enum_map[value]
        except KeyError:
            return "%s<%s>" % (cls.__name__, str(value))


class PacketType(Enum):
    Enum.define("Req")
    Enum.define("Resp")


class Req(Enum):
    # debugging requests
    Enum.define("Echo")
    # admin requests
    Enum.define("Exit")
    Enum.define("GetConferences")
    # Notifications
    Enum.define("Joined")
    Enum.define("Departed")
    # node requests
    Enum.define("CreateConference")
    Enum.define("JoinConference")
    Enum.define("GetConferenceInfo")
    Enum.define("GetParticipants")
    Enum.define("Message")


class Resp(Enum):
    Enum.define("Failure")
    Enum.define("Success")


class Notify(Enum):
    Enum.define("Joined")
    Enum.define("Departed")


#
# Packet transmission and reception code
#


class _Connection:

    HeaderFormat = "III"
    HeaderSize = struct.calcsize(HeaderFormat)

    def __init__(self, s):
        self._socket = s
        self._lock = threading.Lock()

    def getpeername(self):
        with self._lock:
            if self._socket is None:
                return None, None
            return self._socket.getpeername()

    def get(self):
        # Only one thread will read /from/ the socket so no thread-safety
        with self._lock:
            sock = self._socket
            if sock is None:
                return None, None, None
        try:
            header_data = sock.recv(self.HeaderSize)
            if len(header_data) == 0:
                raise ConnectionError("recv")
            # logger.debug("get: %s: received %d bytes", self, len(header_data))
            try:
                s = header_data.decode().lower()
            except Exception:
                # If data is not text, assume it is not HTTP protocol
                pass
            else:
                if (s.startswith("get") or s.startswith("post") or
                    s.startswith("head") or s.startswith("put")):
                    self._send_http_error()
                    return None, None, None
            ptype, serial, count = struct.unpack(self.HeaderFormat, header_data)
            sdata = bytes()
            while len(sdata) < count:
                new_data = sock.recv(count - len(sdata))
                if len(new_data) == 0:
                    self.close()
                    return None, None, None
                sdata += new_data
            data = pickle.loads(sdata) if len(sdata) > 0 else None
            # logger.debug("get: %s %s %s %s", self, PacketType.name(ptype), serial, data)
            return ptype, serial, data
        except (OSError, ConnectionError):
            self.close()
            return None, None, None

    def _send_http_error(self):
        self._send(b"HTTP/1.1 418 I'm a teapot\r\n")
        self._send(b"Content-Type: text/plain; charset=utf-8\r\n")
        self._send(b"Connection: close\r\n")
        self._send(b"\r\n")
        self._send(b"I'm a teapot\r\n")
        self.close()

    def put(self, ptype, serial, data):
        # Serialization code copied from ActiveState recipe
        # logger.debug("put: %s %s %s %s", self, PacketType.name(ptype), serial, data)
        d = pickle.dumps(data)
        sdata = struct.pack(self.HeaderFormat, ptype, serial, len(d)) + d
        self._send(sdata)

    def _send(self, *args, **kw):
        # Make "_send" thread-safe since multiple nodes might want
        # to send simultaneously
        with self._lock:
            if self._socket is None:
                logger.warning("put: socket closed already: %s", self)
                return
            try:
                self._socket.send(*args, **kw)
            except ConnectionError:
                self.close()
            except Exception:
                logger.exception("_send: %s", self._socket)
                raise

    def close(self):
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None


#
# Node code
#

class _Endpoint(threading.Thread):

    def __init__(self):
        super().__init__(daemon=True)
        self._requests = {}
        self._serial = itertools.count(1)
        self.conf_name = None
        self.identity = None

    def __str__(self):
        return "%s/%s" % (self.conf_name, self.identity)

    def set_conference_identity(self, conf_name, identity):
        self.conf_name = conf_name
        self.identity = identity

    def add_request(self, callback):
        serial = next(self._serial)
        self._requests[serial] = callback
        return serial

    def send(self, data, *, receivers=None, callback=None, req=Req.Message):
        packet = {"receivers":receivers, "data":data}
        self.send_request(req, packet, callback)

    def sync_request(self, req, data):
        # Not used much anymore since the main thread is usually where
        # synchronous calls are most useful
        import threading
        if threading.main_thread() == threading.current_thread():
            raise RuntimeError("cannot use synchronous call in main thread")
        ev = threading.Event()
        def cb(status, data, s=self, ev=ev):
            ev.mux_status = status
            ev.mux_data = data
            ev.set()
        self.send_request(req, data, cb)
        ev.wait()
        if ev.mux_status != Resp.Success:
            raise RuntimeError(ev.mux_data)
        return ev.mux_data

    def set_identity(self, ident):
        self.identity = ident

    def _setup_cb(self, status, data):
        if status != Resp.Success:
            raise RuntimeError(data)
        conf_name, identity, addrs = data
        self.set_identity(identity)

    def create_conference(self, conf_name, identity, callback=None):
        if callback is None:
            callback = self._setup_cb
        return self.send_request(Req.CreateConference, (conf_name, identity),
                                 callback=callback)

    def join_conference(self, conf_name, identity, callback=None):
        if callback is None:
            callback = self._setup_cb
        return self.send_request(Req.JoinConference, (conf_name, identity),
                                 callback=callback)

    def get_conference_info(self, callback=None):
        if callback is None:
            return self.sync_request(Req.GetConferenceInfo, None)
        else:
            self.send_request(Req.GetConferenceInfo, None, callback)

    def get_participants(self, callback=None):
        if callback is None:
            return self.sync_request(Req.GetParticipants, None)
        else:
            self.send_request(Req.GetParticipants, None, callback)

    def get_conferences(self, admin_word, callback=None):
        if callback is None:
            return self.sync_request(Req.GetConferences, admin_word)
        else:
            self.send_request(Req.GetConferences, admin_word, callback)

    def handle_packet(self, ptype, serial, packet):
        logger.debug("handle_packet: [%s] %s %s %s", self, PacketType.name(ptype), serial, packet)
        if ptype == PacketType.Req:
            req, data = packet
            logger.debug("handle_packet: [%s] request %s %s", self, Req.name(req), data)
            status, response = self.process_request(req, data)
            logger.debug("handle_packet: [%s] request result %s %s", self, Resp.name(status), response)
            self.send_response(status, response, serial)
        elif ptype == PacketType.Resp:
            status, data = packet
            logger.debug("handle_packet: [%s] response %s %s", self, Resp.name(status), data)
            self.process_response(serial, status, data)
            logger.debug("handle_packet: [%s] response done", self)
        else:
            raise ValueError("unknown packet type: %s" % PacketType.name(ptype))

    def process_request(self, req, req_data):
        logger.debug("process_request: %s %s", Req.name(req), req_data)
        sender, data = req_data
        if req == Req.Message:
            return self.handle_message(data, sender)
        elif req == Req.Echo:
            return Resp.Success, data
        elif req == Req.Joined:
            return self.handle_notification(Notify.Joined, data)
        elif req == Req.Departed:
            return self.handle_notification(Notify.Departed, data)
        else:
            logger.error("unknown request: %s", req)
            return Resp.Failure, "unknown request: %s" % Req.name(req)

    def handle_message(self, data, sender):
        # "data" comes from remote node
        # "sender" is identity of requester
        raise NotImplementedError("handle_message()")

    def handle_notification(self, ntype, data):
        logger.warning("handle_notification missing: %s", self.__class__)
        raise NotImplementedError("handle_notification()")

    def process_response(self, serial, status, data):
        try:
            callback = self._requests.pop(serial)
            logger.debug("process_response: callback %s", callback)
        except KeyError:
            logger.debug("process_response: no callback")
            pass
        else:
            if callback is not None:
                callback(status, data)


class _NetworkEndpoint(_Endpoint):

    def __init__(self):
        super().__init__()
        self._connection = None

    def setup_connection(self, s):
        self._connection = _Connection(s)

    def run(self):
        while True:
            if self._connection is None:
                break
            ptype, serial, packet = self._connection.get()
            logger.debug("run received: [%s] %s %s %s", self, PacketType.name(ptype), serial, packet)
            if ptype is None:
                break
            try:
                self.handle_packet(ptype, serial, packet)
            except Exception:
                logger.error("message handler failed", exc_info=sys.exc_info())
        logger.info("run terminated: %s", self)

    def send_request(self, req, data, callback):
        logger.debug("send_request: [via: %s] %s %s", self, Req.name(req), data)
        serial = self.add_request(callback)
        self._connection.put(PacketType.Req, serial, (req, data))

    def send_response(self, status, data, serial):
        logger.debug("send_response: [via: %s] %s %s %s", self, Resp.name(status), data, serial)
        self._connection.put(PacketType.Resp, serial, (status, data))

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None


class NetworkNode(_NetworkEndpoint):

    def __init__(self, hostname, port, conf_name, identity, create):
        super().__init__()
        self.set_conference_identity(conf_name, identity)
        self._hostname = hostname
        self._port = port
        self._create = create

    def start(self, callback=None):
        self.setup_connection(make_node_socket(self._hostname, self._port))
        super().start()
        if self._create:
            logger.debug("creating conference, callback: %s", callback)
            self.create_conference(self.conf_name, self.identity,
                                   callback=callback)
            logger.debug("waiting to create conference")
        else:
            self.join_conference(self.conf_name, self.identity,
                                 callback=callback)


class _BaseHandler:

    def __init__(self):
        super().__init__()
        self._hub = None
        self.conf_name = None
        self.identity = None

    def __str__(self):
        return "%s/%s" % (self.conf_name, self.identity)

    def set_hub(self, hub):
        self._hub = hub
        hub.register(self)

    def process_request(self, req, data):
        return self._hub.process_req(self, req, data)

    def set_conference_identity(self, conf_name, identity):
        self.conf_name = conf_name
        self.identity = identity


class _NetworkHandler(_BaseHandler, _NetworkEndpoint):

    def __init__(self, s, hub):
        super().__init__()
        self.setup_connection(s)
        self.set_hub(hub)

    def __str__(self):
        return "%s/%s" % (self.conf_name, self.identity)

    def run(self):
        super().run()
        self._hub.notify(self, Req.Departed)

    def forward(self, req, sender, data):
        ident = None if sender is None else sender.identity
        logger.info("handler forward [to %s, from %s]: %s", self, sender, Req.name(req))
        self.send_request(req, (ident, data), None)
        # No callbacks for forwarding

    def get_address(self):
        return "%s:%s" % self._connection.getpeername()

    def make_identity(self):
        addr, port = self._connection.getpeername()
        return "%s_%s" % (addr, port)


#
# Code for a "hosting" node, which starts and connects to
# a hub on the same host.  A simple way to do this is to
# use a network connection between the node and hub, but
# that requires every request on this node (probably the
# hosting node responsible for sending scenes to newly
# joined nodes) to be transmitted twice: once between 
# this node and hub, and another from the hub to the
# receiving node(s).  If SSL is in play as well, the
# overhead is unnecessarily high.  To get around it, we
# create "loopback" node and handler, where sending from
# the node invokes the handler receiving method, and
# vice versa.
#


class LoopbackNode(_Endpoint):

    def __init__(self, conf_name, identity):
        # We do not want to use the base class __init__()s
        # because they set up sockets and threads, so we just
        # do the variable initializations and then set up
        # the hub and handler
        super().__init__()
        self.conf_name = conf_name
        self.identity = identity
        self._mux_hub = Hub("", 0, "chimeraxmux")
        self._mux_hub.start()
        self._mux_handler = _LoopbackHandler(self, self._mux_hub)
        self._closed = False

    def __str__(self):
        return "%s/%s" % (self.conf_name, self.identity)

    def start(self, callback=None):
        self._mux_handler.start()
        self._queue = queue.SimpleQueue()
        super().start()
        self.create_conference(self.conf_name, self.identity,
                               callback=callback)

    def run(self):
        while True:
            if self._closed or self._queue is None:
                break
            ptype, serial, packet = self._queue.get()
            logger.debug("loopback run received: %s %s %s", PacketType.name(ptype), serial, packet)
            if ptype is None:
                break
            try:
                self.handle_packet(ptype, serial, packet)
            except Exception:
                logger.error("message handler failed", exc_info=sys.exc_info())
        logger.info("run terminated: %s", self)

    def send_request(self, req, data, callback):
        logger.debug("send_request: [via: %s] %s %s", self, Req.name(req), data)
        serial = next(self._serial)
        self._requests[serial] = callback
        self._mux_handler._queue.put((PacketType.Req, serial, (req, data)))

    def send_response(self, status, data, serial):
        logger.debug("send_response: [via: %s] %s %s %s", self, Resp.name(status), data, serial)
        self._mux_handler._queue.put((PacketType.Resp, serial, (status, data)))

    def close(self):
        self._closed = True
        self._mux_handler._closed = True


class _LoopbackHandler(_BaseHandler, _Endpoint):

    def __init__(self, node, hub):
        super().__init__()
        self._node = node
        self.set_hub(hub)
        self._closed = False

    def __str__(self):
        return "LoopbackHandler-" + super().__str__()

    def start(self):
        self._queue = queue.SimpleQueue()
        super().start()

    def run(self):
        # No need to run read-loop since we will get called
        # directly by node when there is data available
        while True:
            if self._closed or self._queue is None:
                break
            ptype, serial, packet = self._queue.get()
            logger.debug("run received: [%s] %s %s %s", self, PacketType.name(ptype), serial, packet)
            if ptype is None:
                break
            try:
                self.handle_packet(ptype, serial, packet)
            except Exception:
                logger.error("message handler failed", exc_info=sys.exc_info())
        self._hub.notify(self, Req.Departed)

    def forward(self, req, sender, data):
        ident = None if sender is None else sender.identity
        logger.info("handler forward [to %s, from %s]: %s", self, sender, Req.name(req))
        self.send_request(req, (ident, data), None)
        # No callbacks for forwarding

    def send_request(self, req, data, callback):
        logger.debug("send_request: [via: %s] %s %s", self, Req.name(req), data)
        serial = self.add_request(callback)
        self._node._queue.put((PacketType.Req, serial, (req, data)))

    def send_response(self, status, data, serial):
        logger.debug("send_response: [via: %s] %s %s %s", self, Resp.name(status), data, serial)
        self._node._queue.put((PacketType.Resp, serial, (status, data)))

    def get_address(self):
        return "hub:loopback"


#
# Hub code
#

class Hub(threading.Thread):

    def __init__(self, hostname, port, admin_word):
        super().__init__(daemon=True)
        self._socket = make_hub_socket(hostname, port)
        self._handlers = set()
        self._conferences = {}
        self._stopped = False
        self._hub_lock = threading.RLock()
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
            except Exception:
                logger.error("hub accept() failed", exc_info=sys.exc_info())
            else:
                _NetworkHandler(ns, self).start()

    def stop(self):
        with self._hub_lock:
            logger.debug("hub locked: stop %s", repr(threading.current_thread()))
            self._stopped = True

    def register(self, rc):
        with self._hub_lock:
            logger.debug("hub locked: register %s", repr(threading.current_thread()))
            self._handlers.add(rc)

    def get_conference_info(self, data, handler):
        if handler.conf_name is None:
            return Resp.Failure, "not associated with conference"
        with self._hub_lock:
            logger.debug("hub locked: get_conference_info %s", repr(threading.current_thread()))
            if handler.conf_name in self._conferences:
                return Resp.Success, (self.service_addresses, handler.conf_name)
            else:
                return Resp.Failure, ("conference \"%s\" terminated" %
                                      handler.conf_name)

    def get_participants(self, data, handler):
        if handler.conf_name is None:
            return Resp.Failure, "not associated with conference"
        with self._hub_lock:
            logger.debug("hub locked: get_participants %s", repr(threading.current_thread()))
            try:
                conf = self._conferences[handler.conf_name]
            except KeyError:
                return Resp.Failure, ("conference \"%s\" terminated" %
                                      handler.conf_name)
            else:
                idents = [(handler.identity, handler.get_address())
                          for handler in conf.values()]
                return Resp.Success, idents

    def get_conferences(self, data, handler):
        if data != self._admin_word:
            return Resp.Failure, "permission denied"
        with self._hub_lock:
            logger.debug("hub locked: get_conferences %s", repr(threading.current_thread()))
            data = {}
            for conf_name, conf in self._conferences.items():
                ses_data = []
                for ident, handler in conf.items():
                    node_data = {}
                    node_data["identity"] = handler.identity
                    node_data["address"] = handler.get_address()
                    ses_data.append(node_data)
                data[conf_name] = ses_data
            return Resp.Success, data

    def create_conference(self, handler, conf_name, identity):
        identity = self._make_identity(identity)
        with self._hub_lock:
            logger.debug("hub locked: create_conference %s", repr(threading.current_thread()))
            try:
                conf = self._conferences[conf_name]
            except KeyError:
                conf = self._conferences[conf_name] = weakref.WeakValueDictionary()
            else:
                if len(conf) > 0:
                    return (Resp.Failure,
                            "conference name already in use: %s" % conf_name)
            logger.debug("hub set identity: %s %s %s", handler, conf_name, identity)
            handler.set_conference_identity(conf_name, identity)
            conf[handler.identity] = handler
            return Resp.Success, (conf_name, identity, self.service_addresses)

    def join_conference(self, handler, conf_name, identity):
        identity = self._make_identity(identity)
        with self._hub_lock:
            logger.debug("hub locked: join_conference %s", repr(threading.current_thread()))
            try:
                conf = self._conferences[conf_name]
            except KeyError:
                return Resp.Failure, "no such conference: %s" % conf_name
            else:
                if identity in conf:
                    identity = self._make_unique_identity(identity, conf)
                handler.set_conference_identity(conf_name, identity)
                conf[handler.identity] = handler
        self.notify(handler, Req.Joined)
        return Resp.Success, (conf_name, identity, self.service_addresses)

    def _make_identity(self, identity):
        if identity is not None:
            return identity
        else:
            return handler.make_identity()

    def _make_unique_identity(self, identity, conf):
        parts = identity.rsplit('_', 1)
        if len(parts) == 1:
            base = identity
        else:
            try:
                num = int(parts[1])
            except ValueError:
                # There is an _ but we did not put it there
                base = identity
            else:
                base = parts[0]
        num = 2
        while True:
            identity = base + '_' + str(num)
            if identity not in conf:
                return identity
            num += 1

    def process_req(self, handler, req, data):
        logger.debug("hub process_req [handler %s]: %s", str(handler), Req.name(req))
        if req == Req.Message:
            return self.handle_msg(data, handler)
        elif req == Req.GetParticipants:
            return self.get_participants(data, handler)
        elif req == Req.GetConferenceInfo:
            return self.get_conference_info(data, handler)
        elif req == Req.CreateConference:
            conf_name, identity = data
            return self.create_conference(handler, conf_name, identity)
        elif req == Req.JoinConference:
            conf_name, identity = data
            return self.join_conference(handler, conf_name, identity)
        elif req == Req.Exit:
            return self.terminate(data, handler)
        elif req == Req.GetConferences:
            return self.get_conferences(data, handler)
        elif req == Req.Echo:
            return Resp.Success, data
        else:
            return Resp.Failure, "unknown request: %s" % Req.name(req)

    def handle_msg(self, data, handler):
        logger.debug("hub handle_msg [handler %s]: locking %s", str(handler), repr(threading.current_thread()))
        with self._hub_lock:
            logger.debug("hub locked: handle_message %s", repr(threading.current_thread()))
            # "data" comes from remote node,
            # "handler" is connection handler for node
            if handler.conf_name is None:
                return Resp.Failure, "not associated with conference"
            receivers = data.get("receivers")
            conf = self._conferences[handler.conf_name]
            if receivers is None:
                receivers = set(conf.keys())
                receivers.discard(handler.identity)
            unknowns = [r for r in receivers if r not in conf]
            if unknowns:
                return (Resp.Failure,
                        "not in conference: %s" % ", ".join(unknowns))
            count = 0
            for r in receivers:
                conf[r].forward(Req.Message, handler, data)
                count += 1
            return Resp.Success, count

    def notify(self, handler, req):
        # Send joined or departed message to all others in conference
        with self._hub_lock:
            logger.debug("hub locked: notify %s", repr(threading.current_thread()))
            try:
                conf = self._conferences[handler.conf_name]
            except KeyError:
                # Conference must be gone.  Just ignore.
                pass
            else:
                logger.info("hub notify: %s %s", Req.name(req), str(handler))
                if req == Req.Departed:
                    try:
                        del conf[handler.identity]
                    except KeyError:
                        pass
                data = (handler.identity, handler.get_address())
                for ident, hdlr in conf.items():
                    if ident != handler.identity:
                        logger.debug("hub notify request [to: %s]: %s %s", str(hdlr), Req.name(req), str(handler))
                        hdlr.forward(req, None, data)
                if not conf:
                    logger.info("hub end conference: %s", handler.conf_name)
                    del self._conferences[handler.conf_name]


#
# Server main program
#


def serve(hostname, port, admin_word, pid_file=None):
    # Main program to act as hub
    # Make sure SSL cert is in place first
    get_ctx_hub()
    if pid_file:
        import os
        pid = os.fork()
        if pid > 0:
            logger.info("Forked PID: %d" % pid)
            with open(pid_file, "wt") as f:
                print(pid, file=f)
            raise SystemExit(0)
    hub = Hub(hostname, port, admin_word)
    logger.info("Starting hub on %s:%s" % (hostname, port))
    hub.start()
    logger.info("Waiting for hub to finish")
    hub.join()
    logger.info("Exiting")


#
# Script invocation and test code
#


if __name__ == "__main__":

    def run_main(func, hostname, port, admin_word):
        finished = threading.Event()
        # If something goes wrong (e.g., exception thrown in node/hub
        # code), make sure that we terminate anyway.
        def timeout_cb():
            logger.info("timeout expired")
            finished.set()
        timer = threading.Timer(20, timeout_cb)
        timer.start()
        logger.info("running %s", func.__name__)
        func(hostname, port, admin_word, finished)
        logger.info("waiting for activity to finish: %s", finished)
        finished.wait()
        timer.cancel()
        logger.info("finished")


    def test_basic(hostname, port, admin_word, finished):
        logger.debug("create hub")
        hub = Hub(hostname, port, admin_word)
        logger.info("hub created: %s", hub)
        hub.start()
        logger.info("hub running: %s", hub)

        class MyNetworkNode(NetworkNode):
            def handle_notification(self, ntype, data):
                logger.info("notification: %s: %s %s", self.identity, Notify.name(ntype), data)
                return Resp.Success, None
        nodes = []
        for i in range(2):
            logger.debug("create node %s", i)
            conf_name = "conf-%d" % i
            ident = "node-%d" % i
            n = MyNetworkNode(hostname, port, conf_name, ident, True)
            logger.info("node created: %s", n)
            nodes.append(n)
            ident = "node-%da" % i
            n = MyNetworkNode(hostname, port, conf_name, ident, False)
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
                logger.info("(%s) Response \"%s\": status=%s, data=%s", count, msg, Resp.name(status), data)
            count.increment()
            logger.debug("[%s] Sending \"%s\"", count, msg)
            n.send_request(Req.Echo, msg, cb)
            logger.info("[%s] Request \"%s\"", count, msg)

        for n in nodes:
            def cb(status, data, n=n):
                if status != Resp.Success:
                    raise RuntimeError(data)
                logger.info("participants [%s]: %s", n.identity, data)
            n.get_participants(callback=cb)

        ac = NetworkNode(hostname, port, "admin", "admin", True)
        logger.debug("admin node created: %s", ac)
        ac.start()
        logger.info("admin node started: %s", ac)
        def cb(status, data):
            if status != Resp.Success:
                raise RuntimeError(data)
            logger.info("conferences: %s", data)
        ac.get_conferences(admin_word, callback=cb)
        count.decrement()


    def test_conference(hostname, port, admin_word, finished):
        logger.debug("create hub")
        hub = Hub(hostname, port, admin_word)
        logger.info("hub created: %s", hub)
        hub.start()
        logger.info("hub running: %s", hub)

        class MyNetworkNode(NetworkNode):
            def handle_message(self, data, sender):
                logger.info("handle_message: %s %s %s", self, sender, data)
                return Resp.Success, None
            def handle_notification(self, ntype, data):
                logger.info("notification: %s: %s %s", self.identity, Notify.name(ntype), data)
                return Resp.Success, None

        # Create two conferences, each with three nodes
        n00 = MyNetworkNode(hostname, port, "s0", "n0", True)
        n01 = MyNetworkNode(hostname, port, "s0", "n1", False)
        n02 = MyNetworkNode(hostname, port, "s0", "n2", False)
        # n10 = MyNetworkNode(hostname, port, "s1", "n0", True)
        # n11 = MyNetworkNode(hostname, port, "s1", "n1", False)
        # n12 = MyNetworkNode(hostname, port, "s1", "n2", False)
        nodes = [n00, n01, n02]
        for n in nodes:
            n.start()
        n01.close()
        logger.debug("closed n01")
        def cb(status, data):
            if status != Resp.Success:
                raise RuntimeError(data)
            logger.info("conference info: %s", data)
        n00.get_conference_info(callback=cb)

        # Send a broadcast from n00 (to n01 and n02)
        def cb(status, data):
            if status != Resp.Success:
                raise RuntimeError(data)
            logger.info("broadcast status: %s", Resp.name(status))
            logger.info("broadcast data: %s", data)
            finished.set()
        n00.send("junk", callback=cb)
        def cb(status, data):
            if status != Resp.Success:
                raise RuntimeError(data)
            logger.info("participants: %s", data)
        n00.get_participants(callback=cb)


    def test_loopback(hostname, port, admin_word, finished):
        class NodeMixin:
            def handle_message(self, data, sender):
                logger.info("handle_message: %s %s %s", self, sender, data)
                return Resp.Success, None
            def handle_notification(self, ntype, data):
                logger.info("notification: %s: %s %s", self.identity, Notify.name(ntype), data)
                return Resp.Success, None
        class MyLoopbackNode(NodeMixin, LoopbackNode):
            pass
        class MyNetworkNode(NodeMixin, NetworkNode):
            pass

        q = queue.SimpleQueue()
        conf_name = "unnamed"
        h = MyLoopbackNode(conf_name, "host")
        def cb(status, data):
            if status != Resp.Success:
                raise RuntimeError(data)
            conf_name, identity, addrs = data
            logger.info("hosting %s as %s", conf_name, identity)
            h.set_identity(identity)
            logger.info("addresses: %s", addrs)
            q.put(None)
        h.start(callback=cb)
        q.get()
        logger.info("host started: %s %s", h.conf_name, h.identity)
        def cb(status, data):
            if status != Resp.Success:
                raise RuntimeError(data)
            logger.info("conference info: %s", data)
            q.put(None)
        h.get_conference_info(callback=cb)
        q.get()

        port = h._mux_hub.get_port()
        g = MyNetworkNode("localhost", port, conf_name, "guest", False)
        g.start()
        g.get_conference_info(callback=cb)
        def cb(status, data):
            if status != Resp.Success:
                raise RuntimeError(data)
            logger.info("participants: %s", data)
            q.put(None)
        h.get_participants(callback=cb)
        q.get()
        finished.set()


    def main():
        import sys, getopt
        opts, args = getopt.getopt(sys.argv[1:], "h:p:l:L:a:P:T:")
        hostname = "localhost"
        port = 8443
        admin_word = "chimeraxmux"
        run = None
        log_name = None
        log_level = "INFO"
        pid_file = None
        for opt, val in opts:
            if opt == "-h":
                hostname = val
            elif opt == "-p":
                port = int(val)
            elif opt == "-l":
                log_level = val
            elif opt == "-L":
                log_name = val
            elif opt == "-a":
                admin_word = val
            elif opt == "-P":
                pid_file = val
            elif opt == "-T":
                run = val
        logging.basicConfig(filename=log_name, level=log_level)
        if run is None:
            serve(hostname, port, admin_word, pid_file)
        else:
            run_map = {
                "basic": lambda *args: test_basic(*args),
                "conference": lambda *args: test_conference(*args),
                "loopback": lambda *args: test_loopback(*args),
            }
            if run not in run_map:
                print("-T value must be one of %s" % ", ".join(run_map.keys()))
                raise SystemExit(1)
            run_main(run_map[run], hostname, port, admin_word)

    main()
