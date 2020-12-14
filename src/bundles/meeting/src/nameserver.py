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
class MeetingNameServer:
    def __init__(self, port = 52191, max_connections = 10, time_out = 1.0):
        self._names = {}	# Maps name bytes object to value bytes object
        self._version = b'cxns1'
        self._port = port
        self._max_connections = max_connections
        self._time_out = time_out		# seconds
        self._socket = None

        # Lock when using self._names dictionary
        from threading import Lock
        self._lock = Lock() 

    def close(self):
        s = self._socket
        if s:
            s.close()
            self._socket = None

    def listen(self):
        from socket import socket, AF_INET, SOCK_STREAM
        s = socket(AF_INET, SOCK_STREAM)
        self._socket = s

        host = ''  # Listen on all interfaces of local machine 
        s.bind((host, self._port)) 
  
        s.listen(self._max_connections)
  
    def accept_connections(self):
        while True:
            try:
                conn, addr = self._socket.accept()
            except OSError:
                break
            from _thread import start_new_thread
            start_new_thread(self._handle_connection, (conn,))
  
    def _handle_connection(self, conn):
        '''
        Read a get or set command from the socket conn, send back a result, and close the socket.
        '''
        conn.settimeout(self._time_out)
        self._do_command(conn)
        import socket
        conn.shutdown(socket.SHUT_RDWR)
        conn.close()

    def _do_command(self, conn):
        version = _read_value(conn)
        if version != self._version:
            return
        
        operation = _read_value(conn)
        if operation is None:
            return
        
        if operation == b'get':
            self._do_get(conn)
        elif operation == b'set':
            self._do_set(conn)
        elif operation == b'new':
            self._do_set(conn, replace = False)
        elif operation == b'clear':
            self._do_clear(conn)

    def _do_get(self, conn):
        name = _read_value(conn)
        if name is None:
            return
        with self._lock:
            result = self._names.get(name)
        if result is not None:
            _write_value(result, conn)

    def _do_set(self, conn, replace = True):
        name = _read_value(conn)
        if name is None:
            return
        value = _read_value(conn)
        if value is None:
            return
        with self._lock:
            if not replace and name in self._names:
                return
            self._names[name] = value
        _write_value(b'success', conn)

    def _do_clear(self, conn):
        name = _read_value(conn)
        if name is None:
            return
        with self._lock:
            if name in self._names:
                del self._names[name]
        _write_value(b'success', conn)
                
def _read_value(socket):
    '''
    Read a one byte length.  Then read a value that is that length of bytes.
    Return the value or None if there is a timeout.
    '''
    from socket import timeout
    try:
        length = socket.recv(1)
    except timeout:
        return None
    if len(length) != 1:
        return None
    flen = int.from_bytes(length, 'little')
    if flen == 0:
        return b''
    try:
        value = socket.recv(flen)
    except timeout:
        return None
    return value

def _parse_values(bytes, count):
    values = []
    for i in range(count):
        len = int.from_bytes([bytes[0]], 'little')
        values.append(bytes[1:len+1])
        bytes = bytes[len+1:]
    return values
        
def _write_value(value, socket):
    msg = _message([value])
    socket.sendall(msg)
    
def _message(fields):
    blist = []
    for f in fields:
        if len(f) > 255:
            raise ValueError('Name server message field too long (%f, %s)' % (len(f), f))
        blist.append(bytes([len(f)]))
        blist.append(f)
    return b''.join(blist)

def _send_message(args, host, port, time_out = 2.0, version = b'cxns1'):
    msg = _message((version,) + args)
    from socket import socket, AF_INET, SOCK_STREAM, timeout
    with socket(AF_INET, SOCK_STREAM) as s:
        s.settimeout(time_out)
        try:
            s.connect((host, port))
        except OSError as e:
            raise ConnectionError(str(e))
        try:
            s.sendall(msg)
            reply = _read_value(s)
        except timeout as e:
            raise TimeoutError(str(e))
        return reply

def get_value(name, host, port, time_out = 2.0):
    key = name.encode('utf-8')
    result = _send_message((b'get', key), host, port, time_out = time_out)
    return result
    
def set_value(name, value, host, port, time_out = 2.0, replace = True):
    key = name.encode('utf-8')
    op = b'set' if replace else b'new'
    result = _send_message((op, key, value), host, port, time_out = time_out)
    return result == b'success'

def clear_value(name, host, port, time_out = 2.0):
    key = name.encode('utf-8')
    result = _send_message((b'clear', key), host, port, time_out = time_out)
    return result == b'success'

def set_address(name, host, port, name_server_host, name_server_port, time_out = 2.0, replace = True):
    key = name.encode('utf-8')
    op = b'set' if replace else b'new'
    value = _message((host.encode('utf-8'), str(port).encode('utf-8')))
    result = _send_message((op, key, value),
                         name_server_host, name_server_port,
                         time_out = time_out)
    return result == b'success'

def get_address(name, name_server_host, name_server_port, time_out = 2.0, replace = True):
    key = name.encode('utf-8')
    value = _send_message((b'get', key), name_server_host, name_server_port, time_out = time_out)
    if value is None:
        host = port = None
    else:
        host_bytes, port_bytes = _parse_values(value, 2)
        host = host_bytes.decode('utf-8')
        port = int(port_bytes.decode('utf-8'))
    return host, port

def test(port = 51472, host = 'localhost'):
    from sys import argv
    if len(argv) == 2 and argv[1] == 'start':
        ns = MeetingNameServer(port)
        ns.listen()
        ns.accept_connections()
    elif len(argv) == 3 and argv[1] == 'get':
        name = argv[2]
        value = get_value(name, host, port)
        print ('get %s = %s' % (name, value))
    elif len(argv) == 4 and argv[1] == 'set':
        name, value = argv[2], argv[3]
        result = set_value(name, value.encode('utf-8'), host, port)
        print ('set %s = %s, status %s' % (name, value, result))
    elif len(argv) == 4 and argv[1] == 'new':
        name, value = argv[2], argv[3]
        result = set_value(name, value.encode('utf-8'), host, port, replace = False)
        print ('new %s = %s, status %s' % (name, value, result))
    elif len(argv) == 3 and argv[1] == 'clear':
        name = argv[2]
        result = clear_value(name, host, port)
        print ('clear %s, status %s' % (name, result))
    elif len(argv) == 3 and argv[1] == 'getaddr':
        name = argv[2]
        host, port = get_address(name, host, port)
        print ('getaddr %s = %s %s' % (name, host, port))
    elif len(argv) == 5 and argv[1] == 'setaddr':
        name, ahost, aport = argv[2:5]
        result = set_address(name, ahost, int(aport), host, port)
        print ('setaddr %s = %s %s, status %s' % (name, ahost, aport, result))

if __name__ == "__main__":
    # test if run as a script
    test()
