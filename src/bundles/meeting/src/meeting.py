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
def meeting_join(session, meeting_name = None,
                 name = None, color = None, face_image = None,
                 relay_commands = None, update_interval = None,
                 id = None, host = None, port = 52194, timeout = 5, ssh = False,
                 name_server = None, name_server_port = None):
    '''
    Join a meeting where two or more ChimeraX instances show the same models
    and show each others' VR hands and face or mouse positions.

    Parameters
    ----------
    meeting_name : string
      Name of the meeting.  The host of the meeting creates this name and tells it
      to the participants so they can join.  Names are case insensitive.
      Alternatively if the value contains "." or ":" characters it is treated
      as an IP address or host name of the machine that started the meeting or
      the proxy server being used for the meeting.
    name : string
      Name to identify this participant on remote machines.
    color : r,g,b,a (range 0-255)
      Color of this participant's mouse pointer cone.
    face_image : string
      Path to PNG or JPG image file for image to use for VR face depiction of this partipant.
    relay_commands : bool
      Whether to have every command you run sent to the other participants and their commands
      sent to you and automatically run so changes to the scene are mirrored for all participants.
      Default true.
    update_interval : int
      How often VR hand and head model positions are sent for this ChimeraX instance in frames.
      Value of 1 updates every frame.  Default 1.
    host : string
      To join a meeting that was created without a name the host option is used
      to specify the host name or IP address of the machine that started the meeting,
      for example, "descartes.cgl.ucsf.edu" or "169.230.21.39".
    port : int
      To join a meeting that was created without a name the port option is used
      to specify the port to connect to on the machine that started the meeting.
      Default 52194.
    timeout : float
      Time to wait for connection before giving up in seconds.  Default 5.
    ssh : bool
      Whether to connect using an ssh tunnel to the server.  Default False.
      This nay allow makng a connection if a firewall blocks the direct
      outgoing TCP connection without ssh.
    id : string
      This option is an alternative to specifying meeting_name.
      The value will only be interpreted as a text name and not an IP address or host name.
    name_server : string
      Host name or IP address for the name server which maps meeting names given by the id option
      to the host address and port for connecting to the meeting.
    name_server_port : int
      Port number for the name server given by the name_server option.
    '''

    if meeting_name is not None:
        if '.' in meeting_name or ':' in meeting_name:
            host = meeting_name
        else:
            id = meeting_name

    name_server, name_server_port = _name_server_defaults(session, name_server, name_server_port)
    if id is not None and host is None:
        host, port = _lookup_meeting_name(id, name_server, name_server_port)
        
    if host is None:
        from chimerax.core.errors import UserError
        raise UserError('Must specify meeting name, or host or id options')
    
    p = _join_meeting(session, host, port, timeout = timeout, meeting_name = meeting_name,
                      use_ssh = ssh)

    mname = '"%s"' % id if id else ''
    session.logger.info('Joining meeting%s at %s port %d' % (mname, host, port))

    _set_appearance(session, p, name, color, face_image)

    if relay_commands is not None:
        p.send_and_receive_commands(relay_commands)

    if update_interval is not None:
        p.vr_tracker.update_interval = update_interval

# -----------------------------------------------------------------------------
#
def meeting_start(session, meeting_name = None,
                  name = None, color = None, face_image = None,
                  server = None,
                  copy_scene = None,
                  relay_commands = None, update_interval = None,
                  name_server = None, name_server_port = None):
    '''
    Start a meeting allowing two or more ChimeraX instances to show each others' VR hand-controller
    and headset positions or mouse positions.  One participant starts the meeting with this command
    and the others join the meeting with the meeting command. The log will output the host name or
    IP address that other ChimeraX users should join this meeting.

    Parameters
    ----------
    meeting_name : string
      Name of the meeting that participants use to join the meeting.  If omitted then participants
      need to specify the host address to join the meeting.  The host address will be logged after
      the meeting start command and is typically the IP address of the computer or its domain name.
      Meeting names are case insensitive.  Meeting names should not include "." or ":" characters
      since those are used to distinguish host names from meeting names.
    name : string
      Name to identify this participant on remote machines.
    color : r,g,b,a (range 0-255)
      Color of this participant's mouse pointer cone.
    face_image : string
      Path to PNG or JPG image file for image to use for VR face depiction of this partipant.
    server : string
      Name describing what computer participants will connect to to join the meeting.
      The initial default is chimeraxmeeting.net.  If a different name is specified it
      becomes the default for future sessions.  The name can be "direct" meaning that
      participants will connect directly to the computer that started the meeting.
      Additional server names can be defined with the "meeting server" command.
    copy_scene : bool
      Whether to copy the open models from the ChimeraX that started the meeting
      to other participants when they join the meeting.
    relay_commands, update_interval, name_server, name_server_port :
      See the meeting command documentation for these options.
    '''

    local_port = _server_properties(session, 'direct')['port']
    
    participant, hub = _start_meeting(session, local_port, copy_scene = copy_scene)

    _set_appearance(session, participant, name, color, face_image)

    if relay_commands is not None:
        participant.send_and_receive_commands(relay_commands)

    if update_interval is not None:
        participant.vr_tracker.update_interval = update_interval

    # Create tunnel from server to hub
    server = _server_defaults(session, server)
    tunnel = _create_tunnel_to_hub(session, server, local_port, hub)

    # Register meeting name with the name server
    name_server, name_server_port = _name_server_defaults(session, name_server, name_server_port)
    if meeting_name is not None:
        hub.register_meeting_name(meeting_name, server, name_server, name_server_port)

    # Log meeting info
    addresses, cport = ([tunnel.host], tunnel.remote_port) if tunnel else hub.listening_addresses_and_port()
    _report_start(addresses, cport, meeting_name, session.logger)

# -----------------------------------------------------------------------------
#
def _create_tunnel_to_hub(session, server, local_port, hub):
    if server is None:
        return None

    sprops = _server_properties(session, server)
    try:
        tunnel = _create_ssh_tunnel(sprops, local_port, session.logger, hub.close)
    except BaseException:
        hub.close()		# Close meeting if tunnel to server fails.
        raise
    if tunnel is not None:
        hub._ssh_tunnel = tunnel

    return tunnel
    
# -----------------------------------------------------------------------------
#
def _create_ssh_tunnel(server_properties, local_port, log, close_callback, to_port = None):
    sp = server_properties
    acct = sp.get('account')
    if acct is None:
        return None

    addr = sp.get('address')
    key = sp.get('key')		# SSH authentication file private key.
    if key:
        from os.path import isfile
        if not isfile(key):
            from chimerax.core.errors import UserError
            raise UserError('meeting: SSH key file "%s" for %s@%s not found'
                            % (key, acct, addr))
    elif addr == 'chimeraxmeeting.net' and acct == 'tunnel':
        key = _default_proxy_key_file()
    else:
        from chimerax.core.errors import UserError
        raise UserError('meeting: No ssh key given for using server %s@%s' % (acct, addr))

    from_remote = (to_port is None)
    port_range = sp.get('port_range', (52194, 52203)) if from_remote else (to_port,to_port)
    timeout = sp.get('timeout', 5)
    from .sshtunnel import SSHTunnel
    tunnel = SSHTunnel(local_port, acct, addr, key, port_range,
                       from_remote = from_remote,
                       connection_timeout = timeout,
                       closed_callback = close_callback,
                       log = log)

    return tunnel

# -----------------------------------------------------------------------------
#
def meeting_settings(session,
                     name = None, color = None, face_image = None,
                     server = None, name_server = None, name_server_port = None):
    '''
    Display or set meeting settings that are remembered between sessions.
    With no options the current settings are reported.  Specifying options sets
    the saved value.
    '''
    s = (('name',name), ('color',color), ('face_image',face_image), ('server', server),
         ('name_server',name_server), ('name_server_port',name_server_port))
    values = [(k,v) for k,v in s if v is not None]
    settings = _meeting_settings(session)
    if len(values) == 0:
        msg = '\n'.join('%s: %s' % (attr.replace('_', ' '), getattr(settings,attr))
                        for attr,v in s)
        session.logger.info(msg)
    else:
        for attr, value in values:
            setattr(settings, attr, value)
        settings.save()

# -----------------------------------------------------------------------------
#
def meeting_server(session, name = None, address = None, port = None,
                   account = None, key = None, port_range = None, timeout = None,
                   delete = None):
    '''
    Display or define computer addresses where participants connect to meetings.

    Parameters
    ----------
    address : string
      Computer host name or IP address.
    port : int
      Port number participants use to connect to address.
      If an account is specified for creating an ssh tunnel then a
      port is chosen from port_range instead.
    account : string
      Ssh user name for creating a tunnel between the local machine and the
      machine specified by address.  Connections to the address will be forwarded
      to the local machine.
    key : string
      File path to the ssh identity file (e.g. 'proxy-key-private.pem') used to make an
      ssh tunnel to account / address. This identity file is the private key used to
      connect with the ssh <i>-i</i> option when creating the tunnel.  The file must not
      have access permissions by others or ssh will consider it insecure and not accept it.
    port_range : int, int
      The remote range of ports to use for making an ssh tunnel.  An available port
      from this range will be chosen when the tunnel is created.  Each port hosts one meeting
      forwarding connections from address.
    timeout : int
      Time to wait for a setting up a tunnel before giving up.
    delete : bool
      Delete the server name.
    '''
    settings = _meeting_settings(session)
    if delete:
        sdict = settings.servers
        if name and name in sdict:
            del sdict[name]
            settings.servers = dict(sdict)
            settings.save()
        return
    
    s = (('address',address), ('port',port),
         ('account',account), ('key',key),
         ('port_range',port_range), ('timeout',timeout))
    values = {k:v for k,v in s if v is not None}
    if len(values) == 0:
        lines = ['<pre>']
        for n,vals in settings.servers.items():
            if n == name or name is None:
                lines.append('<b>%s</b>' % n)
                for attr,v in vals.items():
                    if v is not None:
                        lines.append('\t%s = %s' % (attr, v))
        lines.append('</pre>')
        msg = '\n'.join(lines)
        session.logger.info(msg, is_html=True)
    elif name is None:
        from chimerax.core.errors import UserError
        raise UserError('meeting server: Must specify a server name.')
    else:
        # Need to copy dictionary so settings realizes it has changed.
        sdict = dict(settings.servers)
        if name in sdict:
            sdict[name].update(values)
        else:
            sdict[name] = values
        settings.servers = sdict
        settings.save()

# -----------------------------------------------------------------------------
#
def meeting_info(session):
    '''Report info about a current meeting in progress.'''
    p = _meeting_participant(session)
    if p is None:
        session.logger.status('No ChimeraX meeting started', log = True)
    else:
        _report_connection(session, p)

# -----------------------------------------------------------------------------
#
class RegisterMeetingName:
    '''Remember meeting name and clear it when meeting is closed.'''

    def __init__(self, meeting_name, name_server, name_server_port, host, port, log):

        self._meeting_name = meeting_name
        self._name_server = name_server
        self._name_server_port = name_server_port
        self._log = log

        from chimerax.core.errors import UserError
        if host is None:
            raise UserError('meeting: Could not determine host address.')
        
        from .nameserver import set_address
        try:
            success = set_address(meeting_name.casefold(), host, port,
                                  name_server, name_server_port, replace = False)
        except ConnectionError as e:
            raise UserError('meeting: Could not register meeting name "%s"'
                            % meeting_name +
                            ', unable to connect to name server %s port %d'
                            % (name_server, name_server_port))
        except TimeoutError as e:
            raise UserError('meeting: Could not register meeting name "%s"'
                            % meeting_name +
                            ', timed out connecting to name server %s port %d'
                            % (name_server, name_server_port))
        if not success:
            raise UserError('meeting: Meeting name "%s" already in use' % meeting_name)

    @property
    def name(self):
        return self._meeting_name
    
    def close(self):
        '''Remove meeting id from name server.'''
        if self._meeting_name is None:
            return
        from .nameserver import clear_value
        try:
            success = clear_value(self._meeting_name.casefold(),
                                  self._name_server, self._name_server_port)
        except (ConnectionError, TimeoutError):
            success = False
        if success:
            self._meeting_name = None
        else:
            msg = ('meeting close: Failed to remove meeting id "%s" from name server %s port %d'
                   % (self._meeting_name, self._name_server, self._name_server_port))
            self._log.warning(msg)

# -----------------------------------------------------------------------------
#
def _lookup_meeting_name(meeting_name, name_server, name_server_port):
    from chimerax.core.errors import UserError
    from .nameserver import get_address
    try:
        host, port = get_address(meeting_name.casefold(), name_server, name_server_port)
    except ConnectionError as e:
        raise UserError('meeting: Could not lookup meeting name "%s"' % meeting_name +
                        ', unable to connect to name server %s port %d' % (name_server, name_server_port))
    except TimeoutError as e:
        raise UserError('meeting: Could not lookup meeting name "%s"' % meeting_name +
                        ', timed out connecting to name server %s port %d' % (name_server, name_server_port))
    if host is None:
        raise UserError('meeting: Meeting name "%s" not found using name server %s port %d'
                        % (meeting_name, name_server, name_server_port))
    return host, port
            
# -----------------------------------------------------------------------------
#
def _start_meeting(session, port, copy_scene):
    p = _meeting_participant(session)
    if p is not None:
        from chimerax.core.errors import UserError
        raise UserError('To start a meeting you must exit'
                        ' the meeting you are currently in using'
                        ' command "meeting close"')

    p = _meeting_participant(session, create = True, start_hub = True)

    if copy_scene is None:
        copy_scene = True
    h = p.hub
    h.copy_scene(copy_scene)
    h.listen(port)
    return p, h

# -----------------------------------------------------------------------------
#
def _default_proxy_key_file():
    k = b'H4sIAOa8pF8C/22Vt7ajWBREc76ic1Yv4U3QwcWD8MJneOGdcPr6eT2TzklPsmtXUL9//xwnyqr5y32BX7arBsATfz3F+O/jN2SoqjgDlQPgyQNHBJfjSX41PwW/LR4b2+RU9SiEBA2CBM2NwT1FlLCbVCpYFuAP84CKAEeoSvYSwSw+YztlkvK88QUtxpXnzDLng3W5P5kZBMYyyZg3Gq7x5BYzO9gR7mcSCkJ6dtsVdXAzLgl2YkepWR/XnTgar1mld43So55TrLE3OC+nNSpzvTTELBpMx+v0N8Riko2d++Z3sojKO3OdgBbntj7Hd4MxgVt3j93MHgDB/HK9V/1YjfRNz4Y+iKGMgAZSJESySW4JfT92WTqTsMbNRBEUap2GBvxsLcal6ZRFNxHu/C2Ij8Qq+HXtaU6cRuwFGRSHrJji76FDvoGX5ibJXQlNZqDceYI5ai8MnFMVgAM4MP3IVqo8bT5n89I50vEFaJH01/rlpCpMibF0dS+oaj6bTfbLXNGzbBC87+CD7BA3cq+WquWO7jj4Qxq2vl+NsEAchphDxbWFoI1I5JR9rWHf96yTryjwFjwhLrS4yIdW6mZWdElxNLQazELc5/bKbjAF0URAOOF4u46XeWHBR2sU2dG1v7+55lo+VyzpDyr68bn4QOTqFis+Rt+GNXXkxss/EkOK5g7zTX7JbZ45uRlNPrJlj/nuNhoN4kcw3mvVUp098XKiwsgDeeGeVXaqtBQm6zNQUpH3Vwc7Hh1y/HLu7qeadfhQ5HfsEGx4aYw+TfUjCImsqoDmLv7UVhRmKYcUMH6jQUrL7ekp8nUsAvrwJf+Ho8uVxHuGCbCQwWSlBldnxPIqWPviByx4vOPCFHoQ6wuuUuj0Q4oM0kcWKYd8hjDcnOli6HYBItWwgir4iaWXPmtq3+dyLVZlsMJ5kVhG0FrfFRS0scaLuY/dkAGtyUmEYmw9tANnC2Cnr7UTTFDnZVEpjRL1S48x0SO9cuNfYuRSJYgfcAbzqe2p3otSLIfknZVJ6Yqp+nXpPOthHs2OHbcM7lSbezMWalkuDviDUYgblDrEIwRZqqaWt9iAxYvw5R5cfYC4LeFUjQR1iBuP6Jn1PC58GN0g+6J36O3vMS7cRO0OqJNoLQ85U+yCQYuTdtN/mvgwoOYMaaiYgt4yr5z+k1wD9HMYiWnbtD/BZPJg/RCSKvMwb0DzPDX7D+LDLyFtECubuO2xb5h/mLhrVf59ihYsTEiwz6tCTxXOgSgwCoqEtNvRWevyEBFH22RnR+Z1hximLAO/uBMx1lc8T3FC6m+zHRuc4oyWuqm1d6dDH460hJQDnmNM0Cj+SFtB66w7IcjhLzJAHcQLo53hKuynSF0IZObE0CRvZSetGJETGg/ckIgoUUR6/qTDhbU0FgW8T1bfbfMNwsISop6wAx0/R6fan1dlgcjbCiq9p/4JwkIYnhBqJ0ZP7Uny4aqDhKP++SBSDS8e+ihEp6FvEXxyc74F93ot3pRSX6Iiq7kOsBvk5cJBt5lewHlytSO8YufxNhvMk7Kd7+BSS9LmmyEUNR7k0ldw+xxm5W7G16RPFWo/3k4iCJCSkXFLyRVnayYx1HC3xUvPbOIoj3qkv3oayS+lDwqbgIfaRHo7U2bG9PIGBqyL+i6UXsXcL2YnXrf29r4KHjHMZ5Zb7qToOp+SMd702GZqpim6ydntF1f62/nnD/TvrIim8P9z8w/ecc3fjwYAAA=='
    import gzip, base64, tempfile, atexit, os
    key = gzip.decompress(base64.b64decode(k))
    f = tempfile.NamedTemporaryFile(delete = False)
    f.write(key)
    f.close()
    atexit.register(lambda path=f.name: os.remove(path))
    return f.name

# -----------------------------------------------------------------------------
#
def _join_meeting(session, host, port, timeout = None, meeting_name = None, use_ssh = False):
    p = _meeting_participant(session, create = True, meeting_name = meeting_name)
    if p.connected:
        from chimerax.core.errors import UserError
        raise UserError('To join another meeting you must exit'
                        ' the meeting you are currently in using'
                        ' command "meeting close"')
    p.connect(host, port, timeout = timeout, use_ssh = use_ssh)
    return p

# -----------------------------------------------------------------------------
#
def _report_start(addresses, port, meeting_name, log):
    loc = ' or '.join(addresses)
    if port != 52194:
        loc += ' port %d' % port
    if meeting_name is not None:
        status = 'Meeting "%s" started at %s' % (meeting_name, loc)
        from chimerax.core.commands import quote_if_necessary
        cmds = ['meeting %s' % quote_if_necessary(meeting_name)]
    else:
        status = 'Meeting started at %s' % loc
        cmds = [('meeting %s' % host) if port == 52194 else ('meeting %s port %s' % (host, port))
                for host in addresses]
    cmd = ' or '.join(['"%s"' % cmd for cmd in cmds])
    info = status + '\nParticipants can join with command %s' % cmd

    log.status(status)
    log.info(info)

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
                'name': '',	# Name seen by other participants
                'color': (0,255,0,255),	# Hand color seen by others
                'face_image': None,	# Path to image file
                'server': 'chimeraxmeeting.net',
                'servers': {'chimeraxmeeting.net': {'address':'chimeraxmeeting.net',
                                                    'account':'tunnel',
                                                    'port_range':(52194,52203),
                                                    'timeout':5},
                            'direct': {'port':52194}},
                'name_server': 'chimeraxmeeting.net',
                'name_server_port': 51472,
                'access': None,		# Obsolete.  Renamed to server.
                'access_points': None,	# Obsolete.  Renamed to servers.
            }
        settings = _MeetingSettings(session, "meeting")
        session._meeting_settings = settings

        # Rename obsolete 'access' and 'access_points' to 'server' and 'servers'.
        if settings.access is not None:
            settings.server = settings.access
            settings.access = None
            settings.save()
        if settings.access_points is not None:
            settings.servers = settings.access_points
            settings.access_points = None
            settings.save()

    return settings

# -----------------------------------------------------------------------------
#
def _server_properties(session, name):
    settings = _meeting_settings(session)
    return settings.servers.get(name)
    
# -----------------------------------------------------------------------------
#
def _server_properties_for_host(session, host):
    settings = _meeting_settings(session)
    for props in settings.servers.values():
        if props.get('address') == host:
            return props
    return None
    
# -----------------------------------------------------------------------------
#
def _server_defaults(session, server):
    settings = _meeting_settings(session)
    if server is None:
        server = settings.server
    elif server != settings.server:
        settings.server = server
        settings.save()
    return server
    
# -----------------------------------------------------------------------------
#
def _name_server_defaults(session, name_server, name_server_port):
    return _get_defaults(session,
                         (('name_server', name_server),
                          ('name_server_port', name_server_port)))
    
# -----------------------------------------------------------------------------
#
def _get_defaults(session, named_values):
    settings = _meeting_settings(session)
    save_settings = False
    values = dict(named_values)
    for attr, value in tuple(values.items()):
        if value is None:
            values[attr] = getattr(settings, attr, None)
        else:
            setattr(settings, attr, value)
            save_settings = True
    if save_settings:
        settings.save()
    return [values[name] for name,val in named_values]

# -----------------------------------------------------------------------------
#
def meeting_close(session):
    '''Close all connection shared pointers.'''
    p = _meeting_participant(session)
    if p:
        p.close()

# -----------------------------------------------------------------------------
#
def meeting_unname(session, meeting_name = None,
                   name_server = None, name_server_port = None):
    '''
    Remove a meeting name from the name server.
    This is needed to reuse the name if the name was not automatically
    removed when the meeting ended.  This can happen if the meeting host
    loses the network connection before the meeting ends.
    '''

    name_server, name_server_port = _name_server_defaults(session, name_server, name_server_port)
    from .nameserver import clear_value
    try:
        success = clear_value(meeting_name.casefold(), name_server, name_server_port)
    except (ConnectionError, TimeoutError) as e:
        msg = ('meeting unname: Failed to remove meeting name "%s" from name server %s port %d:\n%s'
               % (meeting_name, name_server, name_server_port, str(e)))
        from chimerax.core.errors import UserError
        raise UserError(msg)

    if not success:
        msg = ('meeting unname: Failed to remove meeting name "%s" from name server %s port %d'
               % (meeting_name, name_server, name_server_port))
        from chimerax.core.errors import UserError
        raise UserError(msg)

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
def register_meeting_command(cmd_name, logger):
    '''
    Currently unused.  Registered instead in reg_cmd.py.
    '''
    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import StringArg, IntArg, Color8TupleArg, OpenFileNameArg, BoolArg, Int2Arg, DynamicEnum, NoArg
    ServerArg = DynamicEnum(lambda s=logger.session: tuple(_meeting_settings(s).servers.keys()))

    participant_kw = [
        ('name', StringArg),
        ('color', Color8TupleArg),
        ('face_image', OpenFileNameArg),
    ]
    params_kw = [
        ('relay_commands', BoolArg),
        ('update_interval', IntArg),
        ('port', IntArg),
    ]
    name_server_kw = [
        ('name_server', StringArg),
        ('name_server_port', IntArg),
    ]

    if cmd_name == 'meeting' or cmd_name == 'meeting join':
        desc = CmdDesc(
            optional = [('meeting_name', StringArg)],
            keyword = (participant_kw +
                       [('id', StringArg), ('host', StringArg), ('ssh', BoolArg)] +
                       params_kw +
                       [('timeout', IntArg)] +
                       name_server_kw),
            synopsis = 'Join a ChimeraX meeting')
        register('meeting join', desc, meeting_join, logger=logger)
        register('meeting', desc.copy(), meeting_join, logger=logger)
    elif cmd_name == 'meeting start':
        desc = CmdDesc(
            optional = [('meeting_name', StringArg)],
            keyword = [('server', ServerArg),
                       ('copy_scene', BoolArg)] + participant_kw + params_kw + name_server_kw,
            synopsis = 'Create a ChimeraX meeting')
        register('meeting start', desc, meeting_start, logger=logger)
    elif cmd_name == 'meeting settings':
        desc = CmdDesc(
            keyword = participant_kw + [('server', ServerArg)] + name_server_kw,
            synopsis = 'Report or set meeting default settings')
        register('meeting settings', desc, meeting_settings, logger=logger)
    elif cmd_name == 'meeting server':
        desc = CmdDesc(
            optional = [('name', StringArg)],
            keyword = [('address', StringArg),
                       ('port', IntArg),
                       ('account', StringArg),
                       ('key', OpenFileNameArg),
                       ('port_range', Int2Arg),
                       ('timeout', IntArg),
                       ('delete', NoArg)],
            synopsis = 'Report or define meeting server names')
        register('meeting server', desc, meeting_server, logger=logger)
    elif cmd_name == 'meeting info':
        desc = CmdDesc(synopsis = 'Report meeting info')
        register('meeting info', desc, meeting_info, logger=logger)
    elif cmd_name == 'meeting close':
        desc = CmdDesc(synopsis = 'Close meeting')
        register('meeting close', desc, meeting_close, logger=logger)
    elif cmd_name == 'meeting unname':
        desc = CmdDesc(
            required = [('meeting_name', StringArg)],
            keyword = [('name_server', StringArg),
                       ('name_server_port', IntArg)],
            synopsis = 'Remove a meeting name from name server')
        register('meeting unname', desc, meeting_unname, logger=logger)
    elif cmd_name == 'meeting send':
        desc = CmdDesc(synopsis = 'Copy my scene to all other meeting participants')
        register('meeting send', desc, meeting_send, logger=logger)

# -----------------------------------------------------------------------------
#
def _meeting_participant(session, create = False, start_hub = False, meeting_name = None):
    p = getattr(session, '_meeting_participant', None)
    if p and p.closed:
        session._meeting_participant = p = None
    if p is None and create:
        p = MeetingParticipant(session, start_hub = start_hub, meeting_name = meeting_name)
        session._meeting_participant = p
    return p

# -----------------------------------------------------------------------------
#
class MeetingParticipant:
    def __init__(self, session, start_hub = False, meeting_name = None):
        self._version = 1		# Message protocol version
        self._closed = False
        self._meeting_name = meeting_name  # Used for authentication
        self._session = session
        self._name = 'Remote'
        self._color = (0,255,0,255)	# Tracking model color
        self._ssh_tunnel = None
        self._message_stream = None	# MessageStream for communicating with hub
        self._hub = None		# MeetingHub if we are hosting the meeting
        self._trackers = []
        self._mouse_tracker = None
        self._vr_tracker = None
        self._copy_scene = False
        self._received_scene = start_hub

        self._non_synced_commands = ['meeting', 'vr', 'quit']
        self._command_handlers = []	# Trigger handlers to capture executed commands
        self._running_received_command = False
        self._last_command_frame = 0
        self.send_and_receive_commands(True)

        if start_hub:
            self._hub = h = MeetingHub(session, self)
            self._message_stream = MessageStreamLocal(h._message_received)
            self._initiate_tracking()
        
        # Exit cleanly
        self._app_quit_handler = session.triggers.add_handler('app quit', self._app_quit)

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
    
    def connect(self, host, port, timeout = None, use_ssh = False):
        if self._hub:
            raise RuntimeError('Cannot join a meeting when currently hosting a meeting.')

        if use_ssh:
            local_port = port
            self._ssh_tunnel = self._create_ssh_tunnel_to_join(local_port, host, port)
            host = 'localhost'

        from Qt.QtNetwork import QTcpSocket
        socket = QTcpSocket()
        msg_stream = MessageStream(socket, self._message_received, self._disconnected,
                                   self._session.logger, connection_timeout = timeout)
        self._message_stream = msg_stream
        socket.connectToHost(host, port)
        socket.connected.connect(self._connected)

    def _connected(self):
        mname = '' if self._meeting_name is None else self._meeting_name
        msg = {'join': mname, 'version': self._version}
        self._send_message(msg)

        self._initiate_tracking()

        self._session.logger.status('Waiting for scene data from meeting host')

    def _create_ssh_tunnel_to_join(self, local_port, host, port):
        sprops = _server_properties_for_host(self._session, host)
        if sprops is None:
            from chimerax.core.errors import UserError
            raise UserError("meeting: Don't have ssh key to connect to server %s" % host)
        tunnel = _create_ssh_tunnel(sprops, local_port, self._session.logger,
                                    self.close, to_port = port)
        return tunnel
    
    @property
    def hub(self):
        return self._hub

    @property
    def closed(self):
        return self._message_stream is None
    
    def close(self):
        if self._closed:
            return
        self._closed = True
        
        self.send_and_receive_commands(False)
        
        self._close_trackers()

        msg_stream = self._message_stream
        if msg_stream:
            msg_stream.close()
            self._message_stream = None

        # Close ssh tunnel
        t = self._ssh_tunnel
        if t is not None:
            t.close()
            self._ssh_tunnel = None

        h = self._hub
        if h:
            h.close()
            self._hub = None

        aqh = self._app_quit_handler
        if aqh:
            self._session.triggers.remove_handler(aqh)
            self._app_quit_handler = None
            
    def _close_trackers(self):
        for t in self._trackers:
            t.delete()
        self._trackers = []
    
    def _app_quit(self, tname, tdata):
        # Catch app quit otherwise we get socket closed event after OpenGL is gone
        # and cleaning up meeting models raises errors.
        self.close()

    def send_scene(self):
        if self._session.models.empty():
            return
        session_bytes = self._encode_session()
        # Send size of session to allow progress status messages.
        self._send_message({'session size': len(session_bytes)})
        self._send_message({'session': session_bytes})
            
    def _encode_session(self):
        from io import BytesIO
        stream = BytesIO()
        self._session.save(stream, version=3, include_maps=True)
        from lz4.frame import compress
        sbytes = compress(stream.getbuffer())
        return sbytes

    def _restore_session(self, session_bytes):
        size = len(session_bytes)/2**20
        ses = self._session
        ses.logger.status('Received scene data (%.1f Mbytes) from meeting host'
                          % size)
        from lz4.frame import decompress
        sbytes = decompress(session_bytes)
        from io import BytesIO
        stream = BytesIO(sbytes)
        restore_camera = (ses.main_view.camera.name != 'vr')
        ses.restore(stream, resize_window = False, restore_camera = restore_camera,
                    clear_log = False)
        self._received_scene = True

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

        cmd = command.lstrip()
        for exclude_command in self._non_synced_commands:
            if cmd.startswith(exclude_command):
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
        motion = msg.get('motion')
        frame = self._session.main_view.frame_number
        if motion and frame == self._last_command_frame:
            # Only execute one motion command per frame
            # otherwise we can fall behind and the commands snowball
            # causing massive flicker.
            return
        self._last_command_frame = frame
        self._running_received_command = True
        from chimerax.core.commands import run
        try:
            run(self._session, command, log = not motion)
        finally:
            self._running_received_command = False

    @property
    def vr_tracker(self):
        self._initiate_tracking(send_messages = False)
        return self._vr_tracker
    
    def _initiate_tracking(self, send_messages = True):
        if not self._trackers:
            s = self._session
            self._mouse_tracker = mt = MouseTracking(s, self)
            self._vr_tracker = vrt = VRTracking(s, self)
            self._trackers = [mt, vrt]
        if send_messages:
            for t in self._trackers:
                t.send_messages(True)

    def _message_received(self, msg, msg_stream):
        if 'session' in msg:
            self._restore_session(msg['session'])
        if 'session size' in msg:
            msg_stream.status_message_size = msg['session size']
        if 'command' in msg:
            self._run_command(msg)
        for t in self._trackers:
            t.update_model(msg)
        if 'disconnected' in msg:
            self._participant_left(msg)

    def _send_message(self, msg):
        ms = self._message_stream
        if ms.write_backlogged() and _optional_message(msg):
            return False
        msg_bytes = MessageStream.message_as_bytes(msg)
        ms.send_message_bytes(msg_bytes)
        return True

    def _participant_left(self, msg):
        participant_id = msg['id']
        for t in self._trackers:
            t.remove_model(participant_id)

    def _disconnected(self, msg_stream):
        self.close()

# -----------------------------------------------------------------------------
#
def _optional_message(msg):
    if len(msg) == 2 or (len(msg) == 3 and 'id' in msg):
        if VRTracking._VR_HEAD_POSITION in msg and VRTracking._VR_HAND_POSITIONS in msg:
            return True	# Reporting only head and hand positions.
        if 'command' in msg and msg.get('motion'):
            return True # Motion command
        # TODO: Allow 'vr coords' update to be optional.
        #  But need to make sure everyone eventually gets current coords.
    return False

# -----------------------------------------------------------------------------
#
class MeetingHub:
    def __init__(self, session, host_participant):
        self._session = session
        self._closed = False
        self._server = None		# QTcpServer listens for connections
        msg_stream = MessageStreamLocal(host_participant._message_received)
        self._connections = [msg_stream] # List of MessageStream for each participant
        self._pending_connections = set()  # MessageStreams that have not yet sent join message.
        self._next_participant_id = 1
        self._host = host_participant	# MeetingParticipant that provides session for new participants.
        self._copy_scene = True		# Whether new participants get copy of scene
        self._ssh_tunnel = None		# SSHRemoteTunnel instance for ssh tunnel to proxy.
        self._registered_meeting_name = None	# RegisterMeetingName
        self._debug = True		# Write error messages for refused connections.

    def close(self):
        if self._closed:
            return
        self._closed = True
        
        for msg_stream in tuple(self._connections):
            msg_stream.close()
        self._connections = []

        for msg_stream in tuple(self._pending_connections):
            msg_stream.close()
        self._pending_connections.clear()
        
        self._server.close()	# Close QTcpServer
        self._server = None

        self._host.close()	# Close Participant
        self._host = None

        # Close ssh tunnel
        t = self._ssh_tunnel
        if t is not None:
            t.close()
            self._ssh_tunnel = None

        # Remove meeting id from name server
        rmn = self._registered_meeting_name
        if rmn is not None:
            rmn.close()

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
        from Qt.QtNetwork import QTcpServer, QHostAddress
        self._server = s = QTcpServer()
        a = QHostAddress.Any
        if not s.listen(a, port):
            msg = ('QTcpServer.listen(Any, %d) failed on local machine: %s'
                   % (port, s.errorString()))
            self._session.logger.warning(msg)
        else:
            s.newConnection.connect(self._new_connection)
    
    def listen_host_info(self):
        addr_list, port = self.listening_addresses_and_port()
        if not addr_list:
            return None
        hinfo = '%s port %d' % (' or '.join(addr_list), port)
        return hinfo
    
    def listening_addresses_and_port(self):
        if not self.listening:
            return [], None
        
        s = self._server
        port = s.serverPort()
        addresses = [a.toString() for a in self._available_server_ipv4_addresses()]
        from Qt.QtNetwork import QHostInfo
        host = QHostInfo.localHostName()
        if host:
            addresses.insert(0, host)
        
        return addresses, port
    
    def connected_ip_port_list(self):
        return [c.host_and_port() for c in self._connections
                if not isinstance(c, MessageStreamLocal)]
    
    def _available_server_ipv4_addresses(self):
        from Qt.QtNetwork import QNetworkInterface, QAbstractSocket
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
    
    def meeting_host_and_port(self, server):
        tunnel = self._ssh_tunnel
        if tunnel:
            host, port = tunnel.host, tunnel.remote_port
        else:
            sp = _server_properties(self._session, server)
            if 'address' in sp:
                host = sp.get('address')
                port = sp.get('port', 52194)
            else:
                addresses, port = self.listening_addresses_and_port()
                host = addresses[0] if addresses else None
        return host, port

    def register_meeting_name(self, meeting_name, server, name_server, name_server_port):
        host, port = self.meeting_host_and_port(server)
        try:
            self._registered_meeting_name = RegisterMeetingName(meeting_name, name_server, name_server_port,
                                                                host, port, self._session.logger)
        except BaseException:
            self._host.close()		# Close meeting if name server could not be reached.
            raise
    
    def copy_scene(self, copy):
        self._copy_scene = copy
        
    def _new_connection(self):
        s = self._server
        while s.hasPendingConnections():
            socket = s.nextPendingConnection()
            # TODO: Unclear from docs whether this socket is guaranteed to be in the ConnectedState
            #  or whether we have to check and possibly wait for it to reach that state.
            msg_stream = MessageStream(socket, self._message_received,
                                       self._disconnected, self._session.logger)
            self._pending_connections.add(msg_stream)
            if self._debug:
                self._session.logger.info('Connection from %s port %d established,' % msg_stream.host_and_port() +
                                          ' waiting for join message')


    def _handle_join_message(self, msg, msg_stream):
        if msg_stream not in self._pending_connections:
            return False
        
        self._pending_connections.remove(msg_stream)
        
        if 'join' in msg:
            rmn = self._registered_meeting_name
            if rmn is None or msg['join'].casefold() == rmn.name.casefold():
                self._add_connection(msg_stream)
            else:
                # Did not provide the right meeting name, so disconnect.
                if self._debug:
                    err_msg = ('Connection from %s port %d refused' % msg_stream.host_and_port() +
                               ' because meeting name mismatch "%s" != "%s"' % (msg['join'], rmn.name))
                    self._session.logger.info(err_msg)
                msg_stream.close()
        else:
            # First message did not include "join" key.
            if self._debug:
                err_msg = ('Connection from %s port %d refused' % msg_stream.host_and_port() +
                           ' because first message does not have join key: %s' % str(list(msg.keys())))
                self._session.logger.info(err_msg)
                msg_stream.close()

        return True

    def _add_connection(self, msg_stream):
        msg_stream.participant_id = self._next_participant_id
        self._next_participant_id += 1

        self._connections.append(msg_stream)
        
        self._session.logger.info('Connection accepted from %s port %d'
                                  % msg_stream.host_and_port())

        # Send new participant the initial scene
        if self._copy_scene:
            self._copy_scene_to_participant(msg_stream)

        # Send new participant position of scene in VR room
        self._send_room_coords(msg_stream)

    def _copy_scene_to_participant(self, message_stream):
        if self._session.models.empty():
            return
        session_bytes = self._host._encode_session()
        self._send_message({'session size': len(session_bytes)},
                           message_streams=[message_stream])
        self._send_message({'session': session_bytes},
                           message_streams=[message_stream])

    def _send_room_coords(self, message_stream):
        rts = self._host.vr_tracker.last_room_to_scene
        if rts is not None:
            # Tell peer the current vr room coordinates.
            msg = {'vr coords': _place_matrix(rts)}
            self._send_message(msg, message_streams=[message_stream])

    def _message_received(self, msg, msg_stream):
        if self._handle_join_message(msg, msg_stream):
            return

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
                if msg_stream.write_backlogged() and _optional_message(msg):
                    #                    msg_stream._dropped_messages += 1
                    continue
                msg_stream.send_message_bytes(msg_bytes)

    def _disconnected(self, msg_stream):
        if msg_stream in self._pending_connections:
            self._pending_connections.remove(msg_stream)
            return
        
        self._connections.remove(msg_stream)

        # Sendmessage to other participants that this participant has left
        # so they can remove the head/hand models.
        participant_id = msg_stream.participant_id
        msg = {'id': participant_id, 'disconnected': True}
        self._send_message(msg)

class MessageStream:
    def __init__(self, socket, message_received_cb, disconnected_cb, log,
                 connection_timeout = None):
        self._socket = socket
        self._message_received_cb = message_received_cb
        self._disconnected_cb = disconnected_cb
        
        self._log = log

        # Message Pack can only handle objects < 4 Gbytes in size.
        from msgpack import Unpacker
        self._unpacker = Unpacker(max_buffer_size = 2**32-1)

        # If write buffer grows beyond this limit
        # optional messages will not be sent.
        self._max_write_backlog_bytes = 50000

        # Progress status messages
        self._status_report_interval = 0.5	# seconds
        self._message_start_time = None
        self._last_status_time = None
        self._message_bytes_read = 0
        self.status_message_size = 0

        socket.errorOccurred.connect(self._socket_error)
        socket.disconnected.connect(self._socket_disconnected)

        # Register callback called when data available to read on socket.
        socket.readyRead.connect(self._data_available)

        self._connection_timeout = connection_timeout
        if connection_timeout is not None:
            self._timeout_timer = _set_timer(connection_timeout,
                                             self._check_connection_timeout)

    def close(self):
        # Calling close crashes if it is in disconnect callback.
        #        self._socket.close()
        s = self._socket
        s.deleteLater()
        self._socket = None

    def host_and_port(self):
        s = self._socket
        return (s.peerAddress().toString(), s.peerPort())
    
    def send_message_bytes(self, msg_bytes):
        from Qt.QtCore import QByteArray
        qbytes = QByteArray(msg_bytes)
        self._socket.write(qbytes)

    def write_backlogged(self):
        return self._socket.bytesToWrite() >= self._max_write_backlog_bytes
    
    @staticmethod
    def message_as_bytes(message):
        '''
        The message can be any of the types handles by msgpack.
        It is typically a dictionary with strings as keys and bytes,
        integers, floats, and strings as values.
        '''
        from msgpack import packb
        bytes = packb(message)
        return bytes

    def _data_available(self):
        while True:
            msg = self._read_message()
            if msg is None:
                break	# Don't have full message yet.
            else:
                self._message_received_cb(msg, self)
                
    def _read_message(self):
        socket = self._socket
        if socket is None:
            return None		# Socket was closed
        rbytes = socket.readAll()
        unpacker = self._unpacker
        unpacker.feed(rbytes)
        self._report_message_progress(len(rbytes))
        #        self._report_read_bandwidth(len(rbytes))

        # Get a message if enough data is available.
        try:
            for msg in unpacker:
                if not isinstance(msg, dict):
                    raise ValueError('Message was not a dictionary, got %s' % str(type(msg)))
                self._report_message_received()
                #                self._bandwidth_message_count += 1
                return msg
        except Exception as e:
            msg = ('Meeting received a message with wrong format from %s port %d.\n'
                   % self.host_and_port() +
                   'Possibly they are running a ChimeraX older than Dec 9, 2020 ' +
                   ' which used a different message protocol.\n%s' % str(e))
            self._log.warning(msg)
            socket.close()
    
        return None

    def _report_message_received(self):
        '''Report size and time for large received messages.'''
        st = self._message_start_time
        if st is not None:
            from time import time
            mt = time() - st
            if mt >= self._status_report_interval:
                msg = ('Received %.1f Mbytes in %.1f seconds'
                       % (self._message_bytes_read / 2**20, mt))
                self._log.status(msg)
        self._message_start_time = None
        self._message_bytes_read = 0
        self.status_message_size = 0

    def _report_message_progress(self, bytes_read):
        '''Report progress receiving message.'''
        if bytes_read == 0:
            return
        self._message_bytes_read += bytes_read
        st = self._message_start_time
        from time import time
        t = time()
        if st is None:
            self._message_start_time = t
            self._last_status_time = t
        else:
            et = t - self._last_status_time
            if et >= self._status_report_interval:
                mbytes = self._message_bytes_read / 2**20
                msize = self.status_message_size / 2**20
                if msize:
                    percent = min(100, (100*mbytes/msize))
                    msg = 'Receiving data %.0f%% of %.1f Mbytes' % (percent, msize)
                else:
                    msg = 'Receiving data %.1f Mbytes' % mbytes
                self._log.status(msg)
                self._last_status_time = t

    def _report_read_bandwidth(self, nbytes):
        'Report network bandwidth used by received messages.'

        if not hasattr(self, '_bandwidth_report_interval'):
            self._bandwidth_report_interval = 1	# seconds, 0 = no reporting
            self._bandwidth_bytes_read = 0
            self._bandwidth_last_time = None
            self._bandwidth_message_count = 0

        self._bandwidth_bytes_read += nbytes
        from time import time
        t = time()
        lt = self._bandwidth_last_time
        if lt is None:
            self._bandwidth_last_time = t
        elif t-lt > self._bandwidth_report_interval:
            rmbits = 8e-3 * self._bandwidth_bytes_read  # Kbits
            mc = self._bandwidth_message_count
            self._bandwidth_bytes_read = 0
            self._bandwidth_last_time = t
            self._bandwidth_message_count = 0
            rsec = t-lt
            msg = ('Read %.0f Kbit/sec (%.0f Kbits in %.1f sec), %.1f messages/sec'
                   % (rmbits/rsec, rmbits, rsec, mc/rsec))
            self._log.status(msg)
            
    def _socket_error(self, error_type):
        socket = self._socket
        if socket.error() == socket.RemoteHostClosedError:
            return
        host = socket.peerAddress().toString()
        port = socket.peerPort()
        err = socket.errorString()
        self._log.info('Socket error for message stream to %s port %d: %s' % (host, port, err))
#        self._socket_disconnected()

    def _socket_disconnected(self, report = True):
        socket = self._socket
        if socket is None:
            return
        from Qt import qt_object_is_deleted
        if qt_object_is_deleted(socket):
            return	# Happens when exiting ChimeraX
        if report:
            host, port = (socket.peerAddress().toString(), socket.peerPort())
            msg = 'Disconnected from %s port %d' % (host, port)
            self._log.info(msg)
        # Closing or deallocating the socket in this socket callback causes a crash.
        # So close routine must use deleteLater()
        # self.close()
        self._disconnected_cb(self)

    def _check_connection_timeout(self):
        socket = self._socket
        if socket is None:
            return
        state = socket.state()
        if state != socket.ConnectedState and state != socket.UnconnectedState:
            socket.abort()
            msg = 'Failed to connect after %d seconds' % self._connection_timeout
            self._log.status(msg, color='red')
            self._log.warning(msg)
            self._socket_disconnected(report = False)  # QTcpSocket is not firing the disconnected signal

def _set_timer(timeout, callback):
    from Qt.QtCore import QTimer
    delay_msec = int(1000*timeout)
    return QTimer.singleShot(delay_msec, callback)

class MessageStreamLocal:
    def __init__(self, send_message_cb):
        self._send_message_cb = send_message_cb
        self.participant_id = 0
    def send_message_bytes(self, msg_bytes):
        from msgpack import unpackb
        msg = unpackb(msg_bytes)
        self._send_message_cb(msg, self)
    def write_backlogged(self):
        return False
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

        self._mouse_hover_handler = None
        self._last_camera_position = None
        self._camera_move_handler = None

    def delete(self):
        self.send_messages(False)
        PointerModels.delete(self)

    def send_messages(self, send):
        hh,hc = self._mouse_hover_handler, self._camera_move_handler
        if hh is None and send:
            t = self._session.triggers
            self._mouse_hover_handler = t.add_handler('mouse hover',
                                                      self._mouse_hover_cb)
            self._camera_move_handler = t.add_handler('new frame',
                                                      self._camera_move_cb)
        elif hh and not send:
            t = self._session.triggers
            t.remove_handler(hh)
            t.remove_handler(hc)
            self._mouse_hover_handler = None
            self._camera_move_handler = None

    def update_model(self, msg):
        if 'mouse' in msg:
            PointerModels.update_model(self, msg)
        if ('camera position' in msg and
            _vr_camera(self._session) is None and
            self._participant._received_scene):
            c = self._session.main_view.camera
            c.position = _matrix_place(msg['camera position'])
            self._last_camera_position = c.position
            
    def make_pointer_model(self, session):
        return MousePointerModel(self._session, 'my pointer')

    def _camera_move_cb(self, trigger_name, update_loop):
        if _vr_camera(self._session):
            return
        if not self._participant._received_scene:
            return
                        
        p = self._session.main_view.camera.position
        lp = self._last_camera_position
        if lp is None or p != lp:
            self._last_camera_position = p
            msg = {'camera position': _place_matrix(p)}
            # Tell other participants my new mouse pointer position.
            self._participant._send_message(msg)
            
    def _mouse_hover_cb(self, trigger_name, pick):
        if _vr_camera(self._session):
            return

        xyz = getattr(pick, 'position', None)
        if xyz is None:
            return
        # Show pointer perpendicular to the view direction at 45 degrees
        # for best visibility of the atom being pointed to.
        c = self._session.main_view.camera
        axis = c.position.transform_vector((-0.707, 0.707, 0))
        msg = {'name': self._participant._name,
               'color': tuple(int(r) for r in self._participant._color),
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
                self.name = '%s pointer' % (msg['name'] if msg['name'] else 'Remote')
        if 'color' in msg:
            self.color = msg['color']
        if 'mouse' in msg:
            xyz, axis = msg['mouse']
            from chimerax.geometry import vector_rotation, translation
            p = translation(xyz) * vector_rotation((0,0,1), axis)
            self.position = p

class VRTracking(PointerModels):
    _VR_HEAD_POSITION = 'vH'	# Message key, short to reduce bandwidth.
    _VR_HAND_POSITIONS = 'vh'	# Message key, short to reduce bandwidth.
    def __init__(self, session, participant, sync_coords = True, update_interval = 1):
        PointerModels.__init__(self, session)
        self._participant = participant		# MeetingParticipant instance
        self._sync_coords = sync_coords

        self._vr_tracking_handler = None
        self._update_interval = update_interval	# Send vr position every N frames.
        self._last_vr_camera = c = _vr_camera(self._session)
        from chimerax.geometry import Place
        self._last_room_to_scene = c.room_to_scene if c else Place()
        self._name = None
        self._color = None
        self._new_face_image = None	# Path to image file
        self._face_image = None		# Encoded image
        self._send_face_image = False
        self._gui_state = {'shown':False, 'size':(0,0), 'room position':None, 'image':None}

    def delete(self):
        self.send_messages(False)
        PointerModels.delete(self)

    def send_messages(self, send):
        h = self._vr_tracking_handler
        if h is None and send:
            t = self._session.triggers
            h = t.add_handler('vr update', self._vr_tracking_cb)
            self._vr_tracking_handler = h
        elif h and not send:
            t = self._session.triggers
            t.remove_handler(h)
            self._vr_tracking_handler = None
        
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

        if self._VR_HEAD_POSITION in msg:
            PointerModels.update_model(self, msg)

    def make_pointer_model(self, session):
        # Make sure new meeting participant gets my
        # name, color, head image and button modes.
        self._name = None
        self._color = None
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
        msg = {
            self._VR_HEAD_POSITION: self._head_position(c),	 # In room coordinates
            self._VR_HAND_POSITIONS: self._hand_positions(c),	 # In room coordinates
        }

        nu = self._name_update()
        if nu:
            msg['name'] = nu

        cu = self._color_update()
        if cu:
            msg['color'] = cu
            
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
            # TODO: When the first VR participant joins scene_moved is false
            #       so the room to scene coordinates are unknown and the head and hand
            #       positions won't be shown correctly until the VR joiner moves the scene.
            #       Not great, but adequate.  It is tricky to do better.  Ticket #4438.

        # Report changes in VR GUI panel
        gu = self._gui_updates(c)
        if gu:
            msg.update(gu)

        # Tell other participants my new vr state
        self._participant._send_message(msg)

    def _head_position(self, vr_camera):
        from chimerax.geometry import scale
        return _place_matrix(vr_camera.room_position * scale(1/vr_camera.scene_scale),
                             encoding = 'vr room')

    def _name_update(self):
        name = self._participant._name
        if name != self._name:
            self._name = name
            return name
        return None

    def _color_update(self):
        color = tuple(self._participant._color)
        if color != self._color:
            self._color = color
            return color
        return None
    
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
        return [_place_matrix(h.room_position, encoding = 'vr room')
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
                self.name = '%s VR' % (msg['name'] if msg['name'] else 'Remote')
        if 'color' in msg:
            for h in self._hands:
                h.set_cone_color(msg['color'])
        if 'vr coords' in msg:
            self.room_to_scene = _matrix_place(msg['vr coords'])
        if VRTracking._VR_HEAD_POSITION in msg:
            h = self._head
            hm = msg[VRTracking._VR_HEAD_POSITION]
            h.room_position = rp = _matrix_place(hm, encoding = 'vr room')
            h.position = self.room_to_scene * rp
        if 'vr head image' in msg:
            self._head.update_image(msg['vr head image'])
        if VRTracking._VR_HAND_POSITIONS in msg:
            hpos = msg[VRTracking._VR_HAND_POSITIONS]
            rts = self.room_to_scene
            for h,hm in zip(self._hand_models(len(hpos)), hpos):
                h.room_position = rp = _matrix_place(hm, encoding = 'vr room')
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

        # Don't allow clip planes to hide head models.
        self.allow_clipping = False
        
        r = size / 2
        from chimerax.surface import box_geometry
        va, na, ta = box_geometry((-r,-r,-0.1*r), (r,r,0.1*r))

        if image_file is None:
            from os.path import join, dirname
            image_file = join(dirname(__file__), self.default_face_file)
        from Qt.QtGui import QImage
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

    def update_image(self, image_bytes):
        im_bytes = _decode_face_image(image_bytes)
        from Qt.QtGui import QImage
        qi = QImage()
        qi.loadFromData(im_bytes)
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
        self.allow_clipping = False	# Don't let panels get clipped

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

def _place_matrix(p, encoding = 'float32 matrix'):
    '''Encode Place as bytes for sending over socket.'''
    if encoding == 'float32 matrix':
        from numpy import float32
        bytes = p.matrix.astype(float32).tobytes()
    elif encoding == 'vr room':
        bytes = _encode_vr_room_position(p)
    return bytes

def _matrix_place(bytes, encoding = 'float32 matrix'):
    if encoding == 'float32 matrix':
        from numpy import frombuffer, float32, float64
        m = frombuffer(bytes, float32).astype(float64).reshape((3,4))
        from chimerax.geometry import Place
        p = Place(matrix = m)
    elif encoding == 'vr room':
        # Encode as 16-bit shift and rotation vector.
        p = _decode_vr_room_position(bytes)
    return p

def _encode_vr_room_position(p):
    '''
    Encode room position with minimal bytes (12 instead of 48)
    to lower bandwidth when position is transmitted at high frequency.
    Encode shift and rotation vector as 16-bit integers.
    Shift resolution is 0.5 mm with range  +/- 16 meters.
    Rotation resolution is 0.005 degree (= 180 / 32000).
    '''
    from numpy import empty, float64, int16, clip
    v = empty((6,), float64)
    clip(2000*p.origin(), -32768, 32767, out = v[3:6])
    axis, angle = p.rotation_axis_and_angle()
    v[0:3] = (32000 * angle/180) * axis
    bytes = v.astype(int16).tobytes()
    return bytes

def _decode_vr_room_position(bytes):
    from numpy import frombuffer, int16, float64
    v = frombuffer(bytes, int16)
    origin = 0.0005 * v[3:6]
    ax,ay,az = v[0:3].astype(float64)
    from math import sqrt
    a = sqrt(ax*ax+ay*ay+az*az)
    angle = a * (180/32000)  # degrees
    axis = (ax/a,ay/a,az/a) if a > 0 else (0,0,1)
    from chimerax.geometry import rotation, translation
    p = translation(origin) * rotation(axis, angle)
    return p

def _encode_face_image(path):
    f = open(path, 'rb')
    bytes = f.read()
    f.close()
    return bytes

def _decode_face_image(bytes):
    return bytes

def _encode_numpy_array(array):
    from lz4.frame import compress
    bytes = compress(array.tobytes())
    data = {
        'shape': tuple(array.shape),
        'dtype': array.dtype.str,
        'data': bytes
    }
    return data

def _decode_numpy_array(array_data):
    shape = array_data['shape']
    dtype = array_data['dtype']
    from lz4.frame import decompress
    bytes = decompress(array_data['data'])
    import numpy
    a = numpy.frombuffer(bytes, dtype).reshape(shape)
    return a
