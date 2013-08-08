========================================================
webapp_client: Client side for Chimera 2 web application
========================================================

Overview
========

The Chimera2 Web Application uses several JavaScript libraries,
including public ones such as `jQuery <http://jquery.com>`_ and
`jQuery UI <http://jqueryui.com>`_ as well as custom libraries for
:ref:`session-management` and
:ref:`applet-management`.

.. _session-management:

Session Management
==================

The session management library is divided into two parts:

- a tool layer that provides user interface elements
  for managing the current active session information, and
- a basic layer that interacts with the server without
  maintaining any state information.

Tool Layer
----------

The tool layer API are accessible via ``$c2_session``.

.. js:function:: $c2_session.init(url)

    :param string url: URL to the location of Chimera 2 server.
    :returns: nothing.

    Initialize tool and basic layers.  The basic layer is initialized
    with a call to `$c2_session.server.init`.  To create user interface
    widgets, `init` looks for an HTML element with id `c2_session`.
    A jQuery button is inserted at the start of this element that both
    displays the name of the current active session and, when clicked,
    brings up a session selection dialog.

.. js:data:: $c2_session.user

    Web account used to contact server.

.. js:data:: $c2_session.session

    Name of current active session.

.. js:data:: $c2_session.password

    Password for current active session.

.. js:data:: $c2_session.session_list

    Array of sessions for current `user`.  Each session is an object with
    attributes `name` and `access` (see `list_sessions` above).

Basic Layer
-----------

The basic layer API are accessible via ``$c2_session.server``.

.. js:function:: $c2_session.server.init(url)

    :param string url: URL to the location of Chimera 2 server.
    :returns: nothing.

    Initialize basic layer.  Must be called prior to using
    other functions.

.. js:function:: $c2_session.server.list_sessions(callback)

    :param callback: Invoked when server returns session information.
    :returns: jQuery XHR object for AJAX request.

    Send an AJAX request for session information.  The return value
    is the object returned by jQuery's `getJSON` function, and may be
    useful for adding error handling functionality.  If the request
    is successful, the `callback` function is invoked with a single
    argument of the session data, which is of the form::

        [ "user_name",
            [ { name: "session_name", access: "access_time" },
              { name: "session_name_2", access: "access_time_2" },
              ... ]
        ]

    and describes the list of sessions associated with a web account, where:
    
    - *user_name* is the web login used to access the server,
    - *session_name* is the name of a session, and
    - *access_time* is the last access time associated with the session
      (as a string formatted by the `ctime` function).

.. js:function:: $c2_session.server.create_session(session_name, password, callback)

    :param string session_name: Name of session to be created.
    :param string password: Name of password for session to be created.
    :param callback: Invoked when server returns status information.
    :returns: jQuery XHR object for AJAX request.

    Send an AJAX request to create a new session.  The return value
    is the object returned by jQuery's `get` function, and may be
    useful for adding error handling functionality.  If the request
    is successful, the `callback` function is invoked with no arguments.

.. js:function:: $c2_session.server.delete_session(session_name, password, callback)

    :param string session_name: Name of session to be created.
    :param string password: Name of password for session to be created.
    :param callback: Invoked when server returns status information.
    :returns: jQuery XHR object for AJAX request.

    Send an AJAX request to delete an existing session.  The return value
    is the object returned by jQuery's `get` function, and may be
    useful for adding error handling functionality.  If the request
    is successful, the `callback` function is invoked with no arguments.

.. js:data:: $c2_session.server.url

    Server URL set by `init` function.

.. _applet-management:

Applet Management
=================

TODO
