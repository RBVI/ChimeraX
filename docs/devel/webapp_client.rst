..  vim: set expandtab shiftwidth=4 softtabstop=4:

=======================================================
webapp_client: Client side for ChimeraX web application
=======================================================

Overview
========

The ChimeraX Web Application uses several JavaScript libraries,
including public ones such as `jQuery <http://jquery.com>`_ and
`jQuery UI <http://jqueryui.com>`_ as well as custom libraries for
:ref:`session-management` and
:ref:`applet-management`.

.. _session-management:

Session Management
==================

The session management library is divided into two parts:

- an HTML layer that provides user interface elements
  for managing the current active session information, and
- an AJAX layer that interacts with the server without
  maintaining any state information.

The public API consistents of attributes of one object:

.. js:data:: $c2_session

HTML Layer
----------

The HTML layer API is accessible via :js:data:`$c2_session`.

.. js:function:: $c2_session.ui_init(url)

    :param string url: URL to the location of ChimeraX server.
    :returns: nothing.

    Initialize HTML and AJAX layers.  The AJAX layer is initialized with a
    call to :js:func:`~$c2_session.server.init`.  To create user interface
    widgets, :js:func:`~$c2_session.ui_init` looks for an HTML element
    with id ``c2_session``.  A jQuery button is inserted at the start
    of this element that both displays the name of the current active
    session and, when clicked, brings up a session selection dialog.

.. js:attribute:: $c2_session.user

    Web account used to contact server.

.. js:attribute:: $c2_session.session

    Name of current active session.

.. js:attribute:: $c2_session.password

    Password for current active session.

.. js:attribute:: $c2_session.session_list

    Array of sessions for current `user`.  Each session
    is an object with attributes `name` and `access` (see
    :js:func:`~$c2_session.server.list_sessions`).

AJAX Layer
-----------

The AJAX layer API is accessible via :js:attr:`$c2_session.server`.

.. js:function:: $c2_session.server.init(url)

    :param string url: URL to the location of ChimeraX server.
    :returns: nothing.

    Initialize AJAX layer.  Must be called prior to using other functions.

.. js:function:: $c2_session.server.list_sessions()

    :returns: :jquery:`ajax` jqXHR object.

    Send an AJAX request for session information.  The return value is
    the object returned by jQuery's :jquery:`getJSON` function, and is
    used to invoke callbacks and for adding error handling functionality.
    If the request is successful, the done method's data argument has
    the session data, which is of the form::

        [ "user_name", [
            { name: "session_name", access: "access_time" },
            { name: "session_name_2", access: "access_time_2" },
            ...
        ]]

    and describes the list of sessions associated with a web account,
    where:

    - *user_name* is the web login used to access the server,
    - *session_name* is the name of a session, and
    - *access_time* is the last access time associated with the session
      (as a string formatted by the :py:func:`~time.ctime` function).

.. js:function:: $c2_session.server.create_session(session_name, password)

    :param string session_name: Name of session to be created.
    :param string password: Name of password for session to be created.
    :returns: :jquery:`ajax` jqXHR object.

    Send an AJAX request to create a new session.  The return value
    is the object returned by :jquery:`get` function, and is used to
    invoke callbacks and for adding error handling functionality.  If the
    request is successful, the done method's data argument is empty.

.. js:function:: $c2_session.server.delete_session(session_name, password)

    :param string session_name: Name of session to be created.
    :param string password: Name of password for session to be created.
    :returns: :jquery:`ajax` jqXHR object.

    Send an AJAX request to delete an existing session.  The return value
    is the object returned by :jquery:`get` function, and is used to
    invoke callbacks and for adding error handling functionality.  If the
    request is successful, the done method's data argument is empty.

.. js:attribute:: $c2_session.server.url

    Server URL set by :js:func:`~$c2_session.server.init` function.

.. _applet-management:

Applet Management
=================

TODO
