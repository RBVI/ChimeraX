========================
Chimera2 Web Application
========================

Requirements
============

The typical usage scenario for the Chimera2 web application is that an
end user initiates a *session* by visiting a web page from which he can
launch *applets* that present him with user interfaces for viewing and
analyzing a variety of data.  The session information should be *persistent*
so that the user may temporarily disconnect (perhaps visiting another
web page) and return to continue from the previous state.

Design
======

The web application is split into a server-side process for maintaining
persistent state and a client-side web page for managing and displaying
applets.  Design goals include:

    - Server-side process should not need to know about user interface
      details so that different front ends may be used to access the
      same session.  For example, a client-side web page may have several
      applets that each access different server functionality.
    - Applets should only need to know about data types in which they are
      interested.  For example, a 3D applet needs to know about graphical
      scene description but not about protein sequences; an alignment
      viewer, on the other hand, needs to know about sequence alignments
      but not the atomic coordinates of individual atoms.
    - Multiple applets should be able to co-exist and work cooperatively.
      For example, the 3D applet can show the structure of a protein while the
      alignment viewer can simultaneously show the amino acid sequence
      of the same protein; when an amino acid is selected in the alignment
      viewer, the same amino acid might also be highlighted in the 3D applet.
    - The user should only need to know the URL for the web application.
      He should not need to install any software for Chimera2, although a
      modern browser may be a prerequisite.

Specifications
==============

The data format and application program interface specifications
are in the following sections:

    - The *Wire Protocol* section describes the format of the data
      exchange between client and server.
    - The *Client Framework* section describes how applets send
      and receive data.
    - The *Server Framework* section describes how the server process
      receives, dispatches, and returns data.

`Wire Protocol`_
----------------

The data exchange between client and server will be in
`JSON <http://www.json.org>`_ format.  Typically, the exchange
is communicated using `AJAX <http://www.w3schools.com/ajax/ajax_intro.asp>`_.

Data from client to server are arrays of requests, each of which is a
three-element array:

    *id*
        Request identifier (to be included in return data)
    *tag*
        Request type (used for dispatching request)
    *value*
        Request data (format determined by request type)

For example:

::

    [
      [ 1, "command", "chain @ca" ],
      [ 2, "save", [ "3D applet",
                        { "position": [ 10, 10, 300, 400 ],
                          "scale": 3.4 } ] ],
    ]

may be a batch of two requests, one for executing a typed command
followed by another to save some data.

Data from server to client are arrays of return values, each of which
is a JSON object with the following name/value pairs:

    *id*
        Identifier that matches the identifier of the request that
        generated this return data.
    *status*
        Boolean value that is *true* if the request was successfully
        processed and *false* otherwise.
    *stdout*
        Array of strings that contain text output from processing
        the request.
    *stderr*
        Array of strings that contain error output from processing
        the request.
    *client_data*
        JSON object representing the return data generated from
        processing the request.
    *server_data*
        JSON object containing metadata that map client data to global
        server data.  The name/value pairs of this object map identifiers
        of applet data to identifiers of server data.  This mapping is
        intended to be used by the client to manage inter-applet
        communications.  For example, when an amino acid is selected
        in the alignment viewer, the applet can map its internal identifier
        for the amino acid to its server identifier; it can then notify
        all other applets that the server identifier has been selected;
        the 3D applet can then map the server identifier to *its* internal
        identifier and then highlight the amino acid in its graphical display.

`Client Framework`_
-------------------

The client framework is responsible for accepting requests from
applets, sending them to the server, receiving data from the server,
managing *server_data* maps, and distributing *client_data*, *stdout*
and *stderr* to the appropriate destinations.

`Server Framework`_
-------------------

The server framework is responsible for accepting requests from
the client, dispatching the request to registered functions,
repackaging function return values, and sending the return values
to the client.

The server framework is described in the `webapp_server <webapp_server.html>`_
Python module.
