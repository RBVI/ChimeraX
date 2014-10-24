..  vim: set expandtab shiftwidth=4 softtabstop=4:

Front-/Back-end Split for HTML5 3D Graphics
===========================================

Overview
~~~~~~~~

There are 4 ways to split an application between a remote server
and a local client:
(1) run it entirely on the server and remotely display it using screen grabs,
e.g., with vnc,
(2) run the application on the server but have the GUI on the client,
e.g., with X11/OpenGL or other remote desktop software,
(3) run the application on the client and use the server for remote computation,
(4) run entirely on the client.
Where the split is made
will affect the performance and the maintainability of the application.  

A modern web browser allows for any of the above splits to be made.
For all but first type of split, the GUI is implemented with HTML5 and WebGL.
And since we wish to take advantage of WebGL,
that eliminates running entirely on the server.
Running entirely on the client is out
because there will always be computations that will be to be run remotely.
So the question becomes which which end should be in charge,
the client or the server?

Assumptions:
(1) server-to-client bandwidth is much higher than client-to-server bandwidth
(2) communication between server and client, speed and quality,
is fair to very good (not bad nor excellent)
(3) need to minimize communication between server and client:
round trips, quantity, and frequency
(4) allow manipulation on client without contacting server
(5) take advantage of client's graphics hardware (i.e., WebGL)
(6) session support (persistent state)

These assumptions lead to building an application that runs on the server
and uses the web browser for its GUI and then optimizing the interaction
by migrating or duplicating functionality to the client.

3D Graphics Support
~~~~~~~~~~~~~~~~~~~

Digging deeper to how to split the graphics between the server and the client,
we need:

* graphical primitives:

  - lines, dots, triangles, volumes

  - cylinders, spheres, disks

  - level of detail (LOD)

  - instancing

  - text

* lighting

  - shading

  - shadows

* translucency

  - single layer transparency
  - per-model transparency (order-independent)
  - multi-layer transparency (order-independent)

* selection highlight

* picking

* projection/view/model matrices

* high resolution printing

* 2D overlay

  - e.g., text, arrows, color key

* 2D underlay

  - background gradient

  - background image

And

* reuse split abstractions for desktop version

  - simplifies supporting different platforms

* minimal front-end code

  - easy to duplicate in JavaScript and C++/Objective-C

  - can provide to third-parties to demonstrate bugs
    without providing whole application

Ideal Graphics Split
~~~~~~~~~~~~~~~~~~~~

The front-end would be:

* WebGL / OpenGL ES 2.0 / OpenGL 3.3(?)

* X11-style API where round trip communication is minimal

* buffer/array based

* flat, state-sorting, scene graph of arrays and instances

* LOD primitives

* selection highlights

* translucency

* shadows

* matrices

* shaders

* culling?

The back-end will:

* prepare the arrays of:

  - vertices
  - colors
  - matrices
  - et. al.

* selection and translucency annotations

* shader preparation

Graphics Front-end Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See frontend.h for C++ example.
