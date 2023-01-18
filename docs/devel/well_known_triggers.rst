..  vim: set expandtab shiftwidth=4 softtabstop=4:

:orphan:

.. 
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

Well Known Triggers
===================

Bundles may wish to be able to react when various events occur, such as models being closed,
or a structure chain being deleted.  Bundles can arrange for relevant code to execute by 
registering with the "trigger" that is fired when such an event occurs.  The mechanism for
registering with a trigger is described in :doc:`modules/core/triggerset`.  This document describes
the names and locations of important triggers that a developer might want to register for.

Quick Overview
--------------

Grouped by triggerset location, here are some of the important triggers provided in ChimeraX.
Later sections provide detailed information about accessing the triggerset, exact trigger names,
and their associated data.

:ref:`Session <session_triggers>`

* models opened/closed
* selection changed
* model position changed
* begin/end session restore
* ChimeraX quitting
* new frame drawn

:ref:`Atomic <atomic_triggers>` -- creation/modification/deletion of atoms, bonds, residues, chains, and structures

:ref:`Core <core_triggers>` -- graphics background color change

:ref:`UI <ui_triggers>` -- main window ready

.. _session_triggers:

Session Triggers
================

The triggerset is :code:`session.triggers`.  The most useful available trigger names are:

:code:`chimerax.core.models.ADD_MODELS`
    Models have been added to the session.  The trigger data is a list of the models that were added.

:code:`chimerax.core.models.REMOVE_MODELS`
    Models have been removed from the session.  The trigger data is a list of the models that were removed.

:code:`chimerax.core.selection.SELECTION_CHANGED`
    The current selection has changed.  The trigger data is :code:`None`.
    The selection state of various atomic data types can be fetched by functions in the
    :doc:`chimerax.atomic <modules/atomic/atomic>` module
    (*e.g.* :code:`selected_residues(session)`)
    or more generically via methods of :code:`session.selection`.

:code:`chimerax.core.models.MODEL_POSITION_CHANGED`
    A model has changed position relative to other models in the scene.  Rotations and translations of
    models typically change the position of the camera relative to the models and not the positions
    of the models themselves.  However, things like Match Maker superposition of models will move a
    model relative to other models.  This trigger fires when such a position change occurs.
    The trigger data is the model that changed position.

"begin restore session" / "end restore session"
    A session is about to be restored.  A bundle may want to suspend some kinds of processing during a
    session restore or postpone some processing until after the restore completes and the entire session
    is in a consistent state.  The trigger data is the session object (which is the same as the current
    main session).

"app quit"
    ChimeraX is about to quit.  Tools could kill running subthreads or query about saving unsaved editing
    changes, for instance.  The trigger data is :code:`None`.

"new frame" / "frame drawn"
    A frame of the main graphics is about to be drawn / has been drawn.  Most frequently used to avoid
    updating graphical user interfaces faster than the 3D graphics itself updates, which would be wasteful
    and could bog down the user interface.  The trigger data is a :code:`chimerax.core.updateloop.UpdateLoop`
    instance, which is rarely relevant.

.. _atomic_triggers:

Atomic Triggers
===============

The atomic triggerset is obtained by calling :code:`chimerax.atomic.get_triggers()`.
The two important trigger names are "changes" and "changes done".
"changes done" is used if a bundle wants other bundles to be able to react to a change
before continuing processing, and its trigger data is :code:`None`.
The "changes" trigger encapulates all changes to C++-level atomic data
(and some Python-level changes, see :ref:`Python-Level Atomic Attributes <python_attributes>`)
since the last firing of the "changes" trigger.
The trigger data is an :code:`chimerax.atomic.changes.Changes` instance,
described :ref:`below <changes_methods>`.
If you are only interested in changes to a particular structure,
each structure has a :code:`triggers` attribute with a "changes"
trigger that will deliver a :code:`Changes` instance containing only
changes to that structure.

.. _changes_methods:

Changes Methods
---------------
A Changes object has methods for querying what atomic items of a particular type
have been created or modified, or how many such items have been deleted.
The created/modified methods return the appropriate type of :py:class:`~chimerax.atomic.molarray.Collection`.
The methods are named :code:`created_`\ *X*, :code:`modified_`\ *X*, and :code:`num_deleted_`\ *X*,
where *X* is one of: :code:`atoms`, :code:`bonds`, :code:`residues`, :code:`chains`, :code:`pseudobonds`,
:code:`pseudobond_groups`, :code:`coordsets`, :code:`structures`, or :code:`atomic_structures`.
Note that for efficiency reasons, for newly opened structures only the :code:`created_structures`
(or :code:`created_atomic_structures`) method holds any data, so if you care about items created by
new structures as well as those added to existing structures, you will have to check the
:code:`created_structures` method and use the appropriate Collection property to get the items of interest.

For modified items you can also find out what changed by using the *X*\ :code:`_reasons` method,
where *X* is the same list as before except using the singular form instead of plural.
These methods return a set of strings, each string of the form "\ *item* changed".
*Item* is frequently an attribute name, but can be more generic.
For instance, changing one or more of the coordinates in a Coordset will produce a
"coordset changed" reason for Coordsets, which doesn't correspond to any single attribute.
For each class, here are the changes that are tracked:

+-----------------+----------------------+
| Class           | Reason               |
+=================+======================+
| Atom            | alt_loc              |
|                 | aniso_u              |
|                 | bfactor              |
|                 | color                |
|                 | coord                |
|                 | display              |
|                 | draw_mode            |
|                 | element              |
|                 | hide                 |
|                 | idatm_type           |
|                 | name                 |
|                 | occupancy            |
|                 | selected             |
|                 | serial_number        |
|                 | structure_category   |
+-----------------+----------------------+
| Bond            | color                |
| Pseudobond      | display              |
|                 | halfbond             |
|                 | hide                 |
|                 | radius               |
|                 | selected             |
+-----------------+----------------------+
| Residue         | chain_id             |
|                 | insertion_code       |
|                 | name                 |
|                 | number               |
|                 | ribbon_adjust        |
|                 | ribbon_color         |
|                 | ribbon_display       |
|                 | ribbon_hide_backbone |
|                 | ring_color           |
|                 | ring_display         |
|                 | ring_mode            |
|                 | ss_id                |
|                 | ss_type              |
+-----------------+----------------------+
| Chain           | chain_id             |
|                 | residues             |
|                 | sequence             |
+-----------------+----------------------+
| PseudobondGroup | color                |
|                 | halfbond             |
|                 | radius               |
+-----------------+----------------------+
| Coordset        | coordset             |
+-----------------+----------------------+
| Structure       | active_coordset      |
| AtomicStructure | ball_scale           |
|                 | display              |
|                 | ribbon_display       |
|                 | ribbon_tether        |
|                 | ribbon_orientation   |
|                 | ribbon_mode          |
|                 | scene_coord          |
+-----------------+----------------------+

.. _python_attributes:

Python-Level Atomic Attributes
------------------------------

ChimeraX has no way of knowing when Python-level attributes are defined or changed.
If you define a new attribute of one of the atomic classes, you will have to take
steps to get that attribute saved in sessions and to get changes to that attribute to
show up in :code:`Changes` objects (the latter only necessary if you think other bundles might want
to know about such changes).

Sessions
^^^^^^^^

To get your attribute saved in sessions you will have to "register" it by calling
*atomic_class*\ :code:`.register_attr(session, attr_name, registrant)` where "registrant"
is some string identifying the source of the registration (used in registration-conflict
error messages) -- frequently your bundle name.  :code:`register_attr()` has a few optional
keyword arguments:

attr_type
    If the attribute is always either a particular type (*e.g.* float) or :code:`None`, then specify this.
    This helps interfaces and commands that categorize attributes by type (*e.g.* "color byattr").

can_return_none
    True if :code:`None` is a legal value. Defaults to :code:`False`.

Changes Notifications
^^^^^^^^^^^^^^^^^^^^^

To get changes in attribute values reported in :code:`Changes` objects, call
:code:`session.change_tracker.add_modified(assigned, reason)`.
:code:`assigned` can be a single object, a :code:`Collection` subclass, or an iterable of objects.
:code:`reason` is typically "\ *attr_name* changed".

.. _core_triggers:

Core Triggers
=============

:doc:`Settings <modules/core/settings>` objects each have a triggerset that fires a "setting changed" trigger
whenever a setting is changed.  The associated data is a three-tuple of (setting name, old value,
new value).  The core settings object (:code:`chimerax.core.core_settings.settings`) has one setting of
general interest: "background_color", whose value is a :py:class:`chimerax.core.colors.Color`, whose
:py:meth:`uint8x4` returns its RGBA value as an array of four (:code:`x4`) 8-bit integers (:code:`uint8`).

.. _ui_triggers:

UI Triggers
===========

Bundles may need to delay some operations (*e.g.* adding entries to ChimeraX's main menus)
until the ChimeraX main window has fully initialized.
Such bundles should register code to execute in response to the
:code:`session.ui.triggers` triggerset's "ready" trigger.
