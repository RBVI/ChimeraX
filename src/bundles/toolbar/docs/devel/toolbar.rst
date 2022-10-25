toolbar
-------

The toolbar manager uses the manager/provider (TODO: link) mechanism
for adding buttons to the application toolbar.
Buttons are organized into sections within tabs.
And buttons may be optionally grouped, so only one shows at a time.

Buttons are registered by adding providers in a bundle's **bundle_info.xml** file.
The *manager* must be **toolbar**.
The *manager* may be given in each **Provider** or once in the parent **Providers**.
There can be multiple **Providers** with different managers.
Buttons are invoked by calling the bundle's **run_provider** API (TODO: link)
with the provider's *name*.

Basic Button
============

Example::

  <Providers manager="toolbar">
    <Provider tab="Graphics" section="Undo"
      name="Redo" icon="redo-variant.png" description="Redo last action"/>
  </Providers>

A basic button **Provider** needs a *tab*, *section*, *name*, *icon*,
*description*, and optionally a *display_name*.
The *display_name* defaults to the value of the *name* option.
The *display_name* is used to label the button.
The optional *icon* is searched for the in the bundle's **icons** directory.
Also, buttons can be assigned a *group*, where only one button in the group
is visible at a time.
The order of the buttons may be constrained with the *before* and *after* options.
The *before* and *after* values refer to the *display_name* of other buttons in the same
section.
The constraints may include several other buttons by using a colon separated list.
If the *hidden* option is given, with any value, then the button is not shown,
but is available for adding for adding to custom tags, *e.g.*, the **Home** tab.

Mouse Mode
==========

Example::

  <Providers manager="toolbar">
    <Provider tab="Markers" section="Place markers" name="pm1"
      mouse_mode="mark maximum" display_name="Maximum" description="Mark maximum"/>
  </Providers>

Mouse mode buttons are similar to basic buttons,
with the *mouse_mode* identifying which mode mode to set
and the default *display_name*.
The optional *icon* can be given,
but is normally found with the registered mouse mode.

Link Button
===========

Example::

  <Providers manager="toolbar">
    <Provider tab="Map" section="Undo"
      name="Redo-2" link="BundleName:provider-name"/>
  </Providers>

To place to a button into multiple tabs and/or sections,
a **Provider** may give a *link* to it.
The link value is a colon separated *bundle-name:name* pair.
Its **name** is ignored, but still needs to be unique.
*tab* and *section* are required.
*icon*, and *description*, must not given.
*hidden* is ignored.
*display_name*, *before*, *after*, and *group* are optional.
Currently, links may not be made to mouse mode buttons.

Layout
======

Example::

  <Providers manager="toolbar">
    <Provider tab="Graphics" help="help:..."
      name="layout1" before="Nucleotides" after="Molecule Display"/>
    <Provider tab="Molecule Display" section="Cartoons" help="help:..."
      name="layout2" before="Surfaces" after="Atoms"/>
  </Providers>

Tabs, sections, and buttons can have layout contraints:

Information can also be given at the tab and section levels of the hierachy.
The *name* needs to be unique amoung the bundle's **Provider**'s,
but is otherwise unused.
The *before* and *after* constraints refer to items at that level.
And a URL can be given with the *help* option.
