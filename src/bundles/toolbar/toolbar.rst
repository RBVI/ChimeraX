toolbar
-------

The toolbar manager uses the manager/provider (TODO: link) mechanism
for adding buttons to the application toolbar.
Buttons are organized into sections within tabs.
And buttons may be optionally grouped, so only one shows at a time.

Buttons are registered by adding providers in a bundle's **bundle_info.xml** file.
The *manager* must be **toolbar**.
Buttons are invoked by calling the bundle's **run_provider** API (TODO: link)
with the provider's *name*.

For example:

  <Providers manager="toolbar">
    <Provider tab="Graphics" section="Undo"
      name="Redo" icon="redo-variant.png" description="Redo last action"/>
    <Provider tab="Map" section="Undo"
      name="Redo-2" link="BundleName:provider-name"/>
    <Provider tab="Graphics" help="help:..."
      name="layout1" before="Nucleotides" after="Molecule Display"/>
    <Provider tab="Molecule Display" section="Cartoons" help="help:..."
      name="layout2" before="Surfaces" after="Atoms"/>
  </Providers>

A basic button **Provider** needs a *tab*, *section*, *name*, *icon*,
*description*, and optionally a *display_name*.
The *display_name* defaults to the value of the *name* option.
The *display_name* is used to label the button.
The *icon* is searched for the in the bundle's **icons** directory.
Also, buttons can be assigned a *group*, where only one button in the group
is visible at a time.
The order of the buttons may be constrained with the *before* and *after* options.
The *before* and *after* values refer to the *display_name* of other buttons in the same
section.
The constraints may include several other buttons by using a colon separated list.
If the *hidden* option is given, with any value, then the button is not shown,
but is available for adding for adding to custom tags, *e.g.*, the **Home** tab.

To place to a button into multiple tabs and/or sections,
a **Provider** may give a *link* to it.
The link value is a colon separated *bundle-name:name* pair.
Its **name** is ignored, but still needs to be unique.
*tab* and *section* are required.
*icon*, and *description*, must not given.
*hidden* is ignored.
*display_name*, *before*, *after*, and *group* are optional.

Information can also be given at the tab and section levels of the hierachy.
The *name* needs to be unique amoung the bundle's **Provider**'s,
but is otherwise unused.
The *before* and *after* constraints refer to items at that level.
And a URL can be given with the *help* option.
