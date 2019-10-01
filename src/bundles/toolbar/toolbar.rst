toolbar
-------

The toolbar provides a mechanism to add buttons to the application toolbar.
Buttons are organized into sections within tabs.
And buttons may be optionally grouped, so only one shows at a time.

To register a button, add providers in a bundle's **bundle_info.xml** file.

  <Providers manager="toolbar">
    <Provider tab="Home" section="Undo"
      name="Redo" icon="redo-variant.png" description="Redo last action"/>
    <Provider tab="Home" section="Undo"
      name="Redo-2" link="BundleName:provider-name"/>
    <Provider tab="Graphics" help="help:..."
      name="layout1" before="Nucleotides" after="Molecule Display"/>
    <Provider tab="Molecule Display" section="Cartoons" help="help:..."
      name="layout2" before="Surfaces" after="Atoms"/>
  </Providers>

A **Provider** must have a *manager*.  That *manager* may be given in the
**Providers** element or in each **Provider** element.
A **Provider** must also have an unique *name* that is understood by the
bundle.

A simple button **Provider** needs a *tab*, *section*, *name*, *icon*,
*description*, and optionally a *display_name* (which defaults to the *name*).
The *icon* is searched for the in the bundle's **icons** directory.
Also, buttons can be assigned a *group*, where only one button in the group
is visible at a time.
Also, the order of the buttons may be constrained with the *before* and *after*
options.
The constraints may include several other buttons by using a colon separated list.
These buttons are invoked by calling the bundle's **run_provider** API
with the button's *name*.  The *display_name* is given as a keyword argument,
but is often ignored.

To place to a button into multiple tabs and/or sections,
a **Provider** may give a *link* to it.
The link value is a colon separated *bundle-name:name* pair.
Its **name** is ignored, but still needs to be unique.
*tab* and *section* are required.
*icon*, *description*, and *display_name* must not given.
*before*, *after*, and *group* are optional.

Information can also be given at the tab and section levels of the hierachy.
The *name* needs to be unique amoung the bundle's **Provider**'s,
but is otherwise unused.
The *before* and *after* constraints refer to items at that level.
And a URL can be given with the *help* option.
