# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
settings: Save/access tool interface values
===========================================

Basic Usage
-----------

This module provides a convenient way for tools to remember preferred settings.
To do so you create a subclass of this module's :py:class:`Settings` class and
populate it with setting names and default values.  There are two varieties of
settings, ones that are automatically saved to disk when they're set, and ones
that only save to disk when the Settings.save() method is called.  The former
might be used for minor conveniences in the interface, like remembering which
file format the user last used, whereas the latter would be used if a tool
presents an explicit Save button for preserving configuration information.

The settings are presented as attributes of the Settings instance, so therefore
the setting name has to be legal as an attribute, *i.e.* no spaces, just
alphanumeric characters plus underscore.  Also, leading underscores are not
allowed.

**Example**

*setup*

.. code-block: c++

    from chimerax.core.settings import Settings
    from chimerax.core.configfile import Value
    from chimerax.core.commands import EnumOf
    class FullEnum(EnumOf): allow_truncated = False

    class MyToolSettings(Settings):
        AUTO_SAVE = {
            'gap_penalty': 2,
            'matrix': Value('BLOSUM62',
                FullEnum(('BLOSUM45', 'BLOSUM62', 'BLOSUM80')), str)
        }
        EXPLICIT_SAVE = {
            'alignment_coloring': Value('clustal',
                FullEnum(('black', 'clustal', 'residue')), str),
            'font_size': 16
        }

    my_settings = MyToolSettings(session, "MyToolName")

*use*

.. code-block: c++

    penalty += gap_len * my_settings.gap_penalty

*saving*

.. code-block: c++

    my_settings.matrix = 'BLOSUM80' # auto-saved
    ...
    my_settings.font_size = 12
    my_settings.save() # saves all settings

As mentioned above, the AUTO_SAVE/EXPLICIT_SAVE keys have to be legal to use
as attribute names.  The values can be just plain Python values, in which case
their :py:func:`repr` will be saved to disk, or they can be more complex
entities, like enumerations or class instances.  In the latter case you have
to use the :py:class:`~chimerax.core.configfile.Value` class to tell the settings
mechanism how to save and restore the value and perhaps check that the value is
within its legal range of values.  The three arguments to Value() are:

1. The default value.
2. A function to convert text to the value, or an object that can convert the
   text into the needed value by following the
   :py:class:`~chimerax.core.commands.cli.Annotation` abstract class protocol.
3. A function that converts a value to text.

There are many Annotation subclasses in the :py:mod:`~chimerax.core.commands.cli`
module that can be used as the second argument to Value() and that also
perform range checking.

Advanced Usage
--------------

Adding a setting
    One simply adds the definition into the appropriate dictionary, and
    increases the minor part of the Settings version number.  The version
    number is provided as a keyword argument (named *version*) to the
    Settings constructor.  It defaults to "1".

Moving/deleting settings
    This involves adding properties to your Settings subclass and is
    discussed in detail in the :py:mod:`~chimerax.core.configfile`
    documentation.

"""


from .configfile import ConfigFile, only_use_defaults

class Settings(ConfigFile):
    """Supported API. Save/remember tool interface settings

    A tool remembers interface setting across tool invocations with this
    class.  There are two types of settings supported: ones that save to
    disk immediately when changed, and ones that are saved to disk only
    when the :py:meth:`save` method is called.

    A tools uses Settings by subclassing and defining class dictionaries
    named AUTO_SAVE and EXPLICIT_SAVE containing the names of the settings
    and their default values.  This is explained in detail in the
    :py:mod:`~chimerax.core.settings` module documentation.

    Parameters
    ----------
    session
        The ChimeraX session object
    tool_name : str
        The name of the tool
    version : str, optional
        The version number for the settings, which should be increased
        if settings are added, deleted, or modified.  Discussed in the
        Advanced Usage section of the :py:mod:`settings` module documentation.

    Attributes
    ----------
    AUTO_SAVE: Dict
        Class dictionary containing setting names and default values.
        Such settings will be saved to disk immediately when changed.
    EXPLICIT_SAVE: Dict
        Class dictionary containing setting names and default values.
        Such settings will be saved to disk only when the :py:meth:`save`
        method is called.
    triggers: TriggerSet
        When a setting changes its current value, the 'setting changed'
        trigger will be activated with (attr_name, prev_val, new_val)
        as the data provided with the trigger.
    """

    AUTO_SAVE = EXPLICIT_SAVE = {}

    def __init__(self, session, tool_name, version="1"):
        object.__setattr__(self, '_settings_initialized', False)
        self.__class__.PROPERTY_INFO = {}
        self.__class__.PROPERTY_INFO.update(self.__class__.AUTO_SAVE)
        self.__class__.PROPERTY_INFO.update(self.__class__.EXPLICIT_SAVE)
        self._cur_settings = {}
        ConfigFile.__init__(self, session, tool_name, version=version)
        for attr_name in self.__class__.PROPERTY_INFO.keys():
            if attr_name[0] == '_':
                raise ValueError("setting name cannot start with underscore")
            self._cur_settings[attr_name] = getattr(self, attr_name)
        object.__setattr__(self, '_settings_initialized', True)
        from .triggerset import TriggerSet
        object.__setattr__(self, 'triggers', TriggerSet())
        self.triggers.add_trigger('setting changed')

    def __getattr__(self, name):
        if only_use_defaults or not self._settings_initialized:
            return ConfigFile.__getattr__(self, name)
        try:
            return self._cur_settings[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if (self._settings_initialized and name[0] != '_' and name in self._cur_settings):
            cur_val = self._cur_settings[name]
            import warnings
            import numpy as np
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=np.VisibleDeprecationWarning)
                unequal = not np.array_equal(cur_val, value)
            if unequal:
                self._cur_settings[name] = value
                if name in self.__class__.AUTO_SAVE:
                    ConfigFile.__setattr__(self, name, value)
                self.triggers.activate_trigger('setting changed', (name, cur_val, value))
        else:
            ConfigFile.__setattr__(self, name, value)

    def default_value(self, name):
        return self.PROPERTY_INFO[name].default

    def reset(self):
        '''Supported API.  Reset (revert to default) all settings.'''
        for name in self.PROPERTY_INFO.keys():
            setattr(self, name, self.default_value(name))

    def restore(self):
        '''Supported API.  Restore (revert to saved) all settings.'''
        for name in self.__class__.EXPLICIT_SAVE.keys():
            setattr(self, name, self.saved_value(name))

    def save(self, setting=None, *, settings=None):
        '''Supported API. 
        Save settings to disk.  Only needed for EXPLICIT_SAVE settings.
        AUTO_SAVE settings are immediately saved to disk when changed.

        If 'setting' or 'settings' is specified, save only those settings (don't
        change saved value of any other setting. Otherwise, save all settings.'''
        if setting is not None:
            settings_to_save = [setting]
        elif settings is not None:
            settings_to_save = settings
        else:
            settings_to_save = self.__class__.EXPLICIT_SAVE.keys()
        for name in settings_to_save:
            ConfigFile.__setattr__(self, name, self._cur_settings[name], call_save=False)
        ConfigFile.save(self)

    def saved_value(self, name):
        return ConfigFile.__getattr__(self, name)

    def update(self, *args, **kw):
        raise ValueError("update() disabled for Settings class")
