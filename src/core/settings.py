# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
settings: save/access tool interface values
===========================================

Quick Start
-----------

This module provides a convenient way for tools to remember preferred settings.
To do so you create subclass of this module's Settings class and populate it
with setting names and default values.  There are two varieties of settings,
ones that automatically saved to disk when they're set, and ones that only
save to disk when the Settings.save() method is called.  The former might be
used for minor conveniences in the interface, like remembering which file
format the user last used, whereas the latter would be used if a tool presents
an explicit Save button for preserving configuration information.

The settings are presented as attributes of the Settings instance, so therefore
the setting name has to be legal as an attribute, *i.e.* no spaces, just
alphanumeric characters plus underscore.  Also, leading underscores are not
allowed.

**Example**

*setup*
::

    from chimera.core.settings import Settings
    from chimera.core.configfile import Value
    from chimera.core.cli import EnumOf
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
::
    penalty += gap_len * my_settings.gap_penalty

*saving*
    my_settings.matrix = 'BLOSUM80' # auto-saved
    ...
    my_settings.font_size = 12
    my_settings.save() # saves all settings

As mentioned above, the AUTO_SAVE/EXPLICIT_SAVE keys have to be legal to use
as attribute names.  The values can be just plain Python values, in which case
their :py:func:`repr` will be saved to disk, or more complex entities,
like enumerations or class instances.  In the latter case you have to use the
:py:class:`chimera.core.configfile.Value` class to tell the settings mechanism
how to save and restore the value and perhaps check that the value is within
its legal range of values.  The three arguments to Value() are:

1. The default value.
2. An object that can convert text into the needed value by following the
   :py:class:`chimera.core.cli.Annotation` abstract class protocol.
3. A function that converts a value to text.

There are many Annotation subclasses in the :py:mod:`chimera.core.cli`
module that can be used as the second argument to Value() and that also
perform range checking.

Details
-------
migration; versions
"""


from .configfile import ConfigFile, only_use_defaults

class Settings(ConfigFile):
    AUTO_SAVE = EXPLICIT_SAVE = {}

    def __init__(self, session, tool_name, version="1"):
        object.__setattr__(self, '_settings_initialized', False)
        self.__class__.PROPERTY_INFO = {}
        self.__class__.PROPERTY_INFO.update(self.__class__.AUTO_SAVE)
        self.__class__.PROPERTY_INFO.update(self.__class__.EXPLICIT_SAVE)
        self._cur_settings = {}
        ConfigFile.__init__(self, session, tool_name, version=version)
        for attr_name in self.__class__.PROPERTY_INFO.keys():
            self._cur_settings[attr_name] = getattr(self, attr_name)
        object.__setattr__(self, '_settings_initialized', True)

    def __getattr__(self, name):
        if only_use_defaults or not self._settings_initialized:
            return ConfigFile.__getattr__(self, name)
        try:
            return self._cur_settings[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if (self._settings_initialized and name[0] != '_'
        and name in self._cur_settings):
            self._cur_settings[name] = value
            if name in self.__class__.AUTO_SAVE:
                ConfigFile.__setattr__(self, name, value)
        else:
            ConfigFile.__setattr__(self, name, value)

    def save(self):
        for name in self.__class__.EXPLICIT_SAVE.keys():
            ConfigFile.__setattr__(self, name, self._cur_settings[name])
        ConfigFile.save()

    def update(self, *args, **kw):
        raise ValueError("update() disabled for Settings class")
