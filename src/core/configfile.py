# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
configfile: application preferences support
===========================================

This module provides support for accessing persistent
configuratation information, *a.k.a., preferences.
The information is stored in a file in a human-readable form,
*i.e.*, text, but is not necessarily editable.

The configuration information is considered to be an API
and has a semantic version associated with it.

Configuration information is kept in properties.
And those properties have names and values.

Each tool has its own preferences.
The *MAJOR* part of the semantic version is embedded in its filename.
For the chimera core,
that version does not change during the life of the release,
so patches may only introduce additional information.
Tools are allowed to change and might or might not
implement backwards compatibility.

Accessing Configuration Information
-----------------------------------

Access Tool Configuration::

    prefs = tool.get_preferences()
    if prefs.PROPERTY == 12:  # access a value
        pass
    prefs.PROPERTY = value    # set a value

Access Chimera Core Configuration::

    from chimera.core import preferences
    prefs = preferences.get()
    # (ibid)

Declaring the Configuration API
-------------------------------

The fact that there are configuration files is hidden by an object
that implements the tool's configuration API.

Most tools will only have one section.  So the :py:class:`ConfigFile`
and :py:class:`Section` subclasses (next example) can be combined into one::

    _config = None

    BlastMatrixArg = cli.EnumOf((
        'BLOSUM45', 'BLOSUM62', 'BLOSUM80', 'PAM30', 'PAM70'
    ))

    class _BPPreferences(configfile.ConfigFile):

        PROPERTY_INFO = {
            'e_exp': configfile.Value(3, cli.PositiveIntArg, str),
            'matrix': configfile.Value('BLOSUM62', BlastMatrixArg, str)
            'passes': 1,
        }

        def __init__(self, session):
            ConfigFile.__init__(self, session, "Blast Protein")

    def get_preferences():
        global _prefs
        if _prefs is None:
            _prefs = _BPPreferences()
        return _prefs

    # reusing Annotations for command line arguments
    @cli.register("blast",
        cli.CmdDesc(
            keyword=[('evalue', cli.PostiveIntArg),
                     ('matrix', BlastMatrixArg),]
    ))
    def blast(session, e_exp=None, matrix=None):
        prefs = get_preferences()
        if e_exp is None:
            e_exp = prefs.e_exp
        if matrix is None:
            matrix = prefs.matrix
        # process arguments

Property values can either be a Python literal, which is the default value,
or a :py:class:`Value` has three items associated with it:

    1. A default value.
    2. A function that can parse the value or a
       cli :py:class:`~chimera.core.cli.Annotation` that can parse the value.
       This allows for error checking in the case where a user hand edits
       the configuration.
    3. A function to convert the value to a string.

If the tool configuration API changes,
then the tool can subclass :py:class:`Preferences` with custom code.

Adding a Property
-----------------

If an additional property is needed, just add it the
:py:attr:`~Section.PROPERTY_INFO` class attribute,
and document it.
The minor part of the version number should be increased before
the tool is released again.
That way other tools can use the tool's version number
to tell if the property is available or not.


Renaming a Property
-------------------

Since configuration is an API, properties can not be removed without
changing the major version number.  To prepare for that change, document
that the old name is deprecated and that the new name should be used instead.
Then add a Python property to the section class that forwards the
changes to the old property name.  For example, to rename ``e_exp``, in
the previous example, to ``e_value``, extend the _Params class with::

    class _Params(configfile.Section):

        PROPERTY_INFO = {
            'e_exp': ( cli.PositiveIntArg, str, 3 ),
            'matrix': ( BlastMatrixArg, str, 'BLOSUM62' )
        }

        @property
        def e_value(self):
            return 10 ** -self.e_exp

        @e_value.setter
        def e_value(self, value):
            import math
            self.e_exp = -round(math.log10(value))

Later, when the major version changes,
the existing :py:class:`ConfigFile` subclass
would be renamed with a version suffix
with the version number hardcoded,
and a new subclass would be generated with the ``e_exp``
replaced with ``e_value``.
Then in the new :py:class:`ConfigFile` subclass, after it is initialized,
it would check if its data was on disk or not, and if not, try opening up
previous configuration versions and migrate the settings.
The ``migrate_from`` methods,
:py:meth:`ConfigFile.migrate_from` and :py:meth:`Section.migrate_from`,
may be replaced or can be made more explicit.
See the next section for an example.

Changing the API - Migrating to a New Configuration
---------------------------------------------------

Migrating Example::

    class _BPPreferences(configfile.ConfigFile):

        PROPERTY_INFO = {
            'e_exp': ( cli.PositiveIntArg, str, 3 ),
            'matrix': ( BlastMatrixArg, str, 'BLOSUM62' )
        }

        # additional properties removed

        def __init__(self, session):
            ConfigFile.__init__(self, session, "Blast Protein")


    class _BPPreferences(configfile.ConfigFile):

        PROPERTY_INFO = {
            'e_value': ( float, str, 1e-3 ),
            'matrix': ( BlastMatrixArg, str, 'BLOSUM62' )
        }

        # e_exp is gone

        def migrate_from(self, old, version):
            configfile.Section.migrate_from(self, old, version)
            self.e_value = 10 ** -old._exp

        def __init__(self, session):
            # add version
            ConfigFile.__init__(self, session, "Blast Protein", "2")
            if not self.on_disk():
                old = _BPPreferences()
                self.migrate_from(old, "1")


Migrating a Property Without Changing the Version
-------------------------------------------------

This is similar to renaming a property, with a more sophisticated getter
function::

    class _Params(configfile.ConfigFile):

        PROPERTY_INFO = {
            'e_value': 1e-3,
            'matrix': configfile.Value('BLOSUM62', BlastMatrixArg, str)
        }

        @property
        def e_value(self):
            def migrate_e_exp(value):
                # conversion function
                return 10 ** -value
            return self.migrate_value('e_value', 'e_exp', cli.PositiveIntArg,
                                       migrate_e_exp)

The :py:meth:`~ConfigFile.migrate_value` function looks for the new value,
but if it isn't present,
then it looked for the old value and migrates it.
If the old value isn't present, then the new default value is used.
"""
from .cli import UserError

only_use_defaults = False   # if True, do not read nor write configuration data


class ConfigFile:
    """In-memory handle to persistent configuration information.

    Parameters
    ----------
    session : :py:class:`~chimera.core.session.Session`
        (for ``app_dirs`` and ``logger``)
    tool_name : the name of the tool
    version : configuration file version, optional
        Only the major version part of the version is used.

    Attributes
    ----------
    PROPERTY_INFO : dict of property_name: value
        ``property_name`` must be a legal Python identifier.
        ``value`` is a Python literal or an :py:class:`Item`.
    filename : the name of the file used to store the preferences
    """

    PROPERTY_INFO = {}

    def __init__(self, session, tool_name, version="1"):
        import configparser
        from distlib.version import Version, NormalizedVersion
        import os
        if isinstance(version, Version):
            epoch, ver, *_ = version.parse(str(version))
        else:
            epoch, ver, *_ = NormalizedVersion("1").parse(version)
        major_version = ver[0]
        self._session = session
        self._on_disk = False
        # convert all property information to Values
        assert('save' not in self.PROPERTY_INFO)
        for name, value in self.PROPERTY_INFO.items():
            if not isinstance(value, Value):
                self.PROPERTY_INFO[name] = Value(value)
        if only_use_defaults:
            return
        self._filename = os.path.join(
            session.app_dirs.user_config_dir,
            '%s-%s' % (tool_name, major_version) if version else tool_name)
        self._config = configparser.ConfigParser(
            comment_prefixes=(),
            interpolation=None
        )
        if os.path.exists(self._filename):
            self._on_disk = True
            self._config.read(self._filename)
            # check that all values on disk are valid
            for name in self.PROPERTY_INFO:
                getattr(self, name)

    def on_disk(self):
        """Return True the configuration information was stored on disk.

        This information is useful when deciding whether or not to migrate
        settings from a previous configuration version."""
        return self._on_disk

    @property
    def filename(self):
        return self._filename

    def save(self):
        """Save configuration information of all sections to disk.

        Don't store property values that match default value.
        """
        if only_use_defaults:
            raise UserError("Custom configuration is disabled")
        from .safesave import SaveTextFile
        with SaveTextFile(self._filename) as f:
            self._config.write(f)

    def reset(self):
        """Revert all properties to their default state"""
        self._config['DEFAULT'].clear()
        self.save()

    def migrate_from(self, old, version):
        """Migrate identical settings from old configuration.

        Parameters
        ----------
        old : instance Section-subclass for old section
        version : old version "number"
        """
        for name in self.PROPERTY_INFO:
            if hasattr(old._config['DEFAULT'], name):
                setattr(self, name, old._config['DEFAULT'][name])
        self.save()

    def __getattr__(self, name):
        if name not in self.PROPERTY_INFO:
            raise AttributeError(name)
        if name not in self._config['DEFAULT'] or only_use_defaults:
            value = self.PROPERTY_INFO[name].default
        else:
            try:
                value = self.PROPERTY_INFO[name].convert_from_string(
                    self._config._session, self._section['DEFAULT'][name])
            except ValueError as e:
                self._config._session.logger.warning(
                    "Invalid %s.%s value, using default: %s" %
                    (self._name, name, e))
                value = self.PROPERTY_INFO[name].default
        return value

    def __setattr__(self, name, value):
        if name not in self.PROPERTY_INFO:
            if name[0] == '_':
                return object.__setattr__(self, name, value)
            raise AttributeError("Unknown property name: %s" % name)
        if only_use_defaults:
            raise UserError("Custom configuration is disabled")
        if value == self.PROPERTY_INFO[name].default:
            if name not in self._config['DEFAULT']:
                # name is not in ini file and is default, so don't save it
                return
            del self._config['DEFAULT'][name]
        else:
            try:
                str_value = self.PROPERTY_INFO[name].convert_to_string(
                    self._config._session, value)
            except ValueError:
                raise UserError("Illegal %s.%s value, unchanged" %
                                (self._name, name))
            self._config['DEFAULT'][name] = str_value
        self.save()

    def migrate_value(self, name, old_name, old_from_str, convert):
        """For migrating property from ".old_name" to ".name".

        First look for the new value, but if it isn't present,
        then look for the old value and migrate it.
        If the old value isn't present, then the new default value is used.

        Parameters
        ----------
        name : current name of property
        old_name : previous name of property
        old_from_str : function to parse previous version of property
        convert : function to old value to new value
        """
        if (name in self._config['DEFAULT'] or
                old_name not in self._config['DEFAULT']):
            # already migrated and saved, or not present
            return self.__getattr__(self, name)
        else:
            # old value present, get it
            session = self._session
            try:
                if hasattr(old_from_str, 'parse'):
                    old_value, consumed, rest = old_from_str.parse(
                        self._config['DEFAULT'][old_name], session)
                else:
                    old_value = old_from_str(self._config['DEFAULT'][old_name])
            except ValueError:
                # revert to current default
                session.logger.warning(
                    "Unable to migrate old %s value, using current default" %
                    old_name)
                return self.__getattr__(self, name)
            # and migrate
            value = convert(old_value)
            setattr(self, name, value)
        return value


class Value:
    """Placeholder for default value and conversion functions


    Parameters
    ----------
    default : is the value when the property has not been set.
    from_str : function or Annotation, optional
        can be either a function that takes a string
        and returns a value of the right type, or a cli
        :py:class:`~chimera.core.cli.Annotation`.
        Defaults to py:func:`ast.literal_eval`.
    to_str : function returning a string, optional
        Defaults to :py:func:`repr`.

    Attributes
    ----------
    section : :py:class:`Section` instance

    """

    def __init__(self, default, from_str=None, to_str=None):
        self.default = default
        if from_str is None:
            import ast
            self.from_str = ast.literal_eval
        else:
            self.from_str = from_str
        if to_str is None:
            self.to_str = repr
        else:
            self.to_str = to_str

    def convert_from_string(self, session, str_value):
        if hasattr(self.from_str, 'parse'):
            value, consumed, rest = self.from_str.parse(str_value, session)
            return value
        else:
            return self.from_str(str_value)

    def convert_to_string(self, session, value):
        str_value = self.to_str(value)
        # confirm that value can be restored from disk,
        # by converting to a string and back
        new_value = self.convert_from_string(session, str_value)
        if new_value != value:
            raise ValueError('value changed while saving it')
        return str_value

    def default(self):
        return self.default


if __name__ == '__main__':
    # simple test
    from . import cli
    import os

    class _ToolPreferences(ConfigFile):

        PROPERTY_INFO = {
            'log_level': Value(1, cli.Bounded(cli.IntArg, 1, 9)),
        }

        def __init__(self, session):
            ConfigFile.__init__(self, session, 'test', '1.2.0')
            print('preferences file:', self._filename, flush=True)

    prefs = _ToolPreferences(Chimera2_session)  # noqa

    assert(not prefs.on_disk())

    prefs.log.log_level = 4
    prefs.log.save()
    # confirm log level is in file
    assert(0 == os.system("grep -q log_level '%s'" % prefs._filename))

    prefs.log.log_level = 1
    prefs.log.save()
    # confirm log level is not in file
    assert(0 != os.system("grep -q log_level '%s'" % prefs._filename))

    # finished with configuration file
    os.remove(prefs._filename)
