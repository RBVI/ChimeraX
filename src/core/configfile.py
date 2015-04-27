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

Configuration information is separated into sections with properties.
And those properties have names and values.
This terminology comes from the description of INI files,
http://en.wikipedia.org/wiki/INI_file.
Sections are effectively dictionaries
where the keys and values are all strings.

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
    if prefs.SECTION.PROPERTY == 12:  # access a value
        pass
    prefs.SECTION.PROPERTY = value    # set a value
    prefs.SECTION.save()              # save a section
    prefs.save()                      # save all sections

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

    class _BPPreferences(configfile.SingleSectionPreferences):

        PROPERTY_INFO = {
            'e_exp': configfile.Value(3, cli.PositiveIntArg, str),
            'matrix': configfile.Value('BLOSUM62', BlastMatrixArg, str)
            'passes': 1,
        }

        def __init__(self, session):
            SingleSectionPreferences.__init__(self, "Blast Protein", "params")

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
            e_exp = prefs.e_exp           # can use short form
        if matrix is None:
            matrix = prefs.params.matrix  # or long form
        # process arguments

Multi-section Configuration Example::

    _config = None

    BlastMatrixArg = cli.EnumOf((
        'BLOSUM45', 'BLOSUM62', 'BLOSUM80', 'PAM30', 'PAM70'
    ))

    class _Params(configfile.Section):

        PROPERTY_INFO = {
            'e_exp': configfile.Value(3, cli.PositiveIntArg, str),
            'matrix': configfile.Value('BLOSUM62', BlastMatrixArg, str)
        }

    class _Hidden(configfile.Section):

        PROPERTY_INFO = {
            'private': 'xyzzy',
        }

    class _BPPreferences(configfile.ConfigFile):

        def __init__(self, session):
            ConfigFile.__init__(self, "Blast Protein")
            self.params = _Params(self, 'params')
            self.hidden = _Params(self, 'hidden')

    def get_preferences():
        global _prefs
        if _prefs is None:
            _prefs = _BPPreferences()
        return _config

    def blast(session, e_exp=None, matrix=None):
        prefs = get_preferences()
        if e_exp is None:
            e_exp = prefs.params.e_exp      # must use long form
        if matrix is None:
            matrix = prefs.params.matrix
        # process arguments

Note that each property has three items associated with it:

    1. The cli :py:class:`~chimera.core.cli.Annotation`
       that can parse the value.
       This allows for error checking in the case where a user hand edits
       the configuration.
    2. A function to convert the value to a string.
    3. A default value.

If the tool configuration API changes,
then the tool can subclass :py:class:`Section` with custom code.

Adding a Property
-----------------

If an additional property is needed, just add it the section's
:py:attr:`~Section.PROPERTY_INFO`,
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

        @getter
        def e_value(self):
            return 10 ** -self.e_exp

        @setter
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

    class _Params_V1(configfile.Section):

        PROPERTY_INFO = {
            'e_exp': ( cli.PositiveIntArg, str, 3 ),
            'matrix': ( BlastMatrixArg, str, 'BLOSUM62' )
        }

        # additional properties removed

    class _BPPreferences(configfile.ConfigFile):

        def __init__(self, session):
            ConfigFile.__init__(self, "Blast Protein")
            self.params = _Params_V1(self, 'params')


    class _Params(configfile.Section):

        PROPERTY_INFO = {
            'e_value': ( float, str, 1e-3 ),
            'matrix': ( BlastMatrixArg, str, 'BLOSUM62' )
        }

        # e_exp is gone

        def migrate_from(self, old, version):
            configfile.Section.migrate_from(self, old, version)
            self.e_value = 10 ** -old._exp

    class _BPPreferences(configfile.ConfigFile):

        def __init__(self, session):
            ConfigFile.__init__(self, "Blast Protein", "2")  # added version
            self.params = _Params(self, 'params')
            if not self.on_disk():
                old = _BPPreferences()
                self.migrate_from(old, "1")


Migrating a Property Without Changing the Version
-------------------------------------------------

This is similar to renaming a property, with a more sophisticated getter
function::

    class _Params(configfile.Section):

        PROPERTY_INFO = {
            'e_value': 1e-3,
            'matrix': configfile.Value('BLOSUM62', BlastMatrixArg, str)
        }

        @getter
        def e_value(self):
            def migrate_e_exp(value):
                # conversion function
                return 10 ** -value
            return self.migrate_value('e_value', 'e_exp', cli.PositiveIntArg,
                                       migrate_e_exp)

The :py:meth:`~Section.migrate_value` function looks for the new value,
but if it isn't present,
then it looked for the old value and migrates it.
If the old value isn't present, then the new default value is used.
"""
from .cli import UserError
from . import triggerset
from collections import OrderedDict

only_use_defaults = False   # if True, do not read nor write configuration data
triggers = triggerset.TriggerSet()


def _quote(s):
    """Return representation that will unquote."""
    from urllib.parse import quote
    return quote(s)


def _unquote(s):
    """Return original representation."""
    from urllib.parse import unquote
    return unquote(s)


class ConfigFile:
    """In-memory handle to persistent configuration information.

    A trigger with :py:meth:`trigger_name` is created
    that is activated when a property value is set.
    The trigger data is (section name, property name, value).

    Parameters
    ----------
    session : :py:class:`~chimera.core.session.Session`
        (for ``app_dirs`` and ``logger``)
    tool_name : the name of the tool
    version : configuration file version, optional
        Only the major version part of the version is used.
    """

    def __init__(self, session, tool_name, version="1"):
        import configparser
        from distlib.version import Version, NormalizedVersion
        import os
        if isinstance(version, Version):
            epoch, ver, *_ = version.parse(str(version))
        else:
            epoch, ver, *_ = NormalizedVersion("1").parse(version)
        major_version = ver[0]
        self._trigger_name = "confinfo %s-%s" % (tool_name, major_version)
        triggers.add_trigger(self._trigger_name)
        self._sections = OrderedDict()
        self._session = session
        self._on_disk = False
        if only_use_defaults:
            return
        self._filename = os.path.join(
            session.app_dirs.user_config_dir,
            '%s-%s' % (tool_name, major_version) if version else tool_name)
        self._config = configparser.ConfigParser(
            comment_prefixes=(),
            default_section=None,
            interpolation=None
        )
        if os.path.exists(self._filename):
            self._on_disk = True
            self._config.read(self._filename)

    def on_disk(self):
        """Return True the configuration information was stored on disk.

        This information is useful when deciding whether or not to migrate
        settings from a previous configuration version."""
        return self._on_disk

    def trigger_name(self):
        """Return trigger name to use to monitor for value changes."""
        return self._trigger_name

    def save(self, _all=True):
        """Save configuration information of all sections to disk."""
        if only_use_defaults:
            raise UserError("Custom configuration is disabled")
        if _all:
            # update each section
            for section in self._sections.values():
                section.save(_skip_save=True)
        from .safesave import SaveTextFile
        with SaveTextFile(self._filename) as f:
            self._config.write(f)

    def migrate_from(self, old, version):
        """Migrate identical settings from old configuration."""
        for name, section in self._sections.items():
            if not old._config.has_section(name):
                continue
            section.migrate_from(getattr(old, name))
        self.save()

    def _add_section(self, name, section):
        self._sections[name] = section
        section._validate()


class Value:
    """Placeholder for default value and conversion functions


    Parameters
    ----------
    default : is the value when the property has not been set.
    from_str : function or Annotation, optional
        can be either a function that takes a string
        and returns a value of the right type, or a cli
        :py:class:`~chimera.core.cli.Annotation`.
    to_str : function returning a string, optional

    Attributes
    ----------
    section : :py:class:`Section` instance

    """

    def __init__(self, *args):
        if len(args) == 1:
            import ast
            self.from_str = ast.literal_eval
            self.to_str = repr
            self.default = args[0]
        elif len(args) == 3:
            self.default = args[0]
            self.from_str = args[1]
            self.to_str = args[2]
        else:
            raise ValueError()

    def convert_from_string(self, session, str_value):
        if hasattr(self.from_str, 'parse'):
            return self.from_str.parse(str_value, session)
        else:
            return self.from_str(str_value)

    def convert_to_string(self, session, value):
        str_value = self.to_str(value)
        # confirm that value can be restored from disk,
        # by converting to a string and back
        new_value = self._convert_from_string(session, str_value)
        if new_value != value:
            raise ValueError('value changed while saving it')
        return str_value

    def default(self):
        return self.default


class Section:
    """A logical group of properties with a configuration file.

    Each section only flushes its changes when its :py:meth:`save` method
    is called.

    Parameters
    ----------
    config : :py:class:`ConfigFile` instance
    section_name : str
        The name of the section.

    Attributes
    ----------
    PROPERTY_INFO : dict of property_name: value
        ``property_name`` must be a legal Python identifier.
        ``value`` is a Python literal or an :py:class:`Item`.

    Notes
    -----
    TODO: add documentation string to PROPERTY_INFO
    """

    PROPERTY_INFO = {}

    def __init__(self, config, section_name):
        assert('save' not in self.PROPERTY_INFO)
        self._config = config
        self._name = section_name
        self._cache = {}
        # convert all property information to Values
        for name, value in self.PROPERTY_INFO.items():
            if not isinstance(value, Value):
                self.PROPERTY_INFO[name] = Value(value)
        if only_use_defaults:
            return
        if not self._config._config.has_section(section_name):
            self._config._config.add_section(section_name)
        self._section = self._config._config[section_name]
        config._add_section(section_name, self)

    def __getattr__(self, name):
        if name in self._cache:
            return self._cache[name]
        if name not in self.PROPERTY_INFO:
            raise AttributeError(name)
        if name not in self._section or only_use_defaults:
            value = self.PROPERTY_INFO[name].default()
        else:
            try:
                value = self.PROPERTY_INFO[name].convert_from_string(
                    self._section[name], self._config._session)
            except ValueError as e:
                self._config._session.logger.warning(
                    "Invalid %s.%s value, using default: %s" %
                    (self._name, name, e))
                value = self.PROPERTY_INFO[name].default()
        self._cache[name] = value
        return value

    def __setattr__(self, name, value):
        if name not in self.PROPERTY_INFO:
            return object.__setattr__(self, name, value)
        if only_use_defaults:
            raise UserError("Custom configuration is disabled")
        try:
            self.PROPERTY_INFO[name].convert_to_string(
                value, self._config._session)
        except ValueError:
            raise UserError("Illegal %s.%s value, unchanged" %
                            (self._name, name))
        self._cache[name] = value
        triggers.activate_trigger(self._config._trigger_name,
                                  (self._name, name, value))

    def save(self, _skip_save=False):
        """Store section contents

        Don't store property values that match default value.
        """
        for name in self._cache:
            value = self._cache[name]
            default = self.PROPERTY_INFO[name].default()
            if value == default:
                if name in self._section:
                    del self._section[name]
                continue
            str_value = self.PROPERTY_INFO[name].convert_to_string(
                value, self._config._session)
            self._section[name] = str_value
        if _skip_save:
            return
        self._config.save(_all=False)

    def migrate_from(self, old, version):
        """Migrate identical settings from old section to current section

        Parameters
        ----------
        old : instance Section-subclass for old section
        version : old version "number"
        """
        for name in self.PROPERTY_INFO:
            if hasattr(old._section, name):
                setattr(self, name, getattr(old, name))

    def migrate_value(self, name, old_name, old_from_str, convert):
        """Migrate value within a section

        For migrating property API from ".old_name" to ".name".

        Parameters
        ----------
        name : current name of property
        old_name : previous name of property
        old_from_str : function to parse previous version of property
        convert : function to old value to new value
        """
        if name in self._cache:
            # already figured out
            return self._cache[name]
        if name in self._section or old_name not in self._section:
            # already migrated and saved, or not present
            return self.__getattr__(self, name)
        else:
            # old value present, get it
            session = self._config._session
            try:
                if hasattr(old_from_str, 'parse'):
                    old_value = old_from_str.parse(self._section[old_name],
                                                   session)
                else:
                    old_value = old_from_str(self._section[old_name])
            except ValueError:
                # revert to current default
                session.logger.warning(
                    "Unable to migrate old %s value, using current default" %
                    old_name)
                return self.__getattr__(self, name)
            # and migrate
            value = convert(old_value)
        self._cache[name] = value
        return value

    def _validate(self):
        for name in self.PROPERTY_INFO:
            if name in self._section:
                getattr(self, name)  # pre-populate cache


class SingleSectionPreferences(ConfigFile, Section):
    """Simple case of preferences with one Section

    The parameters are the same as ConfigFile with the addition of the
    Section's section_name.

    Restriction: can not use the section_name as one of the section's
    property names.
    """

    def __init__(self, tool_name, section_name, version="1"):
        assert(section_name not in self.PROPERTY_INFO)
        ConfigFile.__init__(self, tool_name, version)
        Section.__init__(self, self, section_name)

    def save(self, _all=True):
        if _all:
            Section.save(self)
        else:
            ConfigFile.save(_all=False)

    def migrate_from(self, old, version):
        raise NotImplemented

if __name__ == '__main__':
    # simple test
    from . import cli
    import os

    class LogSection(Section):

        PROPERTY_INFO = {
            'log_level': (cli.Bounded(cli.IntArg, 1, 9), str, 1),
        }

    class _ToolPreferences(ConfigFile):

        def __init__(self, session):
            ConfigFile.__init__(self, session, 'test', '1.2.0')
            self.log = LogSection(self, 'log')
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
