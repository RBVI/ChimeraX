# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
configinfo: application configuration support
=============================================

This module provides support for accessing persistent
configuratation information.  The information is stored in
configuration files that are human-readable, *i.e.*, text, but
not necessarily editable.

The configuration information is considered to be an API
and has a semantic version associated with it.

Configuration information is separated into sections, a.k.a., categories,
with properties and values.
This terminology comes from the description of INI files,
http://en.wikipedia.org/wiki/INI_file.
On disk, sections are effectively dictionaries where the values are all
strings.

Each tool has its own configuration information.
The MAJOR part of the semantic version is embedded in its filename.
For the chimera core,
that version does not change during the life of the release,
so patches may only introduce additional information.
Tools are allowed to change and might or might not
implement backwards compatibility.

Accessing Configuration Information
-----------------------------------

Access Tool Configuration::

    config = tool.get_config()
    if config.SECTION.PROPERTY == 12:  # access a value
        pass
    config.SECTION.PROPERTY = value    # set a value
    config.SECTION.save()              # save a section
    config.save()                      # save all sections

Access Chimera Core Configuration::

    from chimera.core import get_config
    config = get_config()
    # (ibid)

Declaring the Configuration API
-------------------------------

The fact that there are configuration files is hidden by an object
that implements the tool's configuration API.

Most tools will only have one section.  So the :py:class:`ConfigInfo`
and :py:class:`Section` subclasses (next example) can be combined into one::

    _config = None

    BlastMatrixArg = cli.EnumOf((
        'BLOSUM45', 'BLOSUM62', 'BLOSUM80', 'PAM30', 'PAM70'
    ))

    class _BPConfigInfo(configinfo.SingleSectionConfigInfo):

        PROPERTY_INFO = {
            'e_exp': ( cli.PositiveIntArg, str, 3 ),
            'matrix': ( BlastMatrixArg, str, 'BLOSUM62' )
        }

        def __init__(self, session):
            SingleSectionConfigInfo.__init__(self, "Blast Protein", "params")

    def get_config():
        global _config
        if _config is None:
            _config = _BPConfigInfo()
        return _config

    # reusing Annotations for command line arguments
    @cli.register("blast",
        cli.CmdDesc(
            keyword=[('evalue', cli.PostiveIntArg),
                     ('matrix', BlastMatrixArg),]
    ))
    def blast(session, e_exp=None, matrix=None):
        c = get_config()
        if e_exp is None:
            e_exp = c.e_exp           # can use short form
        if matrix is None:
            matrix = c.params.matrix  # or long form
        # process arguments

Multi-section Configuration Example::

    _config = None

    BlastMatrixArg = cli.EnumOf((
        'BLOSUM45', 'BLOSUM62', 'BLOSUM80', 'PAM30', 'PAM70'
    ))

    class _Params(configinfo.Section):

        PROPERTY_INFO = {
            'e_exp': ( cli.PositiveIntArg, str, 3 ),
            'matrix': ( BlastMatrixArg, str, 'BLOSUM62' )
        }

    class _Hidden(configinfo.Section):

        PROPERTY_INFO = {
            'private': ( str, str, 'xyzzy' ),
        }

    class _BPConfigInfo(configinfo.ConfigInfo):

        def __init__(self, session):
            ConfigInfo.__init__(self, "Blast Protein")
            self.params = _Params(self, 'params')
            self.hidden = _Params(self, 'hidden')

    def get_config():
        global _config
        if _config is None:
            _config = _BPConfigInfo()
        return _config

    def blast(session, e_exp=None, matrix=None):
        c = get_config()
        if e_exp is None:
            e_exp = c.params.e_exp      # must use long form
        if matrix is None:
            matrix = c.params.matrix
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
the tool is released again.  That way other tools can use the tool's version number
to tell if the property is available or not.


Renaming a Property
-------------------

Since configuration is an API, properties can not be removed without
changing the major version number.  To prepare for that change, document
that the old name is deprecated and that the new name should be used instead.
Then add a Python property to the section class that forwards the
changes to the old property name.  For example, to rename ``e_exp``, in
the previous example, to ``e_value``, extend the LogSection class with::

    class _Params(configinfo.Section):

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
the existing :py:class:`ConfigInfo` subclass
would be renamed with a version suffix
with the version number hardcoded,
and a new subclass would be generated with the ``e_exp``
replaced with ``e_value``.
Then in the new :py:class:`ConfigInfo` subclass, after it is initialized,
it would check if its data was on disk or not, and if not, try opening up
previous configuration versions and migrate the settings.
The ``migrate_from`` methods,
:py:meth:`ConfigInfo.migrate_from` and :py:meth:`Section.migrate_from`,
may be replaced or can be made more explicit.
See the next section for an example.

Changing the API - Migrating to a New Configuration
---------------------------------------------------

Migrating Example::

    class _Params_V1(configinfo.Section):

        PROPERTY_INFO = {
            'e_exp': ( cli.PositiveIntArg, str, 3 ),
            'matrix': ( BlastMatrixArg, str, 'BLOSUM62' )
        }
        
        # additional properties removed

    class _BPConfigInfo_V1(configinfo.ConfigInfo):

        def __init__(self, session):
            ConfigInfo.__init__(self, "Blast Protein")
            self.params = _Params_V1(self, 'params')


    class _Params(configinfo.Section):

        PROPERTY_INFO = {
            'e_value': ( float, str, 1e-3 ),
            'matrix': ( BlastMatrixArg, str, 'BLOSUM62' )
        }

        # e_exp is gone

        def migrate_from(self, old, version):
            configinfo.Section.migrate_from(self, old, version)
            self.e_value = 10 ** -old._exp

    class _BPConfigInfo(configinfo.ConfigInfo):

        def __init__(self, session):
            ConfigInfo.__init__(self, "Blast Protein", "2")  # added version
            self.params = _Params(self, 'params')
            if not self.on_disk():
                old = _BPConfigInfo_V1()
                self.migrate_from(old, "1")


Migrating a Property Without Changing the Version
-------------------------------------------------

This is similar to renaming a property, with a more sophisticated getter
function::

    class _Params(configinfo.Section):

        PROPERTY_INFO = {
            'e_value': ( float, str, 1e-3 ),
            'matrix': ( BlastMatrixArg, str, 'BLOSUM62' )
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


class ConfigInfo:
    """In-memory handle to persistent configuration information.

    A trigger with :py:meth:`trigger_name` is created
    that is activated when a property value is set.
    The trigger data is (section name, property name, value).

    Parameters
    ----------
    session : :py:class:`~chimera.core.session.Session` (for ``app_dirs`` and ``logger``)
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


class Section:
    """A logical group of properties with a configuration file.

    Each section only flushes its changes when its :py:meth:`save` method
    is called.

    Parameters
    ----------
    config : :py:class:`ConfigInfo` instance
    section_name : str
        The name of the section.

    Attributes
    ----------
    PROPERTY_INFO : dict
        property_name: (from_str, to_str, default_value)
        ``property_name`` must be a legal Python identifier.
        ``from_str`` can be either a function that takes a string
        and returns a value of the right type, or a cli
        :py:class:`~chimera.core.cli.Annotation`.
        ``to_str`` is a function that takes a value and returns a string.
        The ``default_value`` is the value when the property has not been set.

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
        if name not in self._section:
            value = self.PROPERTY_INFO[name][2]
        elif not only_use_defaults:
            from_str = self.PROPERTY_INFO[name][0]
            if from_str is None:
                value = self._section[name]
            else:
                try:
                    if hasattr(from_str, 'parse'):
                        value = from_str.parse(self._section[name],
                                               self._config._session)
                    else:
                        value = from_str(self._section[name])
                except ValueError as e:
                    self._config._session.logger.warning(
                        "Invalid %s.%s value, using default: %s" %
                        (self._name, name, e))
                    value = self.PROPERTY_INFO[name][2]
        self._cache[name] = value
        return value

    def __setattr__(self, name, value):
        if name not in self.PROPERTY_INFO:
            return object.__setattr__(self, name, value)
        if only_use_defaults:
            raise UserError("Custom configuration is disabled")
        try:
            # confirm that value can be restored from disk,
            # by converting to a string and back
            to_str = self.PROPERTY_INFO[name][1]
            str_value = to_str(value)  # noqa
            from_str = self.PROPERTY_INFO[name][0]
            if from_str is not None:
                if hasattr(from_str, 'parse'):
                    new_value = from_str.parse(self._section[name],
                                               self._config._session)
                else:
                    new_value = from_str(self._section[name])
            assert(value == new_value)
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
            default = self.PROPERTY_INFO[name][2]
            if value == default:
                if name in self._section:
                    del self._section[name]
                continue
            to_str = self.PROPERTY_INFO[name][1]
            if to_str is not None:
                value = to_str(value)
            elif not isinstance(value, str):
                value = str(value)
            self._section[name] = value
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
        old_from_str : function to parse prevous version of property
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


class SingleSectionConfigInfo(ConfigInfo, Section):
    """Simple case of ConfigInfo with one Section

    The parameters are the same as ConfigInfo with the addtion of the
    Section's section_name.

    Restriction: can not use the section_name as one of the section's
    property names.
    """

    def __init__(self, tool_name, section_name, version="1"):
        assert(section_name not in self.PROPERTY_INFO)
        ConfigInfo.__init__(self, tool_name, version)
        Section.__init__(self, self, section_name)

    def save(self, _all=True):
        if _all:
            Section.save(self)
        else:
            ConfigInfo.save(_all=False)

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

    class _ToolConfigInfo(ConfigInfo):

        def __init__(self, session):
            ConfigInfo.__init__(self, session, 'test', '1.2.0')
            self.log = LogSection(self, 'log')
            print('config file:', self._filename, flush=True)

    config = _ToolConfigInfo(Chimera2_session)  # noqa

    assert(not config.on_disk())

    config.log.log_level = 4
    config.log.save()
    # confirm log level is in file
    assert(0 == os.system("grep -q log_level '%s'" % config._filename))

    config.log.log_level = 1
    config.log.save()
    # confirm log level is not in file
    assert(0 != os.system("grep -q log_level '%s'" % config._filename))

    # finished with configuration file
    os.remove(config._filename)
