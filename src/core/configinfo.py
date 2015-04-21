# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
config: application configuration support
=========================================

The module provides support for accessing persistent
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

Tool Configuration Example::

    config = tool.get_config()
    if config.SECTION.PROPERTY == 12:  # access a value
        pass
    config.SECTION.PROPERTY = value    # set a value
    config.SECTION.save()              # save a section
    config.save()                      # save all sections

Chimera Core Configuration Example::

    from chimera.core import get_config
    config = get_config()
    # (ibid)

Declaring the Configuration API
-------------------------------

The fact that there are configuration files is hidden by an object
that implements the tool's configuration API.

Configuration Example::

    _config = None

    BlastMatrix = cli.EnumOf((
        'BLOSUM45', 'BLOSUM62', 'BLOSUM80', 'PAM30', 'PAM70'
    ))


    class _Params(configinfo.Section):

        PROPERTY_INFO = {
            'e_exp': { cli.PositiveIntArg, str, 3 },
            'matrix': { BlastMatrix, str, 'BLOSUM62' }
        }


    class _BPConfigInfo(configinfo.ConfigInfo):

        def __init__(self, session):
            ConfigInfo.__init__(self, "Blast Protein")
            self.params = _Params(self, 'params')


    def get_config():
        global _config
        if _config is None:
            _config = _BPConfigInfo()
        return _config

    config = get_config()

    config.log.log_level = 4

Note that each property has three items associated with it:

    1. The cli Annotation that can parse the value.  This allows for
       error checking in the case where a user hand edits the configuration.
    2. A function to convert the value to a string.
    3. A default value.

The Section class is for convenience.  If the tool configuration API changes,
then the tool can substitute a class with custom code.

Adding a Property
-----------------

If an additional property is needed, just add it the section's PROPERTY_INFO,
and document it.
The minor part of the version number should be increased before
the tool is released again.  Other tools can use the tool's version number
to tell if the property is available or not.

Single Section Configuration
----------------------------

Most tools will only have one section.  So the ConfigInfo and Section
subclasses can be combined into one::


    class _BPConfigInfo(configinfo.SingleSectionConfigInfo):

        PROPERTY_INFO = {
            'e_exp': { cli.PositiveIntArg, str, 3 },
            'matrix': { BlastMatrix, str, 'BLOSUM62' }
        }

        def __init__(self, session):
            SingleSectionConfigInfo.__init__(self, "Blast Protein", "params")

    config.matrix or config.params.matrix


Renaming a Property
-------------------

Since configuration is an API, properties can not be removed without
changing the major version number.  To prepare for that change, document
that the old name is deprecated and the new name should be used instead.
Then add a Python property to the section class that forwards the
changes to the old property name.  For example, to rename 'e_exp', in
the previous example, to 'e_value', extend the LogSection class with::

    class _Params(configinfo.Section):

        PROPERTY_INFO = {
            'e_exp': { cli.PositiveIntArg, str, 3 },
            'matrix': { BlastMatrix, str, 'BLOSUM62' }
        }

        @getter
        def e_value(self):
            return 10 ** -self.e_exp

        @setter
        def e_value(self, value):
            import math
            self.e_exp = -round(math.log10(value))

Later, when the major version changes,
the existing ConfigInfo subclass would be renamed with a version suffix
with the version number hardcoded,
and a new subclass would be generated with the 'e_exp' replaced with
'e_value'.  Then in the new ConfigInfo subclass, after it is initialized,
it would check if its data was on disk or not, and if not, try opening up
previous configuration versions and migrate the settings.  The 'migrate_from'
methods may be replaced or it can be made more explicit.

Changing the API - Migrating to a New Configuration
---------------------------------------------------

class _Params_V1(configinfo.Section):

    PROPERTY_INFO = {
        'e_exp': { cli.PositiveIntArg, str, 3 },
        'matrix': { BlastMatrix, str, 'BLOSUM62' }
    }

class _BPConfigInfo_V1(configinfo.ConfigInfo):

    def __init__(self, session):
        ConfigInfo.__init__(self, "Blast Protein")
        self.params = _Params_V1(self, 'params')


class _Params(configinfo.Section):

    PROPERTY_INFO = {
        'e_value': { float, str, 1e-3 },
        'matrix': { BlastMatrix, str, 'BLOSUM62' }
    }

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

This is similar to renaming a property, with the getter function much
more sophisticated::

    class _Params(configinfo.Section):

        PROPERTY_INFO = {
            'e_value': { float, str, 3 },
            'matrix': { BlastMatrix, str, 'BLOSUM62' }
        }

        @getter
        def e_value(self):
            def cvt_e_exp(value):
                # conversion function
                return 10 ** -value
            return self.migrate_value('e_value', 'e_exp', cli.PositiveIntArg,
                                       cvt_e_exp)
"""
from .cli import UserError
from . import triggerset
from collections import OrderedDict

only_use_defaults = False   # if True, do not read nor write configuration data
triggers = triggerset.TriggerSet()


def _quote(s):
    """Return representation that will unquote"""
    from urllib.parse import quote
    return quote(s)


def _unquote(s):
    """Return original representation"""
    from urllib.parse import unquote
    return unquote(s)


class ConfigInfo:
    """In-memory handle to persistent configuration information

    Parameters
    ----------
    session : Chimera2 session (for app_dirs and logger)
    tool_name : the name of the tool
    version: configuration file version, optional

    Only the major version part of the version is used.

    Creates trigger that is activated when a property value is set.
    The trigger data is (section, property, value).
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
        """Return True the configuration information was stored on disk

        Only migrate configuration information from on disk data"""
        return self._on_disk

    def trigger_name(self):
        """Return trigger name to monitor for value setting."""
        return self._trigger_name

    def save(self, _all=True):
        """Save configuration information of all sections to disk"""
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
        """Migrate identical settings from old configuration"""
        for name, section in self._sections.items():
            if not old._config.has_section(name):
                continue
            section.migrate_from(getattr(old, name))
        self.save()

    def _add_section(self, name, section):
        self._sections[name] = section
        section._validate()


class Section:
    """Configuration Sections

    Each section only flushes its changes when its save method is called.

    Attributes
    ----------
    PROPERTY_INFO : { name: (from_str, to_str, default_value) }

    The from_str item can be either a function that takes a string and
    returns a value of the right type, or cli Annotation.  The to_str
    is a function that takes a value and returns a string.  The the
    default_value is the default value when the property has not been
    set.
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

        Paramters
        ---------
        old : instance Section-subclass for old section
        version : old version "number"
        """
        for name in self.PROPERTY_INFO:
            if hasattr(old._section, name):
                setattr(self, name, getattr(old, name))

    def migrate_value(self, name, old_name, old_from_str, convert):
        """Migrate value within a section

        name : current name of property
        old_name : previous name of property
        old_from_str : function to parse prevous version of property
        convert : function to old value to new value

        For migrating API from ".old_name" to ".name".
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

    Restriction: can not use the section_name as a section property name.
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
