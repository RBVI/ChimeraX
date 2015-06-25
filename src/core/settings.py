# vi: set expandtab shiftwidth=4 softtabstop=4:

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
