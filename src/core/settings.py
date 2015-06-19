# vi: set expandtab shiftwidth=4 softtabstop=4:

from .configfile import ConfigFile, only_use_defaults

class Settings(ConfigFile):
    AUTO_SAVE = EXPLICIT_SAVE = {}

    def __init__(self, session, tool_name, version="1"):
        self.__class__.PROPERTY_INFO {}
        self.__class__.PROPERTY_INFO.update(self.__class__.AUTO_SAVE)
        self.__class__.PROPERTY_INFO.update(self.__class__.EXPLICIT_SAVE)
        ConfigFile.__init__(self, session, tool_name, version=version)
        self.__cur_values = {}
        for attr_name in self.__class__.PROPERTY_INFO.keys():
            self.__cur_values[attr_name] = getattr(self, attr_name)

    def __getattr__(self, name):
        if only_use_defaults:
            return ConfigFile.__getattr__(self, name)
        try:
            return self.__cur_values[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__cur_values:
            self.__cur_values[name] = value
            if name in self.__class__.AUTO_SAVE:
                ConfigFile.__setattr__(self, name, value)
        else:
            ConfigFile.__setattr__(self, name, value)

    def save(self):
        for name in self.__class__.EXPLICIT_SAVE.keys():
            ConfigFile.__setattr__(self, name, self.__cur_values[name])
        ConfigFile.save()

    def update(self, *args, **kw):
        raise ValueError("update() disabled for Settings class")
