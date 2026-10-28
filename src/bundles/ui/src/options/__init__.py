# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from .options import Option, BooleanOption, ColorOption, OptionalColorOption, EnumOption, \
    OptionalEnumOption, FloatOption, FontOption, HostPortOption, InputFileOption, InputFolderOption, \
    IntOption, OutputFileOption, OutputFolderOption, PasswordOption, ProtocolHostPortOption, RGBA8Option, \
    RGBAOption, OptionalRGBAOption, OptionalRGBAPairOption, OptionalRGBA8Option, OptionalRGBA8PairOption, \
    StringOption, StringIntOption, StringsOption, SymbolicEnumOption, OptionalSymbolicEnumOption, \
    make_optional, make_int_spinbox


# Individual options are used during startup; their containers are only needed
# when a preferences/tool panel is actually created.
_container_names = ('OptionsPanel', 'CategorizedOptionsPanel',
                    'SettingsPanel', 'CategorizedSettingsPanel')


def __getattr__(name):
    if name not in _container_names and name != 'containers':
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    module = import_module('.containers', __name__)
    value = module if name == 'containers' else getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_container_names) | {'containers'})


__all__ = [name for name in globals() if not name.startswith('_')]
__all__ += [*_container_names, 'containers']
