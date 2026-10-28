# Taken from qtpy, who apparently took it from pyqtgraph
# MIT licensed
# Updated 9 Feb 2023, to qtpy commit https://github.com/spyder-ide/qtpy/commit/f09c0068def9d2cec646bda4f351eff04ea5df07
def promote_enums_pyqt(module):
    """
    Search enums in the given module and allow unscoped access.

    Taken from:
    https://github.com/pyqtgraph/pyqtgraph/blob/pyqtgraph-0.12.1/pyqtgraph/Qt.py#L331-L377
    and adapted to also copy enum values aliased under different names.

    """
    from PyQt6.sip import wrappertype
    from enum import EnumMeta
    class_names = [name for name in dir(module) if name.startswith('Q')]
    for class_name in class_names:
        klass = getattr(module, class_name)
        if not isinstance(klass, wrappertype):
            continue
        attrib_names = [name for name in dir(klass) if name[0].isupper()]
        for attrib_name in attrib_names:
            attrib = getattr(klass, attrib_name)
            if not isinstance(attrib, EnumMeta):
                continue
            for name, value in attrib.__members__.items():
                setattr(klass, name, value)

def promote_enums_pyside(module):
    """
    Search enums in the given module and allow unscoped access. Similar to the above function.

    """
    from PySide6 import QtWidgets
    wrappertype = type(QtWidgets.QWidget)
    class_names = [name for name in dir(module) if name.startswith('Q')]
    for class_name in class_names:
        klass = getattr(module, class_name)
        if not isinstance(klass, wrappertype):
            continue
        _promote_pyside_class(klass)


def _promote_pyside_class(klass):
    from enum import EnumType
    for attrib_name in dir(klass):
        if not attrib_name[0].isupper():
            continue
        attrib = getattr(klass, attrib_name)
        if isinstance(attrib, EnumType):
            for name, value in attrib.__members__.items():
                setattr(klass, name, value)


def prepare_pyside_class(klass):
    # Native forgiveness applies to class lookup, but not instance lookup
    # (e.g. table.AdjustToContents). Preserve both forms for exported classes.
    if klass.__module__.startswith('PySide6.'):
        _promote_pyside_class(klass)
    return klass


def native_enum_shortcuts(module):
    """Use PySide's documented forgiveness mode for the standard Qt modules.

    Keep explicit promotion for older bindings, other modules, and customized
    enum modes.  Since 6.6 Python enums are mandatory; the default mode provides
    unscoped members on their enclosing classes without eagerly copying them.
    https://doc.qt.io/qtforpython-6/developer/enumfeatures_doc.html
    """
    import sys
    import PySide6
    return (PySide6.__version_info__ >= (6, 6)
            and getattr(sys, 'pyside6_option_python_enum', None) == 1
            and module.__name__ in (
                'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets'))
