# Taken from qtpy, who apparently took it from pyqtgraph
# MIT licensed
# Updated 9 Feb 2023, to qtpy commit https://github.com/spyder-ide/qtpy/commit/f09c0068def9d2cec646bda4f351eff04ea5df07
def promote_enums(module):
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
