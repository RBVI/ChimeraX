def promote_enums(module):
    """
    Search enums in the given module and allow unscoped access in PyQt6.

    Taken from:
    https://github.com/pyqtgraph/pyqtgraph/blob/pyqtgraph-0.12.1/pyqtgraph/Qt.py
    """
    from PyQt6.sip import wrappertype
    from enum import EnumMeta
    class_names = [name for name in dir(module) if name.startswith('Q')]
    for class_name in class_names:
        klass = getattr(module, class_name)
        if isinstance(klass, wrappertype):
            attrib_names = [name for name in dir(klass) if name[0].isupper()]
            for attrib_name in attrib_names:
                attrib = getattr(klass, attrib_name)
                if isinstance(attrib, EnumMeta):
                    for enum_obj in attrib:
                        setattr(klass, enum_obj.name, enum_obj)
