def export_module(namespace, module, *, prepare=None, prepare_class=None):
    """Forward public binding names while retaining PySide's lazy class loading.

    Per-name prepare callbacks can apply compatibility wrappers on first use.
    Other compatibility aliases should be set up before calling this function.
    Wildcard imports and dir() still expose the full set of public names.
    """
    exports = {name for name in dir(module) if not name.startswith('_')}

    def get_attribute(name):
        if name not in exports:
            raise AttributeError(
                f"module {namespace['__name__']!r} has no attribute {name!r}")
        value = getattr(module, name)
        if prepare_class is not None and isinstance(value, type):
            value = prepare_class(value)
        if prepare and name in prepare:
            value = prepare[name](value)
        namespace[name] = value
        return value

    namespace['__getattr__'] = get_attribute
    namespace['__dir__'] = lambda: sorted(set(namespace) | exports)
    namespace['__all__'] = sorted(
        exports | {name for name in namespace if not name.startswith('_')})
