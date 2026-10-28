"""Avoid PySide source inspection until optional API features are requested."""


def defer_source_scans():
    # PySide's first __feature__ import classifies every loaded module in
    # feature.__init__(). Until then there are no nondefault API selections,
    # so eagerly reading each imported module's source duplicates that work.
    # Leave feature imports and their initialization untouched; resume the
    # original callback as soon as feature initialization has taken place.
    import sys
    feature = sys.modules.get('shibokensupport.feature')
    if feature is None or getattr(feature, '_is_initialized', None) is not False:
        return
    original = getattr(feature, 'feature_imported', None)
    if (not callable(original)
            or getattr(original, '__module__', None) != feature.__name__):
        return  # Already deferred, or another integration owns this callback.

    def feature_imported(module):
        if feature._is_initialized:
            feature.feature_imported = original
            original(module)

    feature.feature_imported = feature_imported
