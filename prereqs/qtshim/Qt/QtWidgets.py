from . import using_pyqt6, using_pyqt5, using_pyside2

if using_pyqt6:
    from PyQt6.QtWidgets import *

    # Put some commonly used enumeration values in class namespaces.
    enum_values = [
        (QAbstractItemView, 'SelectionBehavior', ['SelectRows']),
        (QAbstractItemView, 'SelectionMode', ('ExtendedSelection', 'SingleSelection', 'MultiSelection')),
        (QAbstractItemView, 'EditTrigger', ['NoEditTriggers']),
        (QAbstractSpinBox, 'ButtonSymbols', ['NoButtons']),
        (QAbstractSpinBox, 'StepType', ['AdaptiveDecimalStepType']),
        (QDialogButtonBox, 'StandardButton', ('Ok', 'Apply', 'Cancel', 'Close', 'Help')),
        (QDialogButtonBox, 'ButtonRole', ('ActionRole', 'AcceptRole', 'DestructiveRole')),
        (QFileDialog, 'AcceptMode', ['AcceptSave']),
        (QFileDialog, 'FileMode', ['AnyFile', 'Directory']),
        (QFileDialog, 'Option', ['DontUseNativeDialog']),
        (QFormLayout, 'ItemRole', ('LabelRole', 'FieldRole')),
        (QFrame, 'Shape', ('HLine', 'Panel', 'Box')),
        (QFrame, 'Shadow', ['Raised', 'Sunken', 'Plain']),
        (QLayout, 'SizeConstraint', ['SetFixedSize', 'SetMinAndMaxSize']),
        (QLineEdit, 'EchoMode', ['PasswordEchoOnEdit']),
        (QSizePolicy, 'Policy', ('Expanding', 'Fixed', 'Minimum')),
    ]
    for cls, enum_name, values in enum_values:
        enum = getattr(cls, enum_name)
        for value in values:
            setattr(cls, value, getattr(enum, value))
    
    # Make relocated classes available in their PyQt5 location.
    from PyQt6.QtGui import QAction, QShortcut

elif using_pyqt5:
    from PyQt5.QtWidgets import *

elif using_pyside2:
    from PySide2.QtWidgets import *
