from . import using_pyqt6, using_pyqt5, using_pyside2

if using_pyqt6:
    from PyQt6.QtCore import *
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtCore import pyqtSlot as Slot
    from PyQt6.QtCore import pyqtProperty as Property
    from PyQt6.QtCore import QT_VERSION_STR as __version__

    # Put commonly used enumeration values in Qt namespace.
    enum_values = [
        (Qt.AlignmentFlag, ('AlignLeft', 'AlignRight', 'AlignTop', 'AlignBottom',
                            'AlignHCenter', 'AlignVCenter', 'AlignCenter')),
        (Qt.ArrowType, ['RightArrow', 'DownArrow']),
        (Qt.BrushStyle, ('NoBrush', 'SolidPattern')),
        (Qt.CheckState, ('Checked', 'Unchecked')),
        (Qt.FocusReason, ['ShortcutFocusReason']),
        (Qt.GlobalColor, ('black', 'white', 'blue', 'darkGreen')),
        (Qt.ItemDataRole, ('UserRole', 'DisplayRole', 'TextAlignmentRole', 'FontRole',
                           'ForegroundRole', 'ToolTipRole')),
        (Qt.ItemFlag, ('ItemIsEnabled', 'ItemIsUserCheckable', 'ItemIsAutoTristate',
                       'ItemIsSelectable', 'ItemNeverHasChildren')),
        (Qt.Key, [('Key_'+v) for v in ('T','W','Tab','0','1','2','3','4','5','6','7','8','9',
                                       'Plus', 'Minus', 'ZoomIn', 'ZoomOut', 'HomePage', 'Search',
                                       'Back', 'Forward', 'Reload', 'Equal')]),
        (Qt.KeyboardModifier, ('ShiftModifier', 'ControlModifier')),
        (Qt.Modifier, ('CTRL', 'SHIFT', 'META', 'ALT')),
        (Qt.MouseButton, ('RightButton', 'LeftButton')),
        (Qt.Orientation, ('Vertical', 'Horizontal')),
        (Qt.PenStyle, ('NoPen', 'DashLine', 'SolidLine', 'DotLine')),
        (Qt.SortOrder, ('AscendingOrder', 'DescendingOrder')),
        (Qt.ToolButtonStyle, ['ToolButtonTextBesideIcon']),
        (Qt.WidgetAttribute, ('WA_DeleteOnClose', 'WA_LayoutUsesWidgetRect', 'WA_AlwaysShowToolTips',
                              'WA_AcceptTouchEvents')),
        ]
    for enum, values in enum_values:
        for value in values:
            setattr(Qt, value, getattr(enum, value))
    
    # Those are imported from `import *`
    del pyqtSignal, pyqtSlot, pyqtProperty, QT_VERSION_STR

elif using_pyqt5:
    from PyQt5.QtCore import *
    from PyQt5.QtCore import pyqtSignal as Signal
    from PyQt5.QtCore import pyqtSlot as Slot
    from PyQt5.QtCore import pyqtProperty as Property
    from PyQt5.QtCore import QT_VERSION_STR as __version__

    # Those are imported from `import *`
    del pyqtSignal, pyqtSlot, pyqtProperty, QT_VERSION_STR

elif using_pyside2:
    from PySide2.QtCore import *
    import PySide2.QtCore
    __version__ = PySide2.QtCore.__version__
