# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE VERSION 2.1.
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""Overlap independent session state with main-thread AppKit initialization."""
import sys
from threading import Thread, current_thread, main_thread


def _text_system_initializer(get_class, selector, get_class_method, send_boolean):
    # AppKit queries this public property while preparing its native menus.
    # Resolve it before starting the worker; older macOS versions omit it.
    coordinator = get_class(b'NSWritingToolsCoordinator')
    if not coordinator:
        return None
    available = selector(b'isWritingToolsAvailable')
    if not available or not get_class_method(coordinator, available):
        return None
    return lambda: send_boolean(coordinator, available)


class _NativeApplication:
    def __init__(self):
        import ctypes

        self._pool = None
        self.existing = False
        self._framework = ctypes.CDLL('/System/Library/Frameworks/AppKit.framework/AppKit')
        self._objc = objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        if ctypes.c_void_p.in_dll(self._framework, 'NSApp').value:
            self.existing = True
            return
        get_class = objc.objc_getClass
        get_class.argtypes = [ctypes.c_char_p]
        get_class.restype = ctypes.c_void_p
        selector = objc.sel_registerName
        selector.argtypes = [ctypes.c_char_p]
        selector.restype = ctypes.c_void_p
        self._application_class = get_class(b'NSApplication')
        pool_class = get_class(b'NSAutoreleasePool')
        if not self._application_class or not pool_class:
            raise AttributeError('Native application classes unavailable')
        self._shared_application = selector(b'sharedApplication')
        self._drain = selector(b'drain')
        allocate, initialize = selector(b'alloc'), selector(b'init')
        if not all((self._shared_application, self._drain, allocate, initialize)):
            raise AttributeError('Native application selectors unavailable')
        self._send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ('objc_msgSend', objc))
        self._send_void = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(
            ('objc_msgSend', objc))
        get_class_method = objc.class_getClassMethod
        get_class_method.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        get_class_method.restype = ctypes.c_void_p
        send_boolean = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(
            ('objc_msgSend', objc))
        self._initialize_text_system = _text_system_initializer(
            get_class, selector, get_class_method, send_boolean)
        self._pool = self._send(self._send(pool_class, allocate), initialize)
        if not self._pool:
            raise OSError('Native autorelease pool unavailable')

    def initialize(self):
        # The foreign call releases the GIL. AppKit stays on the main thread;
        # Qt supports an NSApplication that exists before QApplication.
        self._send(self._application_class, self._shared_application)

    def close(self):
        pool, self._pool = self._pool, None
        if pool:
            self._send_void(pool, self._drain)

    def warm_text_system(self):
        # Keep AppKit on the main thread and release the GIL during the query,
        # allowing the remaining session work to finish before we join it.
        if self._initialize_text_system is not None:
            self._initialize_text_system()


def _prepare_native_application():
    if sys.platform != 'darwin' or current_thread() is not main_thread():
        return None
    try:
        native = _NativeApplication()
    except (OSError, AttributeError):
        return None
    if native.existing:
        native.close()
        return None
    return native


def initialize_application(session, construct_application, initialize_settings):
    """Finish state and settings before the application is returned to startup."""
    native = _prepare_native_application()
    if native is None:
        session._initialize_state()
        construct_application()
        return initialize_settings()

    errors = []

    def initialize_state():
        try:
            session._initialize_state()
        except BaseException as error:
            errors.append((error, error.__traceback__))

    worker = None
    started = False
    application_error = None
    try:
        try:
            worker = Thread(target=initialize_state, name='ChimeraX session initialization')
            worker.start()
            started = True
        except RuntimeError:
            # Thread exhaustion must not prevent the ordinary startup path.
            session._initialize_state()
        native.initialize()
        construct_application()
        native.warm_text_system()
    except BaseException as error:
        application_error = (error, error.__traceback__)
    finally:
        try:
            if started:
                worker.join()
        finally:
            native.close()

    if application_error is not None:
        error, traceback = application_error
        if errors:
            raise error.with_traceback(traceback) from errors[0][0]
        raise error.with_traceback(traceback)
    if errors:
        error, traceback = errors[0]
        raise error.with_traceback(traceback)
    return initialize_settings()
