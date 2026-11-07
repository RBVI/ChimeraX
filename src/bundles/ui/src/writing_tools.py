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

"""Avoid automatic Writing Tools loading in the inspected standalone macOS app.

AppKit loads WritingToolsUI from main-menu customization, Siri availability
observation, and the lightweight text affordance. Text-view opt-outs do not
prevent those queries. ChimeraX command entry does not use these features.

The latter two hooks are private AppKit API. Apply this process-local policy
only to the inspected arm64 AppKit image, before NSApplication exists, and
only when every method still has its original implementation and signature.
An OS update or another interceptor keeps the ordinary startup path. No
system preferences, executable instructions, or other processes are changed.
"""
import ctypes
import struct
import sys
from threading import current_thread, main_thread

from .native_application import _image

_APPKIT_UUID = '77e3adb8e4e839128599d43182877c10'
# Class, selector, Objective-C encoding, implementation offset in this image.
_METHODS = (
    (b'NSWritingToolsCoordinator', b'isWritingToolsAvailable', b'B16@0:8', 0x8212e0),
    (b'NSWritingToolsCoordinator', b'addObserverForEnhancedSiriAvailabilityDidChangeUsingBlock:',
     b'@24@0:8@?16', 0x8213ac),
    (b'NSWritingToolsCoordinator', b'removeObserverForEnhancedSiriAvailabilityDidChange:',
     b'v24@0:8@16', 0x821614),
    (b'NSCampoLightweightUIController', b'isEligible', b'B16@0:8', 0x6ec394),
)
_ZERO_RETURN = bytes.fromhex('00008052c0035fd6')


def _function(objc, name, result, *arguments):
    function = getattr(objc, name)
    function.argtypes = list(arguments)
    function.restype = result
    return function


def _prepare_policy(objc, base):
    ptr = ctypes.c_void_p
    get_class = _function(objc, 'objc_getClass', ptr, ctypes.c_char_p)
    selector = _function(objc, 'sel_registerName', ptr, ctypes.c_char_p)
    class_method = _function(objc, 'class_getClassMethod', ptr, ptr, ptr)
    instance_method = _function(objc, 'class_getInstanceMethod', ptr, ptr, ptr)
    encoding = _function(objc, 'method_getTypeEncoding', ctypes.c_char_p, ptr)
    implementation = _function(objc, 'method_getImplementation', ptr, ptr)
    replace = _function(objc, 'method_setImplementation', ptr, ptr, ptr)

    # NSObject.isProxy is a permanent native constant-zero implementation.
    # The verified arm64 instructions clear all of x0 and return, ignoring
    # arguments. They return NO for BOOL, nil for the observer token, and
    # serve as a no-op for removal. No Python callback survives shutdown.
    source = instance_method(get_class(b'NSObject'), selector(b'isProxy'))
    if not source or encoding(source) != b'B16@0:8':
        return None
    native = implementation(source)
    if not native or native >> 48 or ctypes.string_at(native, 8) != _ZERO_RETURN:
        return None

    methods = []
    for name, message, expected, offset in _METHODS:
        cls = get_class(name)
        method = class_method(cls, selector(message)) if cls else None
        if (not method or encoding(method) != expected
                or implementation(method) != base + offset):
            return None
        methods.append(method)
    return replace, methods, native


def disable_writing_tools(framework, objc):
    """Return True only after applying the complete guarded standalone policy."""
    if sys.platform != 'darwin' or current_thread() is not main_thread():
        return False
    try:
        if ctypes.c_void_p.in_dll(framework, 'NSApp').value:
            return False
        base, uuid = _image(framework, 'NSApplicationMain')
        if uuid != _APPKIT_UUID:
            return False
        prepared = _prepare_policy(objc, base)
    except (AttributeError, OSError, ValueError, struct.error):
        return False
    if prepared is None:
        return False
    replace, methods, native = prepared
    for method in methods:
        replace(method, native)
    return True
