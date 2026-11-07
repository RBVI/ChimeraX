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

"""Release the GIL around this bundled binding's native app constructor.

PySide's injected QApplication constructor holds the GIL during native Qt
initialization. Its generated wrapper constructor contains no Python API
calls. For the inspected bundled image, call that same constructor with the
GIL released, then perform the normal ownership/signal bookkeeping with the
GIL held. No binary code is modified. Other images use ordinary construction.

The image identities, instructions and wrapper size below are deliberately
specific to the bundled arm64 PySide 6.11.2 build. A rebuilt binding must be
inspected and validated before adding its identity here. The corresponding
upstream fix is to release threads around new QApplicationWrapper in
PySide6/glue/qtwidgets.cpp.
"""
import ctypes
import os
import struct
import sys
from threading import current_thread, main_thread

_WIDGETS_UUID = 'f2ecfa08ff1536c197c6f70fe57e4a29'
_PYSIDE_UUID = '236be6ecfed63909ba87c396ef8556dc'
_SHIBOKEN_UUID = '543bcd00e262371eb040cac9c10f36de'
_CONSTRUCTOR_OFFSET = 0x3dad80
_CONSTRUCTOR_CODE = bytes.fromhex(
    'fd7bbfa9fd030091403a0194280b00d008c13591080000f900e4006f0'
    '08000ad008001adfd7bc1a8c0035fd6')
_WRAPPER_SIZE = 80


def _image_uuid(data):
    if len(data) < 32:
        raise ValueError('Truncated native image header')
    magic, cpu, subtype, kind, count, size, flags, reserved = struct.unpack_from('<8I', data)
    if magic != 0xfeedfacf or cpu != 0x100000c or size > 65536 or len(data) != 32 + size:
        raise ValueError('Unsupported native image')
    offset, uuid = 32, None
    for _ in range(count):
        if offset + 8 > len(data):
            raise ValueError('Truncated native image command')
        command, length = struct.unpack_from('<2I', data, offset)
        if length < 8 or offset + length > len(data):
            raise ValueError('Invalid native image command')
        if command == 0x1b:
            if length != 24:
                raise ValueError('Invalid native image identity')
            uuid = data[offset+8:offset+24].hex()
        offset += length
    if uuid is None or offset != len(data):
        raise ValueError('Missing native image identity')
    return uuid


class _DlInfo(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p), ('base', ctypes.c_void_p),
                ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]


def _image(library, symbol):
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    dladdr.restype = ctypes.c_int
    info = _DlInfo()
    address = ctypes.cast(getattr(library, symbol), ctypes.c_void_p)
    if not dladdr(address, ctypes.byref(info)) or not info.base:
        raise ValueError('Native image address unavailable')
    header = ctypes.string_at(info.base, 32)
    size = struct.unpack_from('<I', header, 20)[0]
    if size > 65536:
        raise ValueError('Unsupported native image header')
    return info.base, _image_uuid(header + ctypes.string_at(info.base + 32, size))


def _python_api(library, name, result, *arguments):
    # PYFUNCTYPE retains the GIL and propagates Python errors from binding APIs.
    return ctypes.PYFUNCTYPE(result, *arguments)((name, library))


def _destroy_unregistered_application(memory):
    # Match this image's generated initializer error path: virtual deleting
    # destructor at byte offset 0x20 of the verified wrapper's vtable. Keep
    # the GIL, as the generated initializer does for this cleanup.
    vtable = ctypes.c_void_p.from_address(memory).value
    destructor = ctypes.c_void_p.from_address(vtable + 0x20).value
    ctypes.PYFUNCTYPE(None, ctypes.c_void_p)(destructor)(memory)


def prepare_application_constructor(ui, application_type, arguments):
    """Return a constructor for the inspected binding, or None before mutation."""
    if (sys.platform != 'darwin' or sys.implementation.name != 'cpython'
            or current_thread() is not main_thread()
            or application_type.__module__ != 'PySide6.QtWidgets'
            or application_type.__name__ != 'QApplication'
            or type(ui).__bases__ != (application_type,)
            or not arguments or any(type(a) is not str or '\0' in a for a in arguments)):
        return None
    if application_type.instance() is not None:
        return None
    try:
        return _prepare_constructor(ui, application_type, arguments)
    except (AttributeError, OSError, ValueError, struct.error, UnicodeError):
        # All image/symbol checks precede application construction. An unknown
        # bundle or unsupported binding keeps the ordinary PySide path.
        return None


def _prepare_constructor(ui, application_type, arguments):
    widgets = sys.modules['PySide6.QtWidgets']
    widgets_library = ctypes.CDLL(widgets.__file__)
    base, uuid = _image(widgets_library, 'PyInit_QtWidgets')
    if uuid != _WIDGETS_UUID:
        raise ValueError('Unrecognized bundled QtWidgets image')
    address = base + _CONSTRUCTOR_OFFSET
    if ctypes.string_at(address, len(_CONSTRUCTOR_CODE)) != _CONSTRUCTOR_CODE:
        raise ValueError('Native application constructor changed')
    pyside = ctypes.PyDLL(os.path.join(os.path.dirname(sys.modules['PySide6'].__file__),
        'libpyside6.abi3.6.11.dylib'))
    shiboken = ctypes.PyDLL(os.path.join(os.path.dirname(sys.modules['shiboken6'].__file__),
        'libshiboken6.abi3.6.11.dylib'))
    if (_image(pyside, '_ZN6PySide23destroyQCoreApplicationEv')[1] != _PYSIDE_UUID
            or _image(shiboken, '_ZN8Shiboken6Object13setCppPointerEP9SbkObjectP11_typeobjectPv')[1] != _SHIBOKEN_UUID):
        raise ValueError('Unrecognized bundled binding support library')

    ptr = ctypes.c_void_p
    # This is the only call made without the GIL. The verified wrapper body
    # calls QApplication, installs its vtable, and clears override caches.
    constructor = ctypes.CFUNCTYPE(None, ptr, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int)(address)
    cpp = ctypes.CDLL('/usr/lib/libc++.1.dylib')
    nothrow = ctypes.c_byte.in_dll(cpp, '_ZSt7nothrow')
    new = _python_api(cpp, '_ZnwmRKSt9nothrow_t', ptr, ctypes.c_size_t, ptr)
    delete = _python_api(cpp, '_ZdlPv', None, ptr)

    def allocate(size):
        memory = new(size, ctypes.byref(nothrow))
        if not memory:
            raise MemoryError('Native application allocation failed')
        return memory

    release = _python_api(shiboken, '_ZN8Shiboken6Object16releaseOwnershipEP9SbkObject', None, ptr)
    set_pointer = _python_api(shiboken, '_ZN8Shiboken6Object13setCppPointerEP9SbkObjectP11_typeobjectPv', ctypes.c_bool, ptr, ptr, ptr)
    set_valid = _python_api(shiboken, '_ZN8Shiboken6Object11setValidCppEP9SbkObjectb', None, ptr, ctypes.c_bool)
    set_wrapper = _python_api(shiboken, '_ZN8Shiboken6Object16setHasCppWrapperEP9SbkObjectb', None, ptr, ctypes.c_bool)
    manager = _python_api(shiboken, '_ZN8Shiboken14BindingManager8instanceEv', ptr)()
    register_wrapper = _python_api(shiboken, '_ZN8Shiboken14BindingManager15registerWrapperEP9SbkObjectPv', None, ptr, ptr, ptr)
    update_signals = _python_api(pyside, '_ZN6PySide6Signal18updateSourceObjectEP7_object', None, ptr)
    register_cleanup = _python_api(pyside, '_ZN6PySide23registerCleanupFunctionEPFvvE', None, ptr)
    cleanup = ctypes.cast(pyside._ZN6PySide23destroyQCoreApplicationEv, ptr)
    select_feature = _python_api(pyside, '_ZN6PySide7Feature6SelectEP7_object', None, ptr)
    enter_context = _python_api(shiboken, '_ZN8Shiboken19PythonContextMarkerC1Ev', None, ptr)
    leave_context = _python_api(shiboken, '_ZN8Shiboken19PythonContextMarkerD1Ev', None, ptr)
    errors_occurred = _python_api(shiboken, '_ZN8Shiboken6Errors8occurredEv', ptr)

    # Like the normal glue's static argc/argv, these buffers have native
    # process lifetime. Python module teardown must not free them before Qt.
    encoded = [argument.encode('utf-8') for argument in arguments]
    buffers = []
    try:
        argc_memory = allocate(ctypes.sizeof(ctypes.c_int))
        buffers.append(argc_memory)
        argv_memory = allocate(ctypes.sizeof(ptr) * (len(encoded) + 1))
        buffers.append(argv_memory)
        argc = ctypes.cast(argc_memory, ctypes.POINTER(ctypes.c_int))
        argc[0] = len(encoded)
        argv_pointers = ctypes.cast(argv_memory, ctypes.POINTER(ptr))
        for i, argument in enumerate(encoded):
            memory = allocate(len(argument) + 1)
            buffers.append(memory)
            ctypes.memmove(memory, argument + b'\0', len(argument) + 1)
            argv_pointers[i] = memory
        argv_pointers[len(encoded)] = None
        argv = ctypes.cast(argv_memory, ctypes.POINTER(ctypes.c_char_p))
    except BaseException:
        for memory in reversed(buffers):
            delete(memory)
        raise

    called = False

    def construct():
        nonlocal called
        if called:
            raise RuntimeError('Application constructor already used')
        called = True
        select_feature(id(ui))
        context = ctypes.c_byte()
        enter_context(ctypes.byref(context))
        constructed = pointer_set = False
        try:
            memory = allocate(_WRAPPER_SIZE)
            constructor(memory, argc, argv, 0)
            constructed = True
            release(id(ui))
            register_cleanup(cleanup)
            # This restores errors deferred by native callbacks. PYFUNCTYPE
            # then raises the restored Python exception before registration.
            errors_occurred()
            if not set_pointer(id(ui), id(application_type), memory):
                raise RuntimeError('Could not register native application pointer')
            pointer_set = True
            set_valid(id(ui), True)
            set_wrapper(id(ui), True)
            register_wrapper(manager, id(ui), memory)
            update_signals(id(ui))
            errors_occurred()
            ui._native_qapplication = True
            # Keep the loaded support libraries alive with the application.
            ui._native_application_libraries = (widgets_library, pyside, shiboken, cpp)
        except BaseException:
            if constructed and not pointer_set:
                _destroy_unregistered_application(memory)
            raise
        finally:
            leave_context(ctypes.byref(context))

    return construct
