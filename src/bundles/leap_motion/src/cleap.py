# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

import ctypes

class LeapError(Exception):
    pass

# -----------------------------------------------------------------------------
# Wrap the Leap Motion CLeap C library version 4.0.0.
#
class CLeapLibrary:
    def __init__(self, library_path = None):
        self._c_funcs = self._open_c_library(library_path)

    def _open_c_library(self, library_path):
        if library_path is None:
            from os.path import join, dirname
            library_path = join(dirname(__file__), 'LeapC.dll')
        c_funcs = CFunctions(library_path)
        return c_funcs

    def LeapCreateConnection(self):
        f = self._c_function('LeapCreateConnection', args = (ctypes.c_void_p, ctypes.c_void_p),
                             ret = ctypes.c_uint32)
        leap_connection_config = ctypes.c_void_p(0)
        leap_connection = ctypes.c_void_p(0)
        result = f(leap_connection_config, ctypes.pointer(leap_connection))
        self._report_error('LeapCreateConnection', result)
        return leap_connection

    def LeapOpenConnection(self, leap_connection):
        f = self._c_function('LeapOpenConnection', args = (ctypes.c_void_p,),
                             ret = ctypes.c_uint32)
        result = f(leap_connection)
        self._report_error('LeapOpenConnection', result)
        return result
        
    def _report_error(self, func_name, result):
        if result != self.eLeapRS_Success:
            raise LeapError('%s failed: %s' % (func_name, self._result_string(result)))

    def _c_function(self, func_name, args = None, ret = None):
        return self._c_funcs.c_function(func_name, args=args, ret=ret)

    # The operation completed successfully.
    eLeapRS_Success                       = 0x00000000

    # An undetermined error has occurred.
    # This is usually the result of an abnormal operating condition in LeapC,
    # the Leap Motion service, or the host computer itself.
    eLeapRS_UnknownError                  = 0xE2010000

    # An invalid argument was specified.
    eLeapRS_InvalidArgument               = 0xE2010001

    # Insufficient resources existed to complete the request.
    eLeapRS_InsufficientResources         = 0xE2010002

    # The specified buffer was not large enough to complete the request.
    eLeapRS_InsufficientBuffer            = 0xE2010003

    # The requested operation has timed out.
    eLeapRS_Timeout                       = 0xE2010004

    # The operation is invalid because there is no current connection.
    eLeapRS_NotConnected                  = 0xE2010005

    # The operation is invalid because the connection is not complete.
    eLeapRS_HandshakeIncomplete           = 0xE2010006

    # The specified buffer size is too large.
    eLeapRS_BufferSizeOverflow            = 0xE2010007

    # A communications protocol error occurred.
    eLeapRS_ProtocolError                 = 0xE2010008

    # The server incorrectly specified zero as a client ID.
    eLeapRS_InvalidClientID               = 0xE2010009

    # The connection to the service was unexpectedly closed while reading or writing a message.
    # The server may have terminated.
    eLeapRS_UnexpectedClosed              = 0xE201000A

    # The specified request token does not appear to be valid.
    # Provided that the token value which identifies the request itself was, at one point, valid, this
    # error condition occurs when the request to which the token refers has already been satisfied or
    # is currently being satisfied.
    eLeapRS_UnknownImageFrameRequest      = 0xE201000B

    # The specified frame ID is not valid or is no longer valid
    # Provided that frame ID was, at one point, valid, this error condition occurs when the identifier
    # refers to a frame that occurred further in the past than is currently recorded in the rolling
    # frame window.
    eLeapRS_UnknownTrackingFrameID        = 0xE201000C

    # The specified timestamp references a future point in time
    # The related routine can only operate on time points having occurred in the past, and the
    # provided timestamp occurs in the future.
    eLeapRS_RoutineIsNotSeer              = 0xE201000D

    # The specified timestamp references a point too far in the past
    # The related routine can only operate on time points occurring within its immediate record of
    # the past.
    eLeapRS_TimestampTooEarly             = 0xE201000E

    # LeapPollConnection is called concurrently.
    eLeapRS_ConcurrentPoll                = 0xE201000F

    # A connection to the Leap Motion service could not be established.
    eLeapRS_NotAvailable                  = 0xE7010002

    # The requested operation can only be performed while the device is sending data.
    eLeapRS_NotStreaming                  = 0xE7010004

    # The specified device could not be opened. It is possible that the device identifier
    # is invalid, or that the device has been disconnected since being enumerated.
    eLeapRS_CannotOpenDevice              = 0xE7010005

    def _result_string(self, result):
        for name in ('Success', 'UnknownError', 'InvalidArgument', 'InsufficientResources',
                     'InsufficientBuffer', 'Timeout', 'NotConnected', 'HandshakeIncomplete',
                     'BufferSizeOverflow', 'ProtocolError', 'InvalidClientID', 'UnexpectedClosed',
                     'UnknownImageFrameRequest', 'UnknownTrackingFrameID', 'RoutineIsNotSeer',
                     'TimestampTooEarly', 'ConcurrentPoll', 'NotAvailable', 'NotStreaming',
                     'CannotOpenDevice'):
            if getattr(self, 'eLeapRS_'+name) == result:
                return name
        return 'result code %x' % result

    def LeapPollConnection(self, connection_handle, timeout_msec = 0):
        msg = LEAP_CONNECTION_MESSAGE()
        f = self._c_function('LeapPollConnection', args = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p),
                             ret = ctypes.c_uint32)
        result = f(connection_handle, timeout_msec, ctypes.pointer(msg))
        if result == self.eLeapRS_Timeout:
            return None
        self._report_error('LeapPollConnection', result)
        return msg

class LEAP_CONNECTION_MESSAGE(ctypes.Structure):
    _fields_ = [('size', ctypes.c_uint32), 
                ('type', ctypes.c_int), # eLeapEventType enum
                ('pointer', ctypes.c_void_p)]

    @property
    def type_name(self):
        for name in self.event_types:
            if getattr(self, 'eLeapEventType_' + name) == self.type:
                return name
        return 'event %x' % self.type
    
    event_types = ('None', 'Connection', 'ConnectionLost', 'Device', 'DeviceLost',
                   'DeviceFailure', 'Policy', 'Tracking', 'ImageRequestError',
                   'ImageComplete', 'LogEvent', 'DeviceLost', 'ConfigResponse',
                   'ConfigChange', 'DeviceStatusChange', 'DroppedFrame', 'Image',
                   'PointMappingChange', 'LogEvents', 'HeadPose')

    
    # No event has occurred within the timeout period specified when calling LeapPollConnection().
    eLeapEventType_None = 0

    # A connection to the Leap Motion service has been established.
    eLeapEventType_Connection = 1

    # The connection to the Leap Motion service has been lost.
    eLeapEventType_ConnectionLost = 2

    # A device has been detected or plugged-in.
    # A device event is dispatched after a connection is established for any
    # devices already plugged in. (The system currently only supports one
    # streaming device at a time.)
    eLeapEventType_Device = 3

    # A device has failed.
    # Device failure could be caused by hardware failure, USB controller issues, or
    # other system instability. Note that unplugging a device generates an
    # eLeapEventType_DeviceLost event message, not a failure message.
    eLeapEventType_DeviceFailure = 4

    # A policy change has occurred.
    # This can be due to setting a policy with LeapSetPolicyFlags() or due to changing
    # or policy-related config settings, including images_mode.
    # (A user can also change these policies using the Leap Motion Control Panel.)
    eLeapEventType_Policy = 5

    # A tracking frame. The message contains the tracking data for the frame.
    eLeapEventType_Tracking = 0x100

    # The request for an image has failed.
    # The message contains information about the failure. The client application
    # will not receive the requested image set.
    # @since 3.0.0
    eLeapEventType_ImageRequestError = 0x101

    # The request for an image is complete.
    # The image data has been completely written to the application-provided
    # buffer.
    eLeapEventType_ImageComplete = 0x102

    # A system message.
    eLeapEventType_LogEvent = 0x103

    # The device connection has been lost.
    # This event is generally asserted when the device has been detached from the system, when the
    # connection to the service has been lost, or if the device is closed while streaming. Generally,
    # any event where the system can conclude no further frames will be received will result in this
    # message. The DeviceEvent field will be filled with the id of the formerly attached device.
    eLeapEventType_DeviceLost = 0x104

    # The asynchronous response to a call to LeapRequestConfigValue().
    # Contains the value of requested configuration item.
    eLeapEventType_ConfigResponse = 0x105

    # The asynchronous response to a call to LeapSaveConfigValue().
    # Reports whether the change succeeded or failed.
    eLeapEventType_ConfigChange = 0x106

    # Notification that a status change has been detected on an attached device
    eLeapEventType_DeviceStatusChange = 0x107
    eLeapEventType_DroppedFrame = 0x108

    # Notification that an unrequested stereo image pair is available
    eLeapEventType_Image = 0x109

    # Notification that point mapping has changed
    eLeapEventType_PointMappingChange = 0x10A

    # An array of system messages.
    eLeapEventType_LogEvents = 0x10B
    eLeapEventType_HeadPose = 0x10C
                
# -----------------------------------------------------------------------------
#
class CFunctions:
    '''Access C functions from a shared library and create Python properties using these functions.'''

    def __init__(self, library_path):
        self._c_lib = ctypes.PyDLL(library_path)
        
    # -----------------------------------------------------------------------------
    #
    def c_function(self, func_name, args = None, ret = None):
        '''Look up a C function and set its argument types if they have not been set.'''
        f = getattr(self._c_lib, func_name)
        if args is not None and f.argtypes is None:
            f.argtypes = args
        if ret is not None:
            f.restype = ret
        return f

