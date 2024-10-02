# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

#
# Routines to use the Facebook OpenXR passthrough video extension
# XR_FB_passthrough with Meta Quest headsets.
#
#  https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_FB_passthrough
#

def passthrough_extension_available():
    import xr
    extensions = xr.enumerate_instance_extension_properties()
    ext_names = [e.extension_name.decode('UTF-8') for e in extensions]
    #print (f'Available openxr extensions {ext_names}')
    return xr.FB_PASSTHROUGH_EXTENSION_NAME in ext_names

class Passthrough:
    def __init__(self, xr_session, xr_instance):
        _define_passthrough_functions(xr_instance)
        self._passthrough, self._layer, self._composition_layer = \
            _enable_passthrough(xr_session, xr_instance)

    def close(self):
        import xr
        xr.xrDestroyPassthroughFB(self._passthrough)
        xr.xrDestroyPassthroughLayerFB(self._layer)
        self._passthrough = None
        self._pasthrough_layer = None
        self._composition_layer = None


def _enable_passthrough(xr_session, xr_instance):
    '''
    Create a CompositionLayerPassthroughFB instance for rendering
    in xrEndFrame().
    '''
    import xr
    passthrough = xr.PassthroughFB()
    pflags = xr.PassthroughFlagsFB(xr.PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB)
    pinfo = xr.PassthroughCreateInfoFB(pflags)
    from ctypes import byref
    result = xr.xrCreatePassthroughFB(xr_session, pinfo, byref(passthrough))
    result = xr.check_result(xr.Result(result))
    if result.is_exception():
        raise result
    passthrough_layer = xr.PassthroughLayerFB()
    plflags = xr.PassthroughFlagsFB(xr.PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB)
    plinfo = xr.PassthroughLayerCreateInfoFB(passthrough, plflags)
    result = xr.xrCreatePassthroughLayerFB(xr_session, plinfo, byref(passthrough_layer))
    result = xr.check_result(xr.Result(result))
    if result.is_exception():
        raise result
    composition_layer = xr.CompositionLayerPassthroughFB(layer_handle = passthrough_layer)
    return passthrough, passthrough_layer, composition_layer

def _define_passthrough_functions(xr_instance):
    import xr
    if hasattr(xr, 'xrCreatePassthroughFB'):
        return
    func_names = ['xrCreatePassthroughFB', 'xrCreatePassthroughLayerFB',
                  'xrDestroyPassthroughFB', 'xrDestroyPassthroughLayerFB']
    from ctypes import cast
    for func_name in func_names:
        f = cast(xr.get_instance_proc_addr(xr_instance, func_name),
                 getattr(xr, f'PFN_{func_name}'))
        setattr(xr, func_name, f)

def _check_blend_modes(xr_instance, xr_system_id):
    import xr
    view_config_type = xr.ViewConfigurationType(xr.ViewConfigurationType.PRIMARY_STEREO)
    blend_modes = xr.enumerate_environment_blend_modes(xr_instance,
                                                       xr_system_id,
                                                       view_config_type)
    print ('Supported OpenXR environment blend modes', [m for m in blend_modes])
    # Reports only blend mode 1 is supported using AirLink and Quest 2.
    # Modes are OPAQUE = 1, ADDITIVE = 2, ALPHA_BLEND = 3
    # Try QuestLink with Quest 3.
    #   https://community.khronos.org/t/xr-fb-passthrough-without-alpha-blend-environment-blend-mode/110353
    #   https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XrCompositionLayerFlags

    # If ALPHA_BLEND or ADDITIVE is used in xrEndFrame() get
    #  "xr.exception.EnvironmentBlendModeUnsupportedError: The specified environment blend mode is not supported by the runtime or platform."
    # Comment online suggests composition layer blend modes are used instead
    # of environment blend mode.
    #   https://community.khronos.org/t/xr-fb-passthrough-without-alpha-blend-environment-blend-mode/110353/2
