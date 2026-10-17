"""Vulkan runtime negotiation and headset texture submission.

Use explicit ctypes layouts here: the bundled pyopenxr release declares
Vulkan handles as 32-bit integers, which truncates handles on 64-bit hosts.
These layouts follow XR_KHR_vulkan_enable; no Vulkan Python package is needed.
"""

import ctypes as C


class GraphicsBindingVulkan(C.Structure):
    _fields_ = [('type', C.c_int32), ('next', C.c_void_p),
                ('instance', C.c_void_p), ('physical_device', C.c_void_p),
                ('device', C.c_void_p), ('queue_family_index', C.c_uint32),
                ('queue_index', C.c_uint32)]

    def __init__(self, info):
        super().__init__(1000025000, None, info['instance'], info['physical_device'],
                         info['device'], info['queue_family_index'], info['queue_index'])


class SwapchainImageVulkan(C.Structure):
    _fields_ = [('type', C.c_int32), ('next', C.c_void_p), ('image', C.c_uint64)]

    def __init__(self):
        super().__init__(1000025001, None, 0)


class GraphicsRequirementsVulkan(C.Structure):
    _fields_ = [('type', C.c_int32), ('next', C.c_void_p),
                ('min_version', C.c_uint64), ('max_version', C.c_uint64)]

    def __init__(self):
        super().__init__(1000025002, None, 0, 0)


def configure_runtime(render, runtime):
    """Recreate the Vulkan device with the runtime's adapter and extensions."""
    window = render.opengl_context.window
    window.switch_backend('vulkan', force=True, vulkan_runtime=runtime)
    return window.view.render


def device_info(render):
    return render._vulkan.device_info(context=render._ctx)


class OpenXRRuntime:
    def __init__(self, instance, system_id):
        self.instance = instance
        self.system_id = system_id

    def _call(self, name, argtypes, *args):
        import xr
        prototype = getattr(C, 'WINFUNCTYPE', C.CFUNCTYPE)(
            C.c_int32, xr.Instance, C.c_uint64, *argtypes)
        fn = C.cast(xr.get_instance_proc_addr(self.instance, name), prototype)
        result = xr.exception.check_result(xr.Result(fn(self.instance, self.system_id, *args)))
        if result.is_exception():
            raise result

    def _extensions(self, name):
        count = C.c_uint32()
        signature = (C.c_uint32, C.POINTER(C.c_uint32), C.c_void_p)
        self._call(name, signature, 0, C.byref(count), None)
        data = C.create_string_buffer(max(1, count.value))
        self._call(name, signature, len(data), C.byref(count), data)
        return data.value.decode('ascii').split()

    def instance_extensions(self):
        requirements = GraphicsRequirementsVulkan()
        self._call('xrGetVulkanGraphicsRequirementsKHR',
                   (C.POINTER(GraphicsRequirementsVulkan),), C.byref(requirements))
        # XrVersion packs major/minor differently from VkVersion.
        version = (1 << 48) | (2 << 32)
        if not requirements.min_version <= version <= requirements.max_version:
            raise RuntimeError('The OpenXR runtime does not support Vulkan 1.2')
        return self._extensions('xrGetVulkanInstanceExtensionsKHR')

    def physical_device(self, instance):
        physical = C.c_void_p()
        self._call('xrGetVulkanGraphicsDeviceKHR',
                   (C.c_void_p, C.POINTER(C.c_void_p)), instance, C.byref(physical))
        return physical.value or 0

    def device_extensions(self, physical_device):
        return self._extensions('xrGetVulkanDeviceExtensionsKHR')


class OpenVRRuntime:
    def __init__(self, system, compositor):
        self.system, self.compositor = system, compositor

    def instance_extensions(self):
        return self.compositor.getVulkanInstanceExtensionsRequired().split()

    def physical_device(self, instance):
        import openvr
        physical = C.c_uint64()
        # The bundled wrapper supplies a newly allocated fake instance.
        # Call the typed function-table entry with our real VkInstance.
        self.system.function_table.getOutputDevice(
            C.byref(physical), openvr.TextureType_Vulkan,
            C.cast(instance, C.POINTER(openvr.VkInstance_T)))
        return physical.value

    def device_extensions(self, physical_device):
        import openvr
        device = C.cast(physical_device, C.POINTER(openvr.VkPhysicalDevice_T))
        fn = self.compositor.function_table.getVulkanDeviceExtensionsRequired
        count = fn(device, None, 0)
        data = C.create_string_buffer(max(1, count))
        fn(device, data, len(data))
        return data.value.decode('ascii').split()


def openvr_texture(render, framebuffer):
    import openvr
    info = device_info(render)
    texture = framebuffer.color_texture
    data = openvr.VRVulkanTextureData_t()
    data.m_nImage = render._vulkan.texture_identity(texture=texture._handle)[0]
    for field, key, handle_type in (
            ('m_pInstance', 'instance', openvr.VkInstance_T),
            ('m_pDevice', 'device', openvr.VkDevice_T),
            ('m_pPhysicalDevice', 'physical_device', openvr.VkPhysicalDevice_T),
            ('m_pQueue', 'queue', openvr.VkQueue_T)):
        setattr(data, field, C.cast(info[key], C.POINTER(handle_type)))
    data.m_nQueueFamilyIndex = info['queue_family_index']
    data.m_nWidth, data.m_nHeight = framebuffer.width, framebuffer.height
    data.m_nFormat = render._vulkan.texture_format(texture=texture._handle)
    data.m_nSampleCount = 1
    submission = openvr.Texture_t()
    submission.handle = C.cast(C.pointer(data), C.c_void_p)
    submission.eType = openvr.TextureType_Vulkan
    submission.eColorSpace = openvr.ColorSpace_Gamma
    # Keep the pointed-to structure alive as long as the submission object.
    submission._vulkan_data = data
    return submission


def prepare_mirror(render, drawing):
    """Finish offscreen rendering and start a desktop presentation frame."""
    if drawing is None or not hasattr(render, '_vulkan'):
        return
    render.end_frame()
    texture = drawing.texture
    if texture is not None:
        texture.prepare_for_sampling()
    render.draw_background()
