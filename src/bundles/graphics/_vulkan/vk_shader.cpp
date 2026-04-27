/*
 * Helper to wrap SPIR-V bytecode in a VkShaderModule.
 */

#include <Python.h>
#include "vulkan_internal.h"

VkShaderModule create_shader_module(VulkanContext* ctx, const uint32_t* code, size_t size) {
    VkShaderModuleCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode = code;

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx->device, &info, nullptr, &module) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return module;
}
