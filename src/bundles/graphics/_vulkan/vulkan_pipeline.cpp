/*
 * Vulkan pipeline management: compile GLSL shaders to SPIR-V at runtime
 * via shaderc, and create VkPipeline objects with specialization constants.
 */

#include <Python.h>
#include "vulkan_internal.h"
#include <shaderc/shaderc.h>

#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
//  Create a graphics pipeline from GLSL vertex and fragment shader sources.
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_create_pipeline(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr;
    const char* vert_source;
    const char* frag_source;
    int has_depth;
    PyObject* constants;
    int depth_only = 0;
    int color_write = 1;
    int instance_colors = 1;  // 1 = color buffer is per-instance (impostors);
                              // 0 = per-vertex (meshes / ribbons).
    int face_culling = 1;     // 1 = cull back faces; 0 = render both sides
                              // (volume planes, transparent surfaces).
    int blend_op_max = 0;     // 0 = ADD, 1 = MAX (max intensity projection).
    int blend_premult = 0;    // 0 = src=SRC_ALPHA, 1 = src=ONE (used when
                              // dim_transparent_voxels is off).
    int depth_write = 1;      // 0 = leave depth buffer untouched (transparent
                              // volume planes), 1 = write depth.
    int topology = 0;         // 0 = triangle list (Solid), 1 = line list
                              // (Mesh display style), 2 = point list (Dot).
    int oit_mode = 0;         // 0 = normal, 1 = OIT accumulation (2 color
                              // attachments, special blend, uses oit_render_pass).
    static const char* kwlist[] = {"context", "vert_source", "frag_source",
                                   "has_depth", "constants",
                                   "depth_only", "color_write",
                                   "instance_colors", "face_culling",
                                   "blend_op_max", "blend_premult",
                                   "depth_write", "topology",
                                   "oit_mode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nssiO|iiiiiiiii"), (char**)kwlist,
            &ctx_ptr, &vert_source, &frag_source, &has_depth,
            &constants, &depth_only, &color_write, &instance_colors,
            &face_culling, &blend_op_max, &blend_premult, &depth_write,
            &topology, &oit_mode))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VK_TRACE("create_pipeline has_depth=%d depth_only=%d color_write=%d instance_colors=%d face_culling=%d blend_max=%d premult=%d depth_write=%d topology=%d",
             has_depth, depth_only, color_write, instance_colors, face_culling,
             blend_op_max, blend_premult, depth_write, topology);

    // ------------------------------------------------------------------
    //  Build specialization constant data from the Python list.
    //  Each element is an int (0 or 1) that maps to a bool constant in
    //  the GLSL shader declared as:
    //    layout(constant_id = N) const bool SOME_FLAG = false;
    // ------------------------------------------------------------------
    if (!PyList_Check(constants)) {
        PyErr_SetString(PyExc_TypeError,
                        "constants must be a list of ints");
        return NULL;
    }
    Py_ssize_t num_constants = PyList_Size(constants);
    std::vector<uint32_t> spec_data(num_constants);
    std::vector<VkSpecializationMapEntry> spec_entries(num_constants);

    for (Py_ssize_t i = 0; i < num_constants; ++i) {
        PyObject* item = PyList_GetItem(constants, i);
        long v = PyLong_AsLong(item);
        if (v == -1 && PyErr_Occurred())
            return NULL;
        spec_data[i] = v ? 1 : 0;

        spec_entries[i].constantID = static_cast<uint32_t>(i);
        spec_entries[i].offset = static_cast<uint32_t>(i * sizeof(uint32_t));
        spec_entries[i].size = sizeof(uint32_t);
    }

    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = static_cast<uint32_t>(num_constants);
    spec_info.pMapEntries = spec_entries.data();
    spec_info.dataSize = num_constants * sizeof(uint32_t);
    spec_info.pData = spec_data.data();

    // ------------------------------------------------------------------
    //  Compile GLSL to SPIR-V using shaderc.
    // ------------------------------------------------------------------
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    shaderc_compile_options_t options = shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan,
                                          shaderc_env_version_vulkan_1_0);
    shaderc_compile_options_set_optimization_level(options,
                                                   shaderc_optimization_level_performance);

    // -- Vertex shader --
    shaderc_compilation_result_t vert_result = shaderc_compile_into_spv(
        compiler, vert_source, strlen(vert_source),
        shaderc_glsl_vertex_shader, "vertex.glsl", "main", options);

    if (shaderc_result_get_compilation_status(vert_result) !=
        shaderc_compilation_status_success) {
        PyErr_Format(PyExc_RuntimeError, "Vertex shader compilation failed: %s",
                     shaderc_result_get_error_message(vert_result));
        shaderc_result_release(vert_result);
        shaderc_compile_options_release(options);
        shaderc_compiler_release(compiler);
        return NULL;
    }

    // -- Fragment shader --
    shaderc_compilation_result_t frag_result = shaderc_compile_into_spv(
        compiler, frag_source, strlen(frag_source),
        shaderc_glsl_fragment_shader, "fragment.glsl", "main", options);

    if (shaderc_result_get_compilation_status(frag_result) !=
        shaderc_compilation_status_success) {
        PyErr_Format(PyExc_RuntimeError, "Fragment shader compilation failed: %s",
                     shaderc_result_get_error_message(frag_result));
        shaderc_result_release(frag_result);
        shaderc_result_release(vert_result);
        shaderc_compile_options_release(options);
        shaderc_compiler_release(compiler);
        return NULL;
    }

    // ------------------------------------------------------------------
    //  Create VkShaderModules from compiled SPIR-V.
    // ------------------------------------------------------------------
    VkShaderModule vert_module = create_shader_module(
        ctx,
        reinterpret_cast<const uint32_t*>(shaderc_result_get_bytes(vert_result)),
        shaderc_result_get_length(vert_result));

    VkShaderModule frag_module = create_shader_module(
        ctx,
        reinterpret_cast<const uint32_t*>(shaderc_result_get_bytes(frag_result)),
        shaderc_result_get_length(frag_result));

    // Release shaderc resources now that we have VkShaderModules.
    shaderc_result_release(vert_result);
    shaderc_result_release(frag_result);
    shaderc_compile_options_release(options);
    shaderc_compiler_release(compiler);

    if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
        if (vert_module != VK_NULL_HANDLE)
            vkDestroyShaderModule(ctx->device, vert_module, nullptr);
        if (frag_module != VK_NULL_HANDLE)
            vkDestroyShaderModule(ctx->device, frag_module, nullptr);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create shader module");
        return NULL;
    }

    // ------------------------------------------------------------------
    //  Shader stage create infos.
    // ------------------------------------------------------------------
    VkPipelineShaderStageCreateInfo shader_stages[2] = {};

    shader_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stages[0].module = vert_module;
    shader_stages[0].pName = "main";
    shader_stages[0].pSpecializationInfo = num_constants > 0 ? &spec_info : nullptr;

    shader_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shader_stages[1].module = frag_module;
    shader_stages[1].pName = "main";
    shader_stages[1].pSpecializationInfo = num_constants > 0 ? &spec_info : nullptr;

    // ------------------------------------------------------------------
    //  Vertex input state (matching the Metal backend layout).
    // ------------------------------------------------------------------
    VkVertexInputRate color_rate = instance_colors
        ? VK_VERTEX_INPUT_RATE_INSTANCE
        : VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputBindingDescription bindings[] = {
        { 10, 12, VK_VERTEX_INPUT_RATE_VERTEX },    // position
        { 11, 12, VK_VERTEX_INPUT_RATE_VERTEX },    // texcoord (vec3: xy=2D, xyz=3D)
        { 12, 12, VK_VERTEX_INPUT_RATE_VERTEX },    // normal
        { 13,  4, color_rate },                     // color (RGBA8)
        { 14, 16, VK_VERTEX_INPUT_RATE_INSTANCE },   // shift_and_scale
        { 15, 64, VK_VERTEX_INPUT_RATE_INSTANCE },   // instance matrix (4x float4)
    };

    VkVertexInputAttributeDescription attributes[] = {
        // binding 10: position
        { 0, 10, VK_FORMAT_R32G32B32_SFLOAT, 0 },
        // binding 11: texcoord (vec3 - .xy used for 2D textures, .xyz for 3D)
        { 1, 11, VK_FORMAT_R32G32B32_SFLOAT, 0 },
        // binding 12: normal
        { 2, 12, VK_FORMAT_R32G32B32_SFLOAT, 0 },
        // binding 13: color (normalized byte4)
        { 3, 13, VK_FORMAT_R8G8B8A8_UNORM, 0 },
        // binding 14: shift_and_scale
        { 4, 14, VK_FORMAT_R32G32B32A32_SFLOAT, 0 },
        // binding 15: instance matrix columns
        { 5, 15, VK_FORMAT_R32G32B32A32_SFLOAT,  0 },
        { 6, 15, VK_FORMAT_R32G32B32A32_SFLOAT, 16 },
        { 7, 15, VK_FORMAT_R32G32B32A32_SFLOAT, 32 },
        { 8, 15, VK_FORMAT_R32G32B32A32_SFLOAT, 48 },
    };

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = sizeof(bindings) / sizeof(bindings[0]);
    vertex_input.pVertexBindingDescriptions = bindings;
    vertex_input.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]);
    vertex_input.pVertexAttributeDescriptions = attributes;

    // ------------------------------------------------------------------
    //  Input assembly.
    // ------------------------------------------------------------------
    VkPrimitiveTopology vk_topology;
    switch (topology) {
        case 1:  vk_topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case 2:  vk_topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        default: vk_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
    }

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = vk_topology;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    // ------------------------------------------------------------------
    //  Viewport and scissor (dynamic state).
    // ------------------------------------------------------------------
    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    // ------------------------------------------------------------------
    //  Rasterization state.
    // ------------------------------------------------------------------
    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.depthClampEnable = VK_FALSE;
    rasterization.rasterizerDiscardEnable = VK_FALSE;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = face_culling ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    // The GL->Vulkan conversion negates clip-space Y in the projection
    // matrix (vulkan.py set_projection_matrix), which mirrors triangle
    // orientation in framebuffer space: geometry authored with OpenGL's
    // CCW-front convention rasterizes clockwise here. With CCW front faces
    // back-face culling would cull exactly the wrong side (the volume
    // raycast box rendered its far faces instead of near faces).
    rasterization.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterization.depthBiasEnable = VK_FALSE;
    rasterization.lineWidth = 1.0f;

    // ------------------------------------------------------------------
    //  Multisample state (no MSAA).
    // ------------------------------------------------------------------
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample.sampleShadingEnable = VK_FALSE;

    // ------------------------------------------------------------------
    //  Depth/stencil state.
    // ------------------------------------------------------------------
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = has_depth ? VK_TRUE : VK_FALSE;
    depth_stencil.depthWriteEnable = (has_depth && depth_write) ? VK_TRUE : VK_FALSE;
    // LESS_OR_EQUAL so a depth pre-pass + color pass workflow (which writes
    // identical frag_depth values in both passes) doesn't reject the color
    // pass for fragments that exactly match the prepass-written depth.
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;

    // ------------------------------------------------------------------
    //  Color blend state.
    // ------------------------------------------------------------------
    VkPipelineColorBlendAttachmentState blend_attachments[2] = {};
    uint32_t blend_count = 0;

    if (oit_mode) {
        // OIT accumulation: 2 color attachments.
        // Attachment 0 (accum, RGBA16F): additive blending.
        blend_attachments[0].blendEnable = VK_TRUE;
        blend_attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
        blend_attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_attachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
        blend_attachments[0].colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        // Attachment 1 (revealage, R8): dst *= (1 - src).
        blend_attachments[1].blendEnable = VK_TRUE;
        blend_attachments[1].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        blend_attachments[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        blend_attachments[1].colorBlendOp = VK_BLEND_OP_ADD;
        blend_attachments[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blend_attachments[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_attachments[1].alphaBlendOp = VK_BLEND_OP_ADD;
        blend_attachments[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
        blend_count = 2;
    } else if (!depth_only) {
        VkBlendFactor src_factor = blend_premult
            ? VK_BLEND_FACTOR_ONE
            : VK_BLEND_FACTOR_SRC_ALPHA;
        VkBlendOp blend_op = blend_op_max ? VK_BLEND_OP_MAX : VK_BLEND_OP_ADD;
        blend_attachments[0].blendEnable = VK_TRUE;
        blend_attachments[0].srcColorBlendFactor = src_factor;
        blend_attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_attachments[0].colorBlendOp = blend_op;
        blend_attachments[0].srcAlphaBlendFactor = src_factor;
        blend_attachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_attachments[0].alphaBlendOp = blend_op;
        blend_attachments[0].colorWriteMask = color_write
            ? (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)
            : 0;
        blend_count = 1;
    }

    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.logicOpEnable = VK_FALSE;
    color_blend.attachmentCount = blend_count;
    color_blend.pAttachments = blend_count ? blend_attachments : nullptr;

    // ------------------------------------------------------------------
    //  Dynamic state: viewport and scissor.
    // ------------------------------------------------------------------
    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    // ------------------------------------------------------------------
    //  Create the graphics pipeline.
    // ------------------------------------------------------------------
    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterization;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blend;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = ctx->shared_pipeline_layout;
    pipeline_info.renderPass = depth_only ? ctx->depth_only_render_pass
                             : oit_mode ? ctx->oit_render_pass
                                        : ctx->main_render_pass;
    pipeline_info.subpass = 0;

    VkPipeline pipeline;
    VkResult result = vkCreateGraphicsPipelines(ctx->device, ctx->pipeline_cache,
                                                1, &pipeline_info, nullptr, &pipeline);

    // Clean up shader modules regardless of pipeline creation result.
    vkDestroyShaderModule(ctx->device, vert_module, nullptr);
    vkDestroyShaderModule(ctx->device, frag_module, nullptr);

    if (result != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create graphics pipeline");
        return NULL;
    }

    // ------------------------------------------------------------------
    //  Wrap the pipeline in a VulkanPipeline struct.
    // ------------------------------------------------------------------
    VulkanPipeline* vk_pipeline = new VulkanPipeline();
    vk_pipeline->pipeline = pipeline;
    vk_pipeline->layout = ctx->shared_pipeline_layout;

    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(vk_pipeline));
}

// ---------------------------------------------------------------------------
//  Bind a pipeline for subsequent draw calls.
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_set_pipeline(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr, pipeline_ptr;
    static const char* kwlist[] = {"context", "pipeline", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn"), (char**)kwlist,
            &ctx_ptr, &pipeline_ptr))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanPipeline* pipeline = reinterpret_cast<VulkanPipeline*>(pipeline_ptr);
    VK_TRACE("set_pipeline %p", (void*)pipeline);

    // Recording is only valid while a frame is being recorded.
    if (!ctx->in_frame)
        Py_RETURN_NONE;

    VkCommandBuffer cmd = ctx->frames[ctx->current_frame].command_buffer;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);

    Py_RETURN_NONE;
}
