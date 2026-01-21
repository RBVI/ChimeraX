// metal_heap_manager.cpp
// Implementation of Metal heaps for efficient resource allocation

#include "metal_heap_manager.hpp"
#include "metal_context.hpp"
#include <iostream>

namespace chimerax {
namespace graphics_metal {

MetalHeapManager::MetalHeapManager(MetalContext* context)
    : _context(context)
{
}

MetalHeapManager::~MetalHeapManager()
{
    purgeHeaps();
}

bool MetalHeapManager::initialize()
{
    if (!_context) {
        return false;
    }
    
    // Check if device supports heaps
    id<MTLDevice> device = _context->device();
    if (!device) {
        return false;
    }
    
    return true;
}

id<MTLBuffer> MetalHeapManager::createBuffer(
    size_t length,
    MTLResourceOptions options,
    const std::string& label)
{
    if (!_context) {
        return nil;
    }
    
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    // For very small allocations, use device directly
    if (length < 4 * 1024) { // Less than 4KB
        id<MTLBuffer> buffer = [device newBufferWithLength:length options:options];
        
        if (buffer && !label.empty()) {
            NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
            [buffer setLabel:nsLabel];
        }
        
        return buffer;
    }
    
    // For larger allocations, try to use a heap
    std::lock_guard<std::mutex> lock(_mutex);
    
    HeapSizeCategory category = getCategoryForSize(length);
    id<MTLHeap> heap = findOrCreateHeap(category, HeapType::Buffer, options);
    
    if (heap) {
        // Try to allocate from the heap
        MTLSizeAndAlign sizeAndAlign = [device heapBufferSizeAndAlignWithLength:length
                                                                        options:options];
        
        if (canAllocateFromHeap(heap, sizeAndAlign.size, options)) {
            id<MTLBuffer> buffer = [heap newBufferWithLength:length
                                                     options:options
                                                      offset:0];
            
            if (buffer) {
                if (!label.empty()) {
                    NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
                    [buffer setLabel:nsLabel];
                }
                return buffer;
            }
        }
        
        // If we couldn't allocate from this heap, try to create a new one
        heap = findOrCreateHeap(category, HeapType::Buffer, options);
        
        if (heap) {
            id<MTLBuffer> buffer = [heap newBufferWithLength:length
                                                     options:options
                                                      offset:0];
            
            if (buffer) {
                if (!label.empty()) {
                    NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
                    [buffer setLabel:nsLabel];
                }
                return buffer;
            }
        }
    }
    
    // If heap allocation failed, fall back to device allocation
    id<MTLBuffer> buffer = [device newBufferWithLength:length options:options];
    
    if (buffer && !label.empty()) {
        NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
        [buffer setLabel:nsLabel];
    }
    
    return buffer;
}

id<MTLTexture> MetalHeapManager::createTexture(
    MTLTextureDescriptor* descriptor,
    const std::string& label)
{
    if (!_context || !descriptor) {
        return nil;
    }
    
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    // Calculate the texture size
    MTLSizeAndAlign sizeAndAlign = [device heapTextureSizeAndAlignWithDescriptor:descriptor];
    
    // For small textures, use device directly
    if (sizeAndAlign.size < 256 * 1024) { // Less than 256KB
        id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
        
        if (texture && !label.empty()) {
            NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
            [texture setLabel:nsLabel];
        }
        
        return texture;
    }
    
    // For larger textures, try to use a heap
    std::lock_guard<std::mutex> lock(_mutex);
    
    HeapSizeCategory category = getCategoryForSize(sizeAndAlign.size);
    MTLResourceOptions options = descriptor.storageMode << MTLResourceStorageModeShift;
    id<MTLHeap> heap = findOrCreateHeap(category, HeapType::Texture, options);
    
    if (heap) {
        // Try to allocate from the heap
        if (canAllocateFromHeap(heap, sizeAndAlign.size, options)) {
            id<MTLTexture> texture = [heap newTextureWithDescriptor:descriptor
                                                             offset:0];
            
            if (texture) {
                if (!label.empty()) {
                    NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
                    [texture setLabel:nsLabel];
                }
                return texture;
            }
        }
        
        // If we couldn't allocate from this heap, try to create a new one
        heap = findOrCreateHeap(category, HeapType::Texture, options);
        
        if (heap) {
            id<MTLTexture> texture = [heap newTextureWithDescriptor:descriptor
                                                             offset:0];
            
            if (texture) {
                if (!label.empty()) {
                    NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
                    [texture setLabel:nsLabel];
                }
                return texture;
            }
        }
    }
    
    // If heap allocation failed, fall back to device allocation
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    
    if (texture && !label.empty()) {
        NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
        [texture setLabel:nsLabel];
    }
    
    return texture;
}

id<MTLHeap> MetalHeapManager::createHeap(
    const HeapDescriptor& descriptor,
    size_t size)
{
    if (!_context) {
        return nil;
    }
    
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    // Create heap descriptor
    MTLHeapDescriptor* heapDesc = [[MTLHeapDescriptor alloc] init];
    heapDesc.size = size;
    heapDesc.storageMode = descriptor.storageMode;
    heapDesc.cpuCacheMode = descriptor.cacheMode;
    
    if (@available(macOS 10.15, *)) {
        heapDesc.hazardTrackingMode = descriptor.hazardTrackingEnabled ? 
            MTLHazardTrackingModeTracked : MTLHazardTrackingModeUntracked;
    }
    
    // Create the heap
    id<MTLHeap> heap = [device newHeapWithDescriptor:heapDesc];
    [heapDesc release];
    
    if (!heap) {
        std::cerr << "MetalHeapManager::createHeap: Failed to create heap of size " 
                  << size << std::endl;
        return nil;
    }
    
    // Set label if provided
    if (!descriptor.name.empty()) {
        NSString* nsLabel = [NSString stringWithUTF8String:descriptor.name.c_str()];
        [heap setLabel:nsLabel];
    }
    
    // Store the heap
    std::lock_guard<std::mutex> lock(_mutex);
    _customHeaps.push_back(heap);
    
    return heap;
}

id<MTLBuffer> MetalHeapManager::createBufferFromHeap(
    id<MTLHeap> heap,
    size_t length,
    const std::string& label)
{
    if (!heap) {
        return nil;
    }
    
    // Create buffer from heap
    id<MTLBuffer> buffer = [heap newBufferWithLength:length
                                             options:[heap resourceOptions]
                                              offset:0];
    
    if (!buffer) {
        std::cerr << "MetalHeapManager::createBufferFromHeap: Failed to create buffer of size " 
                  << length << std::endl;
        return nil;
    }
    
    // Set label if provided
    if (!label.empty()) {
        NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
        [buffer setLabel:nsLabel];
    }
    
    return buffer;
}

id<MTLTexture> MetalHeapManager::createTextureFromHeap(
    id<MTLHeap> heap,
    MTLTextureDescriptor* descriptor,
    const std::string& label)
{
    if (!heap || !descriptor) {
        return nil;
    }
    
    // Create texture from heap
    id<MTLTexture> texture = [heap newTextureWithDescriptor:descriptor offset:0];
    
    if (!texture) {
        std::cerr << "MetalHeapManager::createTextureFromHeap: Failed to create texture" << std::endl;
        return nil;
    }
    
    // Set label if provided
    if (!label.empty()) {
        NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
        [texture setLabel:nsLabel];
    }
    
    return texture;
}

void MetalHeapManager::purgeHeaps()
{
    std::lock_guard<std::mutex> lock(_mutex);
    
    // Release buffer heaps
    for (auto& pair : _bufferHeaps) {
        for (id<MTLHeap> heap : pair.second) {
            [heap release];
        }
        pair.second.clear();
    }
    
    // Release texture heaps
    for (auto& pair : _textureHeaps) {
        for (id<MTLHeap> heap : pair.second) {
            [heap release];
        }
        pair.second.clear();
    }
    
    // Release custom heaps
    for (id<MTLHeap> heap : _customHeaps) {
        [heap release];
    }
    _customHeaps.clear();
}

HeapSizeCategory MetalHeapManager::getCategoryForSize(size_t size)
{
    if (size < SMALL_RESOURCE_THRESHOLD) {
        return HeapSizeCategory::Small;
    } else if (size < MEDIUM_RESOURCE_THRESHOLD) {
        return HeapSizeCategory::Medium;
    } else if (size < LARGE_RESOURCE_THRESHOLD) {
        return HeapSizeCategory::Large;
    } else {
        return HeapSizeCategory::Huge;
    }
}

id<MTLHeap> MetalHeapManager::findOrCreateHeap(
    HeapSizeCategory category,
    HeapType type,
    MTLResourceOptions options)
{
    if (!_context) {
        return nil;
    }
    
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    // Determine which heap map to use
    std::unordered_map<HeapSizeCategory, std::vector<id<MTLHeap>>>& heapMap =
        (type == HeapType::Buffer) ? _bufferHeaps : _textureHeaps;
    
    // Find a compatible heap with enough space
    auto it = heapMap.find(category);
    if (it != heapMap.end()) {
        for (id<MTLHeap> heap : it->second) {
            MTLStorageMode heapStorageMode = ([heap storageMode] & MTLStorageModeShared);
            MTLStorageMode requestedStorageMode = ((options >> MTLResourceStorageModeShift) & 0x3);
            
            // Check if storage modes are compatible
            if (heapStorageMode == requestedStorageMode) {
                // Check if the heap has enough space
                if (canAllocateFromHeap(heap, 0, options)) {
                    return heap;
                }
            }
        }
    }
    
    // Create a new heap
    size_t heapSize;
    switch (category) {
        case HeapSizeCategory::Small:
            heapSize = SMALL_HEAP_SIZE;
            break;
        case HeapSizeCategory::Medium:
            heapSize = MEDIUM_HEAP_SIZE;
            break;
        case HeapSizeCategory::Large:
            heapSize = LARGE_HEAP_SIZE;
            break;
        case HeapSizeCategory::Huge:
            // For huge resources, we create a dedicated heap
            return nil;
    }
    
    // Create heap descriptor
    MTLHeapDescriptor* heapDesc = [[MTLHeapDescriptor alloc] init];
    heapDesc.size = heapSize;
    heapDesc.storageMode = (MTLStorageMode)((options >> MTLResourceStorageModeShift) & 0x3);
    heapDesc.cpuCacheMode = (MTLCPUCacheMode)((options >> MTLResourceCPUCacheModeShift) & 0x3);
    
    if (@available(macOS 10.15, *)) {
        heapDesc.hazardTrackingMode = MTLHazardTrackingModeTracked;
    }
    
    // Set the resource options based on type
    if (type == HeapType::Texture) {
        heapDesc.type = MTLHeapTypeTexture;
    } else {
        heapDesc.type = MTLHeapTypeAutomatic;
    }
    
    // Create the heap
    id<MTLHeap> heap = [device newHeapWithDescriptor:heapDesc];
    [heapDesc release];
    
    if (!heap) {
        std::cerr << "MetalHeapManager::findOrCreateHeap: Failed to create heap of size " 
                  << heapSize << std::endl;
        return nil;
    }
    
    // Set a label for the heap
    std::string heapTypeStr = (type == HeapType::Buffer) ? "Buffer" : "Texture";
    std::string categorySizeStr;
    switch (category) {
        case HeapSizeCategory::Small:
            categorySizeStr = "Small";
            break;
        case HeapSizeCategory::Medium:
            categorySizeStr = "Medium";
            break;
        case HeapSizeCategory::Large:
            categorySizeStr = "Large";
            break;
        case HeapSizeCategory::Huge:
            categorySizeStr = "Huge";
            break;
    }
    
    std::string heapLabel = heapTypeStr + " " + categorySizeStr + " Heap";
    NSString* nsLabel = [NSString stringWithUTF8String:heapLabel.c_str()];
    [heap setLabel:nsLabel];
    
    // Store the heap
    heapMap[category].push_back(heap);
    
    return heap;
}

bool MetalHeapManager::canAllocateFromHeap(id<MTLHeap> heap, size_t size, MTLResourceOptions options)
{
    if (!heap) {
        return false;
    }
    
    // Check if storage modes are compatible
    MTLStorageMode heapStorageMode = [heap storageMode];
    MTLStorageMode requestedStorageMode = (MTLStorageMode)((options >> MTLResourceStorageModeShift) & 0x3);
    
    if (heapStorageMode != requestedStorageMode) {
        return false;
    }
    
    // Check if there's enough space
    size_t currentAllocatedSize = [heap usedSize];
    size_t heapSize = [heap size];
    
    if (size > 0) {
        return (currentAllocatedSize + size <= heapSize);
    }
    
    // If size is 0, just check if the heap is not completely full
    return (currentAllocatedSize < heapSize);
}

} // namespace graphics_metal
} // namespace chimerax
