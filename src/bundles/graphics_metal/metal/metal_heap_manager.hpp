// metal_heap_manager.hpp
// Manages Metal heaps for efficient resource allocation

#pragma once

#import <Metal/Metal.h>
#include <unordered_map>
#include <memory>
#include <vector>
#include <mutex>
#include <string>

namespace chimerax {
namespace graphics_metal {

// Forward declarations
class MetalContext;

/**
 * Size categories for different heaps
 */
enum class HeapSizeCategory {
    Small,   // For small resources (< 1 MB)
    Medium,  // For medium resources (1-32 MB)
    Large,   // For large resources (32-256 MB)
    Huge     // For huge resources (> 256 MB)
};

/**
 * Types of heaps for different resource uses
 */
enum class HeapType {
    Buffer,  // For buffers
    Texture  // For textures
};

/**
 * Heap descriptor for resource allocation
 */
struct HeapDescriptor {
    HeapSizeCategory sizeCategory;
    HeapType type;
    MTLStorageMode storageMode;
    MTLCPUCacheMode cacheMode;
    bool hazardTrackingEnabled;
    std::string name;
};

/**
 * Manages Metal heaps for efficient resource allocation
 */
class MetalHeapManager {
public:
    MetalHeapManager(MetalContext* context);
    ~MetalHeapManager();
    
    // Initialization
    bool initialize();
    
    // Create a buffer from heap
    id<MTLBuffer> createBuffer(
        size_t length,
        MTLResourceOptions options = MTLResourceStorageModeShared,
        const std::string& label = "");
    
    // Create a texture from heap
    id<MTLTexture> createTexture(
        MTLTextureDescriptor* descriptor,
        const std::string& label = "");
    
    // Create a heap with custom parameters
    id<MTLHeap> createHeap(
        const HeapDescriptor& descriptor,
        size_t size);
    
    // Allocate a resource from a specific heap
    id<MTLBuffer> createBufferFromHeap(
        id<MTLHeap> heap,
        size_t length,
        const std::string& label = "");
        
    id<MTLTexture> createTextureFromHeap(
        id<MTLHeap> heap,
        MTLTextureDescriptor* descriptor,
        const std::string& label = "");
    
    // Purge all heaps
    void purgeHeaps();
    
private:
    MetalContext* _context;
    
    // Maps of heaps for different size categories and types
    std::unordered_map<HeapSizeCategory, std::vector<id<MTLHeap>>> _bufferHeaps;
    std::unordered_map<HeapSizeCategory, std::vector<id<MTLHeap>>> _textureHeaps;
    
    // Custom heaps
    std::vector<id<MTLHeap>> _customHeaps;
    
    // Heap size thresholds
    static const size_t SMALL_HEAP_SIZE = 16 * 1024 * 1024;  // 16 MB
    static const size_t MEDIUM_HEAP_SIZE = 64 * 1024 * 1024; // 64 MB
    static const size_t LARGE_HEAP_SIZE = 256 * 1024 * 1024; // 256 MB
    
    // Resource size thresholds
    static const size_t SMALL_RESOURCE_THRESHOLD = 1 * 1024 * 1024;    // 1 MB
    static const size_t MEDIUM_RESOURCE_THRESHOLD = 32 * 1024 * 1024;  // 32 MB
    static const size_t LARGE_RESOURCE_THRESHOLD = 256 * 1024 * 1024;  // 256 MB
    
    // Mutex for thread safety
    std::mutex _mutex;
    
    // Helper methods
    HeapSizeCategory getCategoryForSize(size_t size);
    id<MTLHeap> findOrCreateHeap(HeapSizeCategory category, HeapType type, MTLResourceOptions options);
    bool canAllocateFromHeap(id<MTLHeap> heap, size_t size, MTLResourceOptions options);
};

} // namespace graphics_metal
} // namespace chimerax
