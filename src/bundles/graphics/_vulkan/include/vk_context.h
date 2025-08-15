#ifdef __APPLE__
#include <MoltenVK/mvk_vulkan.h>
#elif _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#else
#define VK_USE_PLATFORM_WAYLAND_KHR
#define VK_USE_PLATFORM_XLIB_KHR
#define VK_USE_PLATFORM_XCB_KHR
#include <wayland-client.h>
#include <vulkan/vulkan.h>
#endif


#include <iostream>
#include <vector>

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

class VulkanContext {
public:
  VulkanContext(bool debug = false);
  ~VulkanContext();
  bool isDebugContext();
  VkInstance getInstance();
  std::vector<std::string> listGPUs();


private:
  VkInstance instance;
  VkDebugUtilsMessengerEXT debugMessenger;
  bool m_debug;
  void createInstance(bool debug = false);
  bool checkValidationLayerSupport();
  void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
  void setupDebugMessenger();
  VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger);
  void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator);
  std::vector<const char *> getRequiredExtensions(bool debug = false);
};
