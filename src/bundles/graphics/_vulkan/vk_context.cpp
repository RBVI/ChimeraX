#include "vk_context.h"

#include <cstdint>
#include <cstring>

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

VulkanContext::VulkanContext(bool debug) : m_debug{debug} {
    createInstance(debug);
    if (debug) {
        setupDebugMessenger();
    }
}

VulkanContext::~VulkanContext() {
}

VkInstance VulkanContext::getInstance() {
    return instance;
}


void VulkanContext::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
}


bool VulkanContext::isDebugContext() { return m_debug; }

std::vector<std::string> VulkanContext::listGPUs() {
  uint32_t count{0};
  vkEnumeratePhysicalDevices(instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devices(count);
  std::vector<std::string> gpuNames(count);
  vkEnumeratePhysicalDevices(instance, &count, devices.data());
  devices.resize(count);
  gpuNames.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
    gpuNames[i] = deviceProperties.deviceName;
  }
  return gpuNames;
}

bool VulkanContext::checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char *layerName : validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }
  return true;
}


void VulkanContext::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

void VulkanContext::DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

VkResult VulkanContext::CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}




std::vector<const char *> VulkanContext::getRequiredExtensions(bool debug) {
  std::vector<const char *> extensions;
  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef __APPLE__
  extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  extensions.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#elif _WIN32
  extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
  extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif

  if (debug) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

void VulkanContext::createInstance(bool debug) {
  if (debug && !checkValidationLayerSupport()) {
    throw std::runtime_error("validation layers requested, but not available!");
  }

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "UCSF ChimeraX";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 10, 1);
  appInfo.pEngineName = "Placeholder";
  appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  auto extensions = getRequiredExtensions(debug);
  std::cout << "Extensions size: " << extensions.size() << std::endl;
  std::cout << "Extensions required: " << std::endl;
  for (auto ext : extensions) {
    std::cout << "  - " << ext << "\n";
  }
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

#ifdef __APPLE__
  createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  if (debug) {
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
  } else {
    createInfo.enabledLayerCount = 0;

    createInfo.pNext = nullptr;
  }

  // VkInstance* instance;
  if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance!");
  }
}
