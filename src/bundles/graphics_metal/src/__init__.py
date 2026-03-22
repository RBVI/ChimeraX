"""
Metal-accelerated graphics backend for UCSF ChimeraX.
"""

from .metal_graphics import MetalBackend, is_metal_supported

from chimerax.core.toolshed import BundleAPI


class _MetalBundleAPI(BundleAPI):
    @staticmethod
    def initialize(session, bundle_info):
        from .custom_init import init
        init(session, bundle_info)

    @staticmethod
    def finish(session, bundle_info):
        from .custom_init import finish
        finish(session, bundle_info)


bundle_api = _MetalBundleAPI()
