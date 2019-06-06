class _MyAPI(BundleAPI):

    # previously implemented parts of the class here...

    @staticmethod
    def get_class(class_name):
        # class_name will be a string
        if class_name == "TutorialTool":
            from . import tool
            return tool.TutorialTool
        raise ValueError("Unknown class name '%s'" % class_name)
