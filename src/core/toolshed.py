# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
toolshed: keep track of tools
=============================

TODO: placeholder for actual code
"""

class ToolShed:

    def __init__(self, app_dirs):
        self.app_dirs = app_dirs

    def startup_tools(self, session):
        return []

def init(app_dirs):
    return ToolShed(app_dirs)
