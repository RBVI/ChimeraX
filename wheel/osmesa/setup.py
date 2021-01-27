#
# For building a wheel containing the osmesa library for off-screen rendering.
#
#   python3 setup.py bdist_wheel
#
# Tested on macOS 10.15.7.
#
from setuptools import setup, dist
from setuptools.command.install import install

# force setuptools to recognize that this is
# actually a binary distribution
class BinaryDistribution(dist.Distribution):
    def has_ext_modules(foo):
        return True

setup(
    # this package is called mymodule
    name = 'osmesa',

    # Include all subpackages recursively
    packages = ['osmesa'],

    # Include shared libraries and data files
    package_data = {'osmesa': ['libOSMesa.dylib']},

    # Brief description
    description = "Provides off-screen mesa library for OpenGL rendering with PyOpenGL",

    # Long description
    long_description = """\
OpenGL graphics rendering in general requires a window on a display 
because the graphics drivers are tied to the windowing system.  The Mesa project
provides an OpenGL library called OSMesa (off-screen mesa) that can render
without a windowing system or a display.  It does not use hardware acceleration
but does use LLVM compiler tools to accelerate shader programs.
This Python module provides the OSMesa library that can be used by PyOpenGL.
""",

    # See class BinaryDistribution that was defined earlier
    distclass = BinaryDistribution,

    version = '1.0',
    url = 'https://github.com/RBVI/ChimeraX',
    author = 'UCSF Computer Graphics Lab',
    author_email = 'chimera-programmers@cgl.ucsf.edu',
    
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.8',
    ],
)
