#
# This setup.py file creates a ChimeraX wheel for installation in a standard Python
# distribution.
#
#   python3 setup.py bdist_wheel
#
# This allows using headless ChimeraX on servers.  Images can be saved
# using osmesa and osmesa/setup.py will make a wheel for that library.  Only a few of
# the many dependencies needed by ChimeraX are listed (below see install_requires).
# It might be worth making various optional dependency sets.  Currently it is the
# responsibility of the user to install additional dependencies based on their use case.
#
# Tested this on macOS 10.15.7.
#
from setuptools import setup, dist, find_namespace_packages
from setuptools.command.install import install

# force setuptools to recognize that this is
# actually a binary distribution
class BinaryDistribution(dist.Distribution):
    def has_ext_modules(foo):
        return True

# Use README.md as long_description
import os.path
dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(dir, 'README.md')) as f:
    long_description = f.read()

setup(
    # this package is called mymodule
    name = 'chimerax',

    # Include all subpackages recursively
    packages = find_namespace_packages(),

    # Include shared libraries and data files
    include_package_data = True,

    # Brief description
    description = "Analysis and visualization of molecular structures and 3D microscopy",

    # Long description
    long_description = long_description,
    long_description_content_type = "text/markdown",

    # See class BinaryDistribution that was defined earlier
    distclass = BinaryDistribution,

    version = '1.6.0',
    url = 'https://github.com/RBVI/ChimeraX',
    author = 'UCSF Computer Graphics Lab',
    author_email = 'chimera-programmers@cgl.ucsf.edu',
    license_files = ['LICENSE.md'],
    
    install_requires = [
        'numpy',                # For atom coordinate arrays and microscopy images
        'tinyarray',            # For atom coordinates
        'html2text',            # To convert log output from html to plain text
        #'PyOpenGL',             # Render images
        #'PyOpenGL_accelerate',  # Render images
        #'Pillow',               # Render images
        'sortedcontainers',     # Color code uses this
        'packaging',            # is_daily_build() is using this.  TODO: Can remove this dependency?
    ],
    
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'License :: Other/Proprietary License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.9',
    ],
)
