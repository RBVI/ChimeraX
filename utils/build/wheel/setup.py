from setuptools import setup, dist, find_namespace_packages
from setuptools.command.install import install

# force setuptools to recognize that this
# is actually a binary distribution
class BinaryDistribution(dist.Distribution):
    def has_ext_modules(foo):
        return True

setup(
    include_package_data = True,
    distclass = BinaryDistribution,
)
