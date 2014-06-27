from distutils.core import setup, Extension
import numpy

proxy_ext = Extension("_proxy",
        sources=["proxy.c"],
        include_dirs=[numpy.get_include()]
)

setup(name="_proxy", version="1.0",
              ext_modules=[proxy_ext])
