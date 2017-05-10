Build notes for OpenMM
----------------------

On Mac and Linux using the OpenMM 7.1.1 build from https://anaconda.org/omnia/openmm/files.

Formerly used the OpenMM 7.0.1 binary installers on Mac and Linux that include precompiled C++ libraries.
The SWIG Python wrapper is compiled by the install.sh script on our machine.  The C++ ABI used by
our compiler must match the ABI used in the OpenMM C++ libraries.  Specifically on Linux the OpenMM
libraries use a G++ version 4.9(?) compiler and G++ 5.x uses a different ABI.

On Windows the OpenMM 7.0.1 binary installers only support Python 3.3 and 3.4 and use Visual Studio 10
while we use Visual Studio Community 2015 (not sure if these have compatible ABIs).  So instead we use
the OpenMM conda install on Windows for Python 3.5 and copy the resulting binaries to a tar file to
use in our builds.  These will need to be updated when we update to Python 3.6.

