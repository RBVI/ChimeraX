Build notes for OpenMM
----------------------

The 7.5 builds came from https://anaconda.org/omnia-dev/openmm/files.

On Linux, we need OpenMM built with the same C++ compiler as the rest
of ChimeraX so that ISOLDE can link to both ChimeraX and OpenMM libraries.
At this time, that is CentOS 7 with devtoolset-9.  And it is
devtoolset-9 so OpenMM can use newer C++ constructs.  See ticket #3756.
In the openmm_build3.zip file with a build script and a singularity
definition file that is used to create the .tar.gz file.  And that
.tar.gz is placed on the chimerax web site for downloading.

Old Build notes for OpenMM
--------------------------

On Mac, Linux and Windows using the OpenMM builds from
https://anaconda.org/omnia/openmm/files -- Version: 7.3.0 Label: main.

The files are renamed from PLATFORM/openmm-VERSION-PYVER.tar.bz2 to
	openemm-VERSION-PLATFORM-PYVER.tar.bz where PLATFORM changes
	{ win-64: windows, osx-64: mac, linux-64: linux }

Even Older Build notes for OpenMM
---------------------------------

Formerly used the OpenMM 7.0.1 binary installers on Mac and Linux that
include precompiled C++ libraries.  The SWIG Python wrapper is compiled by
the install.sh script on our machine.  The C++ ABI used by our compiler
must match the ABI used in the OpenMM C++ libraries.  Specifically on
Linux the OpenMM libraries use a G++ version 4.9(?) compiler and G++
5.x uses a different ABI.

On Windows the OpenMM 7.0.1 binary installers only support Python 3.3 and
3.4 and use Visual Studio 10 while we use Visual Studio Community 2015
(not sure if these have compatible ABIs).  So instead we use the OpenMM
conda install on Windows for Python 3.5 and copy the resulting binaries
to a tar file to use in our builds.  These will need to be updated when
we update to Python 3.6.

