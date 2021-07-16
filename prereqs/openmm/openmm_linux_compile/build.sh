#!/bin/bash
set -e

cd /tmp
git clone https://github.com/openmm/openmm.git
cd openmm
git checkout 7.5_branch
cd ..


# CFLAGS
export MINIMAL_CFLAGS="-g -O3"
export CFLAGS="$MINIMAL_CFLAGS"
export CXXFLAGS="$MINIMAL_CFLAGS"
export LDFLAGS="$LDPATHFLAGS"

INSTALL=`pwd`/install
if [ -e $INSTALL ]; then
    rm -rf $INSTALL
fi

CMAKE_FLAGS="-DCMAKE_INSTALL_PREFIX=$INSTALL"

# Don't build tests
CMAKE_FLAGS+=" -DBUILD_TESTING=OFF"

# Ensure we build a release
CMAKE_FLAGS+=" -DCMAKE_BUILD_TYPE=Release"

# setting the rpath so that libOpenMMPME.so finds the right libfftw3
#CMAKE_FLAGS+=" -DCMAKE_INSTALL_RPATH=.."
# Use NVIDIA CUDA 
CMAKE_FLAGS+=" -DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so"
CMAKE_FLAGS+=" -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc"
CMAKE_FLAGS+=" -DCUDA_SDK_ROOT_DIR=/usr/local/cuda/"
CMAKE_FLAGS+=" -DCUDA_TOOLKIT_INCLUDE=/usr/local/cuda/include"
CMAKE_FLAGS+=" -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/"
# So that FindCuda correctly finds libcuda.os
CMAKE_FLAGS+=" -DCMAKE_LIBRARY_PATH=/usr/local/cuda/compat/"
echo "CMAKE_FLAGS = $CMAKE_FLAGS"
# Use AMD APP SDK 3.0
#CMAKE_FLAGS+=" -DOPENCL_INCLUDE_DIR=/opt/AMDAPPSDK-3.0/include/"
CMAKE_FLAGS+=" -DOPENCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so"
# Generate API docs
CMAKE_FLAGS+=" -DOPENMM_GENERATE_API_DOCS=ON"
# Set location for FFTW3
CMAKE_FLAGS+=" -DFFTW_INCLUDES=/usr/include"
CMAKE_FLAGS+=" -DFFTW_LIBRARY=/usr/lib64/libfftw3f.so"
CMAKE_FLAGS+=" -DFFTW_THREADS_LIBRARY=/usr/lib64/libfftw3f_threads.so"
# Necessary to find GL headers
# CMAKE_FLAGS+=" -DCMAKE_CXX_FLAGS_RELEASE=-I/usr/include/nvidia/"

if [ -e build ]; then
  rm -rf build
fi
mkdir build
cd build
cmake3 ../openmm $CMAKE_FLAGS
make -j12 all install

export CFLAGS="$MINIMAL_CFLAGS"

export CXXFLAGS="$MINIMAL_CFLAGS"
export LDFLAGS="$LDPATHFLAGS"

make -j20 PythonInstall

make install

export OPENMM_INCLUDE_PATH=$INSTALL/include
export OPENMM_LIB_PATH=$INSTALL/lib
PYTHON_VER=$(python3 -c 'import sys; print("%s.%s" % sys.version_info[0:2])')
(cd python && python3 setup.py install --install-lib $INSTALL/lib/python${PYTHON_VER}/site-packages)
find $INSTALL/lib -name \*.so -exec chrpath -d {} \;

cd ..

(cd install/lib && for so in plugins/*.so; do ln -s $so .; done && strip *.so)

OPENMM_VER=$(python3 -c 'vtxt = open("build/python/simtk/openmm/version.py").read(); exec(vtxt); print(short_version)')
PYTHON_VER_NODOT=$(python3 -c 'import sys; print("%s%s" % sys.version_info[0:2])')
CUDA_VER=$(nvcc --version | awk '$4 == "release" {print $5}' | sed 's/[,.]//g')
TAR_FILE=openmm-${OPENMM_VER}-linux-py${PYTHON_VER_NODOT}_cuda${CUDA_VER}_1.tar.bz2
tar -jcf ${TAR_FILE} -C install .

cp ${TAR_FILE} /host
cd /tmp
rm -fr openmm
rm -fr build
rm -fr $INSTALL

