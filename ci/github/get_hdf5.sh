#!/bin/bash
# inspired by https://github.com/h5py/h5py/blob/master/ci/get_hdf5_if_needed.sh

set -e -x

EXTRA_MPI_FLAGS=''
if [ -z ${HDF5_MPI+x} ]; then
    echo "Building serial"
else
    echo "Building with MPI"
    EXTRA_MPI_FLAGS="--enable-parallel --enable-shared"
fi

export LD_LIBRARY_PATH="$HDF5_DIR/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$HDF5_DIR/lib/pkgconfig:${PKG_CONFIG_PATH}"


LZO_VERSION="2.10"
SNAPPY_VERSION="1.1.9"
ZSTD_VERSION="1.5.2"
LZ4_VERSION="1.9.3"
BZIP_VERSION="1.0.8"
ZLIB_VERSION="1.2.12"


echo "building HDF5"
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install automake cmake pkg-config

    CMAKE_ARCHES="$CIBW_ARCHS"
    ARCH_ARGS="-arch $CIBW_ARCHS"
    NPROC=$(sysctl -n hw.ncpu)
    pushd /tmp

    # lzo
    curl -sLO https://www.oberhumer.com/opensource/lzo/download/lzo-$LZO_VERSION.tar.gz
    tar xzf lzo-$LZO_VERSION.tar.gz
    pushd lzo-$LZO_VERSION
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX="$HDF5_DIR" -DENABLE_SHARED:bool=on -DCMAKE_OSX_ARCHITECTURES="$CMAKE_ARCHES" ../
    make
    make install
    popd

    # zstd
    curl -sLO https://github.com/facebook/zstd/releases/download/v$ZSTD_VERSION/zstd-$ZSTD_VERSION.tar.gz
    tar xzf zstd-$ZSTD_VERSION.tar.gz
    pushd zstd-$ZSTD_VERSION
    cd build/cmake
    cmake -DCMAKE_INSTALL_PREFIX="$HDF5_DIR" -DENABLE_SHARED:bool=on -DCMAKE_OSX_ARCHITECTURES="$CMAKE_ARCHES"
    make
    make install
    popd

    CFLAGS_ORIG="$CFLAGS"
    export CFLAGS="$CFLAGS $ARCH_ARGS"
    export CPPFLAGS="$CPPFLAGS $ARCH_ARGS"
    export CXXFLAGS="$CXXFLAGS $ARCH_ARGS"
    export CC="/usr/bin/clang"
    export CXX="/usr/bin/clang"
    export cc=$CC

    # lz4
    curl -sLO https://github.com/lz4/lz4/archive/refs/tags/v$LZ4_VERSION.tar.gz
    tar xzf v$LZ4_VERSION.tar.gz
    pushd lz4-$LZ4_VERSION
    make install PREFIX="$HDF5_DIR"
    popd

    # bzip2
    curl -sLO https://gitlab.com/bzip2/bzip2/-/archive/bzip2-$BZIP_VERSION/bzip2-bzip2-$BZIP_VERSION.tar.gz
    tar xzf bzip2-bzip2-$BZIP_VERSION.tar.gz
    pushd bzip2-bzip2-$BZIP_VERSION
    cat << EOF >> Makefile

libbz2.dylib: \$(OBJS)
	\$(CC) \$(LDFLAGS) -shared -Wl,-install_name -Wl,libbz2.dylib -o libbz2.$BZIP_VERSION.dylib \$(OBJS)
	cp libbz2.$BZIP_VERSION.dylib \${PREFIX}/lib/
	ln -s libbz2.$BZIP_VERSION.dylib \${PREFIX}/lib/libbz2.1.0.dylib
	ln -s libbz2.$BZIP_VERSION.dylib \${PREFIX}/lib/libbz2.dylib

EOF
    sed -i "" "s/CFLAGS=-Wall/CFLAGS=-fPIC -Wall/g" Makefile
    sed -i "" "s/all: libbz2.a/all: libbz2.dylib libbz2.a/g" Makefile
    make install PREFIX="$HDF5_DIR"
    popd

    # zlib
    curl -sLO https://zlib.net/fossils/zlib-$ZLIB_VERSION.tar.gz
    tar xzf zlib-$ZLIB_VERSION.tar.gz
    pushd zlib-$ZLIB_VERSION
    ./configure --prefix="$HDF5_DIR"
    make
    make install
    popd

    popd

    # snappy
    git clone https://github.com/google/snappy.git --branch $SNAPPY_VERSION --depth 1
    pushd snappy
    git submodule update --init
    mkdir build
    cd build
    CC= CXX= CPPFLAGS= CXXFLAGS= CFLAGS="$CFLAGS_ORIG" cmake -DCMAKE_INSTALL_PREFIX="$HDF5_DIR" -DRUN_HAVE_STD_REGEX=0 \
        -DRUN_HAVE_POSIX_REGEX=0 -DENABLE_SHARED:bool=on -DCMAKE_OSX_ARCHITECTURES="$CMAKE_ARCHES" ../
    CC= CXX= CPPFLAGS= CXXFLAGS= CFLAGS="$CFLAGS_ORIG" make
    make install
    popd
else
    yum -y update
    yum install -y zlib-devel bzip2-devel lzo-devel
    NPROC=$(nproc)
fi

pushd /tmp

#                                   Remove trailing .*, to get e.g. '1.12' ↓
curl -fsSLO "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"
tar -xzvf "hdf5-$HDF5_VERSION.tar.gz"
pushd "hdf5-$HDF5_VERSION"

if [[ "$OSTYPE" == "darwin"* && "$CIBW_ARCHS" = "arm64"  ]]; then
    # from https://github.com/conda-forge/hdf5-feedstock/commit/2cb83b63965985fa8795b0a13150bf0fd2525ebd
    export ac_cv_sizeof_long_double=8
    export hdf5_cv_ldouble_to_long_special=no
    export hdf5_cv_long_to_ldouble_special=no
    export hdf5_cv_ldouble_to_llong_accurate=yes
    export hdf5_cv_llong_to_ldouble_correct=yes
    export hdf5_cv_disable_some_ldouble_conv=no
    export hdf5_cv_system_scope_threads=yes
    export hdf5_cv_printf_ll="l"
    export PAC_FC_MAX_REAL_PRECISION=15
    export PAC_C_MAX_REAL_PRECISION=17
    export PAC_FC_ALL_INTEGER_KINDS="{1,2,4,8,16}"
    export PAC_FC_ALL_REAL_KINDS="{4,8}"
    export H5CONFIG_F_NUM_RKIND="INTEGER, PARAMETER :: num_rkinds = 2"
    export H5CONFIG_F_NUM_IKIND="INTEGER, PARAMETER :: num_ikinds = 5"
    export H5CONFIG_F_RKIND="INTEGER, DIMENSION(1:num_rkinds) :: rkind = (/4,8/)"
    export H5CONFIG_F_IKIND="INTEGER, DIMENSION(1:num_ikinds) :: ikind = (/1,2,4,8,16/)"
    export PAC_FORTRAN_NATIVE_INTEGER_SIZEOF="                    4"
    export PAC_FORTRAN_NATIVE_INTEGER_KIND="           4"
    export PAC_FORTRAN_NATIVE_REAL_SIZEOF="                    4"
    export PAC_FORTRAN_NATIVE_REAL_KIND="           4"
    export PAC_FORTRAN_NATIVE_DOUBLE_SIZEOF="                    8"
    export PAC_FORTRAN_NATIVE_DOUBLE_KIND="           8"
    export PAC_FORTRAN_NUM_INTEGER_KINDS="5"
    export PAC_FC_ALL_REAL_KINDS_SIZEOF="{4,8}"
    export PAC_FC_ALL_INTEGER_KINDS_SIZEOF="{1,2,4,8,16}"

    curl -sLO https://github.com/conda-forge/hdf5-feedstock/raw/2cb83b63965985fa8795b0a13150bf0fd2525ebd/recipe/patches/osx_cross_configure.patch
    curl -sLO https://github.com/conda-forge/hdf5-feedstock/raw/2cb83b63965985fa8795b0a13150bf0fd2525ebd/recipe/patches/osx_cross_fortran_src_makefile.patch
    curl -sLO https://github.com/conda-forge/hdf5-feedstock/raw/2cb83b63965985fa8795b0a13150bf0fd2525ebd/recipe/patches/osx_cross_hl_fortran_src_makefile.patch
    curl -sLO https://github.com/conda-forge/hdf5-feedstock/raw/2cb83b63965985fa8795b0a13150bf0fd2525ebd/recipe/patches/osx_cross_src_makefile.patch
    patch -p0 < osx_cross_configure.patch
    patch -p0 < osx_cross_fortran_src_makefile.patch
    patch -p0 < osx_cross_hl_fortran_src_makefile.patch
    patch -p0 < osx_cross_src_makefile.patch

    ./configure --prefix="$HDF5_DIR" --with-zlib="$HDF5_DIR" "$EXTRA_MPI_FLAGS" --enable-build-mode=production \
        --host=aarch64-apple-darwin --enable-tests=no

    mkdir -p native-build/bin
    pushd native-build/bin
    CFLAGS= $CC ../../src/H5detect.c -I ../../src/ -o H5detect
    CFLAGS= $CC ../../src/H5make_libsettings.c -I ../../src/ -o H5make_libsettings
    popd
    export PATH=$(pwd)/native-build/bin:$PATH
else
    ./configure --prefix="$HDF5_DIR" --with-zlib="$HDF5_DIR" "$EXTRA_MPI_FLAGS" --enable-build-mode=production
fi
make -j "$NPROC"
make install

file "$HDF5_DIR"/lib/*

popd
popd
