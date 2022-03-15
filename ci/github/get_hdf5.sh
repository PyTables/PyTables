#!/bin/bash
# inspired by https://github.com/h5py/h5py/blob/master/ci/get_hdf5_if_needed.sh

set -e -x

extra_arch_flags=()
EXTRA_CMAKE_MPI_FLAGS=""
EXTRA_MPI_FLAGS=''
HDF5_HOST=""
if [ -z ${HDF5_MPI+x} ]; then
    echo "Building serial"
else
    echo "Building with MPI"
    EXTRA_CMAKE_MPI_FLAGS="-DHDF5_ENABLE_PARALLEL:bool=on"
    EXTRA_MPI_FLAGS="--enable-parallel --enable-shared"
fi

export LD_LIBRARY_PATH="$HDF5_DIR/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$HDF5_DIR/lib/pkgconfig:${PKG_CONFIG_PATH}"

MINOR_V=${HDF5_VERSION#*.}
MINOR_V=${MINOR_V%.*}
MAJOR_V=${HDF5_VERSION/%.*.*}

LZO_VERSION="2.10"
SNAPPY_VERSION="1.1.9"
ZSTD_VERSION="1.5.2"
LZ4_VERSION="1.9.3"
BZIP_VERSION="1.0.8"
ZLIB_VERSION="1.2.11"


echo "building HDF5"
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install automake cmake pkg-config

    if [[ "$CIBW_ARCHS" = "universal2" ]]; then
        CMAKE_ARCHES="x86_64;arm64"
        ARCH_ARGS="-arch x86_64 -arch arm64"

        # universal binaries is only supported for v1.14+
        if [[ $MAJOR_V -eq 1 && $MINOR_V -lt 14 ]]; then
            echo "MACOS universal wheels can only be built with HDF5 version 1.14+" 1>&2
            exit 1
        fi
    else
        HDF5_HOST="--host=$CIBW_ARCHS-darwin"
        CMAKE_ARCHES="$CIBW_ARCHS"
        ARCH_ARGS="-arch $CIBW_ARCHS"
    fi
    extra_arch_flags=("-DCMAKE_OSX_ARCHITECTURES=$CMAKE_ARCHES")
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

    # snappy
    git clone https://github.com/google/snappy.git --branch $SNAPPY_VERSION --depth 1
    pushd snappy
    git submodule update --init
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX="$HDF5_DIR" -DRUN_HAVE_STD_REGEX=0 -DRUN_HAVE_POSIX_REGEX=0 -DENABLE_SHARED:bool=on \
        -DCMAKE_OSX_ARCHITECTURES="$CMAKE_ARCHES" ../
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

    export CFLAGS="$CFLAGS $ARCH_ARGS"
    export CPPFLAGS="$CPPFLAGS $ARCH_ARGS"
    export CXXFLAGS="$CXXFLAGS $ARCH_ARGS"
    export CC="/usr/bin/clang"
    export CXX="/usr/bin/clang"

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
    curl -sLO https://zlib.net/zlib-$ZLIB_VERSION.tar.gz
    tar xzf zlib-$ZLIB_VERSION.tar.gz
    pushd zlib-$ZLIB_VERSION
    ./configure --prefix="$HDF5_DIR"
    make
    make install
    popd

    popd

    export CPPFLAGS=
    export CXXFLAGS=
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

if [[ $MAJOR_V -gt 1 || $MINOR_V -ge 14 ]]; then
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX="$HDF5_DIR" -DENABLE_SHARED:bool=on $EXTRA_CMAKE_MPI_FLAGS "${extra_arch_flags[@]}" ../
elif [[ $MAJOR_V -gt 1 || $MINOR_V -ge 12 ]]; then
    ./configure --prefix "$HDF5_DIR" "$EXTRA_MPI_FLAGS" --enable-build-mode=production $HDF5_HOST
else
    ./configure --prefix "$HDF5_DIR" "$EXTRA_MPI_FLAGS"
fi
make -j "$NPROC"
make install

file "$HDF5_DIR"/lib/*

popd
popd
