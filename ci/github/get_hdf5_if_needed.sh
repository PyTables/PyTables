#!/bin/bash
# vendored from https://github.com/h5py/h5py/blob/master/ci/get_hdf5_if_needed.sh

set -e

if [ -z ${HDF5_DIR+x} ]; then
    echo "Using OS HDF5"
else
    echo "Using downloaded HDF5"
    if [ -z ${HDF5_MPI+x} ]; then
        echo "Building serial"
        EXTRA_MPI_FLAGS=''
    else
        echo "Building with MPI"
        EXTRA_MPI_FLAGS="--enable-parallel --enable-shared"
    fi

    MINOR_V=$(sed -e "s/.[0-9]\+\$//; s/^[0-9]\+.//" <<< "$HDF5_VERSION")
    MAJOR_V=${HDF5_VERSION/%.*.*}
    if [[ "$OSTYPE" == "darwin"* ]]; then
        lib_name=libhdf5.dylib
    else
        lib_name=libhdf5.so
    fi

    if [ -f "$HDF5_DIR/lib/$lib_name" ]; then
        echo "using cached build"
    else
        echo "building HDF5"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install automake cmake pkg-config
            export MACOSX_DEPLOYMENT_TARGET="10.9"
            pushd /tmp

            # snappy
            curl -sLO https://github.com/google/snappy/archive/refs/tags/1.1.9.tar.gz
            tar xzf 1.1.9.tar.gz
            pushd snappy-1.1.9
            git submodule update --init
            mkdir build
            cd build
            cmake -DCMAKE_INSTALL_PREFIX="$HDF5_DIR" -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" ../
            make
            make install
            popd

            # zstd
            curl -sLO https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz
            tar xzf zstd-1.5.2.tar.gz
            pushd zstd-1.5.2
            cd build/cmake
            cmake -DCMAKE_INSTALL_PREFIX="$HDF5_DIR" -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
            make
            make install
            popd

            # universal binaries is only supported for v1.13+
            if [[ $MAJOR_V -eq 1 && $MINOR_V -lt 13 ]]; then
                echo "MACOS universal wheels can only be built with HDF5 version 1.13+" 1>&2
                exit 1
            fi

            export CFLAGS="$CFLAGS -arch x86_64 -arch arm64"

            # lz4
            curl -sLO https://github.com/lz4/lz4/archive/refs/tags/v1.9.3.tar.gz
            tar xzf v1.9.3.tar.gz
            pushd lz4-1.9.3
            make install PREFIX="$HDF5_DIR"
            popd

            # lzo
            curl -sLO https://www.oberhumer.com/opensource/lzo/download/lzo-2.10.tar.gz
            tar xzf lzo-2.10.tar.gz
            pushd lzo-2.10
            ./configure --prefix="$HDF5_DIR"
            make
            make install
            popd

            # bzip2
            curl -sLO https://gitlab.com/bzip2/bzip2/-/archive/bzip2-1.0.8/bzip2-bzip2-1.0.8.tar.gz
            tar xzf bzip2-bzip2-1.0.8.tar.gz
            pushd bzip2-bzip2-1.0.8
            make install PREFIX="$HDF5_DIR"
            popd

            # zlib
            curl -sLO https://zlib.net/zlib-1.2.11.tar.gz
            tar xzf zlib-1.2.11.tar.gz
            pushd zlib-1.2.11
            ./configure --prefix="$HDF5_DIR"
            make
            make install
            popd

            popd
        fi

        pushd /tmp

        #                                   Remove trailing .*, to get e.g. '1.12' ↓
        curl -fsSLO "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"
        tar -xzvf "hdf5-$HDF5_VERSION.tar.gz"
        pushd "hdf5-$HDF5_VERSION"

        chmod u+x autogen.sh
        # production is supported from 1.12
        if [[ $MAJOR_V -gt 1 || $MINOR_V -ge 12 ]]; then
          ./configure --prefix "$HDF5_DIR" "$EXTRA_MPI_FLAGS" --enable-build-mode=production
        else
          ./configure --prefix "$HDF5_DIR" "$EXTRA_MPI_FLAGS"
        fi
        make -j "$(nproc)"
        make install

        file "$HDF5_DIR/lib/*"

        popd
        popd
    fi
fi
