#!/bin/bash
# libjpeg-turbo, static and position-independent, for the wheels' re-coder (setup.py: FRAMESTORE_LIBJPEG=<prefix>).
# The re-coder only transcodes (DCT coefficients in and out, no DCT), so the library's SIMD and tools are not needed.
#   bash .github/scripts/build_libjpeg.sh /tmp/libjpeg
set -euo pipefail
V=3.2.0
SHA256=6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e
PREFIX=${1:?install prefix}
command -v cmake > /dev/null || pipx install cmake || python3 -m pip install cmake
cd "$(mktemp -d)"
curl -sSfL -o src.tar.gz "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/$V/libjpeg-turbo-$V.tar.gz"
echo "$SHA256  src.tar.gz" | sha256sum -c - 2> /dev/null || echo "$SHA256  src.tar.gz" | shasum -a 256 -c -
tar xzf src.tar.gz
cmake -S "libjpeg-turbo-$V" -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DENABLE_SHARED=OFF -DENABLE_STATIC=ON \
  -DWITH_TURBOJPEG=OFF -DWITH_TOOLS=OFF -DWITH_TESTS=OFF -DWITH_SIMD=OFF \
  ${MACOSX_DEPLOYMENT_TARGET:+-DCMAKE_OSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET}
cmake --build build -j 4
cmake --install build > /dev/null
ls "$PREFIX/lib/libjpeg.a" "$PREFIX/include/jpeglib.h"
