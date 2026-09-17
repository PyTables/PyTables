# Mirrors the built-in ``x64-windows-release`` triplet, which has no dynamic
# ``arm64`` counterpart.  ``VCPKG_BUILD_TYPE`` skips the unused debug variant.
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)
