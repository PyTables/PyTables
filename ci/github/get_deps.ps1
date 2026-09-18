#!/usr/bin/env pwsh
# Build the Windows native dependencies with vcpkg.
#
# Requires VCPKG_ROOT to point at a bootstrapped vcpkg checkout.  When run
# under GitHub Actions the resulting paths are exported to the job environment;
# otherwise they are set for the current session only.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $env:VCPKG_ROOT) {
    throw 'VCPKG_ROOT is not set; bootstrap vcpkg first.'
}

$arch = $env:PROCESSOR_ARCHITECTURE
$triplet = switch ($arch) {
    'ARM64' { 'arm64-windows-release' }
    'AMD64' { 'x64-windows-release' }
    default { throw "Unsupported architecture: $arch" }
}

# Keep in sync with "Prerequisites" in User's Guide.
# Versions come from the pinned VCPKG_COMMIT, not from HDF5_VERSION.
$packages = @(
    'blosc', 'blosc2', 'bzip2', 'hdf5[core,threadsafe,zlib]',
    'lz4', 'lzo', 'snappy', 'zstd', 'zlib'
)

$vcpkgArgs = @(
    'install',
    '--triplet', $triplet,
    '--overlay-triplets', (Join-Path $PSScriptRoot 'vcpkg-triplets'),
    '--clean-after-build'
) + $packages

Write-Host "Building dependencies for $triplet"
& (Join-Path $env:VCPKG_ROOT 'vcpkg.exe') @vcpkgArgs
if ($LASTEXITCODE -ne 0) {
    throw "vcpkg install failed with exit code $LASTEXITCODE"
}

if ($arch -eq 'ARM64') {
    # cryptography (via twine's keyring) has no win_arm64 wheel and builds from
    # source.  Its openssl-sys probes VCPKG_ROOT for this exact triplet, and
    # needs static libs so the built extension has no runtime DLL dependency.
    # WoA support has been added in https://github.com/pyca/cryptography/pull/15350
    # This block can be removed when they publish a release.
    Write-Host 'Building openssl for arm64-windows-static-md'
    & (Join-Path $env:VCPKG_ROOT 'vcpkg.exe') install --triplet arm64-windows-static-md --clean-after-build openssl
    if ($LASTEXITCODE -ne 0) {
        throw "vcpkg openssl install failed with exit code $LASTEXITCODE"
    }
}

$prefix = Join-Path $env:VCPKG_ROOT "installed\$triplet"

$exports = @(
    "HDF5_DIR=$prefix"
    "BLOSC_DIR=$prefix"
    "BLOSC2_DIR=$prefix"
    "BZIP2_DIR=$prefix"
    "LZO_DIR=$prefix"
    # delvewheel only searches PATH, and the blosc2 wheel keeps libblosc2.dll in
    # a lib/ subdirectory, so link against the vcpkg copy that is on PATH below.
    'PYTABLES_NO_BLOSC2_WHEEL=1'
)

$binDir = Join-Path $prefix 'bin'
if ($env:GITHUB_ENV) {
    $exports | Add-Content -Path $env:GITHUB_ENV
    Add-Content -Path $env:GITHUB_PATH -Value $binDir
} else {
    foreach ($entry in $exports) {
        $name, $value = $entry -split '=', 2
        Set-Item -Path "env:$name" -Value $value
    }
    $env:PATH = "$binDir$([System.IO.Path]::PathSeparator)$env:PATH"
}

Write-Host "Dependencies installed to $prefix"
