# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
# Spack recipe for py-anuga (ANUGA Hydro shallow-water simulator).
#
# To add this package to a local Spack instance, copy or symlink this file to:
#   <spack-root>/var/spack/repos/builtin/packages/py-anuga/package.py
#
# Alternatively, register this directory as a Spack repo:
#   spack repo add <path-to-anuga_core>/spack
# and rename the parent directory from "spack" to "py-anuga".
#
# Quick-start example
# -------------------
# Install latest stable release with MPI and data extras:
#   spack install py-anuga+mpi+data
#
# Developer editable install (requires the source tree):
#   cd /path/to/anuga_core
#   spack dev-build py-anuga@main

from spack.package import *


class PyAnuga(PythonPackage):
    """ANUGA Hydro: A Python package for simulating the shallow water
    equations, primarily used for tsunami and flood modelling.

    ANUGA uses a finite-volume method on unstructured triangular meshes
    and ships performance-critical loops as C/Cython/OpenMP extensions.
    """

    homepage = "https://anuga.readthedocs.io"
    pypi = "anuga/anuga-3.3.2.tar.gz"
    git = "https://github.com/anuga-community/anuga_core.git"

    maintainers("stoiver", "anuga-community")

    license("Apache-2.0")

    # ------------------------------------------------------------------ #
    # Versions — add new releases here as they are published to PyPI.    #
    # ------------------------------------------------------------------ #
    version("main", branch="main")
    version("develop", branch="develop")
    version(
        "3.3.2",
        sha256="8defca4834c488604fb56404c4dcfa3a74279f8f7f8c69a6ffbfe9e06143c926",
        preferred=True,
    )
    version(
        "3.3.1",
        sha256="26ddd0463efbd82133f6b8ab3619651d1eaac5c3b3361ac30c5bdffb7fd5423b",
    )
    version(
        "3.3.0",
        sha256="550138efaabac6a3ec5c52f0d937e3039ddc7d3ded778fe3a5f4a9cfd1b2c1fb",
    )

    # ------------------------------------------------------------------ #
    # Variants                                                            #
    # ------------------------------------------------------------------ #
    variant(
        "mpi",
        default=False,
        description="Enable MPI-based domain decomposition (requires mpi4py + pymetis)",
    )
    variant(
        "data",
        default=False,
        description=(
            "Enable raster/vector data I/O extras "
            "(pyproj, rasterio, fiona, shapely, pandas, openpyxl)"
        ),
    )

    # ------------------------------------------------------------------ #
    # Build-system requirements                                           #
    # ------------------------------------------------------------------ #
    depends_on("python@3.10:3.14", type=("build", "run"))

    # PEP 517 build backend
    depends_on("py-meson-python@0.15:", type="build")
    depends_on("meson@1.1:", type="build")
    depends_on("ninja", type="build")

    # Compiled-extension build tools
    depends_on("py-cython@3:", type="build")
    depends_on("py-pybind11@2.11:", type="build")

    # numpy headers are needed at compile time as well as runtime
    depends_on("py-numpy@2:", type=("build", "run"))

    # ------------------------------------------------------------------ #
    # Core runtime dependencies (mirrors pyproject.toml [project].       #
    # dependencies)                                                       #
    # ------------------------------------------------------------------ #
    depends_on("py-dill@0.3.7:", type="run")
    depends_on("py-matplotlib@3.7:", type="run")
    depends_on("py-netcdf4@1.6:", type="run")
    depends_on("py-scipy@1.11:", type="run")
    depends_on("py-meshpy@2022.1:", type="run")
    depends_on("py-utm", type="run")
    depends_on("py-xarray", type="run")

    # ------------------------------------------------------------------ #
    # Optional: MPI / parallel variant                                    #
    # ------------------------------------------------------------------ #
    depends_on("mpi", when="+mpi")
    depends_on("py-mpi4py", type="run", when="+mpi")
    depends_on("py-pymetis@2023.1:", type="run", when="+mpi")

    # ------------------------------------------------------------------ #
    # Optional: data/GIS variant                                         #
    # ------------------------------------------------------------------ #
    depends_on("py-pyproj@3.6:", type="run", when="+data")
    depends_on("py-affine@2.4:", type="run", when="+data")
    depends_on("py-rasterio", type="run", when="+data")
    depends_on("py-fiona", type="run", when="+data")
    depends_on("py-shapely", type="run", when="+data")
    depends_on("py-pandas", type="run", when="+data")
    depends_on("py-openpyxl", type="run", when="+data")
    # tomli is only needed on Python < 3.11 for TOML scenario config
    depends_on("py-tomli", type="run", when="+data ^python@:3.10")

    # ------------------------------------------------------------------ #
    # OpenMP (optional but strongly recommended for performance)         #
    # ------------------------------------------------------------------ #
    # Meson detects OpenMP automatically; Spack just needs to make the
    # compiler flags available.  On macOS with Apple Clang, install the
    # llvm-openmp conda package or use a GNU/LLVM compiler.
    depends_on("llvm-openmp", type=("build", "run"), when="platform=darwin %apple-clang")

    # ------------------------------------------------------------------ #
    # Build configuration                                                 #
    # ------------------------------------------------------------------ #
    # PythonPackage / meson-python handles the PEP 517 build automatically.
    # No override of install(), setup_run_environment(), etc. is needed.

    # Pass --no-build-isolation equivalent through pip so that the numpy /
    # Cython already resolved by Spack are used rather than freshly downloaded
    # ones (which could differ in version from what Spack resolved).
    # PythonPackage sets pip install flags; we extend them here.
    install_options = ["--no-build-isolation"]

    @run_after("install")
    @on_package_attributes(run_tests=True)
    def install_test(self):
        """Lightweight smoke test: import the package and run fast unit tests."""
        python("-c", "import anuga; print('anuga', anuga.__version__)")
        pytest = which("pytest")
        if pytest:
            pytest(
                "--pyargs", "anuga",
                "--run-fast",
                "-q",
                "--tb=short",
            )
