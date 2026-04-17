#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
LOG_FILE="$ROOT_DIR/opencode.log"

log_step() {
    local message="$1"
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" >> "$LOG_FILE"
}

BUILD_DIR="${DREAMPLACE_BUILD_DIR:-build-readme}"
INSTALL_DIR="${DREAMPLACE_INSTALL_DIR:-install}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$(command -v python || true)}"
CXX_ABI="${DREAMPLACE_CXX_ABI:-0}"
SKIP_PIP_INSTALL="${DREAMPLACE_SKIP_PIP_INSTALL:-0}"
BUILD_JOBS="${DREAMPLACE_BUILD_JOBS:-1}"

if [[ -z "${PYTHON_EXECUTABLE}" ]]; then
    echo "[bootstrap] python executable not found; set PYTHON_EXECUTABLE"
    exit 1
fi

if [[ -f "${INSTALL_DIR}/dreamplace/Placer.py" && -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    log_step "bootstrap skip existing build/install build_dir=${BUILD_DIR} install_dir=${INSTALL_DIR}"
    echo "[bootstrap] existing build/install detected, skip rebuild"
    exit 0
fi

if [[ ! -d thirdparty/pybind11 ]]; then
    log_step "bootstrap initializing git submodules"
    echo "[bootstrap] initializing git submodules..."
    git submodule init
    git submodule update
fi

if [[ "${SKIP_PIP_INSTALL}" -eq 1 ]]; then
    log_step "bootstrap skip pip install"
    echo "[bootstrap] skip pip install (DREAMPLACE_SKIP_PIP_INSTALL=1)"
else
    log_step "bootstrap pip install requirements"
    echo "[bootstrap] installing python dependencies..."
    python -m pip install --no-cache-dir -r requirements.txt >/tmp/dreamplace-pip.log 2>&1 || {
        echo "[bootstrap] pip install failed; check /tmp/dreamplace-pip.log"
        exit 1
    }
fi

echo "[bootstrap] configuring project..."
log_step "bootstrap cmake configure build_dir=${BUILD_DIR} install_dir=${INSTALL_DIR} python=${PYTHON_EXECUTABLE} abi=${CXX_ABI}"
cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX="$ROOT_DIR/${INSTALL_DIR}" \
    -DPython_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    -DCMAKE_CXX_ABI="${CXX_ABI}" >/tmp/dreamplace-cmake-config.log 2>&1 || {
    echo "[bootstrap] cmake configure failed; check /tmp/dreamplace-cmake-config.log"
    exit 1
}

echo "[bootstrap] building project..."
log_step "bootstrap cmake build build_dir=${BUILD_DIR} jobs=${BUILD_JOBS}"
cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS}" >/tmp/dreamplace-cmake-build.log 2>&1 || {
    echo "[bootstrap] cmake build failed; check /tmp/dreamplace-cmake-build.log"
    exit 1
}

echo "[bootstrap] installing project..."
log_step "bootstrap cmake install build_dir=${BUILD_DIR}"
cmake --install "${BUILD_DIR}" >/tmp/dreamplace-cmake-install.log 2>&1 || {
    echo "[bootstrap] cmake install failed; check /tmp/dreamplace-cmake-install.log"
    exit 1
}

echo "[bootstrap] done. install directory is ready at $ROOT_DIR/${INSTALL_DIR}"
log_step "bootstrap done build_dir=${BUILD_DIR} install_dir=${INSTALL_DIR}"
