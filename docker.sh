#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$ROOT_DIR/opencode.log"
IMAGE="${DREAMPLACE_IMAGE:-dreamplace:devcontainer}"
DOCKERFILE="${DREAMPLACE_DOCKERFILE:-$ROOT_DIR/.devcontainer/Dockerfile}"
REBUILD=0
BOOTSTRAP="${DREAMPLACE_BOOTSTRAP:-0}"
CUDA_MODE=0
BUILD_JOBS="${DREAMPLACE_BUILD_JOBS:-1}"
AUTO_BUILD_MISSING="${DREAMPLACE_AUTO_BUILD_MISSING:-0}"
PYTHON_EXEC="/usr/local/bin/python"

image_exists() {
    local img="$1"
    docker image inspect "$img" >/dev/null 2>&1
}

log_step() {
    local message="$1"
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" >> "$LOG_FILE"
}

for arg in "$@"; do
    case "$arg" in
        --rebuild)
            REBUILD=1
            ;;
        --no-bootstrap)
            BOOTSTRAP=0
            ;;
        --bootstrap)
            BOOTSTRAP=1
            ;;
        --cuda)
            CUDA_MODE=1
            ;;
        --build-missing)
            AUTO_BUILD_MISSING=1
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--rebuild] [--build-missing] [--bootstrap|--no-bootstrap] [--cuda]"
            exit 1
            ;;
    esac
done

if [[ "$CUDA_MODE" -eq 1 ]]; then
    if [[ -z "${DREAMPLACE_IMAGE:-}" ]]; then
        if image_exists "dreamplace:cuda-ready"; then
            IMAGE="dreamplace:cuda-ready"
        elif image_exists "dreamplace:readme-cuda-probe"; then
            IMAGE="dreamplace:readme-cuda-probe"
        else
            IMAGE="dreamplace:cuda-ready"
        fi
    else
        IMAGE="${DREAMPLACE_IMAGE}"
    fi

    if [[ -z "${DREAMPLACE_DOCKERFILE:-}" ]]; then
        if [[ "$IMAGE" == "dreamplace:readme-cuda-probe" ]]; then
            DOCKERFILE="$ROOT_DIR/Dockerfile"
        else
            DOCKERFILE="$ROOT_DIR/.devcontainer/Dockerfile.cuda.sm120"
        fi
    else
        DOCKERFILE="${DREAMPLACE_DOCKERFILE}"
    fi

    if [[ "$IMAGE" == "dreamplace:readme-cuda-probe" ]]; then
        PYTHON_EXEC="/opt/conda/bin/python"
    fi
fi

log_step "docker.sh start image=$IMAGE dockerfile=$DOCKERFILE rebuild=$REBUILD bootstrap=$BOOTSTRAP cuda_mode=$CUDA_MODE"

if [[ "$REBUILD" -eq 1 ]]; then
    log_step "rebuilding image $IMAGE"
    echo "Rebuilding image $IMAGE from $DOCKERFILE ..."
    docker build --no-cache -t "$IMAGE" -f "$DOCKERFILE" "$ROOT_DIR"
elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    if [[ "$AUTO_BUILD_MISSING" -eq 1 ]]; then
        log_step "image $IMAGE missing; building"
        echo "Image $IMAGE not found. Building from $DOCKERFILE ..."
        docker build -t "$IMAGE" -f "$DOCKERFILE" "$ROOT_DIR"
    else
        echo "[docker.sh] image $IMAGE not found; skip auto-build to avoid large downloads"
        echo "[docker.sh] use one of:"
        echo "  1) ./docker.sh --cuda --build-missing"
        echo "  2) docker build -t $IMAGE -f $DOCKERFILE $ROOT_DIR"
        exit 1
    fi
fi

GPU_FLAG=""
if [[ "$CUDA_MODE" -eq 1 ]]; then
    if docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
        GPU_FLAG="--gpus all"
    else
        echo "[docker.sh] NVIDIA runtime not detected; starting CUDA image without --gpus"
        log_step "nvidia runtime not detected; continue without --gpus"
    fi
fi

if [[ "$BOOTSTRAP" -eq 1 ]]; then
    log_step "running bootstrap in image $IMAGE"
    BOOTSTRAP_ENV=""
    if [[ "$CUDA_MODE" -eq 1 ]]; then
        BOOTSTRAP_ENV="-e DREAMPLACE_BUILD_DIR=build-cuda-readme -e DREAMPLACE_INSTALL_DIR=install-cuda-readme -e DREAMPLACE_CXX_ABI=1 -e DREAMPLACE_SKIP_PIP_INSTALL=1 -e DREAMPLACE_BUILD_JOBS=${BUILD_JOBS}"
    else
        BOOTSTRAP_ENV="-e DREAMPLACE_BUILD_DIR=build-readme-safe -e DREAMPLACE_INSTALL_DIR=install-readme-safe -e DREAMPLACE_CXX_ABI=0 -e DREAMPLACE_SKIP_PIP_INSTALL=1 -e DREAMPLACE_BUILD_JOBS=${BUILD_JOBS}"
    fi

    docker run ${GPU_FLAG} --rm \
        --user "$(id -u):$(id -g)" \
        -e HOME=/workspace/DREAMPlace \
        -e USER="${USER:-dwindz}" \
        -e PYTHON_EXECUTABLE="${PYTHON_EXEC}" \
        ${BOOTSTRAP_ENV} \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        -v "$ROOT_DIR":/workspace/DREAMPlace \
        -w /workspace/DREAMPlace \
        "$IMAGE" bash -lc "./scripts/bootstrap_readme_env.sh"
fi

log_step "opening interactive shell in image $IMAGE"

docker run ${GPU_FLAG} -it --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace/DREAMPlace \
    -e USER="${USER:-dwindz}" \
    -e PYTHON_EXECUTABLE="${PYTHON_EXEC}" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v "$ROOT_DIR":/workspace/DREAMPlace \
    -w /workspace/DREAMPlace \
    "$IMAGE" bash
