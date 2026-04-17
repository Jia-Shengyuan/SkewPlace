# GPU Runtime Status (RTX 5070 Ti / sm_120)

This note records the currently working Docker usage and the safe upgrade path.

## Legacy baseline (kept for rollback notes)

- Historical image tag: `dreamplace:cuda-ready` (legacy)
- Historical backing image id: `cb0c662e82e2`
- Historical source: README-based `Dockerfile` flow + local probe steps
- Historical purpose: validated CUDA compile probe, but runtime torch arch support was insufficient for `sm_120`

Legacy config is preserved by keeping this document and the old Dockerfiles in git history.

## Current target runtime

- Active Dockerfile for `--cuda`: `.devcontainer/Dockerfile.cuda.sm120`
- Active image tag for `--cuda`: `dreamplace:cuda-ready`

## How to launch now

- GPU shell: `./docker.sh --cuda`
- CPU shell: `./docker.sh`
- Force bootstrap only when needed: `./docker.sh --cuda --bootstrap`

Default behavior in `docker.sh` is intentionally conservative:

- no auto-build when image is missing
- no auto-bootstrap on every launch

So routine launch should not trigger large downloads.

## Why old runtime can still fail on this laptop GPU

`torch==1.7.1` warns that `sm_120` is unsupported for runtime kernels.
That means:

- CMake/NVCC compile can succeed
- but runtime CUDA ops may fail or fallback inconsistently

So for stable GPU execution on this device, update to a CUDA 12.8 + newer PyTorch stack.

## Upgrade path without touching baseline files

To avoid overwriting existing working setup, use a new Dockerfile:

- `.devcontainer/Dockerfile.cuda.sm120`

Suggested build command:

```bash
docker build -t dreamplace:cuda-sm120 -f .devcontainer/Dockerfile.cuda.sm120 .
```

Suggested run command:

```bash
docker run --rm --gpus all -it -v "$(pwd)":/workspace/DREAMPlace -w /workspace/DREAMPlace dreamplace:cuda-sm120 bash
```

Inside the container, verify GPU runtime:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_GPU")
PY
```

## Rollback notes

If future updates fail, rebuild an older stack by selecting an older Dockerfile manually with:

`DREAMPLACE_DOCKERFILE=<path> DREAMPLACE_IMAGE=<tag> ./docker.sh --cuda --build-missing`
