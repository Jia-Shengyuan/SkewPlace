#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE_NAME="shengyuanjia/dreamplace:cuda"
CONTAINER_HOSTNAME="eda_server"

HOST_UID=$(id -u)
HOST_GID=$(id -g)

docker run --gpus all -it --rm \
  --hostname "$CONTAINER_HOSTNAME" \
  -u "${HOST_UID}:${HOST_GID}" \
  -v "$SCRIPT_DIR":/DREAMPlace \
  -e HOME=/DREAMPlace \
  -e PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -w /DREAMPlace \
  "$IMAGE_NAME" \
  bash -lc 'cat > /tmp/container_bashrc <<"RC"
export PATH=/opt/conda/bin:$PATH
export PS1="\\[\\e[1;34m\\]jsy@eda_server:\\w\\$ \\[\\e[0m\\]"
RC
echo -e "\\e[44;97m >>> ENTERED DREAMPlace CONTAINER (jsy@eda_server) <<< \\e[0m"
exec bash --noprofile --rcfile /tmp/container_bashrc -i'