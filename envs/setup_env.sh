#!/usr/bin/env bash
# Build the single environment every model runs in: Python 3.12 + PyTorch 2.10 (cu128).
#
#   bash envs/setup_env.sh            # from the project root
#
# Steps are idempotent; re-running only installs what is missing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
UV="uv pip install --python $PY"
export MAX_JOBS="${MAX_JOBS:-4}"   # 15 GB host RAM: keep native builds narrow

# 1. interpreter + torch
[ -x "$PY" ] || uv venv --python 3.12 .venv
$UV torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# 2. upstream repositories (pinned commits) -------------------------------
clone() {  # url dir commit
  [ -d "third_party/$2/.git" ] || git clone "$1" "third_party/$2"
  git -C "third_party/$2" fetch --depth 1 origin "$3" 2>/dev/null || true
  git -C "third_party/$2" checkout -q "$3"
}
mkdir -p third_party
clone https://github.com/NVlabs/GEM-X.git              GEM-X        4aef1452e5df171deff4f293c08e643eb450981e
clone https://github.com/NVlabs/SOMA-X.git             SOMA-X       v0.3.3   # docs/tools reference; runtime uses the py-soma-x wheel
clone https://github.com/facebookresearch/sam-3d-body.git sam-3d-body b5c765a0d89d789985e186d396315e7590887b94
clone https://github.com/gaomingqi/sam-body4d.git      sam-body4d   21af1020979ef32ddf6be3597ef59a68bad2f1bf
clone https://github.com/facebookresearch/DuoMo.git    DuoMo        cfc1cdc44368228438ff579158f19e135173fa42
clone https://github.com/flowhmr/flowhmr.git           flowhmr      b79fc11fc23d477d8d8cda2cea8635d15c26a9f0
git -C third_party/flowhmr submodule update --init --depth 1 third_party/YOLOX third_party/sam-3d-body
# upstream bug fixes kept as patches (envs/patches/<repo>-NNNN-*.patch); applied once
for patch in envs/patches/*.patch; do
  repo="third_party/$(basename "$patch" | sed -E 's/-[0-9]{4}-.*//')"
  git -C "$repo" apply --reverse --check "$ROOT/$patch" 2>/dev/null || git -C "$repo" apply "$ROOT/$patch"
done
[ -d third_party/DuoMo/third_party/PromptHMR ] || {
  git clone https://github.com/yufu-wang/PromptHMR.git third_party/DuoMo/third_party/PromptHMR
  git -C third_party/DuoMo/third_party/PromptHMR checkout -q 2a1c013f54fe9588d1019ad9779961045cf76f65
}

# 3. python packages -------------------------------------------------------
$UV -e third_party/GEM-X                     # transformers 5.3 + py-soma-x 0.3.3 (shared SOMA runtime)
$UV -r envs/requirements.txt
$UV --no-deps -e third_party/sam-body4d/models/sam3
$UV --no-build-isolation --no-deps -e third_party/flowhmr/third_party/YOLOX
$UV -e . --no-deps                           # this project (vvm)

# opencv-python (pulled by several repos) and GEM-X's opencv-python-headless both
# provide cv2; keep them on the same 4.x build so neither shadows the other.
CV="$($PY -c 'import importlib.metadata as m; print(m.version("opencv-python-headless"))')"
$UV "opencv-python==$CV"

# 4. ffmpeg on PATH for upstream code that shells out to it
ln -sf "$($PY -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" .venv/bin/ffmpeg

echo "environment ready: $($PY -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
