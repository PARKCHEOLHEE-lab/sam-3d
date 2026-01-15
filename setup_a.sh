#!/bin/bash

# simplified from ./doc/setup.md
# environment variables of `PIP_EXTRA_INDEX_URL`, `PIP_FIND_LINKS` are set in the Dockerfile.sam-3d

apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

pip install appdirs
pip install --no-build-isolation nvidia-pyindex==1.0.9
pip install --no-build-isolation "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
pip install --no-build-isolation "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
pip install --no-build-isolation flash_attn==2.8.3

pip install -e .
pip install -e '.[dev]'
pip install -e '.[inference]'

pip install 'huggingface-hub[cli]<1.0'
pip install hydra-core==1.3.2

./patching/hydra 