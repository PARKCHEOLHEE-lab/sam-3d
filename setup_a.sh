#!/bin/bash

# simplified from ./doc/setup.md
# environment variables of PIP_EXTRA_INDEX_URL, PIP_FIND_LINKS are set in the Dockerfile.sam-3d

pip install -e '.[dev]'
pip install -e '.[p3d]'
pip install -e '.[inference]'
pip install 'huggingface-hub[cli]<1.0'
pip install hydra-core==1.3.2

./patching/hydra 