#!/bin/bash

# NOTE(pch): To run this script properly, you need to have access to the Hugging Face repository for SAM 3D Objects.
#            (https://huggingface.co/facebook/sam-3d-objects)

# HUGGINGFACE_TOKEN is the token for the Hugging Face account
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN}"

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "HUGGINGFACE_TOKEN is not set"
    echo "Please export HUGGINGFACE_TOKEN as an environment variable using the command: export HUGGINGFACE_TOKEN=<your_huggingface_token>"
    exit 1
fi

hf auth login --token $HUGGINGFACE_TOKEN

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download