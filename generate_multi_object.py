# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, load_masks

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
masks = load_masks("notebook/images/shutterstock_stylish_kidsroom_1640806567", extension=".png")

# run model
outputs = [inference(image, mask, seed=42) for mask in masks]

# export gaussian splat
# Merge all the gaussian splats and save as one file
from sam3d_objects.model.backbone.tdfy_dit.utils.render_utils import merge_gaussian_clouds

# Collect all gs objects from outputs
gs_list = [output["gs"] for output in outputs]

# Merge the gaussian splats (assume gs objects support this operation via render_utils)
merged_gs = merge_gaussian_clouds(gs_list)
merged_gs.save_ply("splat.ply")

print("Your reconstruction has been saved to splat.ply")
