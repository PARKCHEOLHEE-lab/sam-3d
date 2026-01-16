# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
import pytz
import datetime
import argparse
import PIL.Image

sys.path.append("notebook")

from inference import Inference, load_image, load_single_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", 
        type=str, 
        default="notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
    )
    parser.add_argument(
        "--mask_index", 
        type=int, 
        default=14
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output"
    )
    return parser.parse_args()


def generate(args: argparse.Namespace) -> None:
    
    # make output directory
    timestamp = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # load model
    config_path = f"checkpoints/hf/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # load image (RGBA only, mask is embedded in the alpha channel)
    image = load_image(args.image_path)
    PIL.Image.fromarray(image).save(os.path.join(output_dir, "_image.png"))

    mask = load_single_mask(os.path.dirname(args.image_path), index=args.mask_index)
    masked_image = image.copy()
    masked_image[mask == 0] = 0
    PIL.Image.fromarray(masked_image).save(os.path.join(output_dir, f"_{args.mask_index:03d}.png"))

    # run model
    output = inference(masked_image, mask, seed=42)

    # export gaussian splat and mesh
    output["gs"].save_ply(os.path.join(output_dir, "splat.ply"))
    output["glb"].export(os.path.join(output_dir, "mesh.glb"))


if __name__ == "__main__":
    args = parse_args()
    generate(args)