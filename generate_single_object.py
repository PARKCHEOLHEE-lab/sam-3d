# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
import pytz
import imageio
import datetime
import argparse
import PIL.Image

from loguru import logger

sys.path.append("notebook")

from inference import (
    Inference, 
    load_image, 
    load_single_mask, 
    make_scene, 
    ready_gaussian_for_video_rendering,
    render_video
)


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
    parser.add_argument(
        "--gif",
        type=str,
        default="false"
    )
    return parser.parse_args()


def generate(args: argparse.Namespace) -> None:
    
    assert args.gif in ["true", "false"]
    args.gif = args.gif == "true"
    
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
    PIL.Image.fromarray(masked_image).save(os.path.join(output_dir, f"_masked_{args.mask_index:03d}.png"))

    # run model
    logger.info(f"Running model with image and mask...")
    output = inference(image, mask, seed=42)

    # export gaussian splat and mesh
    logger.info(f"Exporting gaussian splat and mesh...")
    output["gs"].save_ply(os.path.join(output_dir, "splat.ply"))
    output["glb"].export(os.path.join(output_dir, "mesh.glb"))
    
    if args.gif:
        # make a video
        logger.info(f"Making a video...")
        scene_gs = make_scene(output)
        scene_gs = ready_gaussian_for_video_rendering(scene_gs)

        video = render_video(
            scene_gs,
            r=1,
            fov=60,
            pitch_deg=15,
            yaw_start_deg=-45,
            resolution=512,
        )["color"]

        # save video as gif
        imageio.mimsave(
            os.path.join(output_dir, "_video.gif"),
            video,
            format="GIF",
            duration=1000 / 30,  # default assuming 30fps from the input MP4
            loop=0,  # 0 means loop indefinitely
        )


if __name__ == "__main__":
    args = parse_args()
    generate(args)