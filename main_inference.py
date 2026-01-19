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
    load_masks,
    make_scene, 
    ready_gaussian_for_video_rendering,
    render_video
)


CACHE = {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", 
        type=str, 
        default="notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
    )
    parser.add_argument(
        "--mask_index", 
        type=int, 
        default=14,
        help="Index of the mask to use. If it is -1, all masks will be used."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output"
    )
    parser.add_argument(
        "--export_images",
        type=str,
        default="false"
    )
    
    args = parser.parse_args()
    
    assert args.mask_index >= -1
    assert args.export_images in ["true", "false"]

    args.export_images = args.export_images == "true"
    
    return args


def make_video(output: dict | list, output_dir: str) -> None:
    """Make a video from the output.

    Args:
        output (dict | list): Output from the model. It is a list if is_multi is True, otherwise a dict.
        output_dir (str): Directory to save the video.
    """
    
    # make a video
    logger.info(f"Making a video...")
    
    if isinstance(output, list):
        scene_gs = make_scene(*output, in_place=True)
    else:
        scene_gs = make_scene(output, in_place=True)

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


def _make_output_dir(output_dir: str):
    timestamp = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)

    return output_path


def _cache_inference():
    
    if "inference" in CACHE:
        return CACHE["inference"]
    
    config_path = f"checkpoints/hf/pipeline.yaml"
    inference = Inference(config_path, compile=False)
    CACHE["inference"] = inference
    
    return inference


def generate_single_object(args: argparse.Namespace, output_path: str, use_inference_cache: bool = False) -> None:
    
    # load model
    if use_inference_cache:
        inference = _cache_inference()
    else:
        config_path = f"checkpoints/hf/pipeline.yaml"
        inference = Inference(config_path, compile=False)
        
    # load image (RGBA only, mask is embedded in the alpha channel)
    image = load_image(args.image_path)
    mask = load_single_mask(os.path.dirname(args.image_path), index=args.mask_index)
    
    # run model
    output = inference(image, mask, seed=42)

    # export gaussian splat and mesh
    logger.info(f"Exporting gaussian splat and mesh for mask index {args.mask_index:03d}...")
    output["gs"].save_ply(os.path.join(output_path, f"splat_{args.mask_index:03d}.ply"))
    output["glb"].export(os.path.join(output_path, f"mesh_{args.mask_index:03d}.glb"))
    logger.info(f"Gaussian splat and mesh for mask index {args.mask_index:03d} exported")

    if args.export_images:
        masked_image = image.copy()
        masked_image[mask == 0] = 0
        PIL.Image.fromarray(masked_image).save(os.path.join(output_path, f"_masked_{args.mask_index:03d}.png"))
        PIL.Image.fromarray(image).save(os.path.join(output_path, "_image.png"))
        
        make_video(output, output_path)

    del output
    if not use_inference_cache:
        del inference
        

def generate_multi_object(args: argparse.Namespace, output_path: str, use_inference_cache: bool = False) -> None:
    # TODO: implement multi object inference to combine multiple objects into a single scene
    # https://github.com/facebookresearch/sam-3d-objects/issues/36
    
    # load model
    if use_inference_cache:
        inference = _cache_inference()
    else:
        config_path = f"checkpoints/hf/pipeline.yaml"
        inference = Inference(config_path, compile=False)
    
    # load image (RGBA only, mask is embedded in the alpha channel)
    image = load_image(args.image_path)
    masks = load_masks(os.path.dirname(args.image_path), extension=".png")
    
    # run model
    outputs = [inference(image, mask, seed=42) for mask in masks]

    for oi, output in enumerate(outputs):
        # export gaussian splat and mesh
        logger.info(f"Exporting gaussian splat and mesh for object {oi:03d}...")
        output["gs"].save_ply(os.path.join(output_path, f"splat_{oi:03d}.ply"))
        output["glb"].export(os.path.join(output_path, f"mesh_{oi:03d}.glb"))
        logger.info(f"Gaussian splat and mesh exported for object {oi:03d}")
    
    if args.export_images:
        for mi, mask in enumerate(masks):
            masked_image = image.copy()
            masked_image[mask == 0] = 0
            PIL.Image.fromarray(masked_image).save(os.path.join(output_path, f"_masked_{mi:03d}.png"))

        PIL.Image.fromarray(image).save(os.path.join(output_path, "_image.png"))

        make_video(outputs, output_path)
        
    del outputs
    if not use_inference_cache:
        del inference
        

if __name__ == "__main__":
    
    """
    for multi object:
        python main_inference.py \
            --image_path=notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
            --mask_index=-1 \
            --export_images=true
        
    for single object:
        python main_inference.py \
            --image_path=notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
            --mask_index=14 \
            --export_images=true
    """
    
    # parse arguments
    args = _parse_args()
    
    # make output directory
    output_path = _make_output_dir(args.output_dir)
        
    # select generator
    generator = generate_single_object
    if args.mask_index == -1:
        generator = generate_multi_object
        
    generator(args, output_path)
            