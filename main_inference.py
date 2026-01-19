# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import gc
import sys
import time
import pytz
import torch
import imageio
import datetime
import argparse
import PIL.Image

from loguru import logger
from torch.profiler import profile as torch_profile, ProfilerActivity, record_function

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
    parser.add_argument(
        "--profile",
        type=str,
        default="false"
    )
    parser.add_argument(
        "--use_inference_cache",
        type=str,
        default="false"
    )
    args = parser.parse_args()
    
    assert args.mask_index >= -1
    assert args.export_images in ["true", "false"]
    assert args.profile in ["true", "false"]
    assert args.use_inference_cache in ["true", "false"]

    args.export_images = args.export_images == "true"
    args.profile = args.profile == "true"
    args.use_inference_cache = args.use_inference_cache == "true"
    
    return args


def _make_video(output: dict | list, output_dir: str) -> None:
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


def generate_single_object(args: argparse.Namespace, output_path: str) -> None:
    
    # load model
    if args.use_inference_cache:
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
    logger.info(f"Exporting gaussian splat and mesh...")
    output["gs"].save_ply(os.path.join(output_path, "splat.ply"))
    output["glb"].export(os.path.join(output_path, "mesh.glb"))
    logger.info(f"Gaussian splat and mesh exported")

    if args.export_images:
        masked_image = image.copy()
        masked_image[mask == 0] = 0
        PIL.Image.fromarray(masked_image).save(os.path.join(output_path, f"_masked_{args.mask_index:03d}.png"))
        PIL.Image.fromarray(image).save(os.path.join(output_path, "_image.png"))
        
        _make_video(output, output_path)

    if not args.use_inference_cache:
        del inference
        

def generate_multi_object(args: argparse.Namespace, output_path: str) -> None:
    # https://github.com/facebookresearch/sam-3d-objects/issues/36
    
    # load model
    if args.use_inference_cache:
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

        _make_video(outputs, output_path)
        
    if not args.use_inference_cache:
        del inference
        

if __name__ == "__main__":
    
    """
    for multi object:
        python main.py \
            --image_path=notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
            --mask_index=-1 \
            --profile=true \
            --use_inference_cache=true
        
    for single object:
        python main.py \
            --image_path=notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
            --mask_index=14 \
            --profile=true \
            --use_inference_cache=false
    """
    
    # parse arguments
    args = _parse_args()
    
    # make output directory
    output_path = _make_output_dir(args.output_dir)
        
    # select generator
    generator = generate_single_object
    if args.mask_index == -1:
        generator = generate_multi_object
        
    # inference w/o profiling
    if not args.profile:
        generator(args, output_path)
            
    # inference w/ profiling
    else:
        wait = 1
        warmup = 1
        active = 1
        
        with torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
        ) as profiler:
            
            elapsed_times = []
            with record_function("model_inference"):
                for i in range(wait + warmup + active):
                    start = time.perf_counter()

                    generator(args, output_path)

                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    profiler.step()
                    if i + 1 > wait + warmup:
                        elapsed_times.append(end - start)
                        logger.success(f"Active step {i - wait - warmup + 1:03d} elapsed time: {elapsed_times[-1]:.4f} seconds")

            if "inference" in CACHE:
                del CACHE["inference"]
            
            assert len(elapsed_times) == active

            logger.success(f"Average elapsed time of active steps: {sum(elapsed_times) / active:.4f} seconds")