# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
import pytz
import torch
import imageio
import datetime
import argparse
import PIL.Image

from loguru import logger
from copy import deepcopy
from pytorch3d.transforms import quaternion_multiply, quaternion_invert, quaternion_to_matrix

sys.path.append("notebook")

from inference import (
    Inference, 
    load_image, 
    load_single_mask, 
    load_masks,
    make_scene, 
    ready_gaussian_for_video_rendering,
    render_video,
    SceneVisualizer
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


    def _process_output(output: dict, in_place: bool = False, minimum_kernel_size: float = float("inf")) -> dict:
        if not in_place:
            output = deepcopy(output)

        rotation = output["rotation"]
        translation = output["translation"]
        scale = output["scale"]
            
        # process gaussian
        # move gaussians to scene frame of reference
        PC = SceneVisualizer.object_pointcloud(
            points_local=output["gaussian"][0].get_xyz.unsqueeze(0),
            quat_l2c=rotation,
            trans_l2c=translation,
            scale_l2c=scale,
        )
        output["gaussian"][0].from_xyz(PC.points_list()[0])
        # must ... ROTATE
        output["gaussian"][0].from_rotation(
            quaternion_multiply(
                quaternion_invert(rotation),
                output["gaussian"][0].get_rotation,
            )
        )
        scale = output["gaussian"][0].get_scaling
        adjusted_scale = scale * scale
        assert (
            scale[0, 0].item()
            == scale[0, 1].item()
            == scale[0, 2].item()
        )
        output["gaussian"][0].mininum_kernel_size *= scale[0, 0].item()
        adjusted_scale = torch.maximum(
            adjusted_scale,
            torch.tensor(
                output["gaussian"][0].mininum_kernel_size * 1.1,
                device=adjusted_scale.device,
            ),
        )
        output["gaussian"][0].from_scaling(adjusted_scale)
        minimum_kernel_size = min(
            minimum_kernel_size,
            output["gaussian"][0].mininum_kernel_size,
        )
        
        # process glb
        glb: trimesh.Trimesh
        glb = output["glb"]
        
        vertices_torch = torch.from_numpy(np.asarray(glb.vertices)).to(rotation.device).to(rotation.dtype)
        vertices_torch = vertices_torch * scale
        vertices_torch = vertices_torch @ quaternion_to_matrix(rotation)
        vertices_torch = vertices_torch + translation
        
        glb.vertices = vertices_torch.detach().cpu().numpy()
        output["glb"] = glb

        return output
    
    # load model
    if use_inference_cache:
        inference = _cache_inference()
    else:
        config_path = f"checkpoints/hf/pipeline.yaml"
        inference = Inference(config_path, compile=False)
    
    # load image (RGBA only, mask is embedded in the alpha channel)
    image = load_image(args.image_path)
    masks = load_masks(os.path.dirname(args.image_path), extension=".png")
    
    glb_scene = trimesh.Scene()
    all_outputs = []
    minimum_kernel_size = float("inf")
    for mask in masks[:1]:
        # run model
        output = inference(image, mask, seed=42)
        output = _process_output(output, minimum_kernel_size=minimum_kernel_size)
        minimum_kernel_size = min(
            minimum_kernel_size,
            output["gaussian"][0].mininum_kernel_size,
        )
        glb_scene.add_geometry(output["glb"], name=f"object_{mask_index:03d}")
        all_outputs.append(output)
    
    glb_scene.export(os.path.join(output_path, "_merged_scene.glb"))
    logger.info(f"Merged scene exported as GLB")

    # merge gaussians
    scene_gs = all_outputs[0]["gaussian"][0]
    scene_gs.mininum_kernel_size = minimum_kernel_size
    for out in all_outputs[1:]:
        out_gs = out["gaussian"][0]
        scene_gs._xyz = torch.cat([scene_gs._xyz, out_gs._xyz], dim=0)
        scene_gs._features_dc = torch.cat(
            [scene_gs._features_dc, out_gs._features_dc], dim=0
        )
        scene_gs._scaling = torch.cat([scene_gs._scaling, out_gs._scaling], dim=0)
        scene_gs._rotation = torch.cat([scene_gs._rotation, out_gs._rotation], dim=0)
        scene_gs._opacity = torch.cat([scene_gs._opacity, out_gs._opacity], dim=0)

    scene_gs.save_ply(os.path.join(output_path, "_merged_scene.ply"))
    logger.info(f"Merged scene exported as PLY")

    if args.export_images:
        for mi, mask in enumerate(masks):
            masked_image = image.copy()
            masked_image[mask == 0] = 0
            PIL.Image.fromarray(masked_image).save(os.path.join(output_path, f"_masked_{mi:03d}.png"))

        PIL.Image.fromarray(image).save(os.path.join(output_path, "_image.png"))
        
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
            