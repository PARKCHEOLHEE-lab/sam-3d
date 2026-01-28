# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import gc
import sys
import pytz
import torch
import trimesh
import datetime
import argparse
import traceback
import PIL.Image
import numpy as np

from loguru import logger
from copy import deepcopy
from pytorch3d.transforms import quaternion_to_matrix

sys.path.append("notebook")

from inference import (
    Inference, 
    load_image, 
    load_single_mask, 
    load_masks,
)

from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

CACHE = {}

DEVICE = "cuda"
if not torch.cuda.is_available():
    DEVICE = "cpu"

_R_ZUP_TO_YUP = np.array(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ],
    dtype=np.float32,
)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

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
        "--save_all_objects",
        type=str,
        default="false",
        help="Save intermediate result of inference. It will only be used in multi object inference."
    )
    parser.add_argument(
        "--use_re_alignment",
        type=str,
        default="false",
    )
    parser.add_argument(
        "--sam_prompt",
        type=str,
        default="interior objects",
        help="Prompt for SAM. It will only be used in multi object inference with automatic mask generation."
    )
    parser.add_argument(
        "--sam_threshold",
        type=float,
        default=0.4,
        help="Threshold for SAM. It will only be used in multi object inference with automatic mask generation."
    )
    
    args = parser.parse_args()
    
    assert args.mask_index >= -2
    assert args.export_images in ["true", "false"]
    assert args.save_all_objects in ["true", "false"]
    assert args.use_re_alignment in ["true", "false"]
    
    args.export_images = args.export_images == "true"
    args.save_all_objects = args.save_all_objects == "true"
    args.use_re_alignment = args.use_re_alignment == "true"

    return args


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


def generate_single_object(
    args: argparse.Namespace, 
    output_path: str, 
    mask: np.ndarray = None,
    use_inference_cache: bool = False
) -> None:
    
    # load image (RGBA only, mask is embedded in the alpha channel)
    image = load_image(args.image_path)
    
    if mask is None:
        mask = load_single_mask(os.path.dirname(args.image_path), index=args.mask_index)
        
    # load model
    if use_inference_cache:
        inference = _cache_inference()
    else:
        config_path = f"checkpoints/hf/pipeline.yaml"
        inference = Inference(config_path, compile=False)
    
    # run model
    output = inference(image, mask, seed=42)
    
    # export mesh
    output["glb"].export(os.path.join(output_path, f"object_{args.mask_index:03d}.glb"))
    logger.info(f"Gaussian splat and mesh for mask index {args.mask_index:03d} exported")

    if args.export_images:
        PIL.Image.fromarray(mask).save(os.path.join(output_path, f"mask_{args.mask_index:03d}.png"))
        PIL.Image.fromarray(image).save(os.path.join(output_path, "image.png"))
  
    del output
    if not use_inference_cache:
        del inference
        

def generate_multi_object(args: argparse.Namespace, output_path: str, use_inference_cache: bool = False) -> None:

    def _transform_output(output: dict, in_place: bool = False) -> dict:
        
        if not in_place:
            output = deepcopy(output)

        # process glb
        glb: trimesh.Trimesh
        glb = output["glb"]
        
        vertices = glb.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)

        R_l2c = quaternion_to_matrix(output["rotation"])
        l2c_transform = compose_transform(
            scale=output["scale"],
            rotation=R_l2c,
            translation=output["translation"],
        )
        vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        glb.vertices = vertices.squeeze(0).cpu().numpy() @ _R_ZUP_TO_YUP
        
        output["glb"] = glb

        return output
    
    # load image (RGBA only, mask is embedded in the alpha channel)
    image = load_image(args.image_path)
    
    if args.mask_index == -1:
        masks = load_masks(os.path.dirname(args.image_path), extension=".png")
        
    elif args.mask_index == -2:
        # http://huggingface.co/docs/transformers/en/model_doc/sam3
        
        from transformers import Sam3Processor, Sam3Model
        
        model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        
        if PIL.Image.fromarray(image).mode == "RGBA":
            image = np.array(PIL.Image.fromarray(image).convert("RGB"))

        inputs = processor(
            images=image, 
            text=args.sam_prompt, 
            return_tensors="pt",
        ).to(DEVICE)    
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = inputs.get("original_sizes").tolist()
            
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        segmentations = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.sam_threshold,
            mask_threshold=0.4,
            target_sizes=target_sizes
        )[0]
        
        masks = segmentations["masks"].cpu().numpy().copy()
        if masks.shape[0] == 0:
            logger.error(f"No masks found.")
            return
        
        logger.info(f"Found {masks.shape[0]} masks automatically")
        
        del model
        del processor
        del outputs
        del segmentations
        
        gc.collect()
        torch.cuda.empty_cache()
    
    else:
        logger.error(f"Invalid mask index is given: {args.mask_index}")
        logger.error(traceback.format_exc())
        raise ValueError
        
    # load model
    if use_inference_cache:
        inference = _cache_inference()
    else:
        config_path = f"checkpoints/hf/pipeline.yaml"
        inference = Inference(config_path, compile=False)
    
    scene_glb = trimesh.Scene()

    for mask_index, mask in enumerate(masks):
        
        logger.info(f"Running inference for mask index {mask_index:03d}/{len(masks) - 1:03d}...")

        # run model
        output = inference(image, mask, seed=42)
        
        if args.save_all_objects:
            scene_glb_intermediate = trimesh.Scene()
            if isinstance(output["glb"], trimesh.Scene):
                for geom in output["glb"].geometry.values():
                    scene_glb_intermediate.add_geometry(geom)
            else:
                scene_glb_intermediate.add_geometry(output["glb"])
                
            scene_glb_intermediate.export(os.path.join(output_path, f"object_{mask_index:03d}.glb"))

        # apply transformation
        output = _transform_output(output)

        if isinstance(output["glb"], trimesh.Scene):
            for geom in output["glb"].geometry.values():
                scene_glb.add_geometry(geom)
        else:
            scene_glb.add_geometry(output["glb"])
        
        del output
        gc.collect()
        torch.cuda.empty_cache()
        
    scene_mesh = scene_glb.to_mesh()
    scene_mesh_vertices = scene_mesh.vertices.astype(np.float32)
    scene_mesh_centroid = scene_mesh_vertices.mean(axis=0)

    if args.use_re_alignment:
        # pca
        sigma = np.cov(scene_mesh_vertices - scene_mesh_centroid, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)

        # check if the eigenvectors are orthonormal
        assert np.allclose(np.dot(eigenvectors, eigenvectors.T), np.eye(3))
        
        # sort eigenvectors by descending order of eigenvalues
        indices = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices].T
        
        # apply inverse of principal components
        for geometry in scene_glb.geometry.values():
            scene_mesh_vertices = geometry.vertices.astype(np.float32)
            geometry.vertices = (scene_mesh_vertices - scene_mesh_centroid) @ eigenvectors.T
            geometry.vertices = geometry.vertices @ _R_YUP_TO_ZUP
            
    else:
        for geometry in scene_glb.geometry.values():
            scene_mesh_vertices = geometry.vertices.astype(np.float32)
            geometry.vertices = (scene_mesh_vertices - scene_mesh_centroid) @ _R_YUP_TO_ZUP
                
    scene_glb.export(os.path.join(output_path, "scene.glb"))
    logger.info(f"Merged scene exported as GLB")

    if args.export_images:
        for mi, mask in enumerate(masks):
            masked_image = image.copy()
            masked_image[mask == 0] = 0
            PIL.Image.fromarray(masked_image).save(os.path.join(output_path, f"mask_{mi:03d}.png"))

        PIL.Image.fromarray(image).save(os.path.join(output_path, "image.png"))
        
    if not use_inference_cache:
        del inference
        

if __name__ == "__main__":
    # parse arguments
    args = _parse_args()
    
    # make output directory
    output_path = _make_output_dir(args.output_dir)
        
    # select generator
    generator = generate_single_object
    if args.mask_index in (-1, -2):
        generator = generate_multi_object
        
    generator(args, output_path)
            