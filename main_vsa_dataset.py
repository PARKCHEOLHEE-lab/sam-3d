import os
import sys
import shutil
import argparse
import traceback
import subprocess

from loguru import logger

from main_inference import generate_multi_object, generate_single_object

sys.path.append("notebook")

from inference import load_mask

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="./VSA_dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./VSA_output"
    )
    parser.add_argument(
        "--sam_prompt",
        type=str,
        default="furniture"
    )
    parser.add_argument(
        "--sam_threshold",
        type=float,
        default=0.27
        # default=0.7
    )
    
    args = parser.parse_args()
    
    # fixed args
    args.mask_index = -2
    args.export_images = True
    args.save_all_objects = True
    args.use_inference_cache = False
    args.use_re_alignment = True
    
    return args


def generate_3d_from_image(args: argparse.Namespace) -> bool:
    """_summary_

    Args:
        root_dir (str): _description_

    Returns:
        bool: _description_
    """
    
    output_path = os.path.join(args.output_dir, args.root_dir.split("/")[-1])
    os.makedirs(output_path, exist_ok=True)

    # single object
    object_data_list = sorted(os.listdir(args.root_dir))
    for object_index, object_data in enumerate(object_data_list):
        if (
            object_data.startswith(".") 
            or object_data.startswith("R")
            or object_data.startswith("S")
        ):
            continue
        
        object_image_path = os.path.join(
            args.root_dir, 
            object_data, 
            "Image", 
            f"{object_data}_IsoView1.png"
        )
        
        if not os.path.exists(object_image_path):
            object_image_path = object_image_path.replace("IsoView1", "Isoview1")
            if not os.path.exists(object_image_path):
                continue
        
        single_object_output_path = os.path.join(output_path, f"Object{object_index}")
        os.makedirs(single_object_output_path, exist_ok=True)

        # dummy mask index
        args.mask_index = 0
        args.image_path = object_image_path
        generate_single_object(
            args, 
            single_object_output_path, 
            mask=load_mask(object_image_path),
            use_inference_cache=args.use_inference_cache
        )
        
        os.remove(os.path.join(single_object_output_path, "mask_000.png"))
        os.rename(
            os.path.join(single_object_output_path, "object_000.glb"), 
            os.path.join(single_object_output_path, "object.glb")
        )
        
    # multi object
    scene_output_path = os.path.join(output_path, "Scene")
    mask_output_path = os.path.join(scene_output_path, "mask")
    three_d_output_path = os.path.join(scene_output_path, "3d")
    os.makedirs(scene_output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)
    os.makedirs(three_d_output_path, exist_ok=True)
    
    # set the mask_index to -2 for multi object inference with SAM
    args.mask_index = -2
    
    room_number = args.root_dir.split("_")[-1]
    room_image_path = os.path.join(
        args.root_dir, 
        room_number, 
        "Image", 
        f"{room_number}_IsoView1.png"
    )
    
    if not os.path.exists(room_image_path):
        room_image_path = room_image_path.replace("IsoView1", "Isoview1")
        if not os.path.exists(room_image_path):
            return False

    args.image_path = room_image_path
    generate_multi_object(args, output_path)
    
    output_list = sorted(os.listdir(output_path))
    for output_name in output_list:
        
        if output_name.startswith("object"):
            shutil.move(
                os.path.join(output_path, output_name),
                os.path.join(three_d_output_path, output_name)
            )
        
        elif output_name.startswith("mask"):
            shutil.move(
                os.path.join(output_path, output_name),
                os.path.join(mask_output_path, output_name)
            )
            
        elif output_name.startswith("image"):
            shutil.move(
                os.path.join(output_path, output_name),
                os.path.join(scene_output_path, output_name)
            )
            
        elif output_name.startswith("scene"):
            shutil.move(
                os.path.join(output_path, output_name),
                os.path.join(scene_output_path, output_name)
            )
            
            
def plane_to_plane():
    return


def main(args: argparse.Namespace) -> None:
    
    os.makedirs(args.output_dir, exist_ok=True)

    vsa_dataset = sorted(os.listdir(args.dataset_dir))
    for vsa_data in vsa_dataset:
        args.root_dir = os.path.join(args.dataset_dir, vsa_data)
        generate_3d_from_image(args)
        
        # os.makedirs()


if __name__ == "__main__":
    args = _parse_args()
    main(args)