import os
import argparse

from enum import Enum

from loguru import logger

# from main_inference import generate_multi_object


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
        "--output_format",
        type=str,
        default="glb"
    )
    parser.add_argument(
        "--sam_prompt",
        type=str,
        default="interior objects"
    )
    parser.add_argument(
        "--sam_threshold",
        type=float,
        default=0.3
    )
    
    args = parser.parse_args()
    args.mask_index = -2
    args.export_images = "true"
    args.save_intermediate_result = "true"
    
    return args


def generate_3d_from_image(root_dir: str, output_dir: str) -> bool:
    """_summary_

    Args:
        root_dir (str): _description_

    Returns:
        bool: _description_
    """
    
    try:
        pass
    
    
    except Exception as e:
        logger.error(f"Error: {e}")
    
    return True


def plane_to_plane():
    return


def main(args: argparse.Namespace) -> None:
    
    os.makedirs(args.output_dir, exist_ok=True)

    vsa_dataset = sorted(os.listdir(args.dataset_dir))
    
    room_tag = "R"

    for vsa_data in vsa_dataset:
        vsa_data_path = os.path.join(args.dataset_dir, vsa_data)
        
        room_name = vsa_data.split("_")[-1]
        room_image = os.path.join(vsa_data_path, room_name, "Image", f"{room_name}_Isoview1.png")
        breakpoint()
        
        
        os.listdir(vsa_data_path)

        breakpoint()

        args.image_path = None
        # generate_multi_object(args, args.output_dir)


if __name__ == "__main__":

    """
    
    python main_inference.py \
        --image_path=VSA_dataset/250821_R070/R070/Image/R070_Isoview1.png \
        --mask_index=-2 \
        --threshold=0.2 \
        --export_images=true \
        --save_intermediate_result=true
    """
    
    args = _parse_args()
    main(args)