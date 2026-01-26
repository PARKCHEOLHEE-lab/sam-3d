import os
import argparse

from loguru import logger

from main_inference import generate_single_object


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default="./VSA_dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./VSA_output"
    )
    
    return parser.parse_args()


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


def main() -> None:
    pass


if __name__ == "__main__":
    args = _parse_args()
    main(args)