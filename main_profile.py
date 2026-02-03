# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import gc
import sys
import time
import torch
import argparse
import pandas as pd

from loguru import logger
from torch.profiler import profile as torch_profile, ProfilerActivity, record_function

from main_inference import generate_single_object



def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        default="./notebook/images/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/_profile/"
    )
    parser.add_argument(
        "--use_inference_cache",
        type=str,
        default="false"
    )
    parser.add_argument(
        "--save_profile_summary",
        type=str,
        default="false"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1
    )
    parser.add_argument(
        "--active",
        type=int,
        default=3
    )
    
    args = parser.parse_args()
    
    assert args.use_inference_cache in ["true", "false"]
    assert args.save_profile_summary in ["true", "false"]
    
    args.use_inference_cache = args.use_inference_cache == "true"
    args.save_profile_summary = args.save_profile_summary == "true"

    args.export_images = False

    return args


def _make_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


if __name__ == "__main__":
    args = _parse_args()
        
    generator = generate_single_object

    image_names = os.listdir(args.images_dir)
    for image_name in image_names:
        
        if image_name.startswith("_"):
            continue
        
        args.image_path = os.path.join(args.images_dir, image_name, "image.png")
        output_path = os.path.join(args.output_dir, image_name)
        _make_output_dir(output_path)
        
        mask_indices = range(len(os.listdir(os.path.dirname(args.image_path))) - 1)
        if len(mask_indices) == 0:
            continue
        
        columns = ["mask_index"] + [f"elapsed_time_at_active_step_{k:03d}" for k in range(1, args.active + 1)] + ["elapsed_time_average"]
        rows = []
        
        for mask_index in reversed(mask_indices):
            
            try:
                args.mask_index = mask_index
                                
                with torch_profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    schedule=torch.profiler.schedule(wait=args.wait, warmup=args.warmup, active=args.active),
                ) as profiler:
                    
                    elapsed_times = []
                    for i in range(args.wait + args.warmup + args.active):
                        start = time.perf_counter()
                        
                        generator(args, output_path, use_inference_cache=args.use_inference_cache)

                        torch.cuda.synchronize()
                        end = time.perf_counter()
                        
                        gc.collect()
                        torch.cuda.empty_cache()
            
                        profiler.step()
                        if i + 1 > args.wait + args.warmup:                            
                            
                            active_step = i - args.wait - args.warmup + 1
                            elapsed_times.append(end - start)
                            logger.success(f"Active step {active_step:03d} elapsed time: {elapsed_times[-1]:.4f} seconds")

                    assert len(elapsed_times) == args.active
                    elapsed_time_average = sum(elapsed_times) / args.active
                    logger.success(f"Average elapsed time of active steps: {elapsed_time_average:.4f} seconds")

                    row = {"mask_index": mask_index, "elapsed_time_average": elapsed_time_average}
                    row.update({f"elapsed_time_at_active_step_{k + 1:03d}": elapsed_times[k] for k in range(args.active)})
                    rows.append(row)
        
                    df = pd.DataFrame(rows, columns=columns)
                    df = df.sort_values(by="mask_index", ascending=True)
                    
                    df.to_csv(os.path.join(output_path, "_elapsed_time.csv"), index=False)
                    
                    if args.save_profile_summary:
                        with open(os.path.join(output_path, f"_profile_summary_{mask_index:03d}.txt"), "w") as f:
                            f.write(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                
            except Exception as e:
                logger.error(f"Error at mask index {mask_index}: {e}")
        
        df = pd.DataFrame(rows, columns=columns)
        df = df.sort_values(by="mask_index", ascending=True)
        
        mean_s = df.drop(columns=["mask_index"]).mean()
        mean_row = {"mask_index": "mean", **mean_s.to_dict()}

        df = pd.concat([df, pd.DataFrame([mean_row], columns=df.columns)], ignore_index=True)
        df.to_csv(os.path.join(output_path, "_elapsed_time.csv"), index=False)
            
    logger.info("Concatenating elapsed times...")
    
    concatenated = pd.DataFrame()
    for profile_dir in os.listdir(args.output_dir):
        if profile_dir.endswith(".csv"):
            continue
        
        profile_path = os.path.join(args.output_dir, profile_dir)
        if not os.path.exists(os.path.join(profile_path, "_elapsed_time.csv")):
            continue
        
        df = pd.read_csv(os.path.join(profile_path, "_elapsed_time.csv"))
        df["mask_index"] = [f"{mi}_{profile_dir}" for mi in df["mask_index"]]
        concatenated = pd.concat([concatenated, df], ignore_index=True)
        
    concatenated = concatenated[~concatenated["mask_index"].astype(str).str.startswith("mean")]
        
    mean_s = concatenated.drop(columns=["mask_index"]).mean()
    mean_row = {"mask_index": "mean", **mean_s.to_dict()}
    concatenated = pd.concat([concatenated, pd.DataFrame([mean_row], columns=concatenated.columns)], ignore_index=True)
    
    concatenated.to_csv(os.path.join(args.output_dir, "_elapsed_time_concatenated.csv"), index=False)

    logger.success("Elapsed times concatenated")