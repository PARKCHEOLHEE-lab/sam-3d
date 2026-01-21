> [!NOTE] 
> This README was created with AI help to give you clear setup and usage instructions for sam-3d-objects. 
>
> This repository is based on the original code at https://github.com/facebookresearch/sam-3d-objects and the paper https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/.
>
> The original README file has been renamed to [_README.md](_README.md).

<br>

## Installation

### Vessl Environment Setup
To set up the DiffuScene environment in Vessl, set the Custom Image to `docker.io/cjfl2343/sam-3d:0.0.2`. This image was made for this project and has all the required packages already installed. The Docker image comes from the [`Dockerfile.sam-3d`](Dockerfile.sam-3d) file in this repository.
**Since this image uses CUDA 12.1 and SAM 3D requires at least 32GB of VRAM for multi-object inference, it is recommended to use a node with CUDA version 12.1 or higher** (e.g., `eve-s01`).


<div align="center" >
    <img src="./media/vessl-image.png">
    <br><br>
    <i>Set Custom Image to </i> <code>docker.io/cjfl2343/sam-3d:0.0.2</code>
</div>

<br>

### Repository Setup

To get started with sam-3d-objects, first clone this repository:
This will create a folder named `KOCCA-SAM3D` with all necessary source code and scripts.

```bash
git clone https://github.com/KAIST-VML/KOCCA-SAM3D.git
cd KOCCA-SAM3D
```

<br>

## Set Up Environment & Pretined Models 

To set up the environment and pre-trained models, run these scripts in order:

1. Install dependencies:

   ```bash
   bash setup_a.sh
   ```

   <br>

2. Request model checkpoints by providing some information at https://huggingface.co/facebook/sam-3d-objects:

    <div align="center" >
        <img src="./media/requesting-access.png">
        <br><br>
        <i><a href="https://huggingface.co/facebook/sam-3d-objects">Requesting Checkpoints</a></i>
    </div>
    
    <br>


3. Install pre-trained models **(Once your request to access the model checkpoints has been accepted and your Huggingface token has been created)**:

    ```bash
    export HUGGINGFACE_TOKEN=<your_huggingface_token>
    bash setup_b.sh
    ```

<br>

## Inference

The `main_inference.py` script can generate either a single object or an entire scene from an input image using pre-trained model weights.

Arguments:
- `--image_path`: Path to the input image (`.png`) file.
- `--mask_index`: Index of the object mask to process; set to `-1` to process all masks and create a scene.
- `--output_dir`: Directory to save results.
- `--export_images`: If `true`, will save inference visualizations (masked objects and composite video).
- `--output_format`: Output format for 3D objects, `glb` for mesh or `ply` for gaussian splats.

<br> 

### Single Object Inference

To generate a 3D object from a single mask, specify the image path and the index of the mask to use (`--mask_index=N`). For example, to extract the object using mask index 14:

```python
python main_inference.py \
    --image_path=notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --mask_index=14 \
    --output_dir=output \
    --export_images=false \
    --output_format=glb
```

<br>

### Multi Object Inference

To generate 3D meshes for all object masks in an input image and combine them into a scene, set `--mask_index=-1`:

```python
python main_inference.py \
    --image_path=notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --mask_index=-1 \
    --output_dir=output \
    --export_images=false \
    --output_format=glb
```

<br>

## Profiling

The `main_profile.py` script benchmarks inference speed for one or more images and outputs per-mask timing statistics.

Arguments:
- `--images_dir`: Directory containing subfolders for each image, each with its own `image.png` and masks.
- `--output_dir`: Directory in which to save profiling results, including `_elapsed_time.csv`.
- `--use_inference_cache`: If `true`, reuses the loaded model between runs (disables reloading overhead).
- `--save_profile_summary`: If `true`, saves detailed PyTorch profiler summaries at each step.
- `--wait`: Number of initial steps to skip timing (profiling "wait" phase).
- `--warmup`: Number of warmup steps before timing.
- `--active`: Number of measurement steps (timed "active" phase).

<br>

### Profiling All Images

To benchmark inference performance on all images in `./notebook/images/` (approximately 230 total object masks across all images), run:
 
```python
python main_profile.py \
    --images_dir=./notebook/images/ \
    --output_dir=./output/_profile/ \
    --use_inference_cache=false \
    --save_profile_summary=false \
    --wait=0 \
    --warmup=1 \
    --acive=3
```

<br>

### Methods

Inference was executed for approximately 230 individual objects. For each object, the schedule was set to wait = 0, warmup = 1, active = 3, yielding a total of four runs per object. Only the three active-step wall-clock times were recorded and averaged to produce the reported per-object mean. Timing was measured as the difference of `time.perf_counter()` after calling `torch.cuda.synchronize()` at each iteration to enforce GPU synchronization.

<br>

### Results

On an NVIDIA A5000 GPU (24 GB VRAM), the mean wall-clock time per single-object inference is `37.004264873904155` seconds. If we exclude model-loading overhead, the runtime is expected to decrease by approximately 20%. In this benchmark, the configuration was intentionally conservative: the model was reloaded on every run.


| mask_index                                           | elapsed_time_at_active_step_001 | elapsed_time_at_active_step_002 | elapsed_time_at_active_step_003 | elapsed_time_average       |
|------------------------------------------------------|----------------------------------|----------------------------------|----------------------------------|-----------------------------|
| 0_kid_box                                            | 45.33452668134123                | 45.4912094604224                 | 45.65570163633674                | 45.49381259270013           |
| 1_kid_box                                            | 29.45958050340414                | 30.371026386506852               | 30.46039948984981                | 30.097002126586933          |
| 2_kid_box                                            | 36.62481936812401                | 36.83841050881893                | 37.89146111905575                | 37.11823033199956           |
| ...                                                  | ...                              | ...                              | ...                              | ...                         |
| 0_shutterstock_1243680295                            | 31.51510568056256                | 31.225410433486104               | 31.995900759473443               | 31.578805624507368          |
| 1_shutterstock_1243680295                            | 51.53862490598112                | 51.73792759235948                | 51.409426456317306               | 51.56199298488597           |
| 2_shutterstock_1243680295                            | 31.32706823106855                | 31.51491724140942                | 31.40297007188201                | 31.41498518145333           |
| ...                                                  | ...                              | ...                              | ...                              | ...                         |
| **mean**                                                 | 37.063137248682075               | 36.85683911495199                | 37.09281825807841                | **37.004264873904155**          |

<br>
For multi-object inference, the pipeline still performs per-object inference independently and then merges the outputs into a single scene. The merging step increases memory requirements, so a GPU with at least 32 GB VRAM is likely necessary. Because multi-object scene generation time depends on the number of object masks in the image ($M$), a practical estimate is $M \times 30$ seconds if one assumes $30$ seconds for single-object generation.
