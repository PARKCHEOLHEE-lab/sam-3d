> [!NOTE] 
> This README was created with AI help to give you clear setup and usage instructions for sam-3d-objects. 
>
> This repository is based on the original code at https://github.com/facebookresearch/sam-3d-objects and the paper https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/.
>
> The original README file has been renamed to [_README.md](_README.md).


## Installation

### Vessl Environment Setup
To set up the DiffuScene environment in Vessl, set the Custom Image to `docker.io/cjfl2343/diffuscene:0.0.7`. This image was made for this project and has all the required packages already installed. The Docker image comes from the [`Dockerfile.diffuscene`](Dockerfile.diffuscene) file in this repository.
**Since this image uses CUDA 11.6, it is recommended to use a node with CUDA version 11.x or higher** (e.g., `eve-s05`, `character-s05`).


<div align="center" >
    <img src="./media/vessl-image.png">
    <br><br>
    <i>Set Custom Image to </i> <code>docker.io/cjfl2343/diffuscene:0.0.7</code>
</div>

<br>

### Repository Setup

To get started with DiffuScene, first clone this repository:
This will create a folder named `KOCCA-SceneRearrange` with all necessary source code and scripts.

```bash
git clone https://github.com/KAIST-VML/KOCCA-SceneRearrange.git
cd KOCCA-SceneRearrange
```