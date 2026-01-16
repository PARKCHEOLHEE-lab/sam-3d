# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import time

start_time = time.time()

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model with mesh generation enabled
# Access the pipeline directly to enable mesh postprocessing
image_rgba = inference.merge_mask_to_rgba(image, mask)
output = inference._pipeline.run(
    image_rgba,
    None,
    seed=42,
    stage1_only=False,
    with_mesh_postprocess=True,  # Enable mesh postprocessing
    with_texture_baking=True,   # Enable texture baking from Gaussian splats
    with_layout_postprocess=False,
    use_vertex_color=False,
    stage1_inference_steps=None,
    decode_formats=["mesh", "gaussian"],  # Request both mesh and gaussian outputs
)

# export gaussian splat
if "gs" in output:
    output["gs"].save_ply(f"splat.ply")
    print("Gaussian splat saved to splat.ply")

# export mesh
if "glb" in output and output["glb"] is not None:
    # The glb is a trimesh.Trimesh object that can be exported to various formats
    output["glb"].export("mesh.glb")
    print("Mesh saved to mesh.glb")
    
    # You can also export to other formats:
    # output["glb"].export("mesh.ply")  # PLY format
    # output["glb"].export("mesh.obj")  # OBJ format
    # output["glb"].export("mesh.stl")  # STL format
elif "mesh" in output:
    # If mesh exists but glb wasn't created, you can manually convert it
    mesh_result = output["mesh"][0]
    import trimesh
    vertices = mesh_result.vertices.float().cpu().numpy()
    faces = mesh_result.faces.cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export("mesh.ply")
    print("Mesh saved to mesh.ply")
else:
    print("No mesh output available")

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")