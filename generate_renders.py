import argparse 
from pathlib import Path 
from tqdm import tqdm
import torch
import math
import numpy as np 

from utils import read_config, save_imgs, prepare_directories, export_poses
from cv_utils import (generate_trajectory_LBC,
                      generate_trajectory_uniform, 
                      preprocess_pose_and_mesh, 
                      generate_trajectory_SK,
                      generate_trajectory_SK_in_separate_folders,
                      mesh_traj_vis,
                      prepare_coords,
                      render_batch)

import pytorch3d 
import pytorch3d.io
import pytorch3d.utils 
from pytorch3d.renderer import (
    TexturesVertex
)

def main(args):
    config_path = Path(args.config)
    output_path = Path(args.output)

    # path checking
    assert config_path.exists(), 'configuration file does not exist!'
    assert output_path.exists(), 'output directory does not exist!'

    # generate trajectories and process the mesh
    config = read_config(config_path)
    # generate trajectories and process the mesh
    if config['traj_method'] == 'lbc':
        mesh, camera_poses = generate_trajectory_LBC(config) # both poses and mesh is in o3d system
    elif config['traj_method'] == 'sk':
        mesh, camera_poses = generate_trajectory_SK(config)
    elif config['traj_method'] == 'bb':
        mesh, camera_poses = generate_trajectory_uniform(config)
    else:
        raise Exception('invalid traj_method in config')
    mesh_scaled, poses_scaled = preprocess_pose_and_mesh(config, mesh, camera_poses) # both in o3d system
    if config['visualize_traj'] == 1:
        mesh_traj_vis(mesh_scaled, poses_scaled)

    # unfortunately, we have to save the mesh intermediately and load it again
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = Path(output_path)
    assert save_path.exists(), 'invalid path_save_renders directory'
    prepare_directories(save_path)
    mesh_torch3d, poses_scaled = prepare_coords(mesh_scaled, poses_scaled)
    mesh_torch3d = mesh_torch3d.to(device)
    
    force_white = int(config['force_white'])
    if force_white == 1:
        texs = torch.ones_like(mesh_torch3d.verts_packed())
        textures = TexturesVertex(verts_features=texs[None, ...])
        mesh_torch3d.textures = textures

    load_textures = int(config['load_textures'])
    if load_textures == 1:
        print('Loading in textures!')
        assert Path(config['path_to_textures']).exists(), 'path_to_textures does not exist in config'
        texs = torch.load(config['path_to_textures'])
        textures = TexturesVertex(verts_features=texs[None, ...])
        mesh_torch3d.textures = textures
    print('Starting Renders...')

    render_batch_size = int(config['render_batch_size'])
    num_iters = math.ceil(len(poses_scaled) / render_batch_size)
    
    # loop over each "batch" to render images!
    # for i, pose_batch in enumerate(tqdm(poses_scaled, total=len(poses_scaled))):
    #     # if len(poses_scaled) > render_batch_size:
    #     #     pose_batch = poses_scaled[:render_batch_size]
    #     # else:
    #     #     pose_batch = poses_scaled
    #     this_batch_size = len(pose_batch)
        
    #     positions_batch = np.array([pose[:3,3] for pose in pose_batch])
    #     positions_batch = torch.Tensor(positions_batch).to(device)
    #     R_batch = np.array([pose[:3,:3] for pose in pose_batch])
    #     R_batch = torch.Tensor(R_batch).to(device)
    #     render_rgb, render_depth = render_batch(config, mesh_torch3d, R_batch, positions_batch, this_batch_size, device=device)
        
    #     filename = save_path / str(i)
    #     save_imgs(config, filename, render_rgb, i, render_depth, depth=True)
    #     poses_scaled = poses_scaled[render_batch_size:]

    # export_poses(save_path, poses_scaled, 'render')
    # print('...Done!')
    # renderz 
    print('Starting Renders...')
    render_batch_size = int(config['render_batch_size'])
    num_iters = math.ceil(len(poses_scaled) / render_batch_size)
    
    # loop over each "batch" to render images!
    for iter in tqdm(range(num_iters)):
        if len(poses_scaled) > render_batch_size:
            pose_batch = poses_scaled[:render_batch_size]
        else:
            pose_batch = poses_scaled
        this_batch_size = len(pose_batch)
        
        positions_batch = np.array([pose[:3,3] for pose in pose_batch])
        positions_batch = torch.Tensor(positions_batch).to(device)
        R_batch = np.array([pose[:3,:3] for pose in pose_batch])
        R_batch = torch.Tensor(R_batch).to(device)
        render_rgb, render_depth = render_batch(config, mesh_torch3d, R_batch, positions_batch, this_batch_size, device=device)
        save_imgs(config, save_path, render_rgb, iter, render_depth, depth=True)
        poses_scaled = poses_scaled[render_batch_size:]

    export_poses(save_path, poses_scaled, 'render')
    print('...Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/config.yaml")
    parser.add_argument('--output', default="./data/pytorch3d_render")
    args = parser.parse_args()
    main(args)