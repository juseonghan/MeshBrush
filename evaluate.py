import argparse 
from pathlib import Path 
from tqdm import tqdm
import torch
import math
from random import sample 
import copy 
import cv2 as cv 
import matplotlib.pyplot as plt
import os  
import numpy as np 

from models import    TextureLearner
from utils  import   (read_config, 
                      read_loss_fn, 
                      prepare_directories, 
                      record_losses, 
                      export_poses,
                      log_training, 
                      read_mask, 
                      extract_imgs, 
                      read_textures,
                      save_imgs)

from cv_utils import (generate_trajectory_LBC, 
                      preprocess_pose_and_mesh, 
                      generate_trajectory_SK,
                      prepare_styletransfer_model,
                      generate_trajectory_uniform,
                      export_mesh,
                      inference_styletransfer,
                      down_sample_cameras,
                      mesh_traj_vis,

                      train,
                      train2,
                      train3, 
                      prepare_coords,
                      render_batch_with_pointmap,
                      render_batch)

from data import STDataset

from pytorch3d.renderer import TexturesVertex
from torch.utils.data import DataLoader 

def main(args):
    config_path = Path(args.config)

    # path checking
    assert config_path.exists(), 'configuration file does not exist!'

    # prepare style transfer model
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

    # fix the freaking coordinate system
    # all_poses = copy.deepcopy(poses_scaled)
    # if 'num_cameras' in config:
    #     num_iters = int(config['num_cameras'])
    #     poses_scaled = down_sample_cameras(all_poses, num_iters)
    #     mesh_traj_vis(mesh_scaled, poses_scaled)

    # poses_scaled = all_poses
    mesh_torch3d, poses_scaled = prepare_coords(mesh_scaled, poses_scaled)
    # poses_scaled = down_sample_cameras(poses_scaled, 1500)
    mesh_torch3d = mesh_torch3d.to(device)
    texs = read_textures(config['path_to_mesh'])
    texs.append(texs[-1])
    texs = np.stack(texs, axis=0)
    texs = torch.ones_like(mesh_torch3d.verts_packed()) - 0.8
    texs = torch.Tensor(texs)[None].to(device)
    mesh_torch3d.textures = TexturesVertex(verts_features=texs)
    
    save_path = Path('/home/juseonghan/consistent_style_transfer/experiments/example')
    export_mesh(save_path, mesh_torch3d, None, -1)
    breakpoint()
    
    # manual pytorch3d biz
    # verts_rgb = torch.ones_like(mesh_torch3d.verts_packed())[None].float()  # (1, V, 3)
    # # verts_rgb = torch.load('/home/juseonghan/consistent_style_transfer/experiments/030524_newgaussian2/learned_textures.pt')
    # # verts_rgb = verts_rgb[None]
    # tex = TexturesVertex(verts_features=verts_rgb.to(device))
    # mesh_torch3d.textures = tex
    # mesh_torch3d.textures._verts_features_list = torch.ones_like(mesh_torch3d.verts_packed())[None].float()

    # textures = torch.load('/home/juseonghan/Downloads/learned_textures.pt')
    # verts_rgb = textures[None]
    # tex = TexturesVertex(verts_features=verts_rgb.to(device))
    # mesh_torch3d.textures = tex

    # renderz 
    print('Starting Renders...')
    # render_batch_size = int(config['render_batch_size'])
    # if 'num_cameras' not in config:
    #     num_iters = math.ceil(len(poses_scaled) / render_batch_size)
    #     print(f'{num_iters} iterations!')
    
    # initialize training stuff
    # TextureModel = TextureLearner(config, args.resume).to(device)
    # TextureModel.eval()
    # loss_fn = read_loss_fn(config)
    
    # a (num_vertices,), dtype=bool torch tensor that signifies whether or not a vertex has been updated
    # white_mesh = mesh_torch3d.clone()
    # verts_rgb = torch.ones_like(mesh_torch3d.verts_packed())[None].float()  # (1, V, 3)
    # tex = TexturesVertex(verts_features=verts_rgb.to(device))
    # white_mesh.textures = tex
    # learned_textures = torch.ones_like(mesh_torch3d.verts_packed())

    # try training with light sources
    # light_intensities = torch.rand(len(poses_scaled), 6)
    # batch_size = int(config['batch_size'])
    # dataset = STDataset(config, white_mesh, poses_scaled)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # num_epochs = int(config['num_epochs'])

    # scheduler and optim init
    # if config['scheduler'] == 'step':
    #     print('Using StepLR scheduling w/ Adam')
    #     optim = torch.optim.Adam(TextureModel.parameters(), float(config['lr']), weight_decay=0)
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(config['decay_step']), gamma=float(config['lr_decay']))
    # elif config['scheduler'] == 'cosine':
    #     steps = math.floor(len(dataset) / batch_size)
    #     optim = torch.optim.SGD(TextureModel.parameters(), lr=1.)
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)
    #     print('Using Cosine Annealing Scheduler with SGD')
    # else:
    #     raise Exception('invalid scheduler in config, needs to be step or cosine')
    
    # if args.resume:
    #     ckpt_path = config['texlearner_ckpt']
    #     ckpt = torch.load(ckpt_path)
    #     TextureModel.load_state_dict(ckpt['model_state_dict'])
    #     optim.load_state_dict(ckpt['optim_state_dict'])
    #     # lr_scheduler.load_state_dict(ckpt['scheduler'])
    #     print('Resuming!')
    #     # print(f'Resuming from {ckpt['epoch']}!')
    #     poses_scaled = np.load(config['poses_ckpt'])
    #     poses_scaled = list(poses_scaled)
    
    pose_rendered_pair = []
    save_path = Path(args.output)
    assert save_path.exists(), 'save_path does not exist'
    temp = save_path / 'color'
    if not temp.exists():
        os.mkdir(str(temp))
    temp = save_path / 'verts'
    if not temp.exists():
        os.mkdir(str(temp))
    render_batch_size = int(config['batch_size'])
    num_iters = int(len(poses_scaled) / render_batch_size)
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
        render_batch_with_pointmap(config, iter, mesh_torch3d, R_batch, positions_batch, this_batch_size, save_path, device=device)
        # rendered_imgs = extract_imgs(config, save_mesh_path, render_rgb, iter)
        # save_imgs(config, save_mesh_path, render_rgb, iter)
        poses_scaled = poses_scaled[render_batch_size:]

    # losses_vs_epochs = []
    # for epoch in range(num_epochs):
        
    #     mesh_torch3d, losses = eval(config, 
    #                                 mesh_torch3d, 
    #                                 TextureModel,
    #                                 dataloader,
    #                                 loss_fn,
    #                                 optim,  
    #                                 lr_scheduler, 
    #                                 save_path)
        
    #     record_losses(save_path, config, losses, epoch)
    #     # logging
    #     if epoch % int(config['save_logs_every']) == 0:
    #         export_mesh(save_path, mesh_torch3d, learned_textures, epoch)
        
    #     avg_loss = sum(losses) / len(losses)
    #     print(f'Loss for epoch {epoch}/{num_epochs} is {avg_loss}')
    #     losses_vs_epochs.append(avg_loss)

    # log_training(save_path, mesh_torch3d, TextureModel, optim, lr_scheduler, poses_scaled, epoch)
    # export_mesh(save_path, mesh_torch3d, learned_textures, -1)
    # plt.figure()
    # plt.plot(np.array(losses_vs_epochs))
    # save_loss_path = save_path / 'logs' / 'loss_vs_epoch.png'
    # plt.savefig(str(save_loss_path))
    # export_poses(save_path, poses_scaled, 'train')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/config.yaml")
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args)