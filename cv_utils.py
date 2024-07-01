# standard python stuff
import numpy as np
from pc_skeletor import LBC, SLBC
import skeletor
import open3d as o3d
from pathlib import Path 
from tqdm import tqdm 
import copy 
import trimesh
from PIL import Image
import cv2 as cv 
import os 
import random
import matplotlib.pyplot as plt
import time 
from skimage import io, color
# pytorch
import torch 
import torchvision
# from torchvision.utils import save_image
from kornia.color import rgb_to_lab

# pytorch3d
import pytorch3d
import pytorch3d.utils
import pytorch3d.io
from pytorch3d.structures import Meshes 
from pytorch3d.renderer import (PerspectiveCameras, 
                                MeshRasterizer, 
                                MeshRenderer,
                                SoftGouraudShader,
                                HardFlatShader,
                                SoftPhongShader,
                                HardPhongShader,
                                HardGouraudShader,
                                TexturesVertex,
                                AmbientLights, 
                                PointLights,
                                DirectionalLights,
                                RasterizationSettings)

# mine
from utils import read_intrinsics, read_config, to_uint8, device, torch_to_PIL, make_grid, read_mask
from models import RendererWithDepth

# munit trainer 
from I2I.trainer import MUNIT_Trainer

def generate_trajectory_SK(config):
    mesh_path = config['path_to_mesh']
    assert Path(mesh_path).exists(), 'path to mesh in config file invalid'
    mesh = trimesh.load_mesh(str(mesh_path))

    fixed = skeletor.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
    skel = skeletor.skeletonize.by_wavefront(fixed, waves=1, step_size=1)

    pts = np.array(skel.vertices)
    inds = np.array(skel.edges)
    # skel.show(mesh=True)

    cam_poses = []
    for ind in tqdm(inds):
        pt1, pt2 = pts[ind[0]], pts[ind[1]]
        W = (pt2 - pt1) / np.linalg.norm(pt2 - pt1) # the unit vector that we are generating the spiral from
        U = np.cross(W, np.array([1,0,0]))
        U = U / np.linalg.norm(U) # unit vector for sanity's sake
        V = np.cross(W, U) / np.linalg.norm(np.cross(W, U))
        
        # radius a, b/a pitch. let's say we want two full rotations for a line segment. 
        # then the pitch should be half of our line segment length.  
        a = 0.5
        b = 0.5 * a * np.linalg.norm(pt2 - pt1)
        # number of samples should scale according to the length of our line segment 
        num_samples = int(20 * np.linalg.norm(pt2 - pt1))
        for t in np.linspace(0, 4 * np.pi, num_samples):
            cam_pos = (a * np.cos(t)) * U + (a * np.sin(t)) * V + (b*t) * W
            # translate the camera pose to where we were originally
            cam_pos += pt1 

            # check if it's inside to make sure we're not adding any silly points 
            # query_pt = o3d.core.Tensor(cam_pos[np.newaxis,:], dtype=o3d.core.Dtype.Float32)
            # if scene_raycast.compute_signed_distance(query_pt).item() > -1.2:
            #     continue 

            # to get the rotation, use axis angle where the axis is the forward vector, and rotate
            # the left vector. then recalculate the new up vector with cross prod. use rodriguez formula
            new_left = np.cos(t) * U + np.sin(t)*(np.cross(W, U)) + (1 - np.cos(t)) * (np.dot(W, U)) * W
            new_up = np.cross(W, new_left)
            Rot = np.hstack((new_left[:,np.newaxis], new_up[:,np.newaxis], W[:,np.newaxis]))
            SE3 = np.block([[Rot, cam_pos[:,np.newaxis]],
                            [0.,0.,0.,1.]])
            cam_poses.append(SE3)    
        if int(config['one_sequence']) == 1 and len(cam_poses) != 0:
            print('generating only one sequence')
            break

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return mesh, cam_poses 

def get_directional_rots():
    rots = []
    rots.append(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])) # forward
    rots.append(np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])) # left 
    # rots.append(np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])) # up 
    rots.append(np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])) # right
    # rots.append(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])) # down 
    rots.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])) # back 
    return rots
                     
"""
    Using a Laplacian-based Contraction method, skeletonize the point cloud (mesh)
    to generate camera trajectories that "explores" the interiors of the anatomy
"""
def generate_trajectory_LBC(config):

    # configuration reading + preprocessing 
    mesh_path = config['path_to_mesh']
    assert Path(mesh_path).exists(), 'path to mesh in config file invalid'
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pc = mesh.sample_points_uniformly(number_of_points=40_000)
    pc_filt, _ = pc.remove_radius_outlier(nb_points=1, radius=1)

    # using LBC to create skeleton for trajectories
    print('Generating Camera Trajectories...')
    lbc = LBC(point_cloud=pc_filt, down_sample=0.008)
    lbc.extract_skeleton()
    sk = lbc.extract_topology()
    o3d.visualization.draw_geometries([sk, mesh]) # optional 

    # do some fancy math
    N = 5.0 # the amount of interpolation we want to do along the line
    cam_poses = [] # a list of 4x4 homogeneous matrices

    pts = np.array(sk.points)
    inds = np.array(sk.lines)

    # let's get fancy and create a helix about each line segment
    # so that we can create nice renders with varying orientations
    # reference: https://math.stackexchange.com/questions/1723910/helix-along-vector-in-3d-space
    print('Creating an initial trajectory...')
    for ind in tqdm(inds):
        pt1, pt2 = pts[ind[0]], pts[ind[1]]
        W = (pt2 - pt1) / np.linalg.norm(pt2 - pt1) # the unit vector that we are generating the spiral from
        U = np.cross(W, np.array([1,0,0]))
        U = U / np.linalg.norm(U) # unit vector for sanity's sake
        V = np.cross(W, U) / np.linalg.norm(np.cross(W, U))
        
        # radius a, b/a pitch. let's say we want two full rotations for a line segment. 
        # then the pitch should be half of our line segment length.  
        a = 1.
        b = 0.5 * a * np.linalg.norm(pt2 - pt1)
        # number of samples should scale according to the length of our line segment 
        num_samples = int(130 * np.linalg.norm(pt2 - pt1))
        for t in np.linspace(0, 4 * np.pi, num_samples):
            cam_pos = (a * np.cos(t)) * U + (a * np.sin(t)) * V + (b*t) * W
            # translate the camera pose to where we were originally
            cam_pos += pt1 

            # check if it's inside to make sure we're not adding any silly points 
            # query_pt = o3d.core.Tensor(cam_pos[np.newaxis,:], dtype=o3d.core.Dtype.Float32)
            # if scene_raycast.compute_signed_distance(query_pt).item() > -1.2:
            #     continue 

            # to get the rotation, use axis angle where the axis is the forward vector, and rotate
            # the left vector. then recalculate the new up vector with cross prod. use rodriguez formula
            new_left = np.cos(t) * U + np.sin(t)*(np.cross(W, U)) + (1 - np.cos(t)) * (np.dot(W, U)) * W
            new_up = np.cross(W, new_left)
            Rot = np.hstack((new_left[:,np.newaxis], new_up[:,np.newaxis], W[:,np.newaxis]))
            SE3 = np.block([[Rot, cam_pos[:,np.newaxis]],
                            [0.,0.,0.,1.]])
            cam_poses.append(SE3)    
        if int(config['one_sequence']) == 1 and len(cam_poses) != 0:
            print('generating only one sequence')
            break

    return mesh, cam_poses

"""
    This function will center the mesh and the camera poses 
    while making sure that the poses are valid (e.g. not too close
    to the surface and inside the mesh)
"""
def preprocess_pose_and_mesh(config, mesh, poses):

    print('Cleaning up mesh and trajectory...')
    thresh = float(config['include_threshold'])
    assert thresh < 0, 'the threshold has to be negative or the traj will be outside'
    # translate mesh to origin
    center = mesh.get_center()
    subdivide = int(config['subdivide'])
    if subdivide == 1:
        mesh_centered = copy.deepcopy(mesh).translate(-center).subdivide_loop(3)
    else:
        mesh_centered = copy.deepcopy(mesh).translate(-center)
    
    # get raycasting scene to calculate distance to surface
    _mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_centered)
    scene_raycast = o3d.t.geometry.RaycastingScene()
    _ = scene_raycast.add_triangles(_mesh)
    
    pose_centered = []
    for pose in tqdm(poses):
        new_pos = pose[:3,3] - center 
        pos_tensor = o3d.core.Tensor(new_pos[None], dtype=o3d.core.Dtype.Float32)
        if scene_raycast.compute_signed_distance(pos_tensor).item() > thresh:
            continue 
        pose[:3,3] -= center
        pose_centered.append(pose)
    return mesh_centered, pose_centered
        
def down_sample_cameras(poses, num_poses):
    positions_list = [pose[:3,3] for pose in poses]
    positions = np.array([pose[:3,3] for pose in poses])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd_down = pcd.farthest_point_down_sample(num_poses)

    # i apolgize to the creator of python
    result = []
    for point in np.array(pcd_down.points):
        for i, pos in enumerate(positions_list):
            if pos[0] == point[0] and pos[1] == point[1] and pos[2] == point[2]:
                result.append(poses[i])
    return result

def render_batch_with_pointmap(config, iter, mesh, R_batch, positions_batch, this_batch_size, save_path, device=device):
    # save_path = '/home/juseonghan/consistent_style_transfer/experiments/030624_final/featurematch/'

    # load in similar rendering objects 
    K = read_intrinsics(config).to(device)
    img_size = int(config['render_img_size'])
    _img_size = torch.Tensor([[img_size, img_size]]).to(device)
    if int(config['scale_intrinsics']) == 1:
        K = scale_intr(K, img_size)
    # batch_size = int(config['render_batch_size'])
    
    cameras = pytorch3d.utils.cameras_from_opencv_projection(R=R_batch,
                                                             tvec=positions_batch,
                                                             camera_matrix=K[None, ...],
                                                             image_size=_img_size).to(device)
    
    # have to create a rasterizer to get fragments to detect which faces are
    # in view
    img_size = 256
    settings = RasterizationSettings(
        image_size=img_size,
        bin_size=0,
        max_faces_per_bin=10_000,
        faces_per_pixel=1
    )
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=settings
    )
    
    # render
    batch_size = R_batch.shape[0]
    meshes = mesh.extend(batch_size)

    fragments = rasterizer(meshes)

    # get the visible vertices
    # https://github.com/facebookresearch/pytorch3d/issues/126
    visible_face_indices = torch.squeeze(fragments.pix_to_face) # (B, H, W) of indices of visible faces in image
    packed_faces = meshes.faces_packed()
    packed_verts = meshes.verts_packed() 
    face_norms_x, face_norms_y, face_norms_z = packed_faces[:,0], packed_faces[:,1], packed_faces[:,2]
    visible_normx = torch.take(face_norms_x, visible_face_indices)
    visible_normy = torch.take(face_norms_y, visible_face_indices)
    visible_normz = torch.take(face_norms_z, visible_face_indices)
    visible_faces = torch.stack((visible_normx, visible_normy, visible_normz), dim=3) 
    visible_vert_idx = visible_faces[:,:,:,1] # B, H, W, giving index of what vertices you want

    visible_normx = torch.take(packed_verts[:,0], visible_vert_idx)
    visible_normy = torch.take(packed_verts[:,1], visible_vert_idx)
    visible_normz = torch.take(packed_verts[:,2], visible_vert_idx)
    verts_per_pixel = torch.stack((visible_normx, visible_normy, visible_normz), dim=3) 

    lights_batch = PointLights(ambient_color=((1., 1., 1.),),
                                diffuse_color=((0.5, 0.5, 0.5),),
                                # specular_color=lights_config[2], 
                                location=cameras.get_camera_center(),
                                device=device)
    settings = RasterizationSettings(
        image_size=img_size,
        bin_size=0,
        max_faces_per_bin=10_000
    )
    renderer = RendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras, 
            lights=lights_batch
        )
    )
    
    images, _ = renderer(meshes, cameras=cameras, lights=lights_batch)
    images = images[:,:,:,:3].detach().cpu().numpy()


    # save images 
    actual_batch_size = config['batch_size']
    visible_verts = verts_per_pixel.detach().cpu().numpy()

    for i in range(batch_size):
        temp = 'color/' + str(iter * actual_batch_size + i) + '.jpg'
        save_name = str(save_path / temp)
        color = images[i,...]
        color = cv.cvtColor(color, cv.COLOR_RGB2BGR)
        color = to_uint8(color)
        cv.imwrite(save_name, color)

        visible_vert = visible_verts[i, ...]
        temp = 'verts/' + str(iter * actual_batch_size + i) + '.npy'
        save_name = str(save_path / temp)
        np.save(save_name, visible_vert)


"""
    Visualize mesh and trajectory after we zero-center
"""
def mesh_traj_vis(mesh, poses):

    # extract traj positions
    positions = np.array([pose[:3,3] for pose in poses])
    print(f'we have {positions.shape} points in the traj')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    print('PRESS Q TO CLOSE')
    o3d.visualization.draw_geometries([pcd, mesh])

"""
    pretty shoddy, but the original intrinsics were done with 1080x1080 resolution
"""
def scale_intr(K, img_size):
    s = img_size / 1080
    K[0,:] *= s
    K[1,:] *= s
    return K

"""
    Helper function to export mesh into ply format and saving the textures
"""
def export_mesh(save_path, mesh, textures_pred, iter):
    # final_verts, final_faces = mesh.get_mesh_verts_faces(0)
    from pytorch3d.io import IO
    # pytorch3d.io.save_obj(str(log_dir / 'textured_mesh.obj'), final_verts, final_faces)
    # textures = TexturesVertex(verts_features=textures_pred[None, ...])
    # mesh.textures = textures 
    if iter == -1:
        save_name = f'final_textured.ply'
        IO().save_mesh(mesh, str(save_path / 'textured_meshes' / save_name), colors_as_uint8=True, binary=False)
    else:
        save_name = f'textured_mesh_{iter}.ply'
        IO().save_mesh(mesh, str(save_path / 'textured_meshes' / save_name), colors_as_uint8=True, binary=False)

"""
    Open3D has computer vision conventions +x right, +y down, +z into screen
    PyTorch3D has its own nonsense, +x left, +y up, and +z into screen    
"""
def open3d_to_torch3d(mesh):
    verts = mesh.verts_packed()
    verts[:,0] *= -1
    verts[:,1] *= -1
    mesh.verts = verts
    return mesh 

"""
    Render a batch of images (rgb and depth) given mesh and camera pose. 
"""
def render_batch(config, mesh, R_batch, pos_batch, this_batch_size, device, troll=False):
    K = read_intrinsics(config).to(device)
    img_size = int(config['render_img_size'])
    _img_size = torch.Tensor([[img_size, img_size]]).to(device)
    if int(config['scale_intrinsics']) == 1:
        K = scale_intr(K, img_size)
    # batch_size = int(config['render_batch_size'])
    cameras = pytorch3d.utils.cameras_from_opencv_projection(R=R_batch,
                                                             tvec=pos_batch,
                                                             camera_matrix=K[None, ...],
                                                             image_size=_img_size).to(device)

    # lights_config = read_lights(config)
    if troll:
        lights_batch = PointLights(ambient_color=((0.25, 0.25, 0.25),),
                                    diffuse_color=((0.5, 0.5, 0.5),),
                                    # specular_color=lights_config[2], 
                                    location=cameras.get_camera_center(),
                                    device=device)
    else: 
        lights_batch = PointLights(ambient_color=((1., 1., 1.),),
                                    diffuse_color=((0.5, 0.5, 0.5),),
                                    # specular_color=lights_config[2], 
                                    location=cameras.get_camera_center(),
                                    device=device)
    # light_tensor = torch.Tensor([[1., 1., 1.]]).repeat(this_batch_size, 1)

    # lights_batch = PointLights(ambient_color=light_tensor, location=cameras.get_camera_center(), device=device)
    settings = RasterizationSettings(
        image_size=img_size,
        bin_size=0,
        max_faces_per_bin=10_000
    )
    renderer = RendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras, 
            lights=lights_batch
        )
    )
    
    meshes = mesh.extend(this_batch_size)
    images, depth = renderer(meshes, cameras=cameras, lights=lights_batch)

    # calculate visible vertices in the frame
    # verts_world = mesh.verts_padded()
    # verts_screen = cameras.transform_points(verts_world)
    return images, depth
    
"""
    Given mesh and camera extrinsics, calculate the vertices that are
    "in view" or "close enough" to the camera so that we don't update
    the entire mesh at every iteration
"""
def weighted_loss_mask(config, mesh, Rs, ps):

    # load in similar rendering objects 
    K = read_intrinsics(config).to(device)
    img_size = int(config['render_img_size'])
    _img_size = torch.Tensor([[img_size, img_size]]).to(device)
    if int(config['scale_intrinsics']) == 1:
        K = scale_intr(K, img_size)
    # batch_size = int(config['render_batch_size'])
    cameras = pytorch3d.utils.cameras_from_opencv_projection(R=Rs,
                                                             tvec=ps,
                                                             camera_matrix=K[None, ...],
                                                             image_size=_img_size).to(device)
    
    # have to create a rasterizer to get fragments to detect which faces are
    # in view
    img_size = 256
    settings = RasterizationSettings(
        image_size=img_size,
        bin_size=0,
        max_faces_per_bin=10_000,
        faces_per_pixel=1
    )
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=settings
    )
    
    # render
    batch_size = Rs.shape[0]
    meshes = mesh.extend(batch_size)
    fragments = rasterizer(meshes)

    # get the visible vertices
    # https://github.com/facebookresearch/pytorch3d/issues/126
    visible_face_indices = torch.squeeze(fragments.pix_to_face) # (B, H, W) of indices of visible faces in image
    if batch_size == 1:
        visible_face_indices = torch.unsqueeze(visible_face_indices, 0)
    face_norms = meshes.faces_normals_packed()
    face_norms_x, face_norms_y, face_norms_z = face_norms[:,0], face_norms[:,1], face_norms[:,2]
    visible_normx = torch.take(face_norms_x, visible_face_indices)
    visible_normy = torch.take(face_norms_y, visible_face_indices)
    visible_normz = torch.take(face_norms_z, visible_face_indices)
    visible_norms = torch.stack((visible_normx, visible_normy, visible_normz), dim=3) 

    # calculate angle
    cam_poses = cameras.get_world_to_view_transform().inverse().get_matrix()
    cam_poses = [torch.t(cam_poses[i,...]) for i in range(batch_size)]
    cam_pose_in_world = torch.stack(cam_poses, dim=0)
    cam_view_dir_in_world = cam_pose_in_world[:,:3,2] # (B, 3,) tensor

    # need to do dot product of (B, 3) and visible_norms (B, H, W, 3) to get (B, H, W)
    result = []
    for i in range(batch_size):
        A = visible_norms[i,...] # (H,W,3)
        A_mag = torch.norm(A, dim=2) #(H,W)
        B = cam_view_dir_in_world[i] # (3,)
        B_mag = torch.norm(B) #scalar
        result.append(torch.sum(A*B, 2) / (A_mag * B_mag)) # (H, W)

    cosines = torch.stack(result, dim=0) 
    # angles = torch.acos(cosines) * 180.0 / np.pi # convert to degrees
    distances = torch.squeeze(fragments.zbuf)
    if batch_size == 1:
        distances = torch.unsqueeze(distances, 0)
    result_mask = create_heatmap(config, cosines, distances)
    return result_mask
    
def create_heatmap(config, cosines, distances):
    
    # hyperparams
    distance_threshold = float(config['valid_vertices_dist_threshold'])
    # normal_threshold_min = float(config['valid_vertices_normal_threshold_min'])
    # normal_threshold_max = float(config['valid_vertices_normal_threshold_max'])

    # get the masks depending on hyperparams
    # angle_mask = torch.logical_or(angles < normal_threshold_min, angles > normal_threshold_max)
    angle_mask = cosines > 0.3
    distance_mask = distances > distance_threshold

    # angles[angle_mask] = 0.0
    min_angle, max_angle = torch.min(cosines), torch.max(cosines)
    angles_HM = (cosines - min_angle) / (max_angle - min_angle)

    # get it to [0,1]
    distances[distance_mask] = torch.max(distances)
    min_distance = torch.min(distances) 
    max_distance = torch.max(distances) 
    distance_HM = (distances - min_distance) / (max_distance - min_distance) # 0 to 1

    angles_HM = (1.0 - cosines) / 2.0
    distance_HM = 1 - distance_HM
    result = (angles_HM + distance_HM) / 2
    result[distance_mask] = 0.0
    elimination_mask = torch.logical_or(angle_mask, distance_mask)
    
    # result[elimination_mask] = 0.0

    # eliminate where it exceeds distance threshold 

    return result 

"""
    Given camera position, mesh vertices, and which vertices are visible to the camera,
    calculate the indices of vertices that are within a certain threshold to the camera
"""
def get_valid_vertices(config, cameras, mesh_verts, mesh_norms, visibility_map):
    visibility_map = visibility_map.to(device)
    T = torch.t(cameras.get_world_to_view_transform().inverse().get_matrix()[0])
    cam_forward_vec = T[:3,2]
    cam_norm = torch.norm(cam_forward_vec)
    cam_forward_vec = cam_forward_vec.repeat(mesh_verts.shape[0], 1)

    cam_positions = cameras.get_camera_center()
    dist_to_each_vert = torch.sqrt(torch.sum(torch.square(cam_positions - mesh_verts), dim=1)).to(device)
    distance_threshold = float(config['valid_vertices_dist_threshold'])
    normal_threshold_min = float(config['valid_vertices_normal_threshold_min'])
    normal_threshold_max = float(config['valid_vertices_normal_threshold_max'])
    mask1 = torch.logical_and(visibility_map == 1, dist_to_each_vert < distance_threshold)
    mesh_norms_norm = torch.norm(mesh_norms[0], dim=1)

    dot_prods = (mesh_norms[0] * cam_forward_vec).sum(axis = 1)
    angles = torch.acos(dot_prods / (cam_norm * mesh_norms_norm)) * 180 / np.pi    
    mask2 = torch.logical_and(angles < normal_threshold_max, angles > normal_threshold_min)
    mask = torch.logical_and(mask1, mask2)
    return mask

def loss_proper_pixels(loss_fn, render_rgb, ST_image):
    pass 


"""
    Reading lights from our config 
"""
def read_lights(config):
    a, d, s = config['ambient_light'], config['diffuse_light'], config['specular_light']
    a, d, s = float(a), float(d), float(s)
    # render_batch_size = int(config['render_batch_size'])
    ambient = ((a,a,a),)
    diffuse = ((d,d,d),)
    specular = ((s,s,s),)
    return ambient, diffuse, specular 

"""
    Get the trainer model to run inference on the trained style transfer model
"""
def prepare_styletransfer_model(config):

    checkpoint_path = Path(config['styletransfer_ckpt'])
    assert checkpoint_path.exists(), 'styletransfer_ckpt in config path does not exist...'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load in their stuff bc i'm lazy
    conf_path = config['styletransfer_config']
    conf = read_config(conf_path)
    # style_dim = conf['gen']['style_dim'] # should be 5
    trainer = MUNIT_Trainer(conf)
    state_dict = torch.load(checkpoint_path)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    trainer.cuda()
    trainer.eval()
    return trainer 

"""
    Run inference given the trainer model and a batch of images 
    Args:
        batch_imgs:  torch.Tensor shape (B, H, W) 
        trainer:     trainer object, MUNIT_Trainer()
"""
def inference_styletransfer(trainer, batch_imgs, mask):
    # random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    batch_imgs = batch_imgs.detach().cpu().numpy()
    imgs = batch_imgs[0,:,:,:3]
    
    # NEED THIS LATER
    imgs = to_uint8(imgs)
    imgs[mask==255] = 0
    # cv.imwrite('temp.jpg', imgs)
    # imgs = Image.open('temp.jpg').convert('RGB')
    imgs = torchvision.transforms.functional.to_tensor( imgs )
    imgs = torchvision.transforms.functional.normalize( imgs, (0.5,0.5,0.5), (0.5,0.5,0.5) )
    imgs = torchvision.transforms.functional.resize( imgs, (256, 256), Image.BILINEAR )
    imgs = imgs[None]
    with torch.no_grad():
        x_a = imgs.cuda()
        c_a = trainer.gen_a.encode( x_a )
        s_b = torch.randn(1, 8, 1, 1).cuda()
        x_ab = trainer.gen_b.decode( c_a, s_b )
        x_ab = (x_ab.data + 1) / 2
    
    # torchvision.utils.save_image(x_ab, 'temp.jpg', padding=0, normalize=False)
    # img = cv.imread('./temp.jpg')
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # img = torch.Tensor(img).permute(2, 0, 1)
    # img = img[None] / 255.0
    # img = make_grid(x_ab, nrow=1, padding=0)
    # img = torch_to_PIL(img, padding=0, normalize=False)
    return torch.squeeze(x_ab)

"""
    both mesh and poses are currently in open3d convention
    poses can stay because pytorch3d has a built in thing for it 
"""
def prepare_coords(mesh, poses):
    mesh_verts = np.array(mesh.vertices)
    assert mesh_verts.shape[1] == 3, 'incorrect vertices loading, should be (N, 3)'

    # convert mesh from open3d to pytorch3d
    # mesh_verts[:,:2] *= -1
    verts_tensor = torch.Tensor(mesh_verts)
    faces = np.array(mesh.triangles)
    faces_tensor = torch.Tensor(faces)

    verts_rgb = torch.ones_like(verts_tensor)[None].float()  # (1, V, 3)
    tex = TexturesVertex(verts_features=verts_rgb.to(device))

    output_mesh = Meshes(verts=[verts_tensor], faces=[faces_tensor])

    # now convert camera poses from pyrender to opencv 
    output_poses = []
    for pose in poses:
        pose = np.linalg.inv(pose)
        pose[1:3,:] *= -1 
        output_poses.append(pose)
    return output_mesh, output_poses

def train(config, mesh, model, dataloader, loss_fn, optim, scheduler, save_mesh_path):
    
    losses = []
    mask = read_mask(config)
    mask = cv.resize(mask, (256, 256))

    for i, (positions_batch, R_batch, ST_image, _) in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch_size = ST_image.shape[0]
        _mask = np.zeros((mask.shape[0], mask.shape[1], batch_size))
        for i in range(batch_size): 
            _mask[:,:,i] = mask
        _mask = torch.Tensor(_mask).permute(2,0,1)

        # run forward pass and update mesh w/ our predictions
        R_batch, positions_batch = torch.squeeze(R_batch), torch.squeeze(positions_batch)
        if batch_size == 1:
            R_batch, positions_batch = R_batch[None], positions_batch[None]
        positions_batch = positions_batch.to(device).float() 
        R_batch = R_batch.to(device).float() 

        preds = model(mesh.verts_packed())
        tex = TexturesVertex(verts_features=preds[None, ...])
        mesh.textures = tex 
        
        # re-render and make it same size as our style transferred images
        render_rgb, _ = render_batch(config, mesh, R_batch, positions_batch, batch_size, device=device, troll=True)
        render_rgb = render_rgb[:,:,:,:3].permute(0, 3, 1, 2)
        render_rgb = torchvision.transforms.functional.resize( render_rgb, (256, 256), Image.BILINEAR )

        loss, heatmap = view_weighted_loss(config, mesh, loss_fn, render_rgb, _mask, ST_image, R_batch, positions_batch)
        optim.zero_grad()
        loss.backward() 
        optim.step() 
        scheduler.step() 

        # if i % report_loss_every == 0:
        #     print(f'Current Loss for iteration {i}/{num_iterations}: {loss.item()}')
        visualize_st_preds_hm(ST_image, render_rgb, heatmap, _mask, save_mesh_path, batch_size)
        # visualize_training3(OG_render, ST_image, render_rgb, heatmap, _mask, save_mesh_path, batch_size)
        losses.append(loss.item())
    
    return mesh, losses

def run_inference(config, mesh, model, dataloader, loss_fn, optim, scheduler, save_mesh_path):
    
    losses = []
    mask = read_mask(config)
    mask = cv.resize(mask, (256, 256))

    for i, (positions_batch, R_batch, ST_image, OG_render) in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch_size = ST_image.shape[0]
        _mask = np.zeros((mask.shape[0], mask.shape[1], batch_size))
        for i in range(batch_size): 
            _mask[:,:,i] = mask
        _mask = torch.Tensor(_mask).permute(2,0,1)

        # run forward pass and update mesh w/ our predictions
        R_batch, positions_batch = torch.squeeze(R_batch), torch.squeeze(positions_batch)
        if batch_size == 1:
            R_batch, positions_batch = R_batch[None], positions_batch[None]
        positions_batch = positions_batch.to(device).float() 
        R_batch = R_batch.to(device).float() 

        # preds = model(mesh.verts_packed())
        # tex = TexturesVertex(verts_features=preds[None, ...])
        # mesh.textures = tex 
        
        # re-render and make it same size as our style transferred images
        render_rgb, _ = render_batch(config, mesh, R_batch, positions_batch, batch_size, device=device, troll=True)
        render_rgb = render_rgb[:,:,:,:3].permute(0, 3, 1, 2)
        render_rgb = torchvision.transforms.functional.resize( render_rgb, (256, 256), Image.BILINEAR )

        loss, heatmap = view_weighted_loss(config, mesh, loss_fn, render_rgb, _mask, ST_image, R_batch, positions_batch)
        # optim.zero_grad()
        # loss.backward() 
        # optim.step() 
        # scheduler.step() 

        # if i % report_loss_every == 0:
        #     print(f'Current Loss for iteration {i}/{num_iterations}: {loss.item()}')
        # visualize_training2(ST_image, render_rgb, heatmap, _mask, save_mesh_path, batch_size)
        visualize_og_st_preds_hm(OG_render, ST_image, render_rgb, heatmap, _mask, save_mesh_path, batch_size)
        losses.append(loss.item())
    
    return mesh, losses

"""
    Given two rendered images of size (B, C, H, W), calculate the view_weighted_loss
    L = alpha * MSE(img1, img2)
    where alpha is a weighted term that weights "valid pixels" more heavily and invalid pixels = 0
    invalid and valid depends on view angle and vertex distance from camera  
"""
def view_weighted_loss(config, mesh, loss_fn, render_rgb, mask, ST_image, R_batch, positions_batch):
    
    heatmap = weighted_loss_mask(config, mesh, R_batch, positions_batch)
    # ST_image = scale_0_1(ST_image)
    render_rgb, ST_image = rgb_to_lab(render_rgb), rgb_to_lab(ST_image)
    mse = loss_fn(render_rgb, ST_image)
    
    # apply mask
    # heatmap = torch.ones_like(mask).to(device)
    heatmap[mask==255] = 0.0
    heatmap = torch.stack((heatmap, heatmap, heatmap), dim=1) # 3 channel image
    
    weighted_loss = heatmap * mse 
    # weighted_loss = mse 
    result = torch.mean(weighted_loss)
    return result, heatmap

def visualize_og_st_preds_hm(og, st, preds, heatmap, mask, save_mesh_path, batch_size):   
    for i in range(batch_size):
        _mask = mask[i,...]
        _og = og[i,0,:,:,:3].detach().cpu().numpy()
        _og = to_uint8(cv.cvtColor(_og, cv.COLOR_RGB2BGR))
        _og = cv.resize(_og, (256, 256))
        _og[_mask==255] = 255
        _st = st[i, ...].detach().cpu().permute(1, 2, 0).numpy()
        _st = to_uint8(cv.cvtColor(_st, cv.COLOR_RGB2BGR))
        _st[_mask==255] = 255
        _preds = preds[i, ...].detach().cpu().permute(1, 2, 0).numpy()
        _preds = cv.cvtColor(_preds, cv.COLOR_RGB2BGR)
        _preds = to_uint8(_preds)
        _preds[_mask==255] = 255
        _heatmap = heatmap[i, 0, ...].cpu().numpy() # grayscale img
        _heatmap = cv.applyColorMap(to_uint8(_heatmap), cv.COLORMAP_JET)
        cv.imwrite('og.png', _og)
        cv.imwrite('st.png', _st)
        cv.imwrite('preds.png', _preds)
        cv.imwrite('heatmap.png', _heatmap)
        breakpoint()

def visualize_st_preds_hm(st, preds, heatmap, mask, save_mesh_path, batch_size):   
    for i in range(batch_size):
        _mask = mask[i,...]
        _st = st[i, ...].detach().cpu().permute(1, 2, 0).numpy()
        _st = to_uint8(cv.cvtColor(_st, cv.COLOR_RGB2BGR))
        _preds = preds[i, ...].detach().cpu().permute(1, 2, 0).numpy()
        _preds = cv.cvtColor(_preds, cv.COLOR_RGB2BGR)
        _preds = to_uint8(_preds)
        _preds[_mask==255] = 0.0
        _heatmap = heatmap[i, 0, ...].cpu().numpy() # grayscale img
        _heatmap = cv.applyColorMap(to_uint8(_heatmap), cv.COLORMAP_JET)
        # _heatmap = cv.cvtColor(_heatmap, cv.COLOR_RGB2BGR)
        if i == 0:
            base = np.vstack((_st, _preds, _heatmap))
        else:
            add_on = np.vstack((_st, _preds, _heatmap))
            base = np.hstack((base, add_on))
    save_name = str(save_mesh_path / 'visualize_training.jpg')
    cv.imwrite(save_name, base)

def view_ST_outputs(og, st):
    # og is (1, H, W, 4)
    # st is (3, H, W)
    og = og[0,:,:,:3].cpu().numpy()
    st = st.permute(1, 2, 0).cpu().numpy()
    og, st = to_uint8(og), to_uint8(st)
    st = cv.cvtColor(st, cv.COLOR_BGR2RGB)
    og = cv.resize(og, (256, 256))
    img = np.vstack((og, st))
    cv.imwrite('view_ST.png', img)

def rescale_to_255(arr):
    a = 255.0 * arr
    a = np.clip(a, 0, 255)
    return a.astype(np.uint8)

def scale_0_1(arr):
    temp =  (arr - arr.min()) * (1.0 / (arr.max() - arr.min()))
    return temp

# testing
if __name__ == '__main__':
    config = read_config('./configs/config.yaml')
    trainer = prepare_styletransfer_model(config)

    x = torch.rand(1, 3, 340, 340) # make sure size is 256, 256
    x = torchvision.transforms.functional.normalize( x, (0.5,0.5,0.5), (0.5,0.5,0.5) )
    x = torchvision.transforms.functional.resize( x, (256, 256), Image.BILINEAR )
    print(x.shape)
    y = inference_styletransfer(trainer, x) # [1, 3, 256, 256]
    breakpoint()