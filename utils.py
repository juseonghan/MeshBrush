import numpy as np 
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import yaml 
import os 
import cv2 as cv
from PIL import Image
import math 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_mask(config):
    mask_path = Path(config['mask_path'])
    assert mask_path.exists(), 'invalid mask path in config file'
    mask = np.load(str(mask_path))
    mask = mask[:, 420:-420]
    
    # _mask[:,:,3] = mask
    return mask
    
def read_loss_fn(config):
    fn = config['loss']
    if fn == 'mse':
        loss = torch.nn.MSELoss(reduction='none')
    elif fn == 'cos':
        loss = torch.nn.CosineEmbeddingLoss() # plz don't do this yet
    else:
        raise Exception('[ERROR] loss function in config.yaml should be "mse" or "cos"')
    return loss 

def record_losses(save_mesh_path, config, losses, iter):
    losses = np.array(losses)
    report_every = int(config['report_loss_every'])
    plt.figure()
    plt.plot(losses)
    plt.title("training loss")
    x_label = f'epoch / {report_every}'
    plt.xlabel(x_label)
    plt.ylabel('mse')
    name = f'loss_{iter}.png'
    save_fig_name = save_mesh_path / 'logs' / name
    plt.savefig(str(save_fig_name))

def read_intrinsics(config):
    K_path = Path(config['path_to_intrinsics'])
    assert K_path.exists(), 'invalid intrinsics path in config'
    K = np.load(str(K_path))
    return torch.Tensor(K)

def read_config(path):
    with open(str(path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close() 
    return config 

def to_uint8(arr):
    temp =  255 * (arr - arr.min()) * (1.0 /(arr.max() - arr.min()))
    temp = temp.astype(np.uint8)
    return temp

def make_grid(
    tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range=None,
    scale_each: bool = False,
    pad_value: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

def export_poses(path, poses, type):
    # poses is a list of numpy arrays
    # type is 'render' or 'train'
    poses_np = np.stack(poses, axis=0)
    if type == 'render':
        save_name = path / 'render_poses.npy'
    elif type == 'train':
        save_name = path / 'train_poses.npy'
    else:
        print('[WARNING] export_poses should have render or train as its type arg, saving as random')
        save_name = path / 'random.npy'
    np.save(str(save_name), poses_np)

def save_imgs(config, save_path, imgs, iter, depth_imgs=None, depth=False):
    # save_path = Path(config['path_save_renders'])
    # assert save_path.exists(), "path_save_renders invalid in config"
    imgs = imgs.detach().cpu().numpy()
    this_batch_size = imgs.shape[0]
    actual_batch_size = int(config['render_batch_size'])
    for i in range(this_batch_size):
        img = imgs[i,:,:,:3]
        img = to_uint8(img)
        img = img[..., ::-1]
        save_name = str(iter * actual_batch_size + i) + '.jpg'
        cv.imwrite(str(save_path / 'color' / save_name), img)
        if depth: 
            depth_img = depth_imgs[i,:,:,0].detach().cpu().numpy()
            save_name = str(iter * actual_batch_size + i) + '.npy'
            np.save(str(save_path / 'depth' / save_name), depth_img)

def read_textures(path):
    print('READING TEXTURES MANUALLY...')
    with open(path, 'r') as f:
        data = f.readlines()
    f.close()
    start = False 
    result = []
    for i, line in enumerate(data):
        if line[:3] == 'end':
            start = True 
            continue 
        if start:
        # line is end header 
            if i == len(data) - 1:
                break 
            _line = data[i+1][:-1]
            numbers = [float(x) for x in _line.split()]
            if len(numbers) != 6:
                break
            textures = np.array(numbers[-3:]) / 255.0
            result.append(textures)
    return result 

def extract_imgs(config, save_path, imgs, iter):
    # save_path = Path(config['path_save_renders'])
    # assert save_path.exists(), "path_save_renders invalid in config"
    imgs = imgs.detach().cpu().numpy()
    this_batch_size = imgs.shape[0]
    result = []
    for i in range(this_batch_size):
        img = imgs[i,:,:,:3]
        img = to_uint8(img)
        result.append(img)
    return result 

def view_imgs(imgs):

    for i in range(len(imgs)):
        img = torch.squeeze(imgs[i]).permute(1,2,0)
        img = img.detach().cpu().numpy()
        if i == 0:
            base = to_uint8(img)
        else:
            if base.shape != img.shape: 
                img = cv.resize(img, base.shape[:2])
                img = to_uint8(img)
            base = np.hstack((base, img))
    plt.imshow(base)
    plt.show()

def log_training(path, mesh, model, optim, scheduler, poses, epoch):
    poses = np.stack(poses, axis=0)
    poses = torch.Tensor(poses)
    save_path = path / 'model.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'poses': poses,
        'epoch': epoch
    }, str(save_path))
    save_path = path / 'learned_textures.pt'

    textures = model(mesh.verts_packed())
    torch.save(textures, str(save_path))

# from torchvision.utils
def make_grid(
    tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

# from torchvision.utils
def torch_to_PIL(
    tensor,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


"""
    Save open3D triangle mesh
"""
def prepare_directories(path):
    # o3d.io.write_triangle_mesh(str(path / 'scaled_mesh.obj'), mesh)

    if not Path(path / 'color').exists():
        os.mkdir(str(path / 'color'))
    if not Path(path / 'depth').exists():
        os.mkdir(str(path / 'depth'))
    
    log_dir = Path( path / 'logs' )
    if not log_dir.exists():
        os.mkdir(str(log_dir))
    
    mesh_save_dir = Path( path / 'textured_meshes' ) 
    if not mesh_save_dir.exists():
        os.mkdir(str(mesh_save_dir))

class TrainingParams():
    def __init__(self, config):
        self.input_dim = int(config['input_dim'])
        self.sigma = float(config['sigma'])
        self.clamp = config['clamp']
        self.model_width = int(config['width'])
        self.rgb_depth = int(config['rgb_depth'])
        self.model_depth = int(config['depth'])
        self.weight_init = config['weight_init']
        self.lr = float(config['lr'])
        self.encoding = config['encoding']
        self.niter = int(config['n_iters'])
    
    def regurtitate(self):
        print('-------- TRAINING PARAMS --------')
        print(f'Input channel:       {self.input_dim}')
        print(f'Sigma:               {self.sigma}')
        print(f'Clamping function:   {self.clamp}')
        print(f'Model width:         {self.model_width}')
        print(f'Model depth:         {self.model_depth}')
        print(f'RGB head depth:      {self.rgb_depth}')
        print(f'Encoding type        {self.encoding}')
        print(f'Num Iterations:      {self.niter}')
        print(f'Weight init:         {self.weight_init}')
        print(f'lr:                  {self.lr}')
        print('----------- GOOD LUCK -----------')