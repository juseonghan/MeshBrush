import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from utils import TrainingParams, device
from utils import read_config

# custom rendering w/ rasterization to render images and depth simultaneously
class RendererWithDepth(nn.Module):

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader 

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf
    
class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10, exclude=0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self.exclude = exclude
        self.B = torch.randn((num_input_channels, mapping_size)) * scale
        self.B = self.B.to(device)
        # B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        # self._B = torch.stack(B_sort)  # for sape
        self.twopi = 2 * 3.14159265359

    def forward(self, x):

        batches, channels = x.shape
        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)
        # res = x @ self._B.to(x)
        # res = 2 * np.pi * res
        x_proj = torch.matmul(self.twopi * x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)
    
def xavier_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)

def kaiming_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data)

class TextureLearner(nn.Module):

    def __init__(self, config, resume):
        super(TextureLearner, self).__init__()
        layers = [] 
        self.params = TrainingParams(config)
        self.params.regurtitate()
        
        # initialize the Ns network
        if self.params.input_dim != 3:
            print('[WARNING] input dimensions is not 3... did you mean to do this?')
        if self.params.encoding == 'gaussian':
            layers.append(FourierFeatureTransform(self.params.input_dim, self.params.model_width, self.params.sigma, exclude=0))
            layers.append(nn.Linear(self.params.model_width * 2, self.params.model_width))
            layers.append(nn.ReLU())
        else:
            print('[WARNING] The only valid encoding type is gaussian')
            layers.append(nn.Linear(self.params.input_dim, self.params.model_width))
            layers.append(nn.ReLU())
        # layers.append(ProgressiveEncoding(mapping_size=self.params.model_width, T=self.params.niter, d=self.params.input_dim))
        

        for _ in range(self.params.model_depth):
            layers.append(nn.Linear(self.params.model_width, self.params.model_width))
            layers.append(nn.ReLU())
        self.base = nn.ModuleList(layers)

        # RGB head, Nc
        color_head = []
        for _ in range(self.params.rgb_depth):
            color_head.append(nn.Linear(self.params.model_width, self.params.model_width))
            color_head.append(nn.ReLU())
        color_head.append(nn.Linear(self.params.model_width, 3))
        # color_head.append(nn.Softmax())
        self.rgb_head = nn.ModuleList(color_head)
        
        # weights initialization
        if resume:
            pass 
        elif self.params.weight_init == 'kaiming':
            kaiming_init(self.base)
            kaiming_init(self.rgb_head)
        elif self.params.weight_init == 'xavier':
            xavier_init(self.base)
            xavier_init(self.rgb_head)
        
        # for param in self.rgb_head.parameters():
        #     param.data = nn.parameter.Parameter(torch.ones_like(param))

    def forward(self, x):

        # forward pass
        for layer in self.base:
            x = layer(x)
        for layer in self.rgb_head:
            x = layer(x)
        
        # get final layer and apply
        if self.params.clamp == 'tanh': # tanh range is [-1, 1]
            x = (F.tanh(x) + 1) / 2
            x = torch.clamp(x, 0, 1) # just in case for precision errors
        elif self.params.clamp == 'clamp':
            x = torch.clamp(x, 0, 1)
        elif self.params.clamp == 'softmax':
            x = F.softmax(x)
        else:
            print('[WARNING] config has invalid clamp, should be tanh, clamp, or softmax')
        return x


# testing
if __name__ == '__main__':
    config = read_config('configs/config.yaml')
    model = TextureLearner(config).to(device)

    # batch size of 20
    # 3 channel input
    # 400 vertices
    x = torch.rand(20, 3, 400).to(device)
    x = x.permute(0, 2, 1).reshape(20 * 400, 3)
    y = model(x) # also (20*400, 3)
    breakpoint()
    gt = torch.rand(8000, 3).to(device)
    loss = torch.nn.MSELoss()
    l = loss(y, gt)
    l.backward()