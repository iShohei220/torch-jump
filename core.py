import torch
from torch import nn
from torch.nn import functional as F
from conv_lstm import Conv2dLSTMCell

class PriorCore(nn.Module):
    def __init__(self, z_dim):
        super(PriorCore, self).__init__()
        self.core = Conv2dLSTMCell(z_dim, 128, kernel_size=5, stride=1, padding=2)
        
    def forward(self, z, c_pi, h_pi):
        c_pi, h_pi = self.core(z, (c_pi, h_pi))
        
        return c_pi, h_pi

class InferenceCore(nn.Module):
    def __init__(self, z_dim):
        super(InferenceCore, self).__init__()
        self.core = Conv2dLSTMCell(z_dim+256, 128, kernel_size=5, stride=1, padding=2)
        
    def forward(self, z, r, c_e, h_e):
        c_e, h_e = self.core(torch.cat((z, r), dim=1), (c_e, h_e))
        
        return c_e, h_e
    
class GenerationCore(nn.Module):
    def __init__(self, z_dim):
        super(GenerationCore, self).__init__()
        self.upsample_v = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(7+z_dim, 128, kernel_size=5, stride=1, padding=2)
        self.upsample_h = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        self.z_dim = z_dim
        
    def forward(self, v, c_g, h_g, u, z):
        K = v.size(1)
        z = z.view(-1, 1, self.z_dim, 16, 16).repeat(1, K, 1, 1, 1).view(-1, self.z_dim, 16, 16)
        v = self.upsample_v(v.view(-1, 7, 1, 1))
        c_g, h_g = self.core(torch.cat((v, z), dim=1), (c_g, h_g))
        u = self.upsample_h(h_g) + u
        
        return c_g, h_g, u