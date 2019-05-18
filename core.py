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
    def __init__(self, z_dim, v_dim):
        super(GenerationCore, self).__init__()
        self.upsample_v = nn.ConvTranspose2d(v_dim, v_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(v_dim+z_dim, 128, kernel_size=5, stride=1, padding=2)
        self.upsample_h = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        self.z_dim = z_dim
        
    def forward(self, v, c_g, h_g, u, z):
        K = v.size(1)
        z = z.view(-1, 1, self.z_dim, 16, 16).repeat(1, K, 1, 1, 1).view(-1, self.z_dim, 16, 16)
        v = self.upsample_v(v.view(-1, v.size(2), 1, 1))
        c_g, h_g = self.core(torch.cat((v, z), dim=1), (c_g, h_g))
        u = self.upsample_h(h_g) + u
        
        return c_g, h_g, u
    
class InferenceCoreGQN(nn.Module):
    def __init__(self, z_dim, v_dim):
        super(InferenceCoreGQN, self).__init__()
        self.downsample_x = nn.Conv2d(3, 3, kernel_size=4, stride=4, padding=0, bias=False)
        self.upsample_v = nn.ConvTranspose2d(v_dim, v_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.downsample_u = nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        self.core = Conv2dLSTMCell(z_dim+v_dim+256+2*128, 128, kernel_size=5, stride=1, padding=2)
        self.v_dim = v_dim
        
    def forward(self, x, v, r, c_e, h_e, h_g, u):
        K = v.view(r.size(0), -1, self.v_dim).size(1)
        r = r.view(-1, 1, 256, 16, 16).repeat(1, K, 1, 1, 1).view(-1, 256, 16, 16)
        x = self.downsample_x(x.view(-1, 3, 64, 64))
        v = self.upsample_v(v.view(-1, self.v_dim, 1, 1))
        u = self.downsample_u(u)
        c_e, h_e = self.core(torch.cat((x, v, r, h_g, u), dim=1), (c_e, h_e))
        
        return c_e, h_e

    
class GenerationCoreGQN(nn.Module):
    def __init__(self, z_dim, v_dim):
        super(GenerationCoreGQN, self).__init__()
        self.upsample_v = nn.ConvTranspose2d(v_dim, v_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(v_dim+256+z_dim, 128, kernel_size=5, stride=1, padding=2)
        self.upsample_h = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        self.z_dim = z_dim
        self.v_dim = v_dim
        
    def forward(self, v, r, c_g, h_g, u, z):
        K = v.view(r.size(0), -1, self.v_dim).size(1)
        r = r.view(-1, 1, 256, 16, 16).repeat(1, K, 1, 1, 1).view(-1, 256, 16, 16)
        z = z.view(-1, self.z_dim, 16, 16)
        v = self.upsample_v(v.view(-1, self.v_dim, 1, 1))
        c_g, h_g =  self.core(torch.cat((v, r, z), dim=1), (c_g, h_g))
        u = self.upsample_h(h_g) + u
        
        return c_g, h_g, u