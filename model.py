import numpy as np
import random
import torch
from torch import nn
from torch.nn import functional as F
# from torch.distributions import Normal, Uniform
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
# from representation import Pyramid, Tower, Pool
# from core import PriorCore, InferenceCore, GenerationCore, InferenceCoreGQN, GenerationCoreGQN
# from dataset import sample_batch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7+3, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, v, f):
        B, M, C, H, W = f.size()
        f = f.contiguous().view(B*M, C, H, W)
        v = v.contiguous().view(B*M, v.size(2), 1, 1).repeat(1, 1, H, W)
        r = self.net(torch.cat((v, f), dim=1))
        
        return r
    
class Conv2dLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)
        
        in_channels += out_channels
        
        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input  = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state  = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, states):
        (hidden, cell) = states
        input = torch.cat((hidden, input), dim=1)
        
        forget_gate = torch.sigmoid(self.forget(input))
        input_gate  = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate  = torch.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell
    
class Prior(nn.Module):
    def __init__(self, stride_to_hidden, nf_to_hidden, nf_enc, nf_z):
        super(Prior, self).__init__()
        self.conv1 = nn.Conv2d(32, nf_enc, kernel_size=stride_to_hidden, stride=stride_to_hidden)
        self.lstm = Conv2dLSTMCell(nf_z, nf_to_hidden, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(nf_to_hidden, 2*nf_z, kernel_size=5, stride=1, padding=2)
        
    def forward(self, r, z, h, c):
        r = self.conv1(r)
        h, c = self.lstm(torch.cat((r, z), dim=1), (h, c))
        mu, logvar = torch.split(self.conv2(h), z.size(1), dim=1)
        std = torch.exp(0.5*logvar)
        p = Normal(mu, std)
        
        return h, c, p

class Posterior(nn.Module):
    def __init__(self, stride_to_hidden, nf_to_hidden, nf_enc, nf_z):
        super(Posterior, self).__init__()
        self.conv1 = nn.Conv2d(2*32, nf_enc, kernel_size=stride_to_hidden, stride=stride_to_hidden)
        self.lstm = Conv2dLSTMCell(nf_z, nf_to_hidden, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(nf_to_hidden, 2*nf_z, kernel_size=5, stride=1, padding=2)
        
    def forward(self, r, r_prime, h, c):
        lstm_input = self.conv1(torch.cat((r, r_prime), dim=1), (h, c))
        h, c = self.lstm(torch.cat((lstm_input, z), dim=1), (h, c))
        mu, logvar = torch.split(self.conv2(h), z.size(1), dim=1)
        std = torch.exp(0.5*logvar)
        p = Normal(mu, std)
        
        return h, c, p
    
class Renderer(nn.Module):
    def __init__(self, nf_to_hidden, stride_to_obs, nf_to_obs, nf_dec, nf_z, nf_v):
        super(Renderer, self).__init__()
        self.conv = nn.Conv(nf_to_obs, nf_dec, kernel_size=stride_to_obs, stride=stride_to_obs)
        self.lstm = Conv2dLSTMCell(nf_z+nf_v+nf_dec, nf_to_hidden, kernel_size=5, stride=1, padding=2)
        self.transconv = nn.ConvTranspose2d(nf_to_hidden, nf_to_obs, kernel_size=stride_to_obs, stride=stride_to_obs)
        
    def forward(self, z, v, canvas, h, c):
        K = v.size(1)
        z = z.view(-1, 1, z.size(1), z.size(2), z.size(3)).repeat(1, v.size(1), 1, 1, 1).view(-1, z.size(1), z.size(2), z.size(3))
        v = v.view(-1, v.size(2), 1, 1).repeat(1, 1, z.size(2), z.size(3))
        h, c = self.core(torch.cat((z, v, self.conv(canvas)), dim=1), (h, c))
        canvas = canvas + self.transconv(h)
        
        return h, c, canvas

class JUMP(nn.Module):
    def __init__(self, nt=4, stride_to_hidden=2, nf_to_hidden=64, nf_enc=128, stride_to_obs=2, nf_to_obs=128, nf_dec=64, nf_z=3, nf_v=1):
        super(JUMP, self).__init__()
        
        # The number of DRAW steps in the network.
        self.nt = nt
        # The kernel and stride size of the conv. layer mapping the input image to the LSTM input.
        self.stride_to_hidden = stride_to_hidden
        # The number of channels in the LSTM layer.
        self.nf_to_hidden = nf_to_hidden
        # The number of channels in the conv. layer mapping the input image to the LSTM input.
        self.nf_enc = nf_enc
        # The kernel and stride size of the transposed conv. layer mapping the LSTM state to the canvas.
        self.stride_to_obs = stride_to_obs
        # The number of channels in the hidden layer between LSTM states and the canvas
        self.nf_to_obs = nf_to_obs
        # The number of channels of the conv. layer mapping the canvas state to the LSTM input.
        self.nf_dec = nf_dec
        # The number of channels in the stochastic latent in each DRAW step.
        self.nf_z = nf_z
                
        # Encoder network
        self.m_theta = Encoder()
            
        # DRAW
        self.prior = Prior(stride_to_hidden, nf_to_hidden, nf_enc, nf_z)
        self.posterior = Posterior(stride_to_hidden, nf_to_hidden, nf_enc, nf_z)
        
        # Renderer
        self.m_gamma = Renderer(nf_to_hidden, stride_to_obs, nf_to_obs, nf_dec, nf_z, nf_v)
        self.transconv = nn.ConvTranspose2d(nf_to_obs, 3, kernel_size=4, stride=4)
        
    # EstimateELBO
    def forward(self, v_data, f_data, pixel_var):
        B, N, C, H, W = f_data.size()
        
        M = random.randint(1, N-1)
        indices = np.random.permutation(range(N))
        context_idx, target_idx = indices[:M], indices[M:]
        v, f = v_data[:, context_idx], f_data[:, context_idx]
        v_prime, f_prime = v_data[:, target_idx], f_data[:, target_idx]
        
        r = torch.sum(self.m_theta(v, f).view(B, M, 32, H//4, W//4), dim=1)
        r_prime = torch.sum(self.m_theta(v_prime, f_prime).view(B, N-M, 32, H//4, W//4), dim=1)
        
        H_hidden, W_hidden = H//(4*self.stride_to_hidden), W//(4*self.stride_to_hidden)

        # Prior initial state
        h_phi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        c_phi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        
        # Posterior initial state
        h_psi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        c_psi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        z = v.new_zeros((B, self.nf_z, H_hidden, W_hidden))
        
        # Renderer initial state
        h_gamma = v.new_zeros((B*(N-M), self.nf_to_hidden, H_hidden, W_hidden))
        c_gamma = v.new_zeros((B*(N-M), self.nf_to_hidden, H_hidden, W_hidden))
        canvas = v.new_zeros((B*(N-M), self.nf_to_obs, H_hidden*self.stride_to_obs, W_hidden*self.stride_to_obs))
        
        kl = 0
        for t in range(self.nt):
            # Prior
            h_phi, c_phi, p_phi  = self.prior(r, z, h_phi, c_phi)
            
            # Posterior
            h_psi, c_psi, p_psi  = self.posterior(r, r_prime, z, h_psi, c_psi)

            # Posterior sample
            z = p_psi.rsample()
            
            # Generator state update
            h_gamma, c_gamma, canvas = self.m_gamma(z, v_prime, canvas, h_gamma, c_gamma)
                
            # ELBO KL contribution update
            kl += torch.sum(kl_divergence(p_psi, p_phi), dim=[1,2,3])
                
        # Sample frame
        f_hat = self.transconv(canvas).view(B, N-M, C, H, W) + Normal(v.zeros_like(), torch.sqrt(pixel_var)).sample()
        mse_loss = nn.MSELoss(reduction='none')
        mse = torch.sum(mse_loss(f_hat, f_prime), dim=[1,2,3,4])
        elbo = kl + mse / pixel_var
        
        return elbo, kl, mse
    
    def generate(self, v, f, v_prime):
        B, M, C, H, W = f.size()
        N = v_prime.size(1)
        
        # Scene encoder
        r = torch.sum(self.m_theta(v, f).view(B, M, 32, H//4, W//4), dim=1)
        
        H_hidden, W_hidden = H//(4*self.stride_to_hidden), W//(4*self.stride_to_hidden)

        # Prior initial state
        h_phi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        c_phi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        
        z = v.new_zeros((B, self.nf_z, H_hidden, W_hidden))
        
        # Renderer initial state
        h_gamma = v.new_zeros((B*N, self.nf_to_hidden, H_hidden, W_hidden))
        c_gamma = v.new_zeros((B*N, self.nf_to_hidden, H_hidden, W_hidden))
        canvas = v.new_zeros((B*N, self.nf_to_obs, H_hidden*self.stride_to_obs, W_hidden*self.stride_to_obs))
                
        for t in range(self.nt):
            # Prior
            h_phi, c_phi, p_phi  = self.prior(r, z, h_phi, c_phi)

            # Prior sample
            z = p_psi.sample()
            
            # Generator state update
            h_gamma, c_gamma, canvas = self.m_gamma(z, v_prime, canvas, h_gamma, c_gamma)
                
        # Sample frame
        f_hat = self.transconv(canvas).view(B, N, C, H, W)

        return torch.clamp(f_hat, 0, 1)
    
    def reconstruct(self, v, f):
        B, N, C, H, W = f.size()
        
        # Scene encoder
        r = torch.sum(self.m_theta(v, f).view(B, N, 32, H//4, W//4), dim=1)
        
        # Posterior initial state
        h_psi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        c_psi = v.new_zeros((B, self.nf_to_hidden, H_hidden, W_hidden))
        z = v.new_zeros((B, self.nf_z, H_hidden, W_hidden))
        
        for t in range(self.nt):
            # Prior
            h_phi, c_phi, p_phi  = self.prior(r, z, h_phi, c_phi)

            # Prior sample
            z = p_phi.sample()
            
            # Generator state update
            h_gamma, c_gamma, canvas = self.m_gamma(z, v, canvas, h_gamma, c_gamma)
                
        # Sample frame
        f_hat = self.transconv(canvas).view(B, N, C, H, W)

        return torch.clamp(f_hat, 0, 1)
    