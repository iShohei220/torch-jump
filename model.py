import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from representation import Pyramid, Tower, Pool
from core import PriorCore, InferenceCore, GenerationCore
from dataset import sample_batch
    
class NSG(nn.Module):
    def __init__(self, L=12, shared_core=True, z_dim=3, v_dim=6):
        super(NSG, self).__init__()
        
        # Number of generative layers
        self.L = L
        
        self.z_dim = z_dim
                
        # Representation network
        self.phi = Tower(v_dim=v_dim)
            
        # Generation network
        self.shared_core = shared_core
        if shared_core:
            self.prior_core = PriorCore(z_dim=z_dim)
            self.inference_core = InferenceCore(z_dim=z_dim)
            self.generation_core = GenerationCore(z_dim=z_dim, v_dim=v_dim)
        else:
            self.prior_core = nn.ModuleList([PriorCore(z_dim=z_dim) for _ in range(L)])
            self.inference_core = nn.ModuleList([InferenceCore(z_dim=z_dim) for _ in range(L)])
            self.generation_core = nn.ModuleList([GenerationCore(z_dim=z_dim, v_dim=v_dim) for _ in range(L)])
            
        self.eta_pi = nn.Conv2d(128, 2*z_dim, kernel_size=5, stride=1, padding=2)
        self.eta_e = nn.Conv2d(128, 2*z_dim, kernel_size=5, stride=1, padding=2)
        self.eta_g = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
        
        self.sigma = Variance([3, 64, 64])

    # EstimateELBO
    def forward(self, x_data, v_data, D):
        B, N, *_ = x_data.size()
        
        x, v = sample_batch(x_data, v_data, D)
        M = v.size(1)
        
        # Scene encoder
#         r = torch.sum(self.phi(x.view(-1, 3, 64, 64), v.view(-1, 7)).view(B, -1, 256, 16, 16), dim=1)
        r = self.phi(x, v)

        # Prior initial state
        c_pi = x.new_zeros((B, 128, 16, 16))
        h_pi = x.new_zeros((B, 128, 16, 16))
        z_poe = x.new_zeros((B, self.z_dim, 16, 16))
        
        # Inference initial state
#         c_e = x.new_zeros((B, 128, 16, 16))
#         h_e = x.new_zeros((B, 128, 16, 16))
#         z_e = x.new_zeros((B, self.z_dim, 16, 16))
        c_e = x.new_zeros((B*M, 128, 16, 16))
        h_e = x.new_zeros((B*M, 128, 16, 16))
        z_e = x.new_zeros((B*M, self.z_dim, 16, 16))
        
        # Generator initial state
        c_g = x.new_zeros((B*M, 128, 16, 16))
        h_g = x.new_zeros((B*M, 128, 16, 16))
        u = x.new_zeros((B*M, 128, 64, 64))
        
        kl = 0
        for l in range(self.L):
            # Prior state update
            if self.shared_core:
                c_pi, h_pi = self.prior_core(z_poe, c_pi, h_pi)
            else:
                c_pi, h_pi = self.prior_core[l](z_poe, c_pi, h_pi)
                
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_pi), self.z_dim, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(z_e, r, c_e, h_e)
            else:
                c_e, h_e = self.inference_core[l](z_e, r, c_e, h_e)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), self.z_dim, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q.view(B, M, self.z_dim, 16, 16), std_q.view(B, M, self.z_dim, 16, 16))
            q_poe = self.poe(pi, q)
            
            # Posterior sample
            z_e = q.rsample().view(-1, self.z_dim, 16, 16)
            z_poe = q_poe.rsample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v, c_g, h_g, u, z_poe)
            else:
                c_g, h_g, u = self.generation_core[l](v, c_g, h_g, u, z_poe)
                
            # ELBO KL contribution update
            kl += torch.sum(kl_divergence(q_poe, pi), dim=[1,2,3])
                
        # ELBO likelihood contribution update
#         sigma = self.sigma.view(1, 1, 3, 64, 64).repeat(B, K, 1, 1, 1)
        nll = - torch.sum(Normal(self.eta_g(u).view(B, M, 3, 64, 64), self.sigma()).log_prob(x), dim=[1,2,3,4]) / M
        elbo = N * nll + kl

        return elbo, nll, kl
    
    def generate(self, v):
        B, K, *_ = v.size()

        # Prior initial state
        c_pi = v.new_zeros((B, 128, 16, 16))
        h_pi = v.new_zeros((B, 128, 16, 16))
        z_pi = v.new_zeros((B, self.z_dim, 16, 16))
        
        # Generator initial state
        c_g = v.new_zeros((B*K, 128, 16, 16))
        h_g = v.new_zeros((B*K, 128, 16, 16))
        u = v.new_zeros((B*K, 128, 64, 64))
                
        for l in range(self.L):
            # Prior state update
            if self.shared_core:
                c_pi, h_pi = self.prior_core(z_pi, c_pi, h_pi)
            else:
                c_pi, h_pi = self.prior_core[l](z_pi, c_pi, h_pi)
                
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_pi), self.z_dim, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Prior Sample
            z_pi = pi.sample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v, c_g, h_g, u, z_pi)
            else:
                c_g, h_g, u = self.generation_core[l](v, c_g, h_g, u, z_pi)
                
        # Image sample
        mu = self.eta_g(u).view(B, K, 3, 64, 64)

        return torch.clamp(mu, 0, 1)
    
    def predict(self, x, v, v_q):
        B, M, *_ = x.size()
        K = v_q.size(1)
        
        # Scene encoder
#         r = torch.sum(self.phi(x.view(-1, 3, 64, 64), v.view(-1, 7)).view(B, M, 256, 16, 16), dim=1)
        r = self.phi(x, v)
        
        # Prior initial state
        c_pi = x.new_zeros((B, 128, 16, 16))
        h_pi = x.new_zeros((B, 128, 16, 16))
        z_poe = x.new_zeros((B, self.z_dim, 16, 16))
        
        # Inference initial state
#         c_e = x.new_zeros((B, 128, 16, 16))
#         h_e = x.new_zeros((B, 128, 16, 16))
#         z_e = x.new_zeros((B, self.z_dim, 16, 16))
        c_e = x.new_zeros((B*M, 128, 16, 16))
        h_e = x.new_zeros((B*M, 128, 16, 16))
        z_e = x.new_zeros((B*M, self.z_dim, 16, 16))
        
        # Generator initial state
        c_g = x.new_zeros((B*K, 128, 16, 16))
        h_g = x.new_zeros((B*K, 128, 16, 16))
        u = x.new_zeros((B*K, 128, 64, 64))
                
        for l in range(self.L):
            # Prior state update
            if self.shared_core:
                c_pi, h_pi = self.prior_core(z_poe, c_pi, h_pi)
            else:
                c_pi, h_pi = self.prior_core[l](z_poe, c_pi, h_pi)
                
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_pi), self.z_dim, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(z_e, r, c_e, h_e)
            else:
                c_e, h_e = self.inference_core[l](z_e, r, c_e, h_e)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), self.z_dim, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q.view(B, M, self.z_dim, 16, 16), std_q.view(B, M, self.z_dim, 16, 16))
            q_poe = self.poe(pi, q)
            
            # Posterior sample
            z_e = q.rsample().view(-1, self.z_dim, 16, 16)
            z_poe = q_poe.sample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, c_g, h_g, u, z_poe)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, c_g, h_g, u, z_poe)
                
        # Image sample
        mu = self.eta_g(u).view(B, K, 3, 64, 64)

        return torch.clamp(mu, 0, 1)
    
    def reconstruct(self, x, v):
        B, M, *_ = x.size()
        
        # Scene encoder
#         r = torch.sum(self.phi(x.view(-1, 3, 64, 64), v.view(-1, 7)).view(B, -1, 256, 16, 16), dim=1)
        r = self.phi(x, v)
        
        # Prior initial state
        c_pi = x.new_zeros((B, 128, 16, 16))
        h_pi = x.new_zeros((B, 128, 16, 16))
        z_poe = x.new_zeros((B, self.z_dim, 16, 16))
        
        # Inference initial state
#         c_e = x.new_zeros((B, 128, 16, 16))
#         h_e = x.new_zeros((B, 128, 16, 16))
#         z_e = x.new_zeros((B, self.z_dim, 16, 16))
        c_e = x.new_zeros((B*M, 128, 16, 16))
        h_e = x.new_zeros((B*M, 128, 16, 16))
        z_e = x.new_zeros((B*M, self.z_dim, 16, 16))
        
        # Generator initial state
        c_g = x.new_zeros((B*M, 128, 16, 16))
        h_g = x.new_zeros((B*M, 128, 16, 16))
        u = x.new_zeros((B*M, 128, 64, 64))
        
        kl = 0
        for l in range(self.L):
            # Prior state update
            if self.shared_core:
                c_pi, h_pi = self.prior_core(z_poe, c_pi, h_pi)
            else:
                c_pi, h_pi = self.prior_core[l](z_poe, c_pi, h_pi)
                
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_pi), self.z_dim, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(z_e, r, c_e, h_e)
            else:
                c_e, h_e = self.inference_core[l](z_e, r, c_e, h_e)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), self.z_dim, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q.view(B, M, self.z_dim, 16, 16), std_q.view(B, M, self.z_dim, 16, 16))
            q_poe = self.poe(pi, q)
            
            # Posterior sample
            z_e = q.rsample().view(-1, self.z_dim, 16, 16)
            z_poe = q_poe.sample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v, c_g, h_g, u, z_poe)
            else:
                c_g, h_g, u = self.generation_core[l](v, c_g, h_g, u, z_poe)
                
        # Image sample
        mu = self.eta_g(u).view(B, M, 3, 64, 64)

        return torch.clamp(mu, 0, 1)
    
    def poe(self, prior, posterior):
        vars = torch.cat((prior.variance.view(-1,1,self.z_dim,16,16), posterior.variance), dim=1)
        var = 1. / torch.sum(torch.reciprocal(vars), dim=1)
        locs = torch.cat((prior.loc.view(-1,1,self.z_dim,16,16), posterior.loc), dim=1)
        loc = torch.sum(locs/vars, dim=1) * var
        return Normal(loc, torch.sqrt(var))
    
class Variance(nn.Module):
    def __init__(self, size):
        super(Variance, self).__init__()
        self.param = nn.Parameter(torch.Tensor(*size))
        nn.init.constant_(self.param, 0)
        
    def forward(self):
        sigma = torch.exp(0.5*self.param)
        return sigma