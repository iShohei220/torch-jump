import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from representation import Pyramid, Tower, Pool
from core import PriorCore, InferenceCore, GenerationCore
    
class GQN(nn.Module):
    def __init__(self, L=12, shared_core=False, z_dim=3):
        super(GQN, self).__init__()
        
        # Number of generative layers
        self.L = L
        
        self.z_dim = z_dim
                
        # Representation network
        self.phi = Tower()
            
        # Generation network
        self.shared_core = shared_core
        if shared_core:
            self.prior_core = PriorCore(z_dim=z_dim)
            self.inference_core = InferenceCore(z_dim=z_dim)
            self.generation_core = GenerationCore(z_dim=z_dim)
        else:
            self.prior_core = nn.ModuleList([PriorCore(z_dim=z_dim) for _ in range(L)])
            self.inference_core = nn.ModuleList([InferenceCore(z_dim=z_dim) for _ in range(L)])
            self.generation_core = nn.ModuleList([GenerationCore(z_dim=z_dim) for _ in range(L)])
            
        self.eta_pi = nn.Conv2d(128, 2*z_dim, kernel_size=5, stride=1, padding=2)
        self.eta_e = nn.Conv2d(128, 2*z_dim, kernel_size=5, stride=1, padding=2)
        self.eta_g = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)

    # EstimateELBO
    def forward(self, x, v, v_q, x_q, sigma):
        B, M, *_ = x.size()
        K = v_q.size(1)
        
        # Scene encoder
        r = torch.sum(self.phi(x.view(-1, 3, 64, 64), v.view(-1, 7)).view(B, M, 256, 16, 16), dim=1)

        # Prior initial state
        c_pi = x.new_zeros((B, 128, 16, 16))
        h_pi = x.new_zeros((B, 128, 16, 16))
        z_pi = x.new_zeros((B, self.z_dim, 16, 16))
        
        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
        z_e = x.new_zeros((B, self.z_dim, 16, 16))
        
        # Generator initial state
        c_g = x.new_zeros((B*K, 128, 16, 16))
        h_g = x.new_zeros((B*K, 128, 16, 16))
        u = x.new_zeros((B*K, 128, 64, 64))
        
        elbo = 0
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
            z_pi = pi.rsample()
            
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(z_e, r, c_e, h_e)
            else:
                c_e, h_e = self.inference_core[l](z_e, r, c_e, h_e)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), self.z_dim, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z_e = q.rsample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, c_g, h_g, u, z_e)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, c_g, h_g, u, z_e)
                
            # ELBO KL contribution update
            elbo -= kl_divergence(q, pi).sum()
                
        # ELBO likelihood contribution update
        elbo += Normal(self.eta_g(u), sigma).log_prob(x_q.view(-1, 3, 64, 64)).sum()

        return elbo
    
    def generate(self, v_q):
        B, K, *_ = v_q.size()

        # Prior initial state
        c_pi = v_q.new_zeros((B, 128, 16, 16))
        h_pi = v_q.new_zeros((B, 128, 16, 16))
        z_pi = v_q.new_zeros((B, self.z_dim, 16, 16))
        
        # Generator initial state
        c_g = v_q.new_zeros((B*K, 128, 16, 16))
        h_g = v_q.new_zeros((B*K, 128, 16, 16))
        u = v_q.new_zeros((B*K, 128, 64, 64))
                
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
                c_g, h_g, u = self.generation_core(v_q, c_g, h_g, u, z_pi)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, c_g, h_g, u, z_pi)
                
        # Image sample
        mu = self.eta_g(u).view(B, K, 3, 64, 64)

        return torch.clamp(mu, 0, 1)
    
    def predict(self, x, v, v_q):
        B, M, *_ = x.size()
        K = v_q.size(1)
        
        # Scene encoder
        r = torch.sum(self.phi(x.view(-1, 3, 64, 64), v.view(-1, 7)).view(B, M, 256, 16, 16), dim=1)
        
        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
        z_e = x.new_zeros((B, self.z_dim, 16, 16))
        
        # Generator initial state
        c_g = x.new_zeros((B*K, 128, 16, 16))
        h_g = x.new_zeros((B*K, 128, 16, 16))
        u = x.new_zeros((B*K, 128, 64, 64))
                
        for l in range(self.L):
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(z_e, r, c_e, h_e)
            else:
                c_e, h_e = self.inference_core[l](z_e, r, c_e, h_e)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), self.z_dim, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z_e = q.sample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, c_g, h_g, u, z_e)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, c_g, h_g, u, z_e)
                
        # Image sample
        mu = self.eta_g(u).view(B, K, 3, 64, 64)

        return torch.clamp(mu, 0, 1)
    
    def kl_divergence(self, x, v):
        B, M, *_ = x.size()
        
        # Scene encoder
        r = torch.sum(self.phi(x.view(-1, 3, 64, 64), v.view(-1, 7)).view(B, M, 256, 16, 16), dim=1)

        # Prior initial state
        c_pi = x.new_zeros((B, 128, 16, 16))
        h_pi = x.new_zeros((B, 128, 16, 16))
        z_pi = x.new_zeros((B, self.z_dim, 16, 16))
        
        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
        z_e = x.new_zeros((B, self.z_dim, 16, 16))
        
        kl = 0
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
            z_pi = pi.rsample()
            
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(z_e, r, c_e, h_e)
            else:
                c_e, h_e = self.inference_core[l](z_e, r, c_e, h_e)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), self.z_dim, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z_e = q.rsample()
            
            # ELBO KL contribution update
            kl += kl_divergence(q, pi).sum()
            
        return kl