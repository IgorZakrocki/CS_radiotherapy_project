import torch
import torch.nn as nn
import numpy as np
from .physics import NSTEPS, DT, H_VEC, dose_rate_time, K, L

class ODE_Supermodel(nn.Module):
    """
    Rozszerzony NODE, który modeluje nie tylko Masę, ale i Środek Masy (COM).
    Input: [Mass, COM, Dose_Center, Is_Uniform, r(t), rho, beta]
    Output: [dMass_corr, dCOM]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )
    
    def forward(self, x):
        return self.net(x)

def run_supermodel_simulation(model, u0, rho, beta, W, d_center, d_uni):
    """Integracja ODE Supermodel (Mass + COM)"""
    model.eval()
    times = np.linspace(0, NSTEPS*DT, NSTEPS + 1)
    mass_hist = np.zeros(NSTEPS + 1)
    
    # Stan początkowy
    x_grid = np.linspace(0, L, len(u0))
    curr_mass = np.trapz(u0, dx=x_grid[1]-x_grid[0])
    
    # Obliczanie początkowego COM
    if curr_mass > 1e-6:
        curr_com = np.trapz(u0 * x_grid, dx=x_grid[1]-x_grid[0]) / curr_mass
    else:
        curr_com = 0.5
        
    mass_hist[0] = curr_mass
    wh_mean = np.mean(W * H_VEC)
    
    with torch.no_grad():
        for n in range(NSTEPS):
            t = times[n]
            r = dose_rate_time(t)
            
            # Fizyka bazowa dla Masy
            growth = rho * curr_mass * (1.0 - curr_mass / K)
            kill = beta * r * wh_mean * curr_mass
            d_phys = growth - kill
            
            # Predykcja sieci
            inp = torch.tensor([[curr_mass, curr_com, d_center, d_uni, r, rho, beta]], dtype=torch.float32)
            out = model(inp).numpy()[0]
            
            d_corr = out[0]
            d_com = out[1]
            
            # Krok Eulera
            curr_mass += DT * (d_phys + d_corr)
            curr_com += DT * d_com
            
            curr_mass = max(curr_mass, 0.0)
            mass_hist[n+1] = curr_mass
            
    return times, mass_hist