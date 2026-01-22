import torch
import torch.nn as nn
import numpy as np
from .physics import NSTEPS, DT, H_VEC, dose_rate_time, L
from .ode_baseline import physics_ode_derivative

class ResidualMLP(nn.Module):
    """
    Sieć neuronowa ucząca się poprawki (residuum) do modelu ODE.
    Wejście: [Mass, r(t), rho, beta, WH_mean]
    Wyjście: Poprawka do pochodnej dM/dt
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) 
        )

    def forward(self, mass, r_t, rho, beta, wh_mean):
        x_in = torch.stack([mass, r_t, rho, beta, wh_mean], dim=1)
        return self.net(x_in)

def run_node_simulation(model, u0, rho, beta, W):
    """Integracja w czasie używając hybrydowego NODE"""
    model.eval()
    times = np.linspace(0, NSTEPS*DT, NSTEPS + 1)
    mass_hist = np.zeros(NSTEPS + 1)
    
    current_mass = np.trapz(u0, dx=L/len(u0))
    mass_hist[0] = current_mass
    wh_mean = np.mean(W * H_VEC)
    
    with torch.no_grad():
        for n in range(NSTEPS):
            t = times[n]
            r_t = dose_rate_time(t)
            
            # 1. Fizyka bazowa
            d_phys = physics_ode_derivative(current_mass, r_t, rho, beta, wh_mean)
            
            # 2. Poprawka z sieci
            inputs = torch.tensor([current_mass, r_t, rho, beta, wh_mean], dtype=torch.float32)
            # Powielamy wymiar batcha (1, 5) aby pasowało do sieci
            inputs = inputs.unsqueeze(0) 
            
            # Sieć zwraca tensor, bierzemy scalar
            d_corr = model(inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3], inputs[:,4]).item()
            
            # 3. Krok (Euler)
            current_mass += DT * (d_phys + d_corr)
            current_mass = max(current_mass, 0.0)
            
            mass_hist[n+1] = current_mass
            
    return times, mass_hist