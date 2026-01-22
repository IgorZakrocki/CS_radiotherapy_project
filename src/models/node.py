import numpy as np
import torch

from ..methods.physics import NSTEPS, DT, H_VEC, dose_rate_time, K, L
from .ode import physics_ode_derivative

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