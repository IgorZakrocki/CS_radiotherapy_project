import torch
import torch.nn as nn
import numpy as np
from ..methods.physics import NSTEPS, DT, H_VEC, dose_rate_time, L
from .ode import physics_ode_derivative

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


# --- PINN surrogate ---
import torch
import torch.nn as nn
import numpy as np
from ..methods.physics import NSTEPS, T_END, X_GRID

class PINN_SuperNet(nn.Module):
    """
    Model 'SuperNet' (wg nomenklatury projektu).
    Działa jako surogat pola skalarnego u(t, x) sparametryzowanego przez rho, beta itd.
    Input: [t, x, rho, beta, dose_center, is_uniform]
    Output: u(t,x)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softplus() # Gwarancja nieujemności
        )

    def forward(self, coords, params):
        # coords: [t, x], params: [rho, beta, dc, du]
        # Skalowanie czasu (krytyczne dla zbieżności!)
        t_scaled = coords[:, 0:1] / T_END 
        x_in = coords[:, 1:2]
        
        inp = torch.cat([t_scaled, x_in, params], dim=1)
        return self.net(inp)

def predict_pinn_trajectory(model, rho, beta, d_center, d_uni):
    """Rekonstrukcja trajektorii masy z sieci PINN"""
    model.eval()
    times = np.linspace(0, T_END, NSTEPS + 1)
    mass_hist = []
    
    x_tens = torch.tensor(X_GRID, dtype=torch.float32).view(-1, 1)
    
    with torch.no_grad():
        for t_val in times:
            t_vec = torch.ones_like(x_tens) * t_val
            coords = torch.cat([t_vec, x_tens], dim=1)
            
            # Parametry stałe dla całej przestrzeni w danej chwili
            p_vec = torch.tensor([[rho, beta, d_center, d_uni]], dtype=torch.float32).repeat(len(X_GRID), 1)
            
            u_pred = model(coords, p_vec).numpy().flatten()
            
            # Całkowanie przestrzenne
            m = np.trapz(u_pred, dx=X_GRID[1]-X_GRID[0])
            mass_hist.append(m)
            
    return times, np.array(mass_hist)

# --- Supermodel (Mass + COM) ---
import torch
import torch.nn as nn
import numpy as np
from ..methods.physics import NSTEPS, DT, H_VEC, dose_rate_time, K, L

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



# Backward compatible aliases
PINN = PINN_SuperNet
Supermodel = ODE_Supermodel
