import numpy as np
from ..methods.physics import NSTEPS, DT, H_VEC, K, dose_rate_time, L

def physics_ode_derivative(mass, r_t, rho, beta, wh_mean):
    """Pochodna dM/dt wynikająca z czystej fizyki (Logistyka - Śmierć)"""
    # W modelu 1D o długości L=1, średnia gęstość == całkowita masa
    growth = rho * mass * (1.0 - mass / K)
    kill = beta * r_t * wh_mean * mass
    return growth - kill

def run_ode_simulation(u0, rho, beta, W):
    """
    Rozwiązuje uproszczony model ODE (Zadanie 3).
    Ignoruje przestrzeń, operuje na średniej/masie.
    """
    mass_hist = np.zeros(NSTEPS + 1)
    
    # Warunek początkowy: całka z u0
    current_mass = np.trapz(u0, dx=L/len(u0)) 
    mass_hist[0] = current_mass
    
    wh_mean = np.mean(W * H_VEC)
    times = np.linspace(0, NSTEPS*DT, NSTEPS + 1)

    for n in range(NSTEPS):
        t = times[n]
        r_t = dose_rate_time(t)
        
        d_phys = physics_ode_derivative(current_mass, r_t, rho, beta, wh_mean)
        
        current_mass += DT * d_phys
        current_mass = max(current_mass, 0.0)
        mass_hist[n+1] = current_mass
        
    return times, mass_hist