import numpy as np
from .physics import NSTEPS, DT, DX, H_VEC, K, dose_rate_time

def run_pde_simulation(D, rho, beta, u0, W):
    """
    Rozwiązuje równanie PDE metodą różnic skończonych (Explicit Euler).
    Zwraca:
        times: tablica czasów
        u_hist: macierz [czas, przestrzeń] - pełna historia
        mass_hist: wektor masy całkowitej w czasie
    """
    u = u0.copy()
    u_hist = np.zeros((NSTEPS + 1, len(u0)))
    mass_hist = np.zeros(NSTEPS + 1)
    times = np.linspace(0, NSTEPS*DT, NSTEPS + 1)

    u_hist[0, :] = u.copy()
    mass_hist[0] = np.trapz(u, dx=DX)

    for n in range(1, NSTEPS + 1):
        t = n * DT
        
        # Laplacian (Neumann BC)
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / DX**2
        u_xx[0] = 2*(u[1] - u[0]) / DX**2     # BC lewy
        u_xx[-1] = 2*(u[-2] - u[-1]) / DX**2  # BC prawy

        # Reakcja
        r_t = dose_rate_time(t)
        kill_term = beta * (r_t * W) * H_VEC * u
        growth_term = rho * u * (1.0 - u / K)

        du_dt = D * u_xx + growth_term - kill_term
        
        u = u + DT * du_dt
        u = np.clip(u, 0.0, None) # Fizyczne ograniczenie

        u_hist[n, :] = u.copy()
        mass_hist[n] = np.trapz(u, dx=DX)

    return times, u_hist, mass_hist