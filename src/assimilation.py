import numpy as np
from scipy.optimize import minimize
from .physics import get_initial_condition, get_dose_profile, DT, NSTEPS, L
from .ode_baseline import run_ode_simulation

# --- ABC (Approximate Bayesian Computation) ---

def run_abc_assimilation(target_data_pde, u0_kind, dose_kind, 
                         n_samples=1000, tolerance=0.1, 
                         prior_ranges={'rho': (0.1, 0.5), 'beta': (0.2, 1.0)}):
    """
    Proste odrzucanie (Rejection Sampling).
    target_data_pde: wektor masy guza z symulacji PDE (Ground Truth)
    """
    accepted_params = []
    
    u0 = get_initial_condition(u0_kind)
    W, _, _ = get_dose_profile(dose_kind)
    
    # Pobieramy punkty czasowe z PDE (zakładamy zgodność kroków)
    # Jeśli PDE ma gęstszy zapis, trzeba zdownsamplować target_data_pde
    
    print(f"Rozpoczynam ABC (N={n_samples}, tol={tolerance})...")
    
    for _ in range(n_samples):
        # 1. Losowanie z priorów (rozkład jednostajny)
        rho_cand = np.random.uniform(*prior_ranges['rho'])
        beta_cand = np.random.uniform(*prior_ranges['beta'])
        
        # 2. Symulacja modelu w przód (ODE)
        _, mass_ode = run_ode_simulation(u0, rho_cand, beta_cand, W)
        
        # 3. Obliczenie odległości (MSE lub norma Euklidesowa)
        # Porównujemy tylko w punktach wspólnych (tutaj cała trajektoria ma tę samą długość)
        dist = np.mean((mass_ode - target_data_pde)**2)
        
        # 4. Decyzja
        if dist < tolerance:
            accepted_params.append([rho_cand, beta_cand])
            
    return np.array(accepted_params)

# --- 4D-Var (Variational Data Assimilation) ---

def cost_function_4dvar(params, u0, W, target_data):
    """Funkcja kosztu J = ||Obs - Model||^2"""
    rho_val, beta_val = params
    
    # Constraints (soft penalty for negative params)
    if rho_val < 0 or beta_val < 0:
        return 1e6
        
    _, mass_ode = run_ode_simulation(u0, rho_val, beta_val, W)
    
    # MSE loss
    loss = np.mean((mass_ode - target_data)**2)
    return loss

def run_4dvar_assimilation(target_data_pde, u0_kind, dose_kind, initial_guess=[0.25, 0.6]):
    """
    Minimalizacja funkcji kosztu w celu znalezienia optymalnych parametrów.
    """
    u0 = get_initial_condition(u0_kind)
    W, _, _ = get_dose_profile(dose_kind)
    
    print("Rozpoczynam 4D-Var (optymalizacja)...")
    
    result = minimize(
        cost_function_4dvar,
        x0=initial_guess,
        args=(u0, W, target_data_pde),
        method='L-BFGS-B',
        bounds=[(0.01, 1.0), (0.01, 2.0)] # Granice dla rho, beta
    )
    
    return result