import numpy as np
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze
from .physics import get_initial_condition, get_dose_profile, H_VEC, DT, NSTEPS, L
from .ode_baseline import run_ode_simulation

def wrapper_ode_for_sensitivity(params_values, u0_kind, dose_kind):
    """
    Wrapper uruchamiający symulację ODE dla zestawu parametrów z SALib.
    params_values: macierz N x 2 (kolumny: rho, beta)
    Zwraca: wektor wyników (np. masa końcowa) o długości N.
    """
    Y = []
    u0 = get_initial_condition(u0_kind)
    W, _, _ = get_dose_profile(dose_kind)
    
    for row in params_values:
        rho_val, beta_val = row
        
        # Uruchomienie modelu ODE
        _, mass_hist = run_ode_simulation(u0, rho_val, beta_val, W)
        
        # Jako wynik analizy bierzemy masę końcową (lub średnią)
        Y.append(mass_hist[-1])
        
    return np.array(Y)

def run_morris_analysis(problem, u0_kind="center_peak", dose_kind="uniform", num_trajectories=10):
    """Metoda Morrisa (Screening)"""
    # Generowanie próbek
    X = morris_sample.sample(problem, num_trajectories, num_levels=4)
    
    # Ewaluacja modelu
    Y = wrapper_ode_for_sensitivity(X, u0_kind, dose_kind)
    
    # Analiza
    Si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=False)
    return Si

def run_sobol_analysis(problem, u0_kind="center_peak", dose_kind="uniform", samples=512):
    """Metoda Sobola (Variance-based)"""
    # Generowanie próbek (Saltelli)
    X = saltelli.sample(problem, samples)
    
    # Ewaluacja modelu
    Y = wrapper_ode_for_sensitivity(X, u0_kind, dose_kind)
    
    # Analiza
    Si = sobol_analyze.analyze(problem, Y, print_to_console=False)
    return Si