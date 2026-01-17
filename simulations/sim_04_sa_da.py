import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import (
    run_pde_simulation, run_ode_simulation,
    run_morris_analysis, run_sobol_analysis,
    run_abc_assimilation, run_4dvar_assimilation,
    get_initial_condition, get_dose_profile,
    RHO_BASE, BETA_BASE, D_BASE
)

def main():
    # Definicja katalogu wyjściowego
    OUTPUT_DIR = "figures/sim_04"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Wyniki zostaną zapisane w: {OUTPUT_DIR}")
    
    # --- Analiza Wrażliwości ---
    print("--- ZADANIE 4: SA ---")
    
    problem = {
        'num_vars': 2, 'names': ['rho', 'beta'],
        'bounds': [[0.1, 0.4], [0.4, 1.0]]
    }
    
    Si_morris = run_morris_analysis(problem)
    plt.figure()
    plt.bar(['rho', 'beta'], Si_morris['mu_star'], yerr=Si_morris['sigma'])
    plt.title("Morris Analysis")
    plt.savefig(os.path.join(OUTPUT_DIR, "morris.png"))
    
    Si_sobol = run_sobol_analysis(problem)
    plt.figure()
    plt.bar(['rho', 'beta'], Si_sobol['S1'])
    plt.title("Sobol Analysis (S1)")
    plt.savefig(os.path.join(OUTPUT_DIR, "sobol.png"))

    # --- Asymilacja Danych ---
    print("\n--- ZADANIE 5: DA ---")
    true_rho = 0.35
    true_beta = 0.8
    test_ic = "double_peak"
    test_dose = "left_focus"
    
    u0 = get_initial_condition(test_ic)
    W, _, _ = get_dose_profile(test_dose)
    times, _, mass_pde_truth = run_pde_simulation(D_BASE, true_rho, true_beta, u0, W)
    
    # ABC
    posterior = run_abc_assimilation(mass_pde_truth, test_ic, test_dose)
    if len(posterior) > 0:
        est_rho_abc = np.mean(posterior[:, 0])
        est_beta_abc = np.mean(posterior[:, 1])
        print(f"ABC Estimate: {est_rho_abc:.4f}, {est_beta_abc:.4f}")
        
        plt.figure()
        plt.scatter(posterior[:,0], posterior[:,1], alpha=0.5)
        plt.plot(true_rho, true_beta, 'rx', markersize=10, label='Truth')
        plt.title('ABC Posterior')
        plt.savefig(os.path.join(OUTPUT_DIR, "abc_posterior.png"))
    
    # 4D-Var
    res_4dvar = run_4dvar_assimilation(mass_pde_truth, test_ic, test_dose)
    est_rho_4d, est_beta_4d = res_4dvar.x
    print(f"4D-Var Estimate: {est_rho_4d:.4f}, {est_beta_4d:.4f}")
    
    _, mass_4d = run_ode_simulation(u0, est_rho_4d, est_beta_4d, W)
    
    plt.figure()
    plt.plot(times, mass_pde_truth, 'k-', label='PDE Truth')
    plt.plot(times, mass_4d, 'r--', label='Model (4D-Var)')
    plt.legend()
    plt.title("4D-Var Reconstruction")
    plt.savefig(os.path.join(OUTPUT_DIR, "da_result.png"))

if __name__ == "__main__":
    main()