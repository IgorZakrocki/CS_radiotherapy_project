import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import (
    RHO_BASE, BETA_BASE, D_BASE, 
    get_initial_condition, get_dose_profile,
    run_pde_simulation, run_ode_simulation
)

def main():
    # Definicja katalogu wyjściowego
    OUTPUT_DIR = "figures/sim_01"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Wyniki zostaną zapisane w: {OUTPUT_DIR}")
    
    ic_kinds = ["left_peak", "center_peak", "right_peak", "double_peak"]
    dose_kinds = ["uniform", "left_focus", "center_focus", "right_focus"]
    rho_factors = [0.7, 0.9, 1.1, 1.3]
    beta_factors = [0.6, 0.8, 1.0, 1.2]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    print("Rozpoczynam symulację 4x4 (PDE vs ODE)...")

    for i, ic_name in enumerate(ic_kinds):
        for j, dose_name in enumerate(dose_kinds):
            # Parametry
            rho = RHO_BASE * rho_factors[i]
            beta = BETA_BASE * beta_factors[j]
            
            # Symulacja
            u0 = get_initial_condition(ic_name)
            W, _, _ = get_dose_profile(dose_name)
            
            ts, _, m_pde = run_pde_simulation(D_BASE, rho, beta, u0, W)
            _, m_ode = run_ode_simulation(u0, rho, beta, W)
            
            # Wykres
            ax = axes[i, j]
            ax.plot(ts, m_pde, label="PDE (Truth)", linewidth=2)
            ax.plot(ts, m_ode, "--", label="ODE (Base)", linewidth=2)
            
            ax.set_title(f"IC:{ic_name}\nR:{dose_name}", fontsize=8)
            
            if i == 3: ax.set_xlabel("Czas")
            if j == 0: ax.set_ylabel("Masa guza")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Porównanie PDE vs ODE (Baseline)", fontsize=16)
    
    # Zapis do podfolderu
    output_path = os.path.join(OUTPUT_DIR, "grid_comparison.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Zapisano wykres: {output_path}")

if __name__ == "__main__":
    main()