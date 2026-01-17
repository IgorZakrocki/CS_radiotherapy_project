import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import (
    run_pde_simulation, run_ode_simulation,
    get_initial_condition, get_dose_profile,
    L, X_GRID
)

def main():
    # Definicja katalogu wyjściowego
    OUTPUT_DIR = "figures/sim_03"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Wyniki zostaną zapisane w: {OUTPUT_DIR}")

    RHO = 0.25
    BETA = 0.6
    D = 0.001
    ic_kind = "center_peak"
    dose_kind = "center_focus"

    u0 = get_initial_condition(ic_kind)
    W, _, _ = get_dose_profile(dose_kind)

    times, u_hist, mass_pde = run_pde_simulation(D, RHO, BETA, u0, W)
    _, mass_ode = run_ode_simulation(u0, RHO, BETA, W)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(times, mass_pde, label='PDE Model', linewidth=2, color='blue')
    ax1.plot(times, mass_ode, label='ODE Model', linewidth=2, linestyle='--', color='orange')
    ax1.set_title("Masa guza w czasie")
    ax1.set_xlabel("Czas")
    ax1.set_ylabel("Masa")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(X_GRID, u_hist[0], label='Początek', linestyle=':', color='green')
    ax2.plot(X_GRID, u_hist[-1], label='Koniec', color='red')
    scale_factor = np.max(u_hist) / np.max(W) * 0.5
    ax2.fill_between(X_GRID, W * scale_factor, color='gray', alpha=0.2, label='Dawka')
    ax2.set_title("Profil przestrzenny")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Zapis do podfolderu
    output_path = os.path.join(OUTPUT_DIR, "reproduction_result.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Zapisano: {output_path}")

if __name__ == "__main__":
    main()