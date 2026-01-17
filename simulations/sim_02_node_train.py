import sys
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import (
    DT, RHO_BASE, BETA_BASE, D_BASE, H_VEC, 
    get_initial_condition, get_dose_profile, dose_rate_time,
    run_pde_simulation, physics_ode_derivative, run_ode_simulation,
    ResidualMLP, run_node_simulation
)

def main():
    # Definicja katalogu wyjściowego
    OUTPUT_DIR = "figures/sim_02"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Wyniki zostaną zapisane w: {OUTPUT_DIR}")
    
    # 1. Generowanie danych treningowych
    print("--- Generowanie danych (PDE) ---")
    T_TRAIN_START = 10.0
    T_TRAIN_END = 25.0
    
    ic_kinds = ["left_peak", "double_peak"]
    dose_kinds = ["center_focus", "uniform"]
    
    X_train, y_train = [], []
    
    for ic in ic_kinds:
        for dk in dose_kinds:
            u0 = get_initial_condition(ic)
            W, _, _ = get_dose_profile(dk)
            wh_mean = np.mean(W * H_VEC)
            
            ts, _, m_pde = run_pde_simulation(D_BASE, RHO_BASE, BETA_BASE, u0, W)
            
            mask = (ts >= T_TRAIN_START) & (ts < T_TRAIN_END)
            indices = np.where(mask)[0]
            
            for idx in indices[:-1]:
                m_curr = m_pde[idx]
                m_next = m_pde[idx+1]
                t_curr = ts[idx]
                r_curr = dose_rate_time(t_curr)
                
                d_phys = physics_ode_derivative(m_curr, r_curr, RHO_BASE, BETA_BASE, wh_mean)
                m_ode_next = m_curr + DT * d_phys
                target_residual = (m_next - m_ode_next) / DT
                
                X_train.append([m_curr, r_curr, RHO_BASE, BETA_BASE, wh_mean])
                y_train.append([target_residual])
                
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    # 2. Trening NODE
    print(f"--- Trening NODE ({len(X_t)} próbek) ---")
    model = ResidualMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
    
    for epoch in range(150):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb[:,0], xb[:,1], xb[:,2], xb[:,3], xb[:,4])
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            
    # 3. Walidacja
    print("--- Testowanie NODE ---")
    test_ic = "double_peak"
    test_dose = "center_focus"
    u0_test = get_initial_condition(test_ic)
    W_test, _, _ = get_dose_profile(test_dose)
    
    ts, _, m_pde = run_pde_simulation(D_BASE, RHO_BASE, BETA_BASE, u0_test, W_test)
    _, m_ode = run_ode_simulation(u0_test, RHO_BASE, BETA_BASE, W_test)
    _, m_node = run_node_simulation(model, u0_test, RHO_BASE, BETA_BASE, W_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ts, m_pde, 'k-', label='PDE (Ground Truth)', linewidth=2, alpha=0.7)
    plt.plot(ts, m_ode, 'b--', label='ODE Baseline', linewidth=2)
    plt.plot(ts, m_node, 'r:', label='NODE (Hybrid)', linewidth=3)
    
    plt.axvspan(T_TRAIN_START, T_TRAIN_END, color='gray', alpha=0.2, label='Training Zone')
    
    plt.title(f"NODE Ekstrapolacja: {test_ic} + {test_dose}")
    plt.xlabel("Czas")
    plt.ylabel("Masa guza")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zapis do podfolderu
    output_path = os.path.join(OUTPUT_DIR, "node_extrapolation.png")
    plt.savefig(output_path)
    print(f"Zapisano: {output_path}")

if __name__ == "__main__":
    main()