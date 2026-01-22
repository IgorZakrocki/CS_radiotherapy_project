import sys
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import imageio
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import (
    DT, X_GRID, RHO_BASE, BETA_BASE, D_BASE, H_VEC, K,
    get_initial_condition, get_dose_profile, dose_rate_time, get_com,
    run_pde_simulation,
    PINN_SuperNet, predict_pinn_trajectory,
    ODE_Supermodel, run_supermodel_simulation
)

def generate_data():
    X_pinn, y_pinn = [], []
    X_sm, y_sm = [], []
    
    ic_kinds = ["left_peak", "center_peak", "right_peak", "double_peak"]
    dose_kinds = ["uniform", "center_focus"]
    
    print("Generowanie danych symulacyjnych...")
    for ic in ic_kinds:
        for dk in dose_kinds:
            u0 = get_initial_condition(ic)
            W, dc, du = get_dose_profile(dk)
            wh_mean = np.mean(W * H_VEC)
            
            ts, u_hist, _ = run_pde_simulation(D_BASE, RHO_BASE, BETA_BASE, u0, W)
            
            for idx in range(0, len(ts)-1, 5): 
                if idx % 10 == 0:
                    t_val = ts[idx]
                    for si in range(0, len(X_GRID), 5):
                        X_pinn.append([t_val, X_GRID[si], RHO_BASE, BETA_BASE, dc, du])
                        y_pinn.append([u_hist[idx, si]])
                
                u_curr = u_hist[idx]
                u_next = u_hist[idx+1]
                t_val = ts[idx]
                r = dose_rate_time(t_val)
                
                m_curr = np.trapz(u_curr, dx=X_GRID[1]-X_GRID[0])
                com_curr = get_com(u_curr)
                m_next = np.trapz(u_next, dx=X_GRID[1]-X_GRID[0])
                com_next = get_com(u_next)
                
                growth = RHO_BASE * m_curr * (1 - m_curr/K)
                kill = BETA_BASE * r * wh_mean * m_curr
                dm_phys = growth - kill
                
                d_mass_corr = ((m_next - m_curr)/DT) - dm_phys
                d_com = (com_next - com_curr)/DT
                
                X_sm.append([m_curr, com_curr, dc, du, r, RHO_BASE, BETA_BASE])
                y_sm.append([d_mass_corr, d_com])

    return (torch.tensor(X_pinn).float(), torch.tensor(y_pinn).float(),
            torch.tensor(X_sm).float(), torch.tensor(y_sm).float())

def create_gif(times, u_pde, ic_name, filepath):
    print(f"Tworzenie GIF: {filepath} ...")
    images = []
    # Tworzenie tymczasowego pliku klatki w folderze docelowym
    temp_frame = filepath + "_temp_frame.png"
    
    for i in range(0, len(times), 50):
        plt.figure(figsize=(8,4))
        plt.plot(X_GRID, u_pde[i], 'k-', label='PDE (Ground Truth)', linewidth=2)
        plt.title(f"Symulacja PDE: {ic_name}, t={times[i]:.2f}")
        plt.ylim(0, 1.5)
        plt.legend()
        plt.xlabel("Przestrzeń x")
        
        plt.savefig(temp_frame)
        plt.close()
        images.append(imageio.imread(temp_frame))
        
    imageio.mimsave(filepath, images, fps=10)
    if os.path.exists(temp_frame):
        os.remove(temp_frame)

def main():
    # Definicja katalogu wyjściowego
    OUTPUT_DIR = "figures/sim_05"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Wyniki zostaną zapisane w: {OUTPUT_DIR}")

    X_pinn, y_pinn, X_sm, y_sm = generate_data()
    
    print("--- Trening SuperNet (PINN) ---")
    pinn = PINN_SuperNet()
    opt_pinn = optim.Adam(pinn.parameters(), lr=0.002)
    dl_pinn = DataLoader(TensorDataset(X_pinn, y_pinn), batch_size=4096, shuffle=True)
    
    for epoch in range(100):
        for xb, yb in dl_pinn:
            opt_pinn.zero_grad()
            loss = torch.mean((pinn(xb[:,:2], xb[:,2:]) - yb)**2)
            loss.backward()
            opt_pinn.step()
            
    print("--- Trening ODE Supermodel ---")
    supermodel = ODE_Supermodel()
    opt_sm = optim.Adam(supermodel.parameters(), lr=0.002)
    dl_sm = DataLoader(TensorDataset(X_sm, y_sm), batch_size=1024, shuffle=True)
    
    for epoch in range(100):
        for xb, yb in dl_sm:
            opt_sm.zero_grad()
            loss = torch.mean((supermodel(xb) - yb)**2)
            loss.backward()
            opt_sm.step()
            
    # Test
    test_ic = "right_peak"
    test_dose = "left_focus"
    u0 = get_initial_condition(test_ic)
    W, dc, du = get_dose_profile(test_dose)
    
    ts, u_hist, m_pde = run_pde_simulation(D_BASE, RHO_BASE, BETA_BASE, u0, W)
    _, m_pinn = predict_pinn_trajectory(pinn, RHO_BASE, BETA_BASE, dc, du)
    _, m_sm = run_supermodel_simulation(supermodel, u0, RHO_BASE, BETA_BASE, W, dc, du)
    
    # Wykres
    plt.figure(figsize=(10,6))
    plt.plot(ts, m_pde, 'k-', linewidth=2, label="PDE (Truth)")
    plt.plot(ts, m_pinn, 'm-.', linewidth=2, label="PINN (SuperNet)")
    plt.plot(ts, m_sm, 'g:', linewidth=3, label="ODE Supermodel")
    
    plt.title(f"SuperNet vs Supermodel: {test_ic} + {test_dose}")
    plt.legend()
    
    # Zapis wykresu
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison.png"))
    
    # Zapis animacji
    gif_path = os.path.join(OUTPUT_DIR, "pde_animation.gif")
    create_gif(ts, u_hist, test_ic, gif_path)
    
    print("Zakończono.")

if __name__ == "__main__":
    main()