import numpy as np

# --- Konfiguracja Przestrzenno-Czasowa ---
L = 1.0
N = 101
X_GRID = np.linspace(0.0, L, N)
DX = X_GRID[1] - X_GRID[0]

T_END = 35.0
DT = 0.01
NSTEPS = int(T_END / DT)
TIMES = np.linspace(0.0, T_END, NSTEPS + 1)

# --- Parametry Biofizyczne (Bazowe) ---
D_BASE = 0.001
RHO_BASE = 0.25
BETA_BASE = 0.6
K = 1.0

# --- Profil Hipoksji (Stały) ---
# Lepsze natlenienie w pobliżu x ~ 0.5
H_VEC = 0.4 + 0.6 * np.exp(-((X_GRID - 0.5 * L) / 0.5) ** 2)

# --- Funkcja Dawki w Czasie r(t) ---
def dose_rate_time(t):
    dose_amp = 4.0
    fraction_duration = 0.2
    # 3 kursy po 5 frakcji
    course_starts = [5.0, 15.0, 25.0]
    
    for cs in course_starts:
        for n in range(5):
            t_start = cs + n * 1.0
            t_end = t_start + fraction_duration
            if t_start <= t <= t_end:
                return dose_amp
    return 0.0

# --- Warunki Początkowe u0(x) ---
def get_initial_condition(kind):
    x = X_GRID
    if kind == "left_peak":
        u0 = 0.1 + 0.5 * np.exp(-((x - 0.2) / 0.1) ** 2)
    elif kind == "center_peak":
        u0 = 0.1 + 0.5 * np.exp(-((x - 0.5) / 0.1) ** 2)
    elif kind == "right_peak":
        u0 = 0.1 + 0.5 * np.exp(-((x - 0.8) / 0.1) ** 2)
    elif kind == "double_peak":
        u0 = (0.05 + 0.35 * np.exp(-((x - 0.25) / 0.07) ** 2)
                   + 0.35 * np.exp(-((x - 0.75) / 0.07) ** 2))
    else: # uniform
        u0 = 0.2 * np.ones_like(x)
    
    # Normalizacja do stałej masy początkowej (dla porównywalności)
    u0 = np.clip(u0, 0.0, 1.0)
    return u0 / np.sum(u0) * 20.0

# --- Profile Dawki W(x) ---
def get_dose_profile(kind):
    x = X_GRID
    center = 0.5
    is_uniform = 0.0
    
    if kind == "uniform":
        W = np.ones_like(x)
        is_uniform = 1.0
    elif kind == "left_focus":
        W = 0.2 + 0.8 * np.exp(-((x - 0.2) / 0.1) ** 2)
        center = 0.2
    elif kind == "center_focus":
        W = 0.2 + 0.8 * np.exp(-((x - 0.5) / 0.1) ** 2)
        center = 0.5
    elif kind == "right_focus":
        W = 0.2 + 0.8 * np.exp(-((x - 0.8) / 0.1) ** 2)
        center = 0.8
    else:
        W = np.ones_like(x)
        is_uniform = 1.0

    # Normalizacja energii
    W = W / np.sum(W) * 200.0
    return W, center, is_uniform

# --- Funkcja pomocnicza: Środek Masy (COM) ---
def get_com(u):
    mass = np.sum(u)
    if mass < 1e-6: return 0.5
    return np.sum(u * X_GRID) / mass