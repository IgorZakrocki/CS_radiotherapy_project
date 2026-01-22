# 1D PDE solver (FD Explicit Euler)
import numpy as np
from ..methods.physics import NSTEPS, DT, DX, H_VEC, K, dose_rate_time

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

# 2D PDE toy model used in sim_01
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Tuple

# --- KONFIGURACJA I PARAMETRY (z params.py) ---

@dataclass
class Grid:
    L_mm: float = 50.0
    N: int = 100
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False)

    def __post_init__(self) -> None:
        self.dx = self.L_mm / self.N
        self.x = np.linspace(0.0, self.L_mm, self.N)
        self.y = np.linspace(0.0, self.L_mm, self.N)

@dataclass
class TumorKinetics:
    # Parametry z lab01.zip (main.py defaults)
    D_mm2_per_day: float = 0.137     
    rho_per_day: float   = 0.0274    
    K: float = 1.0                   

@dataclass
class Radiobiology:
    mode: Literal["pde_kill", "lq_pulses"] = "lq_pulses"
    beta_rt_per_Gy: float = 0.06
    alpha: float = 0.06              
    beta: float  = 0.006             
    OER_max: float = 3.0
    K_m_mmHg: float = 3.0

@dataclass
class DoseSchedule:
    start_day: float = 10.0
    n_fractions: int = 30
    d_per_frac_Gy: float = 2.0
    frac_duration_day: float = 0.004  # ~6 min

@dataclass
class Beam:
    profile: Literal["uniform", "gaussian"] = "uniform"
    sigma_mm: float = 10.0

@dataclass
class SimConfig:
    T_days: float = 60.0
    dt_day: float = 0.05
    grid: Grid = field(default_factory=Grid)
    tumor: TumorKinetics = field(default_factory=TumorKinetics)
    rtd: Radiobiology = field(default_factory=Radiobiology)
    schedule: DoseSchedule = field(default_factory=DoseSchedule)
    beam: Beam = field(default_factory=Beam)

    def stability_ok(self) -> bool:
        dx = self.grid.dx
        D = self.tumor.D_mm2_per_day
        return self.dt_day <= (dx*dx) / (4.0*D)

# --- NUMERYKA I FIZYKA (z numerics.py, hypoxia.py, schedule.py) ---

def laplacian_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    """Oblicza Laplacian 2D z warunkami brzegowymi Neumanna (odbicie)."""
    up = np.pad(u, 1, mode="edge")
    return (up[2:,1:-1] + up[:-2,1:-1] + up[1:-1,2:] + up[1:-1,:-2] - 4*up[1:-1,1:-1]) / (dx*dx)

def init_tumor_gaussian(N: int, L_mm: float, peak: float = 0.1, sigma_mm: float = 5.0) -> np.ndarray:
    x = np.linspace(0.0, L_mm, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return peak * np.exp(-((X - L_mm/2)**2 + (Y - L_mm/2)**2) / (2*sigma_mm**2))

def radial_pO2(N: int, p_center: float = 1.0, p_edge: float = 30.0) -> np.ndarray:
    """Fantazmat hipoksji: niedotleniony środek, natlenione brzegi."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing="ij")
    r = np.clip(np.sqrt(xx*xx + yy*yy), 0.0, 1.0)
    return p_center + (p_edge - p_center) * r

def omf_from_pO2(pO2: np.ndarray, OER_max: float = 3.0, K_m_mmHg: float = 3.0) -> np.ndarray:
    """Oxygen Modification Factor"""
    return (1.0 + (OER_max - 1.0) * (pO2 / (pO2 + K_m_mmHg))) / OER_max

def weekday_fractions(start_day: float, n: int) -> np.ndarray:
    """Harmonogram: Pn-Pt (przerwa weekendowa)."""
    times = []
    d = start_day
    for i in range(n):
        times.append(d)
        d += 1.0
        if (i+1) % 5 == 0:
            d += 2.0
    return np.array(times)

def beam_profile(N: int, L_mm: float, kind: str, sigma_mm: float) -> np.ndarray:
    if kind == "uniform":
        return np.ones((N, N))
    x = np.linspace(0.0, L_mm, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    r2 = (X - L_mm/2)**2 + (Y - L_mm/2)**2
    g = np.exp(-r2 / (2*sigma_mm**2))
    return g / g.max()

# --- GŁÓWNA PĘTLA SYMULACJI (z simulation.py) ---

@dataclass
class SimOutput:
    times: np.ndarray
    u_final: np.ndarray
    total_mass: np.ndarray
    dose_rate_series: np.ndarray
    beam: np.ndarray

def run_2d_simulation(cfg: SimConfig, save_hook=None) -> SimOutput:
    g = cfg.grid
    kin = cfg.tumor
    rb = cfg.rtd
    sch = cfg.schedule
    b = cfg.beam

    # Walidacja stabilności
    if not cfg.stability_ok():
        print(f"Ostrzeżenie: dt={cfg.dt_day} może być niestabilne!")

    # Inicjalizacja pól
    u = init_tumor_gaussian(g.N, g.L_mm)
    pO2 = radial_pO2(g.N)
    H = omf_from_pO2(pO2, rb.OER_max, rb.K_m_mmHg)
    beam = beam_profile(g.N, g.L_mm, b.profile, b.sigma_mm)

    dt = cfg.dt_day
    times = np.arange(0.0, cfg.T_days + 1e-12, dt)
    total_mass = np.zeros_like(times)
    dose_rate_series = np.zeros_like(times)

    # Harmonogram
    fr_t = weekday_fractions(sch.start_day, sch.n_fractions)
    fr_end = fr_t + sch.frac_duration_day
    dose_rate_val = sch.d_per_frac_Gy / sch.frac_duration_day

    # Parametry LQ
    alpha_eff = rb.alpha * H
    beta_eff  = rb.beta * (H**2)

    def in_window(t: float) -> bool:
        return np.any((t >= fr_t) & (t < fr_end))

    max_u_init = np.max(u)

    # Pętla czasowa
    for k, t in enumerate(times):
        # 1. Dyfuzja (PDE)
        Lu = laplacian_neumann(u, g.dx)
        growth = kin.rho_per_day * u * (1.0 - u/kin.K)
        du = kin.D_mm2_per_day * Lu + growth

        # 2. Radioterapia (Continuous PDE kill lub LQ pulses)
        current_dose_rate = 0.0
        if in_window(t):
            current_dose_rate = dose_rate_val
        
        dose_rate_series[k] = current_dose_rate

        if rb.mode == "pde_kill" and current_dose_rate > 0:
            kill = rb.beta_rt_per_Gy * (current_dose_rate * beam) * H * u
            du -= kill

        # Euler step
        u = np.clip(u + dt*du, 0.0, kin.K)

        # 3. Model LQ (dyskretny skok przeżywalności na koniec frakcji)
        if rb.mode == "lq_pulses":
            # Sprawdzamy czy właśnie skończyła się frakcja
            for te in fr_end:
                if (t - dt) < te <= t + 1e-12:
                    d_map = sch.d_per_frac_Gy * beam
                    S = np.exp(-(alpha_eff*d_map + beta_eff*d_map**2))
                    u *= S
                    break

        total_mass[k] = u.sum()

        if save_hook is not None:
            save_hook(k, t, u, times[:k+1], total_mass[:k+1], dose_rate_series[:k+1], max_u_init)

    return SimOutput(times, u, total_mass, dose_rate_series, beam)