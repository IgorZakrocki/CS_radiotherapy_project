"""Reusable utilities (physics, metrics, visualization, assimilation)."""

from .physics import (
    L, N, DX, DT, T_END, NSTEPS, TIMES, X_GRID,
    D_BASE, RHO_BASE, BETA_BASE, K, H_VEC,
    get_initial_condition, get_dose_profile, dose_rate_time, get_com
)
from .viz_style import set_style, COLORS
from .rrms import mse, rrms
from .assimilation import (
    run_abc_assimilation, run_4dvar_assimilation,
    run_morris_analysis, run_sobol_analysis, wrapper_ode_for_sensitivity
)
