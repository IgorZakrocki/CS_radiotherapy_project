"""CS Radiotherapy Project - Python package."""

from .methods.physics import (
    L, N, DX, DT, T_END, NSTEPS, TIMES, X_GRID,
    D_BASE, RHO_BASE, BETA_BASE, K, H_VEC,
    get_initial_condition, get_dose_profile, dose_rate_time, get_com
)
from .models.pde import run_pde_simulation, Grid, SimConfig, run_2d_simulation
from .models.ode import run_ode_simulation, physics_ode_derivative
from .models.mlp import ResidualMLP, PINN_SuperNet, ODE_Supermodel, PINN, Supermodel
from .models.node import run_node_simulation, run_supermodel_simulation
from .methods.viz_style import set_style, COLORS
