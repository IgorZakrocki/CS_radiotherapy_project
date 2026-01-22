import matplotlib.pyplot as plt
import matplotlib as mpl

def set_style():
    """Ustawia profesjonalny styl wykresów do publikacji/prezentacji."""
    # Fallback jeśli brak seaborn
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('fast')
    
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 11,
        'lines.linewidth': 2.5,
        'grid.alpha': 0.3,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'image.cmap': 'magma' 
    })

COLORS = {
    'pde': '#004488',      # Ciemny niebieski (Explicit)
    'rk45': '#BB5566',     # Magenta (Solver RK45)
    'dose': '#BBBBBB',     # Szary (Dawka)
    'truth': 'black'
}