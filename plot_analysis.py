import matplotlib.pyplot as plt
import numpy as np
import os
from base_model import BaseModel

COL_FORWARD  = "#1f77b4"   # blue  – forward trajectory
COL_BACKWARD = "#ff7f0e"   # orange – backward trajectory

def plot_network_structure(model: BaseModel):
    """
    Display the adjacency matrix (nested network) and the forbidden-links matrix
    for the given model's generated network.
    Species are sorted by degree (descending) so the characteristic nested
    triangular pattern is visible, matching Fig A of the supporting information.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Sort species by degree (descending) so the nested triangular pattern is visible
    pollinator_order = np.argsort(-model.adj_matrix.sum(axis=1))  # SC indices, most connected first
    plant_order      = np.argsort(-model.adj_matrix.sum(axis=0))  # SP indices, most connected first

    for ax, mat, title in zip(
        axes,
        [model.adj_matrix, model.forbidden_network],
        [f"Nestedness = {model.nestedness:.1f}",
         f"Forbidden Links = {model.forbidden_links:.1f}"],
    ):
        # Reorder rows (pollinators) and columns (plants) by degree, then transpose
        # so that plant species appear on y-axis and pollinator species on x-axis
        sorted_mat = mat[pollinator_order, :][:, plant_order].T
        ax.imshow(
            sorted_mat, aspect="auto", cmap="Greys",
            interpolation="none", vmin=0, vmax=1, origin="lower"
        )
        ax.set_xlabel("Pollinator species", fontsize=11)
        ax.set_ylabel("Plant species",      fontsize=11)
        ax.set_title(title,                 fontsize=12)

    fig.suptitle(
        f"Network structure  (nestedness={model.nestedness:.1f})",
        fontsize=13,
    )
    fig.tight_layout()
    return fig

def plot_C_P_alpha_dynamics(model: BaseModel, critical_data: dict, title: str='System Dynamics'):
    """
    Plot a 3-panel figure showing:
    - Panel 1: Country (C) dynamics
    - Panel 2: Product (P) dynamics  
    - Panel 3: Specialization (alpha) dynamics
    """
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))
    
    # Panel 1: Country Dynamics
    ax = axes[0]
    
    # Plot individuals
    for i in range(critical_data['C_forward'].shape[1]):
        ax.plot(critical_data['d_C_forward'], critical_data['C_forward'][:, i],
                color=COL_FORWARD, alpha=0.7, linewidth=0.8, zorder=1)
    for i in range(critical_data['C_backward'].shape[1]):
        ax.plot(critical_data['d_C_backward'], critical_data['C_backward'][:, i],
                color=COL_BACKWARD, alpha=0.7, linewidth=0.8, ls='--', zorder=1)
    
    # Critical points
    ax.axvline(critical_data['d_collapse'], color='red', linestyle='--', 
              linewidth=2, label=f"Collapse", zorder=2)
    ax.axvline(critical_data['d_recovery'], color='green', linestyle='--', 
              linewidth=2, label=f"Recovery", zorder=2)
    
    ax.set_ylabel('Country Activity (C)', fontsize=12, fontweight='bold')
    ax.set_title('(A) Country Dynamics', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Panel 2: Product Dynamics
    ax = axes[1]
    
    # Plot individuals
    for i in range(critical_data['P_forward'].shape[1]):
        ax.plot(critical_data['d_C_forward'], critical_data['P_forward'][:, i],
                color=COL_FORWARD, alpha=0.7, linewidth=0.8, zorder=1)
    for i in range(critical_data['P_backward'].shape[1]):
        ax.plot(critical_data['d_C_backward'], critical_data['P_backward'][:, i],
                color=COL_BACKWARD, alpha=0.7, linewidth=0.8, ls='--', zorder=1)
    
    # Critical points
    ax.axvline(critical_data['d_collapse'], color='red', linestyle='--', 
              linewidth=2, zorder=2)
    ax.axvline(critical_data['d_recovery'], color='green', linestyle='--', 
              linewidth=2, zorder=2)
    
    ax.set_ylabel('Product Output (P)', fontsize=12, fontweight='bold')
    ax.set_title('(B) Product Dynamics', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Panel 3: Alpha Dynamics
    ax = axes[2]
    
    # Aggregate alpha measure: mean of max alpha per country
    # This shows how specialized countries are (high value = very specialized)
    def compute_mean_specialization(alpha_array):
        """
        For each step, compute mean of max alpha across countries.
        """
        max_alpha_per_country = np.max(alpha_array, axis=2)  # Max alpha for each country
        return max_alpha_per_country.mean(axis=1)  # Mean across countries
    
    spec_forward = compute_mean_specialization(critical_data['alpha_forward'])
    spec_backward = compute_mean_specialization(critical_data['alpha_backward'])
    
    ax.plot(critical_data['d_C_forward'],  spec_forward,
            color=COL_FORWARD,  lw=2.0, label='forward trajectory', zorder=3)
    ax.plot(critical_data['d_C_backward'], spec_backward,
            color=COL_BACKWARD, lw=2.0, ls='--', label='backward trajectory', zorder=3)

    
    # Critical points
    ax.axvline(critical_data['d_collapse'], color='red', linestyle='--', 
              linewidth=2, zorder=2)
    ax.axvline(critical_data['d_recovery'], color='green', linestyle='--', 
              linewidth=2, zorder=2)
    
    ax.set_xlabel('Driver of Decline (d_C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Specialization\n(max α per country)', fontsize=12, fontweight='bold')
    ax.set_title('(C) Specialization Dynamics', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Annotation
    ax.text(0.98, 0.02, 'High value = countries specialized in few products\nLow value = countries diversified across products',
        transform=ax.transAxes, fontsize=9, 
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Main title
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def verify_base_model():
    """
    Run verification and output 3-panel plots.
    """

    # Without adaptive foraging 
    nu = 1.0
    q = 0.0
    print(f"Testing WITHOUT adaptive foraging (nu={nu}, q={q})...")
    model_no_adapt = BaseModel(nu=nu, q=q, seed=133)
    fig_net_no_adapt = plot_network_structure(model_no_adapt)
    critical_data_no_adapt = model_no_adapt.find_critical_points()
    fig1 = plot_C_P_alpha_dynamics(
        model_no_adapt, 
        critical_data_no_adapt, 
        'Complete Dynamics (no adaptive foraging)'
    )

    # With adaptive foraging
    nu = 0.6
    q = 0.0
    print(f"Testing WITH adaptive foraging (nu={nu}, q={q})...")
    model_adapt = BaseModel(nu=nu, q=q, seed=133)
    critical_data_adapt = model_adapt.find_critical_points()
    fig2 = plot_C_P_alpha_dynamics(
        model_adapt, 
        critical_data_adapt,
        'Complete Dynamics (with adaptive foraging)'
    )

    # Summary
    print("RESULTS SUMMARY")
    
    print("Without adaptive foraging:")
    print(f"  Collapse:  d_C = {critical_data_no_adapt['d_collapse']:.3f}")
    print(f"  Recovery:  d_C = {critical_data_no_adapt['d_recovery']:.3f}")
    print(f"  Hysteresis = {critical_data_no_adapt['d_collapse'] - critical_data_no_adapt['d_recovery']:.3f}")
    
    print("With adaptive foraging:")
    print(f"  Collapse:  d_C = {critical_data_adapt['d_collapse']:.3f}")
    print(f"  Recovery:  d_C = {critical_data_adapt['d_recovery']:.3f}")
    print(f"  Hysteresis = {critical_data_adapt['d_collapse'] - critical_data_adapt['d_recovery']:.3f}")

    return {
        'figures': {'no_adapt': fig1, 'adapt': fig2, 'network_no_adapt': fig_net_no_adapt},
        'data': {'no_adapt': critical_data_no_adapt, 'adapt': critical_data_adapt}
    }


if __name__ == "__main__":
    
    results = verify_base_model()
    
    # Folder for figures
    os.makedirs('Figures', exist_ok=True)
    
    # Save figures
    results['figures']['network_no_adapt'].savefig('Figures/network_structure_no_adapt.png', dpi=300, bbox_inches='tight')
    results['figures']['no_adapt'].savefig('Figures/dynamics_no_adapt.png', dpi=300, bbox_inches='tight')
    results['figures']['adapt'].savefig('Figures/dynamics_adapt.png', dpi=300, bbox_inches='tight')
    
    print("Figures saved")