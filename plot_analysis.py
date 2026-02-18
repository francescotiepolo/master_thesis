import matplotlib.pyplot as plt
import numpy as np
import os
from base_model import BaseModel


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
    
    # Plot mean
    C_forward_mean = critical_data['C_forward'].mean(axis=1)
    C_backward_mean = critical_data['C_backward'].mean(axis=1)
    ax.plot(critical_data['d_C_forward'], C_forward_mean, 
            'b-', linewidth=2.5, label='Forward', zorder=3)
    ax.plot(critical_data['d_C_backward'], C_backward_mean, 
            'g-', linewidth=2.5, label='Backward', zorder=3)
    
    # Plot individuals
    for i in range(critical_data['C_forward'].shape[1]):
        ax.plot(critical_data['d_C_forward'], critical_data['C_forward'][:, i], 
                'b-', alpha=0.15, linewidth=0.8, zorder=1)
    for i in range(critical_data['C_backward'].shape[1]):
        ax.plot(critical_data['d_C_backward'], critical_data['C_backward'][:, i], 
                'g-', alpha=0.15, linewidth=0.8, zorder=1)
    
    # Critical points
    ax.axvline(critical_data['d_collapse'], color='orange', linestyle='--', 
              linewidth=2, label=f"Collapse", zorder=2)
    ax.axvline(critical_data['d_recovery'], color='red', linestyle='--', 
              linewidth=2, label=f"Recovery", zorder=2)
    
    ax.set_ylabel('Country Activity (C)', fontsize=12, fontweight='bold')
    ax.set_title('(A) Country Dynamics', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Panel 2: Product Dynamics
    ax = axes[1]
    
    # Plot mean
    P_forward_mean = critical_data['P_forward'].mean(axis=1)
    P_backward_mean = critical_data['P_backward'].mean(axis=1)
    ax.plot(critical_data['d_C_forward'], P_forward_mean, 
            'b-', linewidth=2.5, label='Forward', zorder=3)
    ax.plot(critical_data['d_C_backward'], P_backward_mean, 
            'g-', linewidth=2.5, label='Backward', zorder=3)
    
    # Plot individuals
    for i in range(critical_data['P_forward'].shape[1]):
        ax.plot(critical_data['d_C_forward'], critical_data['P_forward'][:, i], 
                'b-', alpha=0.15, linewidth=0.8, zorder=1)
    for i in range(critical_data['P_backward'].shape[1]):
        ax.plot(critical_data['d_C_backward'], critical_data['P_backward'][:, i], 
                'g-', alpha=0.15, linewidth=0.8, zorder=1)
    
    # Critical points
    ax.axvline(critical_data['d_collapse'], color='orange', linestyle='--', 
              linewidth=2, zorder=2)
    ax.axvline(critical_data['d_recovery'], color='red', linestyle='--', 
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
    
    ax.plot(critical_data['d_C_forward'], spec_forward, 
            'b-', linewidth=2.5, label='Forward', zorder=3)
    ax.plot(critical_data['d_C_backward'], spec_backward, 
            'g-', linewidth=2.5, label='Backward', zorder=3)
    
    # Critical points
    ax.axvline(critical_data['d_collapse'], color='orange', linestyle='--', 
              linewidth=2, zorder=2)
    ax.axvline(critical_data['d_recovery'], color='red', linestyle='--', 
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
    critical_data_no_adapt = model_no_adapt.find_critical_points()
    fig1 = plot_C_P_alpha_dynamics(
        model_no_adapt, 
        critical_data_no_adapt, 
        'Complete Dynamics (no adaptive foraging)'
    )

    # With adaptive foraging
    nu = 0.3
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
        'figures': {'no_adapt': fig1, 'adapt': fig2},
        'data': {'no_adapt': critical_data_no_adapt, 'adapt': critical_data_adapt}
    }


if __name__ == "__main__":
    
    results = verify_base_model()
    
    # Folder for figures
    os.makedirs('Figures', exist_ok=True)
    
    # Save figures
    results['figures']['no_adapt'].savefig('Figures/dynamics_no_adapt.png', dpi=300, bbox_inches='tight')
    results['figures']['adapt'].savefig('Figures/dynamics_adapt.png', dpi=300, bbox_inches='tight')
    
    print("Figures saved")