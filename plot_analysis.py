import matplotlib.pyplot as plt
import numpy as np
from base_model import BaseModel

def plot_critical(model: BaseModel, critical_data: dict, title: str='Hysteresis Diagram'):
    '''
    Plot critical diagram
    '''

    fig, ax = plt.subplots(figsize=(10, 6))

    # Forward
    for i in range(critical_data['C_forward'].shape[1]):
        ax.plot(critical_data['d_C_forward'], critical_data['C_forward'][:, i], color='blue', alpha=0.3, linewidth=1.5)
    
    # Backward
    for i in range(critical_data['C_backward'].shape[1]):
        ax.plot(critical_data['d_C_backward'], critical_data['C_backward'][:, i], color='green', alpha=0.3, linewidth=1.5)

    # Show critical points
    ax.axvline(critical_data['d_collapse'], color='blue', linestyle='--', label=f'Forward - Collapse (d_C={critical_data["d_collapse"]:.2f})')
    ax.axvline(critical_data['d_recovery'], color='green', linestyle='--', label=f'Backward - Recovery (d_C={critical_data["d_recovery"]:.2f})')

    ax.set_xlabel('Driver of decline (d_C)')
    ax.set_ylabel('Abundance (C)')
    ax.set_title(title)
    ax.legend()
    ax.grid()
    ax.set_ylim(bottom=-5)
   
    return fig

def verify_base_model():

    # Without adaptive foraging 
    model_no_adapt = BaseModel(nu=1.0)
    critical_data_no_adapt = model_no_adapt.find_critical_points()
    fig1 = plot_critical(model_no_adapt, critical_data_no_adapt, title='Hysteresis Diagram (no adaptive foraging)')

    # With adaptive foraging
    model_adapt = BaseModel(nu=0.3)
    critical_data_adapt = model_adapt.find_critical_points()
    fig2 = plot_critical(model_adapt, critical_data_adapt, title='Hysteresis Diagram (with adaptive foraging)')

    print(f"No adaptive foraging: Collapse at d_C={critical_data_no_adapt['d_collapse']:.3f}, Recovery at d_C={critical_data_no_adapt['d_recovery']:.2f}")
    print(f"With adaptive foraging: Collapse at d_C={critical_data_adapt['d_collapse']:.3f}, Recovery at d_C={critical_data_adapt['d_recovery']:.2f}")

    return {
        'figures': {'no_adapt': fig1, 'adapt': fig2},
        'data': {'no_adapt': critical_data_no_adapt, 'adapt': critical_data_adapt}
    }

if __name__ == "__main__":
    np.random.seed(133)

    results = verify_base_model()

    results['figures']['no_adapt'].savefig('hysteresis_no_adapt.png', dpi=300)
    results['figures']['adapt'].savefig('hysteresis_adapt.png', dpi=300)