# -*- coding: utf-8 -*-
"""
This script is used to plot classification weights using circle format.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from mne.viz import plot_connectivity_circle, circular_layout


def test_plot_connectivity_circle():
    """
    Test plotting connectivity circle.
    """
    node_order = ['Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'];
    
    label_names = ['Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'];

    group_boundaries = [0, 2, 4, 6, 8, 10]

    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)
    
    con = np.random.RandomState(0).randn(11, 11)
    # con[con < 1] = 0
    
    con2 = -con
    
    n_lines = 5
    
    figs, ax = plt.subplots(1,2, facecolor ='k')
    
    plot_connectivity_circle(con, label_names, n_lines=n_lines,
                             node_angles=node_angles, title='test',
                             colormap='Reds', vmin=0, vmax=2, linewidth=2,
                             fontsize_names=12, textcolor='k', facecolor='w', 
                             subplot=121, fig=figs, colorbar=True,
                             node_colors=['r', 'b'])

    plot_connectivity_circle(con2, label_names, n_lines=n_lines,
                         node_angles=node_angles, title='test',
                         colormap='Blues', vmin=-2, vmax=0, linewidth=1.5,
                         fontsize_names=12, textcolor='w', facecolor='k',
                         subplot=122, fig=figs, colorbar=True)

    # plt.tight_layout()
    plt.subplots_adjust(wspace = 0.2, hspace = 0)   
    

    pytest.raises(ValueError, circular_layout, label_names, node_order,
                  group_boundaries=[-1])
    pytest.raises(ValueError, circular_layout, label_names, node_order,
                  group_boundaries=[20, 0])
    # plt.close('all')


if __name__ == "__main__":
    test_plot_connectivity_circle()