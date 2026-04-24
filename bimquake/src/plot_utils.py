import matplotlib.pyplot as plt
import numpy as np

def _plot_capacity_curve(vr_ult_TOT, Hult_TOT, bilinear_x, bilinear_y):
    """ Generate a plot comparing pushover capacity curves with their bilinear approximations.

        Parameters
        ----------
        vr_ult_TOT : list of np.ndarray
            List of ultimate displacements for X and Y directions.

        Hult_TOT : list of np.ndarray
            List of ultimate base shear forces for X and Y directions.  
        
        bilinear_x : list of list
            List of x-coordinates for bilinear curves in X and Y directions.

        bilinear_y : list of list
            List of y-coordinates for bilinear curves in X and Y directions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the pushover plot. """
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    directions = ['X', 'Y']

    for i in range(len(directions)):
        ax[i].plot(vr_ult_TOT[i], Hult_TOT[i], 'g--', label='Capacity curve')

        ax[i].plot(bilinear_x[i], bilinear_y[i], label='Equivalent bi-linear curve')
        
        ax[i].set_ylabel('H [kN]')
        ax[i].set_xlabel('$δ_{x}$ [mm]')
        ax[i].title.set_text('Earthquake {} - Capacity curve'.format(directions[i]))
    fig.tight_layout()
    ax[0].legend()
    return fig


def _plot_ADRS(ADRS, Sda, Saa, delta_ult_eq, S_eq, dxstar_t, Tr, IR, ADRS_TR, Sda_TR, Saa_TR):
    """ Generate a plot comparing pushover capacity curves with ADRS and TR spectra.

        Parameters
        ----------
        ADRS : np.ndarray
            Array of acceleration-displacement response spectrum values.

        Sda : list of np.ndarray
            List of spectral displacement values for X and Y directions.

        Saa : list of np.ndarray
            List of spectral acceleration values for X and Y directions.
        
        delta_ult_eq : list of np.ndarray
            List of ultimate equivalent displacements for X and Y directions.

        S_eq : list of np.ndarray
            List of equivalent spectral acceleration values for X and Y directions.

        dxstar_t : list of float
            List of adjusted displacement values for X and Y directions.

        Tr : list of np.ndarray
            List of TR response spectra for X and Y directions.

        IR : list of float
            List of safety indicies for X and Y directions.

        ADRS_TR : list of np.ndarray
            List of TR acceleration-displacement response spectra for X and Y directions.

        Sda_TR : list of np.ndarray
            List of TR spectral displacement values for X and Y directions.

        Saa_TR : list of np.ndarray
            List of TR spectral acceleration values for X and Y directions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the ADRS comparison plot. """
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    directions = ['X', 'Y']

    for i in range(len(directions)):

        ax[i].plot(delta_ult_eq[i], S_eq[i], 'r-', label='Capacity curve - EPUSH')

        ax[i].plot(ADRS[:, 0], ADRS[:, 1], 'b-', label='$SLV_e$ (TR=712)')
        ax[i].plot(Sda[i], Saa[i], 'b--', label='$SLV_a$ (TR=712)')

        ax[i].plot(ADRS_TR[i][:, 0], ADRS_TR[i][:, 1], 'g-', label='$S_e$ (TR={})'.format(Tr[i]))
        ax[i].plot(Sda_TR[i], Saa_TR[i], 'g--', label='$S_a$ (TR={})'.format(Tr[i]))

        ax[i].axvline(delta_ult_eq[i][2], color='g', ls='--', linewidth=0.7, label='$d_u^*$')
        ax[i].axvline(dxstar_t[i], color='m', ls='--', linewidth=0.7, label='$d_t^*$')

        ax[i].set_ylim([0, 0.7])
        ax[i].set_xlim([0, np.max(delta_ult_eq[i])*2])

        ax[i].set_ylabel('$α_g$ [g]')
        ax[i].set_xlabel('$δ_x$ [mm]')

        if Tr == 2475:
            ax[i].title.set_text('Earthquake {} - ADRS - Safety Index>{}'.format(directions[i], IR[i]))
        else:
            ax[i].title.set_text('Earthquake {} - ADRS - Safety Index={}'.format(directions[i], IR[i]))
        ax[i].legend()

    return fig
