import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def plot_capacity_curve(vr_ult_TOT, Hult_TOT, bilinear_x, bilinear_y):
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


def plot_ADRS_demand_and_capacity(v_bl, pushover_results):
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
    
    Sda = pushover_results["Sda"]
    Saa = pushover_results["Saa"]
    S_eq = pushover_results["S_eq"]
    dxstar_t = pushover_results["dxstars"]
    Tr = pushover_results["Tr"]
    IR = pushover_results["IR"]
    ADRS_TR = pushover_results["ADRS_TR"]
    Sda_TR = pushover_results["Sda_TR"]
    Saa_TR = pushover_results["Saa_TR"]
    tstep = pushover_results["t_step"]
    ADRS = pushover_results["ADRS"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    directions = ['X', 'Y']

    for i in range(len(directions)):

        ax[i].plot(v_bl[i], S_eq[i], 'r-', label='Capacity curve - EPUSH')

        ax[i].plot(ADRS[:, 0], ADRS[:, 1], 'b-', label='$SLV_e$ (TR=712)')
        ax[i].plot(Sda[i], Saa[i], 'b--', label='$SLV_a$ (TR=712)')

        ax[i].plot(ADRS_TR[i][:, 0], ADRS_TR[i][:, 1], 'g-', label='$S_e$ (TR={})'.format(Tr[i]))
        ax[i].plot(Sda_TR[i], Saa_TR[i], 'g--', label='$S_a$ (TR={})'.format(Tr[i]))

        ax[i].axvline(v_bl[i][2], color='g', ls='--', linewidth=0.7, label='$d_u^*$')
        ax[i].axvline(dxstar_t[i], color='m', ls='--', linewidth=0.7, label='$d_t^*$')

        ax[i].set_ylim([0, 0.7])
        ax[i].set_xlim([0, np.max(v_bl[i])*2])

        ax[i].set_ylabel('$α_g$ [g]')
        ax[i].set_xlabel('$δ_x$ [mm]')

        if Tr == 2475:
            ax[i].title.set_text('Earthquake {} - ADRS - Safety Index>{}'.format(directions[i], IR[i]))
        else:
            ax[i].title.set_text('Earthquake {} - ADRS - Safety Index={}'.format(directions[i], IR[i]))
        ax[i].legend()

    return fig


def _get_wall_vertices(cx, cy, l, w, alpha_deg):
    """ Helper to calculate rotated rectangle corners for Plotly. """
    alpha = np.radians(alpha_deg)
    
    # Local coordinates of corners relative to center
    # (assuming l is length along local x, w is thickness along local y)
    dx = np.array([l/2, -l/2, -l/2, l/2])
    dy = np.array([w/2, w/2, -w/2, -w/2])
    
    # Rotate and translate
    x_rot = cx + dx * np.cos(alpha) - dy * np.sin(alpha)
    y_rot = cy + dx * np.sin(alpha) + dy * np.cos(alpha)
    
    return x_rot, y_rot


def get_color_from_safety_factor(sf):   
    if np.isnan(sf):
        return "#E5E4E2"  # Neutral light gray
    if sf < 1.0:
        norm = max(0, (sf - 0.5) / 0.5)
        color = px.colors.sample_colorscale(['#4B0000', '#FF0000'], [norm])[0]
    elif sf < 1.5:
        norm = max(0, (sf - 1.0) / 0.5)
        color = px.colors.sample_colorscale(['#FF8C00', '#CCFF00'], [norm])[0]
    else:
        norm = min(1, (sf - 1.5) / 0.5)
        color = px.colors.sample_colorscale(['#00FF00', '#006400'], [norm])[0]
    return color


def add_safety_factor_colorbar(fig, c_min=0.5, c_max=2.0):
    marker=dict(
                colorscale=[
                    [0, '#4B0000'], [0.3333, '#FF0000'],
                    [0.3334, '#FF8C00'], [0.6666, '#CCFF00'],
                    [0.6667, '#00FF00'], [1.0, '#006400']
                ],
                showscale=True, cmin=c_min, cmax=c_max,
                colorbar=dict(
                    title="Safety Factor", thickness=20, outlinewidth=0,
                    tickvals=[0.5, 1.0, 1.5, 2.0],
                    ticktext=["< 0.5 (Crit.)", "1.0 (Lim.)", "1.5 (Warn.)", "> 2.0 (Safe)"]
                )
            )
    fig.add_trace(go.Scatter(
              x=[None], y=[None], mode='markers',
              marker=marker,
              hoverinfo='none', showlegend=False
          ))
    # return marker    


def plot_colored_layout(wall_props, colors, hover_texts, bounds, title=None):
    x_min, x_max, y_min, y_max = bounds
    # pack wall properties for iteration
    wall_prop_iter = zip(wall_props["wall_id"],
                         wall_props["Cx"],
                         wall_props["Cy"],
                         wall_props["L"],
                         wall_props["w"],
                         wall_props["alpha"],
                         colors,
                         hover_texts)
    fig = go.Figure()

    for idx, (w_id, cx, cy, l, w, alpha, col_i, text_i) in enumerate(wall_prop_iter):
        x_v, y_v = _get_wall_vertices(cx, cy, l, w, alpha)
        wall_label = f"Wall {w_id}"

        fig.add_trace(go.Scatter(
            x=np.append(x_v, x_v[0]), y=np.append(y_v, y_v[0]),
            fill="toself", fillcolor=col_i, line=dict(color="black", width=1),
            mode='lines', name=wall_label, text=f"<b>{wall_label}</b><br>{text_i}", hoveron='fills',
            hovertemplate="%{text}<extra></extra>", hoverlabel=dict(namelength=0),
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[x_min-1, x_max+1], title='x [m]', scaleanchor="y", scaleratio=1, gridcolor='lightgray', zerolinecolor='gray'),
        yaxis=dict(range=[y_min-1, y_max+1], title='y [m]', gridcolor='lightgray', zerolinecolor='gray'),
        width=1100, height=800, plot_bgcolor='white'
    )
    return fig
