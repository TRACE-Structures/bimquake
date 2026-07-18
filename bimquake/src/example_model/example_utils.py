import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_wall_types(S):
    wall_types = []
    for i in range(len(S.floors)):
        wall_type = np.zeros(len(S.floors[i].walls))
        for j in range(len(S.floors[i].walls)):
            if S.floors[i].walls[j].material.E != 780*10**6:
                wall_type[j] = 1
        wall_types.append(wall_type)
    return wall_types


def plot_sobol_sensitivity(param_name, y_var, partial_variance, colors, color_map):
    y_var_df = pd.DataFrame(y_var, index=[param_name])

    fig, ax = plt.subplots(figsize=(8, 6))
    threshold = 0.01
    loc_par_var = partial_variance.loc[param_name]
    df = pd.DataFrame(loc_par_var)
    df[df < 0] = 0
    df['percentage'] = df[param_name] / df[param_name].sum()
    under_threshold = df.loc[df['percentage'] < threshold].sum()
    remaining = [y_var_df.loc[param_name][0], 1] - df.sum()
    others = under_threshold + remaining
    colors['others'] = color_map[-1]
    df = df[df['percentage'] >= threshold]
    df.loc['others'] = others
    pie_colors = [colors[x] for x in df.index]

    ax.set_title(f"Sobol sensitivity for: {param_name}", fontsize=16)
    ax.pie(df['percentage'], labels=df.index, colors=pie_colors, wedgeprops={"alpha": 0.5})
    return fig

