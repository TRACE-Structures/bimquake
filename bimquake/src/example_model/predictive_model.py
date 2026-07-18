import numpy as np
import scipy.linalg as la
import pandas as pd
from SALib.analyze import sobol
from .example_utils import *

class BimquakePredictiveModel:
    def __init__(self, model, Q):
        self.model = model
        self.Q = Q
        self.masonry_idx = self.get_masonry_idx()
        self.D, _, _, self.V, self.alpha, _, self.NZ = self.model.get_wall_data()
        self.N, _, _, _, _, self.Masses, _ = self.model.get_floor_data()
        self.M = 10**3*np.diag(np.flip(self.Masses))
        self.damp=0.05

    def get_masonry_idx(self):
        masonry_idx = []
        for i in range(4):
            idx_ = np.where(np.array(self.model.get_wall_data()[5][i])[:, 2] == 75600)[0]
            masonry_idx.append(idx_)
        return masonry_idx
    
    def predict(self, q):
        q = q.values[0]
        e_stone = q[0]
        e_masonry = q[1]
        r_stone = q[2]
        r_masonry = q[3]

        N = self.N
        NZ = self.NZ


        # Initialize main floor(storey) properties (center of mass, center of rigidity) 
        # KXel=np.zeros(N) #Rigidezza Totale Elastica X
        KYel=np.zeros(N) #Rigidezza Totale Elastica Y
        K = []

        for j in range(N):
            D0=self.D[j] #;%Dimensioni Pareti di Piano
            alpha0=self.alpha[j]
            V0=self.V[j]

            #Stifness of the walls
            K0=np.zeros((NZ[j],2))
            for i in range(NZ[j]):
                if i in self.masonry_idx[j]:
                    r = r_masonry
                    e = e_masonry
                else:
                    r = r_stone
                    e = e_stone
                g = e*r
                theta=alpha0[i]*np.pi/180
                K0[i,0]=g*D0[i,0]*D0[i,1]/(1.2*V0[i]*(1+1/(1.2*e/g)*(V0[i]/D0[i,0])**2))*np.cos(theta)**2
                K0[i,1]=g*D0[i,0]*D0[i,1]/(1.2*V0[i]*(1+1/(1.2*e/g)*(V0[i]/D0[i,0])**2))*np.sin(theta)**2 

            K.append(K0)
            KYel[j]=np.sum(K0[:,1])
        

        if N == 1:
            frequencies_Y_damp = ((KYel * 10**4 / self.M)**0.5) * (1 - self.damp**2)**0.5 / (2 * np.pi)

        elif N > 1:
            KY_mat = np.zeros((N, N))
            for i in range(N):
                if i == 0:
                    KY_mat[i, i] = KYel[i] + KYel[i+1]
                elif i == N-1:
                    KY_mat[i, i] = KYel[i]
                    KY_mat[i, i-1] = -KYel[i]
                else:
                    KY_mat[i, i] = KYel[i] + KYel[i+1]
                    KY_mat[i, i-1] = -KYel[i]
                if i < N-1:
                    KY_mat[i, i+1] = -KYel[i+1]

            KY_mat = 10**4 * KY_mat

            eigenvalues_Y, _ = la.eig(KY_mat, self.M)

            eigenvalues_Y = np.diag(eigenvalues_Y.real)

            eigenvalues_Y = np.diag(eigenvalues_Y)

            frequencies_Y = np.sqrt(eigenvalues_Y)/(2*np.pi)

            frequencies_Y_damp = frequencies_Y*(1-self.damp**2)**0.5

        return frequencies_Y_damp[0]
    
    def compute_partial_vars(self, max_index, QoI_names):
        '''
        Computes partial variances using Sobol sensitivity analysis.

        Parameters
        ----------
        model_obj : LinRegModel
            The linear regression model object.
        max_index : int
            Maximum index for Sobol analysis (1 or 2).

        Returns
        -------
        partial_var_df : pd.DataFrame
            DataFrame containing partial variances for each parameter and QoI.
        sobol_index_df : pd.DataFrame
            DataFrame containing Sobol indices for each parameter and QoI.
        y_var : np.ndarray
            Variance of the model outputs.
        '''

        model_obj = self

        variableset = model_obj.Q

        problem = {
            'num_vars': variableset.num_variables(), 'names': variableset.variable_names(), 'dists': variableset.get_dist_types(), 'bounds': variableset.get_dist_params()
            } 
        
        d = variableset.num_variables()
        q = pd.DataFrame(variableset.sample(method='Sobol_saltelli', n=2048)) # saltelli working only for uniform distribution # N * (2D + 2)
        ys = []
        for i in range(len(q)):
            y = model_obj.predict(pd.DataFrame([q.iloc[i, :]]))
            ys.append(y)

        ys = np.array(ys).reshape(-1, len(QoI_names))
        
        # Run model
        S1 = []
        S2 = []
        for i in range(ys.shape[1]):
            y_i = ys[:, i]

            # Sobol analysis
            Si_i = sobol.analyze(problem, y_i)
            T_Si, first_Si, (idx, second_Si) = sobol.Si_to_pandas_dict(Si_i)
            df = Si_i.to_df()
            cols_S1 = list(df[1].index)
            cols_S2 = list(df[2].index)

            S1.append(first_Si['S1'])
            S2.append(second_Si['S2'])

        S1 = np.array(S1)
        S2 = np.array(S2)

        col_names = cols_S1
        sobol_index = S1
        if max_index == 2:
            sobol_index = np.concatenate([S1, S2], axis=1)
            col_names = cols_S1 + cols_S2
            col_names = [f"{x[0]} {x[1]}" if isinstance(x, tuple) else x for x in col_names]
                    
        # Compute partial variances
        y_var = ys.var(axis=0).reshape(-1, 1)
        partial_variance = sobol_index * y_var
                
        partial_var_df, sobol_index_df = pd.DataFrame(partial_variance, columns=col_names, index=QoI_names), pd.DataFrame(sobol_index, columns=col_names, index=QoI_names)

        return partial_var_df, sobol_index_df, y_var

    def get_sobol_sensitivity(self, QoI_names, max_index=2):
        partial_var_df, sobol_index_df, y_var = self.compute_partial_vars(max_index, QoI_names)
        fig = plot_sobol_sensitivity(QoI_names[0], y_var, partial_var_df, colors={'E_stone': '#1f77b4', 'r_stone': '#ff7f0e', 'E_stone r_stone': '#2ca02c', 'others': '#d62728'}, color_map=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        return sobol_index_df, fig
    
    def update_moduli(self, values):
        N = self.N
        model = self.model
        for i in range(len(model.floors)):
            for j in range(len(model.floors[i].walls)):
                if j in self.masonry_idx[N-i-1]:
                    e = values[1]/100*10**6
                    g = e*values[3]
                    model.floors[i].walls[j].material.E = e
                    model.floors[i].walls[j].material.G = g
                else:
                    e = values[0]/100*10**6
                    g = e*values[2]
                    model.floors[i].walls[j].material.E = e
                    model.floors[i].walls[j].material.G = g
        self.model = model

