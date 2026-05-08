from .frame_element import FrameElement
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np

from .plot_utils import(
    plot_colored_layout,
    get_color_from_safety_factor,
    add_safety_factor_colorbar
)


@dataclass
class Floor:
    def __init__(self, id:int, height: float, weight: float):
        self.id = id
        self.height = height
        self.weight = weight
        self.walls: List[FrameElement] = []

        self.floor_alt = None

        # relativ seismic accelerations acting on the floor
        self.acc_rel = None
        self.acc_rel_ort = None

#-------------------------------------------------------------------------------
    @property
    def walls_geometric_properties(self):
      records = []
      for W_i in self.walls:
          records.append({
            "wall_id": W_i.global_id,
            "Cx": W_i.geometry.Cx,  # X coord of center of wall
            "Cy": W_i.geometry.Cy,  # Y coord of center of wall
            "L": W_i.geometry.L,   # Length
            "w": W_i.geometry.w,  # Width
            "alpha": W_i.geometry.alpha  #angle of the wall with x axis
           })
      return pd.DataFrame(records)

#-------------------------------------------------------------------------------

    def add_wall(self, wall: 'FrameElement'):
        self.walls.append(wall)

    def __repr__(self):
        return f"Floor(height={self.height}, walls={len(self.walls)})"

    def set_altitude(self, floor_alt:float):
        self.floor_alt = floor_alt


#--------------------------------------------------------------------------
#  Compute wall safety factors
#--------------------------------------------------------------------------

    def set_relative_accelerations(self, seismic_params:dict, H:float):

        h = self.height
        H_f = self.floor_alt

        # Retrieve seismic parameters
        Se_SLV_T = seismic_params["Se_SLV_T"]
        ag_SLV = seismic_params["ag_SLV"]
        S_geo = seismic_params["S_geo"]
        # q_ort =  seismic_params["q_ort"]
        # q = seismic_params["q"]  # behavior factor EFM.bahvior_factor

        # relative horizontal acceleration
        Se_SLV_ort = S_geo * ag_SLV * (1.5 * (1 + ( H_f - h/2 ) / H) - 0.5) 
        
        self.acc_rel = Se_SLV_T
        self.acc_rel_ort = Se_SLV_ort

#--------------------------------------------------------------------------
#  Compute wall safety factors
#--------------------------------------------------------------------------

    def compute_wall_safety_factors(self, q, q_ort, gamma_m) -> pd.DataFrame:
        """
        Compute safety factors for all walls on this floor and return a DataFrame.
        Each row is a wall, columns are safety factors.
        """
        records = []

        for W_i in self.walls:
            factors = W_i.compute_safety_factors(
                acc_rel = self.acc_rel / q,
                acc_rel_ort = self.acc_rel_ort / q_ort,
            )

            # Prepare a row: wall_id + all safety factors
            row = {}
            row.update(factors)
            records.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(records)
        return df
#--------------------------------------------------------------------------
#  Compute wall ductility limits
#--------------------------------------------------------------------------

    def set_wall_limit_displacement(self, mu=1.5, drift_lim=0.004, method="Ductility"):
        """
        Compute ductility or drift limits for all walls on the floor.
        """

        for W_i in self.walls:
            if method == "Ductility":
              W_i.set_ultimate_displacement_based_on_ductility_limit(mu)
            else:
              W_i.set_ultimate_displacement_based_on_ductility_limit(drift_lim)


    def get_wall_properties(self):
        """
        Compute floor-level properties: KX, KY, center of mass, center of rigidity.
        Returns dictionary of values needed for pushover.
        """
        records = []
        for W_i in self.walls:
            k_i, H_Rd_i = W_i.get_stiffness_and_resistance()
            records.append({
              "wall_id": W_i.id,
              "kx": k_i[0],      #stiffness in dir x
              "ky": k_i[1],       # Stiffness in dir y
              "H_Rd_x": H_Rd_i[0],  # Shear resistance in dir x
              "H_Rd_y": H_Rd_i[1],  # Shear resistance in dir y
              "Cx": W_i.geometry.Cx,  # X coord of center of wall
              "Cy": W_i.geometry.Cy,  # Y coord of center of wall
              "A": W_i.geometry.L * W_i.geometry.w,  # Area of wall cross section
              "sigma": W_i.stress.midheight["SLS"]   # stress SLE limit state
            })
        return pd.DataFrame(records)

    def compute_redistribution_coefficients(self):
        df = self.get_wall_properties()
        # Compute coordinate of center of mass
        SX = (df["sigma"] * df["A"] * df["Cx"]).sum()
        SY = (df["sigma"] * df["A"] * df["Cy"]).sum()
        SS = (df["sigma"] * df["A"] ).sum()
        
        x_M = SX / SS           ##center of mass TX in old code
        y_M = SY / SS           ##center of mass TY in old code
        
        # Compute coordinate of center of rigidity
        K_x = (df["kx"]).sum()       #KXe in the old code
        K_y = (df["ky"]).sum()       #KXe in the old code

        x_R = (df["ky"] * df["Cx"]).sum()/K_y  #center of rigidity, XP in the old code
        y_R = (df["kx"] * df["Cy"]).sum()/K_x  #center of rigidity,YP in the old code

        # Compute eccentrities of the story
        e_x = x_M - x_R                       #EX in the old code
        e_y = y_M - y_R                       #EY in the old code

        # Compute the polar moment of intertia of the stiffness
        Jy = (df["ky"]* ( df["Cx"]-x_R )**2).sum() 
        Jx = (df["kx"]* ( df["Cy"]-y_R )**2).sum()
        J_R = Jx + Jy          # Maybe this should be PM in the code???????
        
        # It does not match what is in the old code, that I can not make up (does not seem to match with paper)
        #X1=np.sum(K0[:,0]*X0[:,1]**2) 
        #Y1=np.sum(K0[:,1]*X0[:,0]**2) 
        #JX[j]=X1-KXel[j]*YP[j]**2 
        #JY[j]=Y1-KYel[j]*XP[j]**2 
        #IY=JY 
        #IX=JX 
        #PM[j]=JX[j]+JY[j]

        # Compute redistribution coefficients for each wall
        #rho_x = 1 + K_y/J_R * e_x * df["Cx"]
        #rho_y = 1 + K_x/J_R * e_x * df["Cx"]
        df["rho_x"] = 1 + (e_y * K_x / J_R) * (df["Cy"] - y_R)
        df["rho_y"] = 1 + (e_x * K_y / J_R) * (df["Cx"] - x_R)

        return df


#--------------------------------------------------------------------------
#  Plot Utils
#--------------------------------------------------------------------------

    def plot_2d_layout(self, bounds):
        wall_props = self.walls_geometric_properties
        color = np.where((wall_props["alpha"] < 45) | (wall_props["alpha"] >= 135), "#00649e", "#ff6600")
        hoover_text = np.where((wall_props["alpha"] < 45) | (wall_props["alpha"] >= 135), "shear wall in dir X", "shear wall in dir Y")
        floor_id = self.id
        title = f'Shear Walls - Floor {floor_id + 1}'
        return plot_colored_layout(wall_props, color, hoover_text, bounds, title=title)

    
    def plot_failing_walls(self, bounds, failing_walls, direction):
        wall_props = self.walls_geometric_properties
        status = np.array([False]*len(wall_props))
        if len(failing_walls) > 0:
           status[failing_walls.astype(int)] = True
        hover_texts = np.where(status, "Failing", "Not failing")
        colors = np.where(status, "#FF0000", "#E5E4E2")
          
        floor_id = self.id
        title = f"Direction {direction} - Floor {floor_id + 1} - Failure"
        # status = wall_props.index.isin(f_set).map({True: "Failing", False: "Not failing"})
        # wall_props, colors, hover_texts, bounds, title=None
        return plot_colored_layout(wall_props, colors, hover_texts, bounds, title=title)


    def plot_safety_factors(self, bounds, sfs, direction, type):
        wall_props = self.walls_geometric_properties
        floor_id = self.id
        title = f"Safety factors for Direction {direction} - Floor {floor_id + 1} - {type}"
        if sfs is None:
          print("No plot is created because safety factors are none")
        else:
          valid_sfs = sfs.dropna()
          if not valid_sfs.empty:
                min_sf = valid_sfs.min()
                title += f" - min. SF = {min_sf:.2f}"
          else:
                title += " - min. SF = N/A"
          colors = sfs.apply(get_color_from_safety_factor)
          hover_texts = "Safety factor: " + sfs.map(lambda x: f"{x:.2f}")
          fig = plot_colored_layout(wall_props, colors, hover_texts, bounds, title)
          # Add colorbar
          add_safety_factor_colorbar(fig)          
        return fig 
