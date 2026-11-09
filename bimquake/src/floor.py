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
    """ Class representing a floor in a building model.
    
        Attributes
        ----------
        id : int
            Unique identifier for the floor.
        height : float
            Height of the floor.
        weight : float
            Weight of the floor.
        walls : List[FrameElement]
            List of walls on the floor.
        floor_alt : float
            Altitude of the floor.
        acc_rel : float
            Relative seismic acceleration in the main direction.
        acc_rel_ort : float
            Relative seismic acceleration in the orthogonal direction. """
    
    def __init__(self, id:int, height: float, weight: float):
        """ Initialize a Floor object with the given id, height, and weight.

            Parameters
            ----------
            id : int
                Unique identifier for the floor.
            height : float
                Height of the floor.
            weight : float
                Weight of the floor. """
        
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
      """ Return a DataFrame containing geometric properties of all walls on this floor. 

          Returns
          -------
          records : pd.DataFrame
              DataFrame with columns: wall_id, Cx, Cy, L, w, alpha. """
      
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
        """ Add a wall to the floor.

            Parameters
            ----------
            wall : FrameElement
                The wall to be added to the floor. """
        
        self.walls.append(wall)

    def __repr__(self):
        """ String representation of the Floor object. """

        return f"Floor(height={self.height}, walls={len(self.walls)})"

    def set_altitude(self, floor_alt:float):
        """ Set the altitude of the floor.

            Parameters
            ----------
            floor_alt : float
                Altitude of the floor. """
        
        self.floor_alt = floor_alt


#--------------------------------------------------------------------------
#  Compute wall safety factors
#--------------------------------------------------------------------------

    def set_relative_accelerations(self, seismic_params:dict, H:float):
        """ Set the relative seismic accelerations acting on the floor.

            Parameters
            ----------
            seismic_params : dict
                Dictionary containing seismic parameters.
            H : float
                Total height of the building. """

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

    def compute_wall_safety_factors(self, q, q_ort, gamma_m, stress_location: str="midheight", limit_state: str="SLS") -> pd.DataFrame:
        """ Compute safety factors for all walls on this floor and return a DataFrame. Each row is a wall, columns are safety factors.

            Parameters
            ----------
            q : float
                Behavior factor in the main direction.
            q_ort : float
                Behavior factor in the orthogonal direction.
            gamma_m : float
                Partial safety factor for the material.

            Returns
            -------
            df : pd.DataFrame
                DataFrame containing safety factors for all walls on this floor. """
        
        records = []

        for W_i in self.walls:
            factors = W_i.compute_safety_factors(
                acc_rel = self.acc_rel / q,
                acc_rel_ort = self.acc_rel_ort / q_ort,
                stress_location = stress_location,
                limit_state = limit_state
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

    def set_wall_limit_displacement(self, mu=1.5, drift_lim=0.004, method="Ductility", stress_location: str="midheight", limit_state: str="SLS"):
        """ Compute ductility or drift limits for all walls on the floor.
            
            Parameters
            ----------
            mu : float
                Ductility factor.
            drift_lim : float
                Drift limit.
            method : str
                Method to compute wall limit displacement. Options are "Ductility" or "Drift". """

        for W_i in self.walls:
            if method == "Ductility":
              W_i.set_ultimate_displacement_based_on_ductility_limit(mu, stress_location=stress_location, limit_state=limit_state)
            else:
              W_i.set_ultimate_displacement_based_on_ductility_limit(drift_lim, stress_location=stress_location, limit_state=limit_state)


    def get_wall_properties(self, stress_location: str="midheight", limit_state: str="SLS"):
        """ Compute floor-level properties: KX, KY, center of mass, center of rigidity. Returns dictionary of values needed for pushover.

            Returns
            -------
            records : pd.DataFrame
                DataFrame containing properties of all walls on this floor. """
        
        records = []
        for W_i in self.walls:
            k_i, H_Rd_i = W_i.get_stiffness_and_resistance(stress_location=stress_location, limit_state=limit_state)
            records.append({
              "wall_id": W_i.id,
              "kx": k_i[0],      #stiffness in dir x
              "ky": k_i[1],       # Stiffness in dir y
              "H_Rd_x": H_Rd_i[0],  # Shear resistance in dir x
              "H_Rd_y": H_Rd_i[1],  # Shear resistance in dir y
              "Cx": W_i.geometry.Cx,  # X coord of center of wall
              "Cy": W_i.geometry.Cy,  # Y coord of center of wall
              "A": W_i.geometry.L * W_i.geometry.w,  # Area of wall cross section
              "sigma": W_i.get_stress(stress_location=stress_location, limit_state=limit_state)   # stress SLE limit state
            })
        return pd.DataFrame(records)

    def compute_redistribution_coefficients(self, stress_location: str="midheight", limit_state: str="SLS"):
        """ Compute redistribution coefficients for the walls on this floor.

            Returns
            -------
            df : pd.DataFrame
                DataFrame containing the redistribution coefficients for each wall. """
        
        df = self.get_wall_properties(stress_location=stress_location, limit_state=limit_state)
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
        """ Plot the 2D layout of the walls on this floor.

            Parameters
            ----------
            bounds : tuple
                The bounds of the plot.

            Returns
            -------
            fig : plotly.graph_objects.Figure
                The plotly figure object representing the 2D layout. """
        
        wall_props = self.walls_geometric_properties
        color = np.where((wall_props["alpha"] < 45) | (wall_props["alpha"] >= 135), "#00649e", "#ff6600")
        hoover_text = np.where((wall_props["alpha"] < 45) | (wall_props["alpha"] >= 135), "shear wall in dir X", "shear wall in dir Y")
        floor_id = self.id
        title = f'Shear Walls - Floor {floor_id + 1}'
        fig = plot_colored_layout(wall_props, color, hoover_text, bounds, title=title)
        return fig

    def plot_wall_type_layout(self, bounds, wall_type):
        """ Plot the layout of wall types on this floor.

            Parameters
            ----------
            bounds : tuple
                The bounds of the plot.
            wall_type : array-like
                Array indicating the type of each wall.

            Returns
            -------
            fig : plotly.graph_objects.Figure
                The plotly figure object representing the wall type layout. """
        
        wall_props = self.walls_geometric_properties
        color = np.where(wall_type == 1,"#ff6600", "#00649e")
        hoover_text = np.where(wall_type == 1, "masonry wall", "stone wall")
        floor_id = self.id
        title = f'Wall types - Floor {floor_id + 1}'
        fig = plot_colored_layout(wall_props, color, hoover_text, bounds, title=title)
        return fig
    
    def plot_failing_walls(self, bounds, failing_walls, direction):
        """ Plot the layout of failing walls on this floor.

            Parameters
            ----------
            bounds : tuple
                The bounds of the plot.
            failing_walls : array-like
                Array indicating which walls are failing.
            direction : str
                The direction of the failure.

            Returns
            -------
            fig : plotly.graph_objects.Figure
                The plotly figure object representing the failing walls layout. """
        
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
        fig = plot_colored_layout(wall_props, colors, hover_texts, bounds, title=title)
        return fig


    def plot_safety_factors(self, bounds, sfs, direction, type):
        """ Plot the layout of safety factors for walls on this floor.

            Parameters
            ----------
            bounds : tuple
                The bounds of the plot.
            sfs : pandas.Series
                Series containing the safety factors for each wall.
            direction : str
                The direction of the analysis.
            type : str
                The type of safety factor.

            Returns
            -------
            fig : plotly.graph_objects.Figure
                The plotly figure object representing the safety factors layout. """
        
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
