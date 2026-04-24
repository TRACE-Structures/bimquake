from os import error
import numpy as np
import pandas as pd
import warnings
from typing import Optional
import pushover_utils as pushover_utils
import coordinates as coords


from dataclasses import dataclass, field
from typing import List, Dict


import numpy as np
import pandas as pd
from typing import List

def create_equivalent_frame_model_from_excel(file_name):
  # TODO: Write function to convert wall properties to dictionary of dataframes
  # from excile file, and numpy array or list with heights and weightss:
  #  props = 
  # floor_heights = 
  # floor_weights =
  # return create_equivalent_frame_model_from_dataframes(props, floor_heights, floor_weights)

  return None

def create_equivalent_frame_model_from_dataframes(
  props_dict,  # Dictionary with keys 1,2,3... (floor numbers) and values of dataframes
  floor_heights,  #numpy or list with heights in m
  floor_weights   #numpy or list with weights in kN
  ) -> 'EquivalentFrameModel':
    """
    Create an EquivalentFrame instance from a list of pandas DataFrames (one per floor).
    Assumes the following columns in each DataFrame:
        ["Wall", "L [m]", "w [m]", "H [m]", "Cx [m]", "Cy [m]", "α",
         "σ [N/mm²]", "τ [N/mm²]", "fₘ [N/mm²]", "γ [kN/m³]", "E [N/mm²]", "G [N/mm²]"]
    Floor heights are given in [m], weights in [kN]
    All stresses will be converted to N/m² (Pa), densities to N/m³.
    """
    eq_frame = EquivalentFrameModel()

    for i, df in props_dict.items():   #loop over floor
        floor = Floor(id=i, height=floor_heights[i], weight=floor_weights[i]*1000)

        for _, row in df.iterrows():
            # Convert units to SI
            sigma = row["σ [N/mm²]"] * 1e6  # N/mm² → N/m²
            tau = row["τ [N/mm²]"] * 1e6    # N/mm² → N/m²
            f_m = row["fₘ [N/mm²]"] * 1e6   # N/mm² → N/m²
            gamma = row["γ [kN/m³]"] * 1000 # kN/m³ → N/m³
            E = row["E [N/mm²]"] * 1e6      # N/mm² → N/m²
            G = row["G [N/mm²]"] * 1e6      # N/mm² → N/m²

            # Create material properties
            material = MaterialProperties(tau=tau, f_m=f_m, gamma=gamma, E=E, G=G)

            # Geometry
            geometry = GeometricProperties(
              L= row["L [m]"], 
              w = row["w [m]"], 
              h = row["H [m]"], 
              Cx = row["Cx [m]"], 
              Cy = row["Cy [m]"], 
              alpha = row["α"]
              )

            # Stress state: 
            stress = StressState(
                top={"ULS": None, "SLS": None},
                midheight={"ULS": None, "SLS": sigma}
            )

            # Create the frame element
            frame_elem = FrameElement(
                id=int(row["Wall"]),
                global_id=f"F{i}_W{int(row['Wall'])}",
                floor=i,
                material=material,
                geometry=geometry,
                stress=stress,
                k=None,     # To be computed later
                H_Rd=None   # To be computed later
            )

            floor.add_wall(frame_elem)

        eq_frame.floors.append(floor)

    return eq_frame
           
###############################################################################
###############################################################################
# Material properties
###############################################################################
###############################################################################
@dataclass
class MaterialProperties:
    E: float        # Young modulus
    G: float        # Shear modulus
    tau: float      # Shear strength
    f_m: float       # Masonry compressive strength
    gamma: float   # Density [N/m3] 
    nu: Optional[float] = None  # Poisson ratio
    mu: Optional[float] = None  # Friction coefficients
    f_u: Optional[float] = None  # tensile strength
      

###############################################################################
###############################################################################
# Geometrical properties
###############################################################################
###############################################################################
@dataclass
class GeometricProperties:
    L: float                       # Length
    w: float                       # Thickness
    h: float                       # Height
    Cx: float                      # Centroid x
    Cy: float                      # Centroid y
    alpha: float                   # Orientation angle
    points: Optional[np.ndarray]=None  # expected shape (8,3) - Vertext point coordinates


###############################################################################
###############################################################################
# Stress state
###############################################################################
###############################################################################
@dataclass
class StressState:
    top: Dict[str, float] = field(default_factory=dict)   #str: expected options "ULS" or "SLS"
    midheight: Dict[str, float] = field(default_factory=dict)  #str: expected options "ULS" or "SLS"
    #bottom: float


###############################################################################
###############################################################################
# Linkage
###############################################################################
###############################################################################
@dataclass
class Linkage:
    element: "FrameElement"   # linked element
    contact_length: float     # length of contact

###############################################################################
###############################################################################
# Linkages
###############################################################################
###############################################################################
@dataclass
class Linkages:
    connections: List[Linkage] = field(default_factory=list)
#------------------------------------------------------------------------------
    def add(self, link):
        self.connections.append(link) 

    def __add__(self, other):
        if not isinstance(other, Linkages):
            return NotImplemented
        return Linkages(self.union(other))

    def __getitem__(self, index):
        return self.connections[index]

    def __len__(self):
        return len(self.connections)

    def __iter__(self):
        return iter(self.connections)


###############################################################################
###############################################################################
# Frame element (wall)
###############################################################################
###############################################################################
@dataclass
class FrameElement:
    id: int
    global_id: str
    floor: int 
    material: MaterialProperties
    geometry: GeometricProperties
    stress: StressState
    supported_by: Linkages = field(default_factory=Linkages)
    loaded_by: Linkages = field(default_factory=Linkages)
    delta_u_duct: np.ndarray = None  # max diplacement based on ductility 2 values (X,Y)
    delta_u_drift: np.ndarray = None # max diplacement based on drift 2 values (X,Y)
    k: np.ndarray = None # lateral stiffness of the wall in direction x and y 
    H_Rd: np.ndarray = None # Shear resistace of the wall in direction x and y 

###############################################################################
# Methods for linear static analysis
###############################################################################

    #--------------------------------------------------------------------------
    #  Compute resistance
    #--------------------------------------------------------------------------

    def compute_resistances(self, gamma_m: float=2):
        """Compute shear and bending resistances for this wall."""
        # gamma_m is the safety factor (resistance is computed resistance divided by gamma_m)
        # stress in the midsection of the wall
        sigma = self.stress.midheight['SLS']        #????SLE?
        f_m = self.material.f_m
        tau = self.material.tau
        L, w = self.geometry.L, self.geometry.w
        A = L * w

        # Shear resistance
        V_Rd = A * tau / gamma_m * np.sqrt(1 + sigma / (1.5 * tau / gamma_m))

        # In-plane bending
        M_Rd = (L**2 * w * sigma / 2) * (1 - sigma / (0.85 * f_m / gamma_m))
        M_Rd = max(M_Rd, 0)

        # Out-of-plane bending
        M_Rd_ort = (L * w**2 * sigma / 2) * (1 - sigma / (0.85 * f_m / gamma_m))
        M_Rd_ort = max(M_Rd_ort, 0)

        return V_Rd, M_Rd, M_Rd_ort

    #--------------------------------------------------------------------------
    #  Compute safety factors
    #--------------------------------------------------------------------------

    def compute_safety_factors(self, Se_SLV_T: float, q: float, q_ort: float,
                                floor_alt: float, total_height: float):
        """Compute safety indexes for this wall using its own properties."""
        sigma = self.stress.midheight['SLS']
        gamma = self.material.gamma
        H = total_height
        L, w, h = self.geometry.L, self.geometry.w, self.geometry.h          #Do I need the floor height or is it fine to use height of the wall?
        alpha = self.geometry.alpha

        # Resistances
        V_Rd, M_Rd, M_Rd_ort = self.compute_resistances()

        # Demands
        T_Ed = Se_SLV_T / q * sigma * L * w
        M_Ed = Se_SLV_T / q * ((sigma - gamma * h) * L * w * h / 2 + gamma * h * L * w * h / 3)

        # Acceleration to be used for out of plane
        Se_SLV_ort = Se_SLV_T * (1.5 * (1 + (floor_alt-h/2)/H) - 0.5)   # !!!! Here a comparison is missing:
        M_Ed_ort = Se_SLV_ort / q_ort * ((sigma - gamma * h) * L * w * h / 2 + gamma * h * L * w * h / 3)

        # Assign to X or Y based on orientation
        if alpha < 45 or alpha >= 135:
            return {
                "Wall_ID": self.id,        #????? which Id should we use here
                "SF_VX": V_Rd / T_Ed,
                "SF_MX": M_Rd / M_Ed,
                "SF_MOX": M_Rd_ort / M_Ed_ort
            }
        else:
            return {
                "Wall_ID": self.id,         #????? which Id should we use here
                "SF_VY": V_Rd / T_Ed,
                "SF_MY": M_Rd / M_Ed,
                "SF_MOY": M_Rd_ort / M_Ed_ort
            }

###############################################################################
# Methods for pushover analysis
###############################################################################

#--------------------------------------------------------------------------
#  Set-get ultimate displacement of the wall
#--------------------------------------------------------------------------
    def set_ultimate_displacement_based_on_ductility_limit_alt(self, mu=1.5):
        k, H_Rd = self.get_stiffness_and_resistance()
        d_e = H_Rd/k
        return mu*d_e
       


    def set_ultimate_displacement_based_on_ductility_limit(self, mu=1.5):          # THIS could be directly computed from kx/H_rdx and k_y/H_rdy, isn't it?
        """
        Set hoirzontal displacement limit of the wall based on  
        ductility (limiting delta_ult/delta_elastic) limit
        Sets ultimate displacement arrays of size (2,) - dir x and y.
        mu is the ductility factor delta_u = mu*delta_e
        """

        L = self.geometry.L                                                    
        w = self.geometry.w
        h = self.geometry.h
        alpha = self.geometry.alpha

        sigma = self.stress.midheight["SLS"]

        tau = self.material.tau
        E = self.material.E
        G = self.material.G

        a = G / tau
        b = E / G
       
       # Initiate vector for displacement limit
        d_u = np.zeros(2)
        
        for j in range(2):         # for x an y direction
            if alpha < 45 or alpha >= 135:
                l = L
            else:
                l = w if j == 0 else l
            d_e = np.sqrt(1 + sigma / (1.5*tau) )* 1.2/a *h * (1 + 1/(1.2*b)*(h/l)**2)
            d_u[j] = mu * d_e

        self.delta_u_duct = d_u
        return d_u

    def get_ultimate_diplacement_based_on_ductility_limit(self):
        d_u = self.delta_u_duct
        if d_u is None:
          d_u = self.set_ultimate_displacement_based_on_ductility_limit()
        return d_u

    def get_ultimate_displacement(self, method="ductility"):
        if method == "ductility":
          return self.get_ultimate_diplacement_based_on_ductility_limit()
        elif method == "drift":
          return self.get_ultimate_diplacement_based_on_drift_limit()
        else:
          warnings.warn(
            f"Unknown method '{method}'. Defaulting to 'ductility'.",
            UserWarning
            )
          return self.get_ultimate_diplacement_based_on_ductility_limit()
          
    def set_ultimate_displacement_based_on_drift_limit(self, drift_lim=0.004):
        """
        Compute hoirzontal displacement limit of the wall based on  
        drift (limiting delta_ult/height) limits.
        Sets ultimate displacement arrays of size (2,) - dir x and y.
        mu is the ductility factor delta_u = mu*delta_e
        """
        h = self.geometry.h
        d_u = drift_lim * np.ones(2)
        self.delta_u_drift = d_u
        return d_u

    def get_ultimate_diplacement_based_on_drift_limit(self):
        d_u = self.delta_u_ddrift
        if d_u is None:
          d_u = self.set_ultimate_displacement_based_on_drift_limit()
        return d_u

#--------------------------------------------------------------------------
#  Set-get stiffness values
#--------------------------------------------------------------------------

    def set_stiffness_and_resistance(self):
        """
        Compute wall stiffness in 'X' or 'Y' direction for pushover analysis.
        Returns k (float) and effective horizontal force H0 (float).
        """
        # Wall's geometric properties
        L = self.geometry.L 
        w = self.geometry.w 
        h = self.geometry.h  
        theta = np.radians(self.geometry.alpha)

        # Wall's material properties
        tau = self.material.tau
        G = self.material.G 
        E = self.material.E

        sigma = self.stress.midheight["SLS"]

        K0 = G * L * w / (1.2 * h * (1 + G/(1.2*E)*(h/L)**2))

        # Stiffness and resistance in direction X
        k_x = K0 * np.cos(theta)**2 
        H_Rd_x = L * w * tau * np.sqrt(1 + sigma/(1.5*tau))
      # Stiffness of the wall
        k_y = K0 * np.sin(theta)**2
        # Resistance of the wall
        H_Rd_y = L * w * tau * np.sqrt(1 + sigma/(1.5*tau))
        self.k = np.array([k_x, k_y])
        self.H_Rd = np.array([H_Rd_x, H_Rd_y])
        return self.k, self.H_Rd

    def get_stiffness_and_resistance(self):
        k = self.k
        H_Rd = self.H_Rd
        if k is None or H_Rd is None:
          k, H_Rd = self.set_stiffness_and_resistance()
        return k, H_Rd


###############################################################################
###############################################################################
# Set of frame elements
###############################################################################
###############################################################################
#@dataclass
#class FrameElementSet:
#    elements: List[FrameElement] = field(default_factory=list)
#
#    def add_element(self, element: FrameElement):
#        self.elements.append(element)
#
#    def __iter__(self):
#        return iter(self.elements)

    
###############################################################################
###############################################################################
# Floor
###############################################################################
###############################################################################
@dataclass
class Floor:
    def __init__(self, id:int, height: float, weight: float):
        self.id = id
        self.height = height
        self.weight = weight
        self.walls: List[FrameElement] = []

#-------------------------------------------------------------------------------

    def add_wall(self, wall: 'FrameElement'):
        self.walls.append(wall)

    def __repr__(self):
        return f"Floor(height={self.height}, walls={len(self.walls)})"

#--------------------------------------------------------------------------
#  Compute wall safety factors
#--------------------------------------------------------------------------

    def compute_wall_safety_factors(self, Se_SLV_T: float, q: float, q_ort: float,
                             floor_alt: float, total_height: float) -> pd.DataFrame:
        """
        Compute safety factors for all walls on this floor and return a DataFrame.
        Each row is a wall, columns are safety factors.
        """
        records = []

        for W_i in self.walls:
            factors = W_i.compute_safety_factors(
                Se_SLV_T=Se_SLV_T,
                q=q,
                q_ort=q_ort,
                floor_alt=floor_alt,
                total_height=total_height
            )

            # Prepare a row: wall_id + all safety factors
            row = {}
            row.update(factors)
            records.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(records)
        return df

###############################################################################
###############################################################################
# Equivalent frame model
###############################################################################
###############################################################################
class EquivalentFrameModel:
      def __init__(self):
        self.floors: List[Floor] = []
         # --- location ---
        self.latitude = None
        self.longitude = None

        # --- seismic / design parameters ---
        self.soil_category = None
        self.service_life = None
        self.topographic_category = None
        self.importance_class = None
        self.behaviour_factor = None

        # seismic loading
        self.paraTR = None

#------------------------------------------------------------------------------        
      @property
      def num_floors(self) -> float:
          return len(self.floors)

      @property
      def total_height(self)-> float:
          return sum(f_i.height for f_i in self.floors)

      @property
      def floor_heights(self)-> List:
          return [f_i.height for f_i in self.floors]

      @property 
      def floor_heights_from_bottom(self) -> np.ndarray:   # alt_s
       hs = self.floor_heights
       return np.cumsum(hs)       

      @property
      def masses(self)-> np.ndarray:         
        weights_r = [f_i.weight for f_i in self.floors]
        masses = np.array(weights_r)/9.81
        return masses

      @property
      def red_F(self) -> np.ndarray:    #distribution factor from top to bottom to distribute base shear force, sums to 1 
        alt_s = self.floor_heights_from_bottom
        Ms = self.masses
        red_F = Ms*alt_s/np.dot(alt_s, Ms)  #Ms*alt /\sum (Ms*alt)  #is this reversed?????
        return red_F

#--------------------------------------------------------------------------
#  Set properties
#--------------------------------------------------------------------------

      def set_location(self, latitude: float, longitude: float):
          self.latitude = latitude
          self.longitude = longitude
          self.paraTR, _, _ = coords.get_Parameters(latitude, longitude)

      def set_seismic_design_parameters(
          self,
          soil_category: str,
          nominal_life: float,
          topographic_category: str,
          importance_class: str,
          behaviour_factor: float
        ):
          """Set seismic design parameters."""
          self.soil_category = soil_category
          self.service_life = nominal_life                      #???????
          self.topographic_category = topographic_category
          self.importance_class = importance_class
          self.behaviour_factor = behaviour_factor

#--------------------------------------------------------------------------
#  Add floor, floor walls
#--------------------------------------------------------------------------

      def add_floor(self, floor: Floor):
          self.floors.append(floor)

      def add_wall_to_floor(self, floor_index: int, wall: 'FrameElement'):
          if floor_index < 0 or floor_index >= len(self.floors):
              raise IndexError("Floor index out of range")
          self.floors[floor_index].add_wall(wall)

#--------------------------------------------------------------------------
#  Run Linear Static Analysis
#--------------------------------------------------------------------------

      def run_linear_static_analysis(self, gamma_m=2):
          #-------
          H = self.total_height   #instead of hTOT      
          q = self.behaviour_factor
          q_ort = 3
          # safety factor on the resistance side
          # gamma_m = 2
          # Get local seismic loading
          Se_SLV_T = self.get_seismic_loading()
          # Get height of the floors (re)
          alt_s = self.floor_heights_from_bottom
          
          # Initiate list of dataframes with safety factors 
          SFs = []
          for j, floor in enumerate(self.floors):
            SFs_j = floor.compute_wall_safety_factors(
                Se_SLV_T=Se_SLV_T,   #accelearation
                q=q,                 # behaviour factor
                q_ort=q_ort,
                floor_alt=alt_s[j],      #?
                total_height = H
            )
            SFs.append(SFs_j)
          return SFs
      
      #--------------------------------------------------------------------------
      #  Run pushover analysis
      #--------------------------------------------------------------------------

      def run_pushover_analysis(self, method="Ductility", algorithm="incr", mu=1.5):
          k_ult_TOT, Hult_TOT, vr_ult_TOT, L_organized = pushover_utils.pushover_analysis_calculation(self, algorithm, method, mu)
          return k_ult_TOT, Hult_TOT, vr_ult_TOT, L_organized
      
      def get_pushover_dataframe(self, k_ult_TOT, Hult_TOT, vr_ult_TOT):
          dataframe_results = pushover_utils.return_pushover_dataframe(self, k_ult_TOT, Hult_TOT, vr_ult_TOT)
          return dataframe_results
      
      def get_capacity_plot(self, Hult_TOT, vr_ult_TOT):
          fig = pushover_utils.plot_capacity_curve(vr_ult_TOT, Hult_TOT)
          return fig
      
      def get_ADRS_plot(self, k_ult_TOT, Hult_TOT, vr_ult_TOT):
           fig = pushover_utils.plot_ADRS_curve(self, k_ult_TOT, Hult_TOT, vr_ult_TOT)
           return fig

      def get_floor_data(self):
        N = len(self.floors)
        floorID = ["Floor{}".format(floor.id+1) for floor in self.floors]
        alt = np.array([floor.height for floor in reversed(self.floors)])
        alt_s = np.zeros(N)

        for i in range(N):
            alt_s[i] = np.sum(alt[i:])

        hTOT = np.sum(alt)
        Masses = np.array([floor.weight for floor in reversed(self.floors)]) / 10000
        # Masses = np.array(list(reversed(xlsx.parse(0)['W [kN]'].values))) / 10

        denominator = np.dot(alt_s, Masses)
        red_F=Masses*alt_s/denominator

        return N, floorID, alt, alt_s, hTOT, Masses, red_F
    

      def get_wall_data(self):
        N = len(self.floors)
        D = []
        X = []
        S = []
        V = []
        alpha = []
        G = []
        NZ = []

        df = self.get_wall_initial_properties()
        for k in range(N):
            D.append(np.array([df[N-k-1]['L [m]'], df[N-k-1]['w [m]']]).T)
            X.append(np.array([df[N-k-1]['Cx [m]'], df[N-k-1]['Cy [m]']]).T)
            S.append(df[N-k-1]['σ [N/mm²]'] * 100)
            V.append(df[N-k-1]['H [m]'])
            alpha.append(df[N-k-1]['α'])
            # mud.append(df[N-k-1]['μ'])
            G.append(np.array([df[N-k-1]['G [N/mm²]'] *100, df[N-k-1]['τ [N/mm²]'] *100, df[N-k-1]['E [N/mm²]'] *100, df[N-k-1]['fₘ [N/mm²]'] *100, df[N-k-1]['γ [kN/m³]'] / 10]).T)
            NZ.append(len(D[k]))
        return D, X, S, V, alpha, G, NZ

#--------------------------------------------------------------------------
#  Get Seismic Loading
#--------------------------------------------------------------------------
      def is_seismic_loading_input_valid(self, ParaTR):
        #TODO check whether necessary data is available
          return True

      
      def set_seismic_loading(self, ParaTR=None):
          if not None in (self.latitude, self.longitude) and ParaTR==None:
            ParaTR, _, _ = coords.get_Parameters(self.latitude, self.longitude)
          else:
            if ParaTR is None:
              error("Either first set latitude and longitude, or give valid paraTR table")
            elif not self.is_seismic_loading_input_valid(ParaTR):
              error("ParaTR seismic loading table in is not valid")
          self.ParaTR = ParaTR

      
      def get_seismic_loading(self):
          ParaTR = self.paraTR
          _, TrSLV = pushover_utils._get_Vr_and_Tr(self.service_life, self.importance_class)
        #   ParaTR = ParaTR.values[:, 1:]
          params = pushover_utils._iterate_return_periods_hazard_params(ParaTR, TrSLV)
          S_Sgeo, _ = pushover_utils._get_Sgeo_and_Cc(self.soil_category, params)
          S_t = pushover_utils._get_S_t(self.topographic_category)

          S_geo=S_t*S_Sgeo
          ag_SLV, p_1, p_2 = params
          TC_SLV = (p_2**(-0.2))*1.1*p_2
          TD = ag_SLV/9.81*4+1.6
          H = self.total_height   #instead of hTOT      
          T = (H**(3/4))*0.05
          if T<TC_SLV/3:
              Se_SLV_T=ag_SLV*S_geo*p_1*(T/(TC_SLV/3)+1/p_1*(1-T/(TC_SLV/3)))
          elif T>TC_SLV/3 and T<TC_SLV:
              Se_SLV_T=ag_SLV*S_geo*p_1
          elif T>TC_SLV and T<TD:
              Se_SLV_T=ag_SLV*S_geo*p_1*TC_SLV/T
          elif T>TD:
              Se_SLV_T=ag_SLV*S_geo*p_1*TC_SLV*TD/T
          return Se_SLV_T
      
      def get_wall_initial_properties(self):
          # initiate list of dataframes for the different floors
          prop_dfs = {}
          # Loop over floors from top to bottom
          for i, floor in enumerate(self.floors):
            walls = floor.walls
            records = []
            for W_j in walls:              
              record = {}
              record['Wall'] = W_j.id
              record["GlobalId"] = W_j.global_id
              record["L [m]"] = W_j.geometry.L
              record["w [m]"] = W_j.geometry.w
              record["H [m]"] = W_j.geometry.h
              record["Cx [m]"] = W_j.geometry.Cx
              record["Cy [m]"] = W_j.geometry.Cy
              record["α"] = W_j.geometry.alpha
              record["γ [kN/m³]"] = W_j.material.gamma /1e3
              sigma_SLS = W_j.stress.midheight["SLS"]
              record["σ [N/mm²]"] = sigma_SLS/1e6 if sigma_SLS is not None else None
              # Wall elastic moduli
              record["E [N/mm²]"] = W_j.material.E / 1e6 if W_j.material.E is not None else None
              # wall's ShearModulus
              record["G [N/mm²]"] = W_j.material.G / 1e6 if W_j.material.G is not None else None
              # wall's TensileStrength
              # record["f_u [N/mm²]"] = W_j.material.f_u/1e6 if W_j.material.f_u is not None else None
              # wall's ShearStrength
              record["τ [N/mm²]"] = W_j.material.tau/1e6 if W_j.material.tau is not None else None
              # wall's compressive strength
              record["fₘ [N/mm²]"] = W_j.material.f_m/1e6 if W_j.material.f_m is not None else None
              # wall's PoissonRatio
              record["nu"] = W_j.material.nu
              # Add record to list
              records.append(record)
            df = pd.DataFrame(records).sort_values("Wall")
            df = df[["Wall", "L [m]", "w [m]", "H [m]", "Cx [m]", "Cy [m]", 
              "α", "σ [N/mm²]", "τ [N/mm²]", "fₘ [N/mm²]",
              "γ [kN/m³]", "E [N/mm²]", "G [N/mm²]"]]
            prop_dfs[i] = df

          return prop_dfs

      def __repr__(self):
          return f"EquivalentFrame(floors={self.num_floors})"